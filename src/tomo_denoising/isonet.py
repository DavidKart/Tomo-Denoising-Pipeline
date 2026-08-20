"""Steps 4 and 5: IsoNet training and prediction.

``train_isonet`` performs the whole preparation and training chain:

  0. CTFFIND4 defocus estimate on the first collected tilt of each series
     (cached in ``isonet_config/<tomo>_PS.txt``)
  1. ``isonet.py prepare_star``  + per-tomogram defocus / tilt range patching
  2. ``isonet.py denoise``       noise2noise denoiser
  3. ``isonet.py predict``       apply the denoiser (needed for the masks)
  4. ``isonet.py make_mask``     masks from the denoised volumes
  5. ``isonet.py refine``        missing-wedge network

``predict_isonet`` is the final step: apply the refined network to the
prediction set. Its output - the only directory holding the finished,
deliverable tomograms - is ``isonet/corrected_tomos_final/`` (see the output
layout below ``DENOISE_DIR`` and friends).
"""

import glob
import os
import re

from .ctffind import run_ctffind
from .mdoc import collect_tilt_series
from .steplog import StepLog, step_completed
from .utils import (
    banner,
    ensure_dir,
    info,
    link_file,
    require_executable,
    run,
    warn,
    gpu_flag_isonet,
)

# ------------------------------------------------------ isonet output layout
#
# Passed explicitly as --output_dir / --output_prefix on every isonet.py call
# below, rather than left to isonet.py's own defaults: two different calls
# (the intermediate predict needed only for make_mask, and the final predict
# on the refined network) both invoke ``isonet.py predict``, and without an
# explicit, distinct --output_dir they would land in the very same directory
# under the same names - the final result silently indistinguishable from,
# and liable to mix with, the intermediate one.
#
#   <processing_directory>/isonet/
#     denoise/                    noise2noise networks         (train step 2)
#     predict_denoise_only/       denoise_*.mrc - intermediate, feeds
#                                  make_mask ONLY, not a deliverable
#     mask/                       masks from the denoised volumes (step 4)
#     isonet_maps/                missing-wedge networks        (step 5)
#     corrected_tomos_final/      final_*.mrc - THE deliverable: denoised AND
#                                  missing-wedge corrected
DENOISE_DIR = "denoise"
DENOISE_PREDICT_DIR = "predict_denoise_only"
DENOISE_PREDICT_PREFIX = "denoise_"
MASK_DIR = "mask"
REFINE_DIR = "isonet_maps"
FINAL_PREDICT_DIR = "corrected_tomos_final"
FINAL_PREDICT_PREFIX = "final_"


# ------------------------------------------------------------------ half maps


def _linked_half_maps(config, series_list, subset):
    """Link the half maps of *series_list* into ``isonet/<subset>/{even,odd}``.

    Both halves are linked under the SAME base name (``<tomo>.mrc``) so that
    IsoNet pairs them unambiguously regardless of how it matches the two
    directories. Linking avoids copying tens of gigabytes and never touches
    the originals in ``raw_halfMaps``.

    The returned list is sorted by tomogram name, which is the order
    ``prepare_star`` sees the linked files in - the per-tomogram ``--defocus``
    list must be built in exactly this order.
    """
    even_dir = ensure_dir(config.isonet_dir, subset, "even")
    odd_dir = ensure_dir(config.isonet_dir, subset, "odd")

    usable = []
    for series in sorted(series_list, key=lambda s: s.name):
        even = os.path.join(config.raw_halfmaps_dir,
                            "even_%s_rec_tomo.mrc" % series.name)
        odd = os.path.join(config.raw_halfmaps_dir,
                           "odd_%s_rec_tomo.mrc" % series.name)
        if not (os.path.exists(even) and os.path.exists(odd)):
            warn("no half maps for %s - excluded from IsoNet." % series.name)
            continue
        link_file(even, os.path.join(even_dir, series.name + ".mrc"))
        link_file(odd, os.path.join(odd_dir, series.name + ".mrc"))
        usable.append(series)

    if not usable:
        raise SystemExit(
            "ERROR: none of the selected tomograms have half maps in %s.\n"
            "Run 'tomo_denoising create_halfMaps' first."
            % config.raw_halfmaps_dir
        )
    return even_dir, odd_dir, usable


# ---------------------------------------------------------------- tilt range


def tilt_range(config, tomo_name):
    """Minimum and maximum tilt angle from ``rec_<tomo>/<tomo>.rawtlt``."""
    rawtlt = os.path.join(config.rec_dir(tomo_name), tomo_name + ".rawtlt")
    if not os.path.isfile(rawtlt):
        warn("no .rawtlt for %s (expected %s)" % (tomo_name, rawtlt))
        return None
    angles = []
    with open(rawtlt, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                angles.append(float(line.split()[0]))
            except ValueError:
                continue
    if not angles:
        warn("%s is empty" % rawtlt)
        return None
    return min(angles), max(angles)


# ------------------------------------------------------------- gather metadata


def gather_metadata(config, series_list, force_ctf=False, dry_run=False):
    """Per-tomogram defocus and tilt range for the IsoNet star file."""
    metadata = {}
    for series in series_list:
        defocus = run_ctffind(config, series, force=force_ctf, dry_run=dry_run)
        tilts = tilt_range(config, series.name)
        if defocus is None or tilts is None:
            warn("%s: incomplete metadata (defocus=%s, tilt range=%s) - "
                 "global values will be used for this tomogram."
                 % (series.name, defocus, tilts))
        metadata[series.name] = {
            "defocus": defocus,
            "tilt_min": tilts[0] if tilts else None,
            "tilt_max": tilts[1] if tilts else None,
        }
    return metadata


#: Tilt ranges closer than this are treated as identical (rounding in mdocs).
TILT_TOLERANCE = 2.0


def resolve_tilt_range(config, usable, metadata):
    """Pick the single tilt range ``prepare_star`` will be given.

    ``prepare_star`` accepts one ``--tilt_min``/``--tilt_max`` for the whole
    set, so tomograms whose ranges differ have to be reconciled. The WIDEST
    range is used, because of how the two errors fail:

    * declaring a range wider than a tomogram really has -> IsoNet believes
      the outer shells contain data, leaves them alone, and that tomogram is
      merely under-corrected;
    * declaring a range narrower than it really has -> IsoNet believes those
      shells are missing and synthesises them, overwriting measured data.

    The first is recoverable, the second destroys information, so the union
    is the safe default. Outliers are reported by name so the mismatch is
    never silent.

    This is only what ``prepare_star`` is called with - ``prepare_star``
    afterwards corrects ``tomograms.star`` back to a per-tomogram range read
    from each tomogram's own .aln (see ``_edit_star_tilt_range``), UNLESS
    ``isonet.fix_tilt_range`` overrides the whole calculation, in which case
    the fixed range returned here is what ends up in the star file too.

    Returns ``(tilt_min, tilt_max, is_override)``.
    """
    override = config["isonet"].get("fix_tilt_range")
    if override:
        low, high = float(override[0]), float(override[1])
        info("isonet.fix_tilt_range override: %.1f to %.1f deg "
             "(used for every tomogram, unedited)" % (low, high))
        return low, high, True

    ranges = {s.name: (metadata[s.name]["tilt_min"], metadata[s.name]["tilt_max"])
              for s in usable
              if metadata[s.name]["tilt_min"] is not None}
    if not ranges:
        warn("no .rawtlt files found - falling back to -60 to 60 deg. "
             "Set isonet.fix_tilt_range in your configuration.")
        return -60.0, 60.0, False

    minima = [low for low, _ in ranges.values()]
    maxima = [high for _, high in ranges.values()]
    tilt_min, tilt_max = min(minima), max(maxima)

    spread_low = max(minima) - min(minima)
    spread_high = max(maxima) - min(maxima)
    if spread_low <= TILT_TOLERANCE and spread_high <= TILT_TOLERANCE:
        info("tilt range %.1f to %.1f deg (consistent across %d tomograms)"
             % (tilt_min, tilt_max, len(ranges)))
        return tilt_min, tilt_max, False

    warn("tilt ranges differ across %d tomograms:" % len(ranges))
    outliers = []
    for name in sorted(ranges):
        low, high = ranges[name]
        narrower_low = low - tilt_min
        narrower_high = tilt_max - high
        if narrower_low > TILT_TOLERANCE or narrower_high > TILT_TOLERANCE:
            outliers.append(name)
            warn("    %-24s %6.1f to %6.1f   <- %.1f/%.1f deg narrower"
                 % (name, low, high, narrower_low, narrower_high))
        else:
            warn("    %-24s %6.1f to %6.1f" % (name, low, high))
    warn("Using the widest range for prepare_star: %.1f to %.1f deg."
         % (tilt_min, tilt_max))
    warn("%s would be under-corrected by that range alone, because IsoNet "
         "would assume tilt data they do not have - but tomograms.star gets "
         "each tomogram's own range from its .aln right after, so this only "
         "matters for the call to prepare_star itself." % ", ".join(outliers))
    return tilt_min, tilt_max, False


# --------------------------------------------------------- per-tomogram .aln


def _aln_tilt_range(config, tomo_name):
    """Actual min/max of the TILT column in ``rec_<tomo>/aretomo_full/*.aln``.

    This is the ground truth for what AreTomo2 actually used: dark tilts it
    dropped are not in this file at all, and if ``aretomo.tilt_axis_correction``
    is on, TILT holds the corrected angle rather than the nominal one from the
    .rawtlt. TILT is always the last column of a data row, so it is read
    positionally from the end rather than by counting columns from the start.
    """
    search_dir = os.path.join(config.rec_dir(tomo_name), "aretomo_full")
    candidates = sorted(glob.glob(os.path.join(search_dir, "*.aln")))
    if not candidates:
        return None
    preferred = os.path.join(search_dir, tomo_name + "_stacked.aln")
    aln_path = preferred if preferred in candidates else candidates[0]

    angles = []
    with open(aln_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            try:
                angles.append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
    if not angles:
        return None
    return min(angles), max(angles)


_STAR_COLUMN_RE = re.compile(r"^_(\S+)\s+#(\d+)")


def _edit_star_tilt_range(config, star_path, usable):
    """Replace the dataset-wide TiltMin/TiltMax in *star_path* with each
    tomogram's own range, read from its .aln (see ``_aln_tilt_range``).

    ``prepare_star`` writes the same ``--tilt_min``/``--tilt_max`` on every
    row (``resolve_tilt_range`` had to pick one value for the whole call), so
    this is what makes the star file per-tomogram again. Rows are matched to
    tomograms positionally, in the same order ``usable`` was handed to
    ``prepare_star`` (sorted by name) - the star file has no other way to
    identify which row is which tomogram, since ``rlnTomoName`` is "None".
    """
    with open(star_path, encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    columns = {}
    header_end = None
    for index, line in enumerate(lines):
        match = _STAR_COLUMN_RE.match(line.strip())
        if match:
            columns[match.group(1)] = int(match.group(2)) - 1
            header_end = index

    if header_end is None or "rlnTiltMin" not in columns or "rlnTiltMax" not in columns:
        warn("%s has no rlnTiltMin/rlnTiltMax column - per-tomogram tilt "
             "ranges were not applied." % star_path)
        return
    min_col, max_col = columns["rlnTiltMin"], columns["rlnTiltMax"]

    data_rows = [index for index in range(header_end + 1, len(lines))
                 if lines[index].strip()]
    if len(data_rows) != len(usable):
        warn("%s has %d rows but %d tomograms were prepared - per-tomogram "
             "tilt ranges were not applied." % (star_path, len(data_rows), len(usable)))
        return

    for row_index, series in zip(data_rows, usable):
        aln_range = _aln_tilt_range(config, series.name)
        if aln_range is None:
            warn("no .aln tilt range for %s - that row keeps the "
                 "dataset-wide range prepare_star wrote." % series.name)
            continue
        fields = lines[row_index].split()
        fields[min_col] = "%.6f" % aln_range[0]
        fields[max_col] = "%.6f" % aln_range[1]
        lines[row_index] = "\t".join(fields) + "\n"

    with open(star_path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    info("tomograms.star: per-tomogram tilt ranges applied from each .aln")


def _half_map_pixel_size(config, series):
    from .mdoc import resolve_pixel_size

    first_frame = (
        os.path.join(config.input_frames, series.frames[0])
        if series.frames else None
    )
    apix = resolve_pixel_size(config, series, first_frame)
    return apix * int(config["motioncor"]["binning"]) * int(config["aretomo"]["binning"])


# ------------------------------------------------------------------- star file


def prepare_star(config, series_list, subset, metadata, dry_run=False):
    """Run ``isonet.py prepare_star`` with a per-tomogram defocus list.

    ``--defocus`` is given one value per tomogram, in the same order as the
    linked half maps (sorted by tomogram name), so every tomogram gets the
    defocus CTFFIND4 measured on its own first tilt. ``--tilt_min`` and
    ``--tilt_max`` take a single value each, so the overall range across the
    selected tomograms is used for that call; the per-tomogram ranges are
    logged.

    Once IsoNet has written ``tomograms.star``, ``rlnTiltMin``/``rlnTiltMax``
    on each row are corrected back to that tomogram's own range, read from its
    AreTomo2 .aln (see ``_edit_star_tilt_range``) - unless
    ``isonet.fix_tilt_range`` is set, in which case the fixed range is left as
    IsoNet wrote it on every row.
    """
    # A tomogram without a defocus would shift every later entry of the
    # --defocus list onto the wrong tomogram, so drop it from the run.
    incomplete = [name for name, entry in metadata.items()
                  if entry["defocus"] is None]
    if incomplete:
        warn("no defocus estimate for %s - excluded from IsoNet so that the "
             "--defocus list stays aligned with the tomograms."
             % ", ".join(sorted(incomplete)))
    series_list = [s for s in series_list
                   if metadata.get(s.name, {}).get("defocus") is not None]
    if not series_list:
        raise SystemExit(
            "ERROR: CTFFIND4 produced no usable defocus for any tomogram.\n"
            "Check %s and the ctffind section of your configuration."
            % config.isonet_config_dir
        )

    even_dir, odd_dir, usable = _linked_half_maps(config, series_list, subset)

    pixel_size = _half_map_pixel_size(config, usable[0])
    defoci = [metadata[s.name]["defocus"] for s in usable]

    info("half-map pixel size %.4f A" % pixel_size)
    info("tomograms, in the order prepare_star receives them:")
    for series in usable:
        entry = metadata[series.name]
        tilts = ("%6.1f to %6.1f" % (entry["tilt_min"], entry["tilt_max"])
                 if entry["tilt_min"] is not None else "     no .rawtlt")
        info("    %-24s defocus %9.1f A   tilt %s"
             % (series.name, entry["defocus"], tilts))

    tilt_min, tilt_max, fixed_tilt_range = resolve_tilt_range(config, usable, metadata)

    defocus_list = "[%s]" % ",".join("%.2f" % value for value in defoci)

    work_dir = ensure_dir(config.isonet_config_dir, "prepare_" + subset)
    run([config.exe("isonet"), "prepare_star",
         "--even", even_dir,
         "--odd", odd_dir,
         "--pixel_size", "%.4f" % pixel_size,
         "--defocus", defocus_list,
         "--tilt_min", "%.2f" % tilt_min,
         "--tilt_max", "%.2f" % tilt_max],
        cwd=work_dir,
        log_file=os.path.join(config.isonet_config_dir,
                              "prepare_star_%s.log" % subset),
        dry_run=dry_run)

    if dry_run:
        return os.path.join(work_dir, "tomograms.star"), usable

    produced = sorted(glob.glob(os.path.join(work_dir, "*.star")),
                      key=os.path.getmtime, reverse=True)
    if not produced:
        raise SystemExit(
            "ERROR: isonet.py prepare_star produced no star file in %s\n"
            "See %s"
            % (work_dir,
               os.path.join(config.isonet_config_dir,
                            "prepare_star_%s.log" % subset))
        )
    info("star file: %s" % produced[0])

    if fixed_tilt_range:
        info("isonet.fix_tilt_range is set - tomograms.star keeps that fixed "
             "range on every row.")
    else:
        _edit_star_tilt_range(config, produced[0], usable)

    return produced[0], usable


# -------------------------------------------------------------------- networks


def resolve_network(config, key, default_dir):
    """Locate a trained network: an explicit setting, or the newest in a directory."""
    setting = str(config["isonet"][key] or "auto").strip()
    if setting and setting.lower() != "auto":
        path = setting if os.path.isabs(setting) else os.path.join(
            config.isonet_dir, setting)
        if not os.path.exists(path):
            raise SystemExit("ERROR: isonet.%s points at a missing file: %s"
                             % (key, path))
        return path

    search = os.path.join(config.isonet_dir, default_dir)
    found = sorted(glob.glob(os.path.join(search, "network_*.pt")),
                   key=os.path.getmtime, reverse=True)
    if not found:
        found = sorted(glob.glob(os.path.join(search, "*.pt")),
                       key=os.path.getmtime, reverse=True)
    if not found:
        raise SystemExit(
            "ERROR: no trained network found in %s\n"
            "Either training has not finished, or set isonet.%s to the "
            "network file explicitly." % (search, key)
        )
    info("using network %s" % found[0])
    return found[0]


def _gpu_args(config):
    return ["--gpuID", gpu_flag_isonet(config.gpus)]


def _extra_args(config):
    extra = str(config["isonet"]["extra_args"]).strip()
    return extra.split() if extra else []


# ------------------------------------------------------------------ entrypoints


def train(config, force_ctf=False, force=None, dry_run=False):
    """``tomo_denoising train_isonet``"""
    force = config.overwrite if force is None else force
    with StepLog(config, "train_isonet", force=force) as log:
        code = _train(config, log, force_ctf=force_ctf, force=force,
                      dry_run=dry_run)
        return code


def _train(config, log, force_ctf=False, force=False, dry_run=False):
    banner("IsoNet training")
    require_executable(config.exe("isonet"), "IsoNet")

    settings = config["isonet"]
    work = ensure_dir(config.isonet_dir)
    logs = ensure_dir(work, "logs")
    ensure_dir(config.isonet_config_dir)

    if not force and step_completed(config, "train_isonet"):
        try:
            network = resolve_network(config, "refine_network", REFINE_DIR)
        except SystemExit:
            network = None
        if network:
            info("IsoNet is already trained: %s" % network)
            log.note("skipped: the previous training completed and %s exists. "
                     "Use --force to retrain." % network)
            return 0

    series_list = collect_tilt_series(config.mdoc_dir("train"), "train")
    info("%d tilt series selected for IsoNet training" % len(series_list))
    log.note("training on %d tilt series from %s"
             % (len(series_list), config.mdoc_dir("train")))

    banner("IsoNet 0/5  defocus estimation (CTFFIND4)")
    metadata = gather_metadata(config, series_list, force_ctf=force_ctf,
                               dry_run=dry_run)

    banner("IsoNet 1/5  prepare_star")
    star, _ = prepare_star(config, series_list, "train", metadata, dry_run=dry_run)

    banner("IsoNet 2/5  denoise (%d epochs)" % int(settings["denoise_epochs"]))
    run([config.exe("isonet"), "denoise", star,
         "--CTF_mode", str(settings["ctf_mode"]),
         "--epochs", str(int(settings["denoise_epochs"])),
         "--output_dir", DENOISE_DIR]
        + _gpu_args(config) + _extra_args(config),
        cwd=work, log_file=os.path.join(logs, "denoise.log"), dry_run=dry_run)

    banner("IsoNet 3/5  apply the denoiser (intermediate - feeds make_mask only)")
    denoise_network = (
        os.path.join(work, DENOISE_DIR, "network.pt") if dry_run
        else resolve_network(config, "denoise_network", DENOISE_DIR)
    )
    run([config.exe("isonet"), "predict", star, denoise_network,
         "--output_dir", DENOISE_PREDICT_DIR,
         "--output_prefix", DENOISE_PREDICT_PREFIX]
        + _gpu_args(config) + _extra_args(config),
        cwd=work, log_file=os.path.join(logs, "predict_denoise.log"),
        dry_run=dry_run)

    banner("IsoNet 4/5  make_mask")
    run([config.exe("isonet"), "make_mask", star,
         "--density_percentage", str(settings["density_percentage"]),
         "--std_percentage", str(settings["std_percentage"]),
         "--input_column", "rlnDenoisedTomoName",
         "--output_dir", MASK_DIR],
        cwd=work, log_file=os.path.join(logs, "make_mask.log"), dry_run=dry_run)

    banner("IsoNet 5/5  refine (%d epochs)" % int(settings["refine_epochs"]))
    run([config.exe("isonet"), "refine", star,
         "--CTF_mode", str(settings["ctf_mode"]),
         "--mw_weight", str(settings["mw_weight"]),
         "--bfactor", str(settings["bfactor"]),
         "--epochs", str(int(settings["refine_epochs"])),
         "--output_dir", REFINE_DIR]
        + _gpu_args(config) + _extra_args(config),
        cwd=work, log_file=os.path.join(logs, "refine.log"), dry_run=dry_run)

    info("IsoNet training finished. Networks are in %s"
         % os.path.join(work, REFINE_DIR))
    log.note("networks: %s" % os.path.join(work, REFINE_DIR))
    for entry in series_list:
        log.ok(entry.name)
    return 0


def predict(config, force_ctf=False, force=None, dry_run=False):
    """``tomo_denoising predict_isonet``"""
    force = config.overwrite if force is None else force
    with StepLog(config, "predict_isonet", force=force) as log:
        return _predict(config, log, force_ctf=force_ctf, force=force,
                        dry_run=dry_run)


def _predict(config, log, force_ctf=False, force=False, dry_run=False):
    banner("IsoNet prediction")
    require_executable(config.exe("isonet"), "IsoNet")

    work = ensure_dir(config.isonet_dir)
    logs = ensure_dir(work, "logs")
    ensure_dir(config.isonet_config_dir)

    series_list = collect_tilt_series(config.mdoc_dir("predict"), "predict")
    info("%d tilt series selected for IsoNet prediction" % len(series_list))
    log.note("predicting %d tilt series from %s"
             % (len(series_list), config.mdoc_dir("predict")))

    if not force and step_completed(config, "predict_isonet"):
        info("IsoNet prediction already complete.")
        log.note("skipped: the previous prediction completed. "
                 "Use --force to redo it.")
    else:
        metadata = gather_metadata(config, series_list, force_ctf=force_ctf,
                                   dry_run=dry_run)
        star, _ = prepare_star(config, series_list, "predict", metadata,
                               dry_run=dry_run)

        network = (
            os.path.join(work, REFINE_DIR, "network.pt") if dry_run
            else resolve_network(config, "refine_network", REFINE_DIR)
        )

        run([config.exe("isonet"), "predict", star, network,
             "--output_dir", FINAL_PREDICT_DIR,
             "--output_prefix", FINAL_PREDICT_PREFIX]
            + _gpu_args(config) + _extra_args(config),
            cwd=work, log_file=os.path.join(logs, "predict_refine.log"),
            dry_run=dry_run)

    final_dir = os.path.join(work, FINAL_PREDICT_DIR)
    info("IsoNet prediction finished. Final denoised tomograms: %s" % final_dir)
    log.note("final denoised tomograms: %s" % final_dir)
    for entry in series_list:
        log.ok(entry.name)

    if not dry_run:
        from . import report
        report.refresh_after_prediction(config, "isonet", log=log)
    return 0
