"""Steps 4 and 5: IsoNet training and prediction.

``train_isonet`` performs the whole preparation and training chain:

  1. ``isonet.py prepare_star``  + per-tomogram tilt range patching
  2. ``isonet.py denoise``       noise2noise denoiser
  3. ``isonet.py predict``       apply the denoiser (needed for the masks)
  4. ``isonet.py make_mask``     masks from the denoised volumes
  5. ``isonet.py refine``        missing-wedge network

No defocus estimate is needed anywhere in this chain: ``isonet.ctf_mode`` is
``"None"`` (see the config comment - the half maps are already CTF-corrected
by WarpTools, before IsoNet ever sees them), and IsoNet2 does not use a
defocus value at all in that mode. CTFFIND4 is not run by this pipeline.

``predict_isonet`` is the final step: apply the refined network to the
prediction set. Its output - the only directory holding the finished,
deliverable tomograms - is ``isonet/corrected_tomos_final/`` (see the output
layout below ``DENOISE_DIR`` and friends).
"""

import glob
import os
import re

from . import tiltfilter
from .halfmap_sets import ensure_halfmaps_available, resolve_frame_halfmaps_dir
from .mdoc import collect_tilt_series
from .steplog import StepLog, step_completed
from .warptools import tiltseries_tilt_range
from .utils import (
    banner,
    ensure_dir,
    gpu_env,
    gpu_flag_isonet,
    gpu_id_note,
    gpu_ids_for,
    info,
    link_file,
    run,
    warn,
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


def _linked_half_maps(config, series_list, subset, log=None, force=False,
                      dry_run=False):
    """Link the half maps of *series_list* into ``isonet/<subset>/{even,odd}``.

    Both halves are linked under the SAME base name (``<tomo>.mrc``) so that
    IsoNet pairs them unambiguously regardless of how it matches the two
    directories. Linking avoids copying tens of gigabytes and never touches
    the source half maps.

    Sourced from ``halfmap_sets.resolve_frame_halfmaps_dir`` - the
    MissAlignment-corrected half maps by default, falling back to
    ``raw_halfMaps`` (with a logged notice) if those are not available. If
    NEITHER exists yet (this step run on its own, before MissAlignment or
    reconstruct_tomograms ever produced any), ``ensure_halfmaps_available``
    reconstructs the plain ones on demand first.

    The returned list is sorted by tomogram name, which is the order
    ``prepare_star`` sees the linked files in.

    *force* re-points links that already exist. It matters for two real
    cases, both of which leave the link NAME unchanged while changing what it
    should resolve to: a forced re-reconstruction at a different
    ``warptools.recon_angpix``, and MissAlignment's half maps appearing where
    a previous run linked the plain ones.
    """
    ensure_halfmaps_available(config, log=log, dry_run=dry_run)

    even_dir = ensure_dir(config.isonet_dir, subset, "even")
    odd_dir = ensure_dir(config.isonet_dir, subset, "odd")

    source_dir = resolve_frame_halfmaps_dir(config, log=log)
    usable = []
    for series in sorted(series_list, key=lambda s: s.name):
        even = os.path.join(source_dir, "even_%s_rec_tomo.mrc" % series.name)
        odd = os.path.join(source_dir, "odd_%s_rec_tomo.mrc" % series.name)
        if not (os.path.exists(even) and os.path.exists(odd)):
            warn("no half maps for %s - excluded from IsoNet." % series.name)
            continue
        link_file(even, os.path.join(even_dir, series.name + ".mrc"),
                  force=force)
        link_file(odd, os.path.join(odd_dir, series.name + ".mrc"),
                  force=force)
        usable.append(series)

    if not usable:
        raise SystemExit(
            "ERROR: none of the selected tomograms have half maps in %s.\n"
            "Run 'tomo_denoising reconstruct_tomograms' first."
            % source_dir
        )
    return even_dir, odd_dir, usable


# ---------------------------------------------------------------- tilt range


def tilt_range(config, tomo_name):
    """Minimum and maximum tilt angle actually used by *tomo_name*.

    Thin wrapper around ``warptools.tiltseries_tilt_range`` - see that
    function for exactly where this comes from and why (the FINAL
    ``warp_tiltseries/<tomo>.xml``, never an AreTomo-generated file).
    """
    result = tiltseries_tilt_range(config, tomo_name)
    if result is None:
        warn("no usable tilt range for %s (no warp_tiltseries/%s.xml, or it "
             "has no angles)" % (tomo_name, tomo_name))
    return result


def tilt_ranges(config, series_list):
    """``{tomo_name: (min, max) or None}`` for every series in *series_list*."""
    return {series.name: tilt_range(config, series.name) for series in series_list}


#: Tilt ranges closer than this are treated as identical (rounding in mdocs).
TILT_TOLERANCE = 2.0


def resolve_tilt_range(config, usable, ranges):
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
    is the safe default.

    This is only what ``prepare_star`` is called with - it is corrected
    afterwards, back to a per-tomogram range read from each tomogram's own
    ``warp_tiltseries/<tomo>.xml`` (see ``_edit_star_tilt_range``), UNLESS
    ``isonet.fix_tilt_range`` overrides the whole calculation, in which case
    the fixed range returned here is what ends up in the star file too.

    *ranges* is ``tilt_ranges()``'s return value. Returns ``(tilt_min,
    tilt_max, is_override)``.
    """
    override = config["isonet"].get("fix_tilt_range")
    if override:
        low, high = float(override[0]), float(override[1])
        info("isonet.fix_tilt_range override: %.1f to %.1f deg "
             "(used for every tomogram, unedited)" % (low, high))
        return low, high, True

    ranges = {s.name: ranges[s.name] for s in usable if ranges.get(s.name) is not None}
    if not ranges:
        warn("no warp_tiltseries/<tomo>.xml tilt ranges found - falling back "
             "to -60 to 60 deg. Set isonet.fix_tilt_range in your configuration.")
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

    # Not a problem worth a warning: this range only goes to prepare_star,
    # and tomograms.star gets each tomogram's own range right after.
    info("tilt ranges differ across %d tomograms - prepare_star is called "
         "with the widest, %.1f to %.1f deg; each tomogram then gets its own "
         "range in tomograms.star" % (len(ranges), tilt_min, tilt_max))
    return tilt_min, tilt_max, False


_STAR_COLUMN_RE = re.compile(r"^_(\S+)\s+#(\d+)")


def _edit_star_tilt_range(star_path, usable, ranges):
    """Replace the dataset-wide TiltMin/TiltMax in *star_path* with each
    tomogram's own range, read from its ``warp_tiltseries/<tomo>.xml`` (see
    ``tilt_range``). *ranges* is ``tilt_ranges()``'s return value.

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
        own_range = ranges.get(series.name)
        if own_range is None:
            warn("no warp_tiltseries/%s.xml tilt range - that row keeps the "
                 "dataset-wide range prepare_star wrote." % series.name)
            continue
        fields = lines[row_index].split()
        fields[min_col] = "%.6f" % own_range[0]
        fields[max_col] = "%.6f" % own_range[1]
        lines[row_index] = "\t".join(fields) + "\n"

    with open(star_path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    info("tomograms.star: per-tomogram tilt ranges applied from each "
         "warp_tiltseries/<tomo>.xml")


def _half_map_pixel_size(config, series):
    """The half maps' actual voxel size: ts_reconstruct is always called
    with an explicit --angpix (warptools.recon_angpix), so that is the
    ground truth directly - no need to re-derive it from independent binning
    factors the way the old MotionCor3+AreTomo2 pipeline had to.

    Resolved through ``warptools.resolve_recon_angpix`` rather than read
    verbatim, since recon_angpix may be written relative to the dataset's own
    pixel size ("2*pixel_size") - exactly as the ts_reconstruct call itself
    resolves it.
    """
    from .warptools import resolve_recon_angpix
    return resolve_recon_angpix(config)


# ------------------------------------------------------------------- star file


def prepare_star(config, series_list, subset, log=None, force=False,
                 dry_run=False):
    """Run ``isonet.py prepare_star``.

    No defocus is passed: ``isonet.ctf_mode`` is ``"None"`` (the half maps
    are already CTF-corrected by WarpTools before IsoNet ever sees them), and
    IsoNet2 does not use a defocus value in that mode. ``--tilt_min``/
    ``--tilt_max`` take a single value each, so the overall range across the
    selected tomograms is used for that call; the per-tomogram ranges are
    logged.

    Once IsoNet has written ``tomograms.star``, ``rlnTiltMin``/``rlnTiltMax``
    on each row are corrected back to that tomogram's own range, read from its
    ``warp_tiltseries/<tomo>.xml`` (see ``_edit_star_tilt_range``) - unless
    ``isonet.fix_tilt_range`` is set, in which case the fixed range is left as
    IsoNet wrote it on every row.
    """
    even_dir, odd_dir, usable = _linked_half_maps(config, series_list, subset,
                                                  log=log, force=force,
                                                  dry_run=dry_run)

    pixel_size = _half_map_pixel_size(config, usable[0])
    ranges = tilt_ranges(config, usable)

    info("half-map pixel size %.4f A" % pixel_size)
    info("tomograms, in the order prepare_star receives them:")
    for series in usable:
        own_range = ranges.get(series.name)
        tilts = ("%6.1f to %6.1f" % own_range if own_range is not None
                 else "     no warp_tiltseries/%s.xml" % series.name)
        info("    %-24s tilt %s" % (series.name, tilts))

    tilt_min, tilt_max, fixed_tilt_range = resolve_tilt_range(config, usable, ranges)

    work_dir = ensure_dir(config.isonet_config_dir, "prepare_" + subset)
    run(config.command("isonet", "prepare_star",
                       "--even", even_dir,
                       "--odd", odd_dir,
                       "--pixel_size", "%.4f" % pixel_size,
                       "--tilt_min", "%.2f" % tilt_min,
                       "--tilt_max", "%.2f" % tilt_max),
        cwd=work_dir, env=_gpu_env(config),
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
        _edit_star_tilt_range(produced[0], usable, ranges)

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
    """``--gpuID`` for IsoNet: the REAL GPU ids from the config, not remapped.

    This is the one program in the pipeline that must not be given the
    0..N-1 logical ids every other GPU argument uses, and getting it wrong
    fails silently on the wrong cards rather than erroring.

    IsoNet does not treat --gpuID as an index into the already-visible
    devices. It ASSIGNS the value straight to the environment variable,
    before importing its GPU backend - confirmed in IsoNet's own source
    (``bin/refine.py``, ``bin/predict.py``)::

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuID

    So whatever we pass here REPLACES the CUDA_VISIBLE_DEVICES we set, and
    is then interpreted against the full set of physical GPUs. Handing it
    remapped ids (``--gpuID 0,1`` for a config of ``gpus: [2, 3]``) would
    therefore put IsoNet on physical GPUs 0 and 1 - exactly the GPUs the
    config asked it to stay off.

    Contrast miss-alignment, which does the opposite: it uses its device
    arguments as ``cuda:<index>`` against the visible set and never touches
    the environment variable, so there the ids MUST be remapped (see
    ``missalign.resolve_devices``).
    """
    return ["--gpuID", gpu_flag_isonet(gpu_ids_for("isonet", config.gpus))]


def _gpu_env(config):
    """Restrict IsoNet to the configured GPUs.

    --gpuID alone is not quite enough: it only covers the steps we pass it
    to (denoise/predict/refine), while prepare_star and make_mask get no GPU
    argument at all, and any library in the process that initialises CUDA on
    its own would otherwise default to device 0.

    This and the real-id --gpuID above agree rather than fight: IsoNet
    overwrites CUDA_VISIBLE_DEVICES with the same real ids we set here, so
    the value is replaced by an identical one.
    """
    return gpu_env(config.gpus)


def _extra_args(config):
    extra = str(config["isonet"]["extra_args"]).strip()
    return extra.split() if extra else []


# ------------------------------------------------------------------ entrypoints


def train(config, force=None, dry_run=False):
    """``tomo_denoising train_isonet``"""
    force = config.overwrite if force is None else force
    with StepLog(config, "train_isonet", force=force) as log:
        code = _train(config, log, force=force, dry_run=dry_run)
        return code


def _train(config, log, force=False, dry_run=False):
    banner("IsoNet training")
    config.require("isonet", "IsoNet")

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

    series_list = tiltfilter.drop_excluded(
        config, collect_tilt_series(config.mdoc_dir("train"), "train"),
        "IsoNet training", log=log)
    if not series_list:
        log.fail("train_isonet", "every tilt series in %s is excluded (see "
                 "%s)" % (config.mdoc_dir("train"), tiltfilter.record_path(config)))
        return 1
    info("%d tilt series selected for IsoNet training" % len(series_list))
    log.note("training on %d tilt series from %s"
             % (len(series_list), config.mdoc_dir("train")))

    log.note(gpu_id_note("isonet", config.gpus))

    banner("IsoNet 1/5  prepare_star")
    star, _ = prepare_star(config, series_list, "train", log=log, force=force,
                           dry_run=dry_run)

    banner("IsoNet 2/5  denoise (%d epochs)" % int(settings["denoise_epochs"]))
    run(config.command("isonet", "denoise", star,
                       "--CTF_mode", str(settings["ctf_mode"]),
                       "--epochs", str(int(settings["denoise_epochs"])),
                       "--output_dir", DENOISE_DIR,
                       _gpu_args(config), _extra_args(config)),
        cwd=work, env=_gpu_env(config),
        log_file=os.path.join(logs, "denoise.log"), dry_run=dry_run)

    banner("IsoNet 3/5  apply the denoiser (intermediate - feeds make_mask only)")
    denoise_network = (
        os.path.join(work, DENOISE_DIR, "network.pt") if dry_run
        else resolve_network(config, "denoise_network", DENOISE_DIR)
    )
    run(config.command("isonet", "predict", star, denoise_network,
                       "--output_dir", DENOISE_PREDICT_DIR,
                       "--output_prefix", DENOISE_PREDICT_PREFIX,
                       _gpu_args(config), _extra_args(config)),
        cwd=work, env=_gpu_env(config),
        log_file=os.path.join(logs, "predict_denoise.log"),
        dry_run=dry_run)

    banner("IsoNet 4/5  make_mask")
    run(config.command("isonet", "make_mask", star,
                       "--density_percentage", str(settings["density_percentage"]),
                       "--std_percentage", str(settings["std_percentage"]),
                       "--input_column", "rlnDenoisedTomoName",
                       "--output_dir", MASK_DIR),
        cwd=work, env=_gpu_env(config),
        log_file=os.path.join(logs, "make_mask.log"), dry_run=dry_run)

    banner("IsoNet 5/5  refine (%d epochs)" % int(settings["refine_epochs"]))
    run(config.command("isonet", "refine", star,
                       "--CTF_mode", str(settings["ctf_mode"]),
                       "--mw_weight", str(settings["mw_weight"]),
                       "--bfactor", str(settings["bfactor"]),
                       "--epochs", str(int(settings["refine_epochs"])),
                       "--output_dir", REFINE_DIR,
                       _gpu_args(config), _extra_args(config)),
        cwd=work, env=_gpu_env(config),
        log_file=os.path.join(logs, "refine.log"), dry_run=dry_run)

    info("IsoNet training finished. Networks are in %s"
         % os.path.join(work, REFINE_DIR))
    log.note("networks: %s" % os.path.join(work, REFINE_DIR))
    for entry in series_list:
        log.ok(entry.name)
    return 0


def predict(config, force=None, dry_run=False):
    """``tomo_denoising predict_isonet``"""
    force = config.overwrite if force is None else force
    with StepLog(config, "predict_isonet", force=force) as log:
        return _predict(config, log, force=force, dry_run=dry_run)


def _predict(config, log, force=False, dry_run=False):
    banner("IsoNet prediction")
    config.require("isonet", "IsoNet")

    work = ensure_dir(config.isonet_dir)
    logs = ensure_dir(work, "logs")
    ensure_dir(config.isonet_config_dir)

    series_list = tiltfilter.drop_excluded(
        config, collect_tilt_series(config.mdoc_dir("predict"), "predict"),
        "IsoNet prediction", log=log)
    if not series_list:
        log.fail("predict_isonet", "every tilt series in %s is excluded (see "
                 "%s)" % (config.mdoc_dir("predict"), tiltfilter.record_path(config)))
        return 1
    info("%d tilt series selected for IsoNet prediction" % len(series_list))
    log.note("predicting %d tilt series from %s"
             % (len(series_list), config.mdoc_dir("predict")))

    log.note(gpu_id_note("isonet", config.gpus))

    if not force and step_completed(config, "predict_isonet"):
        info("IsoNet prediction already complete.")
        log.note("skipped: the previous prediction completed. "
                 "Use --force to redo it.")
    else:
        star, _ = prepare_star(config, series_list, "predict", log=log,
                               force=force, dry_run=dry_run)

        network = (
            os.path.join(work, REFINE_DIR, "network.pt") if dry_run
            else resolve_network(config, "refine_network", REFINE_DIR)
        )

        run(config.command("isonet", "predict", star, network,
                           "--output_dir", FINAL_PREDICT_DIR,
                           "--output_prefix", FINAL_PREDICT_PREFIX,
                           _gpu_args(config), _extra_args(config)),
            cwd=work, env=_gpu_env(config),
            log_file=os.path.join(logs, "predict_refine.log"),
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
