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
prediction set.
"""

import glob
import os

from .ctffind import run_ctffind
from .mdoc import collect_tilt_series
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
    never silent. ``isonet.tilt_range`` overrides the whole calculation.
    """
    override = config["isonet"].get("tilt_range")
    if override:
        low, high = float(override[0]), float(override[1])
        info("isonet.tilt_range override: %.1f to %.1f deg" % (low, high))
        return low, high

    ranges = {s.name: (metadata[s.name]["tilt_min"], metadata[s.name]["tilt_max"])
              for s in usable
              if metadata[s.name]["tilt_min"] is not None}
    if not ranges:
        warn("no .rawtlt files found - falling back to -60 to 60 deg. "
             "Set isonet.tilt_range in your configuration.")
        return -60.0, 60.0

    minima = [low for low, _ in ranges.values()]
    maxima = [high for _, high in ranges.values()]
    tilt_min, tilt_max = min(minima), max(maxima)

    spread_low = max(minima) - min(minima)
    spread_high = max(maxima) - min(maxima)
    if spread_low <= TILT_TOLERANCE and spread_high <= TILT_TOLERANCE:
        info("tilt range %.1f to %.1f deg (consistent across %d tomograms)"
             % (tilt_min, tilt_max, len(ranges)))
        return tilt_min, tilt_max

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
    warn("Using the widest range: %.1f to %.1f deg." % (tilt_min, tilt_max))
    warn("%s will be under-corrected, because IsoNet will assume tilt data "
         "they do not have. That is the safe direction - declaring a narrower "
         "range would make IsoNet synthesise over real data instead."
         % ", ".join(outliers))
    warn("To avoid it, process those tomograms in a separate run, or set an "
         "explicit range:  isonet.tilt_range: [%.0f, %.0f]"
         % (max(minima), min(maxima)))
    return tilt_min, tilt_max


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
    selected tomograms is used; the per-tomogram ranges are logged.

    The star file IsoNet writes is used exactly as produced - nothing rewrites
    it afterwards.
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

    tilt_min, tilt_max = resolve_tilt_range(config, usable, metadata)

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


def train(config, force_ctf=False, dry_run=False):
    """``tomo_denoising train_isonet``"""
    banner("IsoNet training")
    require_executable(config.exe("isonet"), "IsoNet")

    settings = config["isonet"]
    work = ensure_dir(config.isonet_dir)
    logs = ensure_dir(work, "logs")
    ensure_dir(config.isonet_config_dir)

    series_list = collect_tilt_series(config.mdoc_dir("train"), "train")
    info("%d tilt series selected for IsoNet training" % len(series_list))

    banner("IsoNet 0/5  defocus estimation (CTFFIND4)")
    metadata = gather_metadata(config, series_list, force_ctf=force_ctf,
                               dry_run=dry_run)

    banner("IsoNet 1/5  prepare_star")
    star, _ = prepare_star(config, series_list, "train", metadata, dry_run=dry_run)

    banner("IsoNet 2/5  denoise (%d epochs)" % int(settings["denoise_epochs"]))
    run([config.exe("isonet"), "denoise", star,
         "--CTF_mode", str(settings["ctf_mode"]),
         "--epochs", str(int(settings["denoise_epochs"]))]
        + _gpu_args(config) + _extra_args(config),
        cwd=work, log_file=os.path.join(logs, "denoise.log"), dry_run=dry_run)

    banner("IsoNet 3/5  apply the denoiser")
    denoise_network = (
        os.path.join(work, "denoise", "network.pt") if dry_run
        else resolve_network(config, "denoise_network", "denoise")
    )
    run([config.exe("isonet"), "predict", star, denoise_network]
        + _gpu_args(config) + _extra_args(config),
        cwd=work, log_file=os.path.join(logs, "predict_denoise.log"),
        dry_run=dry_run)

    banner("IsoNet 4/5  make_mask")
    run([config.exe("isonet"), "make_mask", star,
         "--density_percentage", str(settings["density_percentage"]),
         "--std_percentage", str(settings["std_percentage"]),
         "--input_column", "rlnDenoisedTomoName"],
        cwd=work, log_file=os.path.join(logs, "make_mask.log"), dry_run=dry_run)

    banner("IsoNet 5/5  refine (%d epochs)" % int(settings["refine_epochs"]))
    run([config.exe("isonet"), "refine", star,
         "--CTF_mode", str(settings["ctf_mode"]),
         "--mw_weight", str(settings["mw_weight"]),
         "--bfactor", str(settings["bfactor"]),
         "--epochs", str(int(settings["refine_epochs"]))]
        + _gpu_args(config) + _extra_args(config),
        cwd=work, log_file=os.path.join(logs, "refine.log"), dry_run=dry_run)

    info("IsoNet training finished. Networks are in %s"
         % os.path.join(work, "isonet_maps"))
    return 0


def predict(config, force_ctf=False, dry_run=False):
    """``tomo_denoising predict_isonet``"""
    banner("IsoNet prediction")
    require_executable(config.exe("isonet"), "IsoNet")

    work = ensure_dir(config.isonet_dir)
    logs = ensure_dir(work, "logs")
    ensure_dir(config.isonet_config_dir)

    series_list = collect_tilt_series(config.mdoc_dir("predict"), "predict")
    info("%d tilt series selected for IsoNet prediction" % len(series_list))

    metadata = gather_metadata(config, series_list, force_ctf=force_ctf,
                               dry_run=dry_run)
    star, _ = prepare_star(config, series_list, "predict", metadata,
                           dry_run=dry_run)

    network = (
        os.path.join(work, "isonet_maps", "network.pt") if dry_run
        else resolve_network(config, "refine_network", "isonet_maps")
    )

    run([config.exe("isonet"), "predict", star, network]
        + _gpu_args(config) + _extra_args(config),
        cwd=work, log_file=os.path.join(logs, "predict_refine.log"),
        dry_run=dry_run)

    info("IsoNet prediction finished. Output is under %s" % work)
    return 0
