"""Selecting the even/odd half maps belonging to a set of mdoc files."""

import os

from .mdoc import collect_tilt_series
from .utils import info, warn


def _dir_has_halfmaps(directory):
    return os.path.isdir(directory) and any(os.scandir(directory))


def resolve_frame_halfmaps_dir(config, log=None):
    """Which frame-split half-maps directory denoising should actually use.

    Prefers MissAlignment-corrected half maps (``run_missalignment``'s
    output) when they are present - that is the pipeline's default
    reconstruction path. Falls back to the plain-AreTomo ``raw_halfMaps``
    otherwise, but never silently: the fallback is always warned AND, when a
    ``StepLog`` is given, recorded in that step's own report, so a run using
    uncorrected half maps is always visible after the fact.
    """
    if config["missalign"]["enabled"] and _dir_has_halfmaps(config.missalignment_halfmaps_dir):
        return config.missalignment_halfmaps_dir

    message = (
        "using raw_halfMaps (plain WarpTools reconstruction, no MissAlignment "
        "correction) - missalign.enabled is false, or 'tomo_denoising "
        "run_missalignment' has not produced half maps yet"
    )
    warn(message)
    if log is not None:
        log.note(message)
    return config.raw_halfmaps_dir


def resolve_tiltsplit_halfmaps_dir(config, log=None):
    """Which tilt-split half-maps directory quality_check should actually use.

    Mirrors :func:`resolve_frame_halfmaps_dir`: prefers MissAlignment's
    tilt-split half maps when they are present, falls back to the plain-
    WarpTools ``raw_halfMaps_tiltSplit`` otherwise - never silently.
    """
    if (config["missalign"]["enabled"]
            and _dir_has_halfmaps(config.missalignment_tiltsplit_halfmaps_dir)):
        return config.missalignment_tiltsplit_halfmaps_dir

    message = (
        "using raw_halfMaps_tiltSplit (plain WarpTools reconstruction, no "
        "MissAlignment correction) - missalign.enabled is false, or "
        "'tomo_denoising run_missalignment' has not produced tilt-split "
        "half maps yet"
    )
    warn(message)
    if log is not None:
        log.note(message)
    return config.tiltsplit_halfmaps_dir


def ensure_halfmaps_available(config, log=None, dry_run=False):
    """On-demand half-map reconstruction for a standalone step.

    ``quality_check``/``train_isonet``/``predict_isonet`` may be run on their
    own, without ``run_pipe``'s own orchestration ever having produced any
    half maps - neither MissAlignment's (``run_missalignment`` was never run)
    nor the plain ones (``reconstruct_tomograms`` was skipped, or was run by
    ``run_pipe`` with ``halfmaps=False`` in anticipation of MissAlignment,
    which then never completed). If NEITHER kind exists yet, reconstruct the
    plain ones now from the uncorrected alignment, so the step has something
    to work with instead of failing outright.

    A no-op whenever either kind already exists - this never re-reconstructs
    half maps that are already there, and never runs in place of an explicit
    ``run_missalignment``.
    """
    if _dir_has_halfmaps(config.missalignment_halfmaps_dir):
        return
    if _dir_has_halfmaps(config.raw_halfmaps_dir):
        return

    message = ("no half maps found yet (neither MissAlignment-corrected nor "
               "plain) - reconstructing raw_halfMaps from the uncorrected "
               "alignment now")
    info(message)
    if log is not None:
        log.note(message)

    from . import warptools
    code = warptools.reconstruct_tomograms(config, force=None, dry_run=dry_run,
                                           halfmaps=True)
    if code != 0 and not dry_run:
        raise SystemExit(
            "ERROR: on-demand half-map reconstruction failed - see %s"
            % os.path.join(config.logs_dir, "reconstruct_tomograms.log")
        )


def half_map_paths(config, tomo_name, log=None):
    """The canonical half-map locations for one tomogram."""
    directory = resolve_frame_halfmaps_dir(config, log=log)
    even = os.path.join(directory, "even_%s_rec_tomo.mrc" % tomo_name)
    odd = os.path.join(directory, "odd_%s_rec_tomo.mrc" % tomo_name)
    return even, odd


def collect_half_maps(config, which, log=None):
    """Return ``[(tomo_name, even_path, odd_path), ...]`` for an mdoc set.

    *which* is ``"train"`` or ``"predict"``; the mdoc directory comes from the
    ``mdocs`` section of the configuration.
    """
    directory = config.mdoc_dir(which)
    series = collect_tilt_series(directory, which)

    source_dir = resolve_frame_halfmaps_dir(config, log=log)
    pairs = []
    missing = []
    for entry in series:
        even = os.path.join(source_dir, "even_%s_rec_tomo.mrc" % entry.name)
        odd = os.path.join(source_dir, "odd_%s_rec_tomo.mrc" % entry.name)
        if os.path.exists(even) and os.path.exists(odd):
            pairs.append((entry.name, even, odd))
        else:
            missing.append(entry.name)

    if missing:
        warn(
            "no half maps for: %s\n"
            "  Run 'tomo_denoising reconstruct_tomograms' first, or check "
            "whether alignment/reconstruction failed for these tilt series."
            % ", ".join(missing)
        )
    if not pairs:
        raise SystemExit(
            "ERROR: none of the tomograms listed in %s have half maps in %s."
            % (directory, source_dir)
        )

    info("%d tomograms selected for %s (from %s)"
         % (len(pairs), which, source_dir))
    return pairs
