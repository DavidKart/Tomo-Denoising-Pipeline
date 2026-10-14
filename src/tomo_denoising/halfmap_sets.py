"""Selecting the even/odd half maps belonging to a set of mdoc files.

Two independent choices are made here, and they are easy to confuse:

* WHICH ALIGNMENT - MissAlignment-corrected (``missalignment/``) or plain
  WarpTools (``warp_tiltseries/``). Whichever exists; MissAlignment wins.
* WHICH SPLIT - frame-split (``raw_halfMaps``, what IsoNet trains on) or
  tilt-split (``raw_halfMaps_tiltSplit``). Only ``quality_check`` has a
  choice here, through ``quality_check.enable_tilt_split``: with it off,
  RESOLVE measures the frame-split half maps that are reconstructed anyway
  and the tilt-split reconstruction is never made at all.
"""

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


def resolve_quality_check_halfmaps_dir(config, log=None):
    """Which half maps ``quality_check`` runs RESOLVE over.

    ``quality_check.enable_tilt_split: true`` -> the dedicated tilt-split
    reconstruction, which exists only for this.

    ``false`` -> the FRAME-split half maps, which are reconstructed anyway for
    IsoNet. RESOLVE needs two genuinely independent reconstructions of the
    same volume, and frame-split halves are exactly that - they just differ
    from each other far less than tilt-split ones, so the resolution it
    reports is the (more optimistic) frame-split number and is labelled as
    such everywhere, including in the results figure (see
    :func:`resolution_label`).
    """
    if config.tilt_split:
        return resolve_tiltsplit_halfmaps_dir(config, log=log)
    return resolve_frame_halfmaps_dir(config, log=log)


def resolution_label(config):
    """How RESOLVE's number should be named wherever it is shown."""
    return "Tilt-split resolution" if config.tilt_split else "Frame-split resolution"


# ------------------------------------------------ on-demand reconstruction


def _reconstruct_missing_tiltsplit(config, log=None, dry_run=False):
    """Reconstruct ONLY the missing tilt-split half maps, from the alignment
    the existing frame-split half maps already came from.

    Which destination that is matters: ``run_missalignment`` refines
    ``warp_tiltseries/<tomo>.xml`` IN PLACE, so reconstructing from
    ``warp_tiltseries.settings`` now yields the MissAlignment-corrected
    geometry. Writing that into ``warp_tiltseries/raw_halfMaps_tiltSplit``
    would file corrected reconstructions under the name this pipeline uses
    for uncorrected ones - so when MissAlignment's own half maps exist, the
    tilt-split set belongs next to them, under ``missalignment/``.
    """
    from . import warptools

    series_names = warptools.aligned_series_names(config)
    if not series_names and not dry_run:
        raise SystemExit(
            "ERROR: quality_check needs tilt-split half maps "
            "(quality_check.enable_tilt_split is true) and none exist yet, "
            "but there are no aligned tilt series (*.xml) in %s to "
            "reconstruct them from.\n"
            "Run 'tomo_denoising align_tiltseries' first."
            % config.warp_tiltseries_dir
        )

    corrected = (config["missalign"]["enabled"]
                 and _dir_has_halfmaps(config.missalignment_halfmaps_dir))
    if corrected:
        halfmaps_dir = config.missalignment_halfmaps_dir
        tiltsplit_dir = config.missalignment_tiltsplit_halfmaps_dir
        reconstruction_dir = config.missalignment_reconstruction_dir
        logs_dir = config.missalignment_logs_dir
    else:
        halfmaps_dir = config.raw_halfmaps_dir
        tiltsplit_dir = config.tiltsplit_halfmaps_dir
        reconstruction_dir = config.uncorrected_reconstruction_dir
        logs_dir = config.logs_dir

    message = ("tilt-split half maps are missing - reconstructing them now "
               "into %s (at %.1f A, quality_check."
               "tilt_split_target_pixel_size)"
               % (tiltsplit_dir, config.tilt_split_angpix))
    info(message)
    if log is not None:
        log.note(message)

    # do_halfmaps=True with do_full=False: the frame-split half maps are
    # already there, so run_ts_reconstruct skips that call by itself and only
    # the tilt-split one actually runs. force stays False for the same
    # reason - this is filling a gap, never redoing finished work.
    ok = warptools.run_ts_reconstruct(
        config, config.warp_tiltseries_settings_path,
        halfmaps_dir, tiltsplit_dir, logs_dir, series_names,
        warptools.resolve_recon_angpix(config), reconstruction_dir,
        full_dir=None, do_full=False, do_halfmaps=True, do_tiltsplit=True,
        log=log, force=False, dry_run=dry_run,
    )
    if not ok and not dry_run:
        raise SystemExit(
            "ERROR: on-demand tilt-split half-map reconstruction failed - "
            "see %s" % os.path.join(logs_dir, "ts_reconstruct_tilt.log")
        )


def ensure_halfmaps_available(config, log=None, dry_run=False,
                              need_tiltsplit=False):
    """On-demand half-map reconstruction for a standalone step.

    ``quality_check``/``train_isonet``/``predict_isonet`` may be run on their
    own, without ``run_pipe``'s own orchestration ever having produced any
    half maps - neither MissAlignment's (``run_missalignment`` was never run)
    nor the plain ones (``reconstruct_tomograms`` was skipped, or was run by
    ``run_pipe`` with ``halfmaps=False`` in anticipation of MissAlignment,
    which then never completed). If NEITHER kind exists yet, reconstruct the
    plain ones now from the uncorrected alignment, so the step has something
    to work with instead of failing outright.

    *need_tiltsplit* (``quality_check`` passes ``config.tilt_split``) also
    requires the TILT-split half maps. Those are a separate reconstruction
    that nothing but RESOLVE uses, so they are commonly absent even when the
    frame-split ones are there - the whole point of
    ``quality_check.enable_tilt_split`` is that the pipeline does not make
    them unless asked. Running ``quality_check`` later with it switched on
    reconstructs them at that point (see
    :func:`_reconstruct_missing_tiltsplit`), rather than failing.

    A no-op whenever everything needed already exists - this never
    re-reconstructs half maps that are already there, and never runs in place
    of an explicit ``run_missalignment``.
    """
    have_frame = (_dir_has_halfmaps(config.missalignment_halfmaps_dir)
                  or _dir_has_halfmaps(config.raw_halfmaps_dir))
    have_tiltsplit = (
        _dir_has_halfmaps(config.missalignment_tiltsplit_halfmaps_dir)
        or _dir_has_halfmaps(config.tiltsplit_halfmaps_dir))

    if have_frame:
        if need_tiltsplit and not have_tiltsplit:
            _reconstruct_missing_tiltsplit(config, log=log, dry_run=dry_run)
        return

    message = ("no half maps found yet (neither MissAlignment-corrected nor "
               "plain) - reconstructing raw_halfMaps%s from the uncorrected "
               "alignment now"
               % ("(_tiltSplit)" if need_tiltsplit else ""))
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
