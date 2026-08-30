"""``tomo_denoising run_pipe``: the whole workflow from one configuration.

    motion_correct -> align_tiltseries -> reconstruct_tomograms
        -> [run_missalignment] -> quality_check -> train_isonet -> predict_isonet

MissAlignment runs by default - pass ``--skip-missalignment`` to disable it.
``reconstruct_tomograms`` always reconstructs the full, non-split tomogram -
that is a standalone deliverable, not an intermediate towards the half maps.
Whenever MissAlignment is going to run, it does NOT also reconstruct the
pre-MissAlignment half maps (``halfmaps=False``) - there is no reason to,
since ``run_missalignment`` reconstructs its own frame-split (for IsoNet) and
tilt-split (for RESOLVE) half maps from the refined alignment right after.
Pass ``--produce-intermediates`` to reconstruct the pre-MissAlignment half
maps here too, alongside MissAlignment's later corrected ones - useful for
comparing before/after. When MissAlignment is skipped entirely, the
plain-WarpTools half maps are always reconstructed here, for
``quality_check``/IsoNet to use directly.

A MissAlignment failure is NOT tolerated: unlike the other steps, silently
falling back to the uncorrected alignment would be a difficult-to-notice
change in what actually gets denoised, so ``run_pipe`` stops immediately and
reports the failure. ``halfmap_sets.resolve_frame_halfmaps_dir``/
``resolve_tiltsplit_halfmaps_dir`` still prefer MissAlignment's half maps
whenever they exist, which matters when a step is instead run on its own.

The quality check sits between reconstruction and training on purpose: it is
what picks the training set when none has been chosen yet, so by the time
training starts the mdocs.train directory is populated. Skipping it with
``--skip-quality`` requires you to have filled that directory yourself.

With ``overwrite: false`` every stage resumes rather than repeats, so re-running
after an interruption picks up where it stopped.
"""

import os

from .mdoc import find_mdocs
from .utils import banner, info, warn


def _training_set_ready(config):
    return bool(find_mdocs(config.mdoc_dir("train")))


def run_pipe(config, skip_quality=False, skip_missalignment=False,
            produce_intermediates=False, force=None, dry_run=False):
    """Run every step in order. Returns a process exit code."""
    from . import isonet, missalign, resolve, warptools

    force = config.overwrite if force is None else force

    banner("run_pipe: IsoNet denoising%s"
           % ("" if not skip_quality else ", quality check skipped"))
    info("processing directory: %s" % config.processing_dir)
    info("resume mode: %s"
         % ("off - everything will be redone" if force
            else "on - finished work is skipped"))

    # --- 1. WarpTools reconstruction ----------------------------------------
    banner("run_pipe 1/6  motion_correct")
    code = warptools.motion_correct(config, force=force, dry_run=dry_run)
    if code != 0:
        warn("motion_correct reported problems - see %s"
             % os.path.join(config.logs_dir, "motion_correct.log"))
        return code

    banner("run_pipe 2/6  align_tiltseries")
    code = warptools.align_tiltseries(config, force=force, dry_run=dry_run)
    if code != 0:
        warn("align_tiltseries reported problems - see %s"
             % os.path.join(config.logs_dir, "align_tiltseries.log"))
        return code

    will_run_missalignment = config["missalign"]["enabled"] and not skip_missalignment
    # The full tomogram is always reconstructed regardless; only the
    # pre-MissAlignment half maps are conditional - skipped by default when
    # MissAlignment is about to redo them from the refined alignment, unless
    # --produce-intermediates asks for both sets.
    do_halfmaps = (not will_run_missalignment) or produce_intermediates

    banner("run_pipe 3/6  reconstruct_tomograms")
    code = warptools.reconstruct_tomograms(config, force=force, dry_run=dry_run,
                                           halfmaps=do_halfmaps)
    if code != 0:
        warn("reconstruct_tomograms reported problems - see %s"
             % os.path.join(config.logs_dir, "reconstruct_tomograms.log"))
        # ts_defocus_hand/ts_ctf/the full tomogram all run unconditionally,
        # and MissAlignment's own reconstruction needs the first two of
        # those too, so any failure here is a hard stop when MissAlignment
        # is about to run. Otherwise the existing tolerance applies: partial
        # half maps are enough to continue.
        if will_run_missalignment or (not dry_run and not os.path.isdir(config.raw_halfmaps_dir)):
            return code

    # --- 2. MissAlignment ---------------------------------------------------
    if not will_run_missalignment:
        info("run_missalignment skipped (--skip-missalignment, or "
             "missalign.enabled is false) - quality_check/IsoNet use "
             "raw_halfMaps")
    else:
        banner("run_pipe 4/6  run_missalignment")
        code = missalign.run(config, force=force, dry_run=dry_run)
        if code != 0:
            warn("run_missalignment failed - stopping. Falling back to "
                 "denoising the uncorrected alignment silently is not done "
                 "here; fix the problem and re-run, or pass "
                 "--skip-missalignment to deliberately proceed without it. "
                 "See %s" % os.path.join(config.logs_dir, "run_missalignment.log"))
            return code

    # --- 3. quality check ----------------------------------------------------
    if skip_quality:
        info("quality_check skipped (--skip-quality)")
        if not _training_set_ready(config) and not dry_run:
            raise SystemExit(
                "ERROR: no .mdoc files in %s and the quality check - which is "
                "what would populate it - was skipped.\nEither drop the "
                "training mdocs in yourself, or run without --skip-quality."
                % config.mdoc_dir("train")
            )
    else:
        banner("run_pipe 5/6  quality_check")
        code = resolve.quality_check(config, force=force, dry_run=dry_run)
        if code == 2:
            # A training set could not be chosen; that needs a human.
            warn("run_pipe stops here: no training set could be chosen "
                 "automatically. See %s"
                 % os.path.join(config.logs_dir, "quality_check.log"))
            return 2
        if code != 0:
            warn("quality_check reported problems - continuing anyway.")

    # --- 4. IsoNet training and prediction ----------------------------------
    banner("run_pipe 6/6  train_isonet + predict_isonet")
    code = isonet.train(config, force=force, dry_run=dry_run)
    if code != 0:
        warn("training failed - stopping.")
        return code

    code = isonet.predict(config, force=force, dry_run=dry_run)
    if code != 0:
        warn("prediction failed.")
        return code

    banner("run_pipe finished")
    info("logs:    %s" % config.logs_dir)
    info("results: %s" % config.processing_dir)
    return 0
