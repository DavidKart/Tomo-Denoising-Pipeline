"""``tomo_denoising run_pipe``: the whole workflow from one configuration.

    create_halfMaps  ->  quality_check  ->  train  ->  predict

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

DENOISERS = ("cryocare", "isonet")


def _training_set_ready(config):
    return bool(find_mdocs(config.mdoc_dir("train")))


def run_pipe(config, denoiser="cryocare", skip_quality=False, force=None,
             force_ctf=False, dry_run=False):
    """Run every step in order. Returns a process exit code."""
    from . import cryocare, halfmaps, isonet, resolve

    denoiser = denoiser.lower()
    if denoiser not in DENOISERS:
        raise SystemExit("ERROR: --denoiser must be one of %s, got %r"
                         % (" / ".join(DENOISERS), denoiser))

    force = config.overwrite if force is None else force

    banner("run_pipe: %s denoising%s"
           % (denoiser, "" if not skip_quality else ", quality check skipped"))
    info("processing directory: %s" % config.processing_dir)
    info("resume mode: %s"
         % ("off - everything will be redone" if force
            else "on - finished work is skipped"))

    # --- 1. reconstruction -------------------------------------------------
    banner("run_pipe 1/4  create_halfMaps")
    code = halfmaps.create_half_maps(config, force=force, dry_run=dry_run)
    if code != 0:
        warn("create_halfMaps reported problems - see %s"
             % os.path.join(config.logs_dir, "create_halfMaps.log"))
        if not dry_run and not os.path.isdir(config.raw_halfmaps_dir):
            return code

    # --- 2. quality check --------------------------------------------------
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
        banner("run_pipe 2/4  quality_check")
        code = resolve.quality_check(config, force=force, dry_run=dry_run)
        if code == 2:
            # A training set could not be chosen; that needs a human.
            warn("run_pipe stops here: no training set could be chosen "
                 "automatically. See %s"
                 % os.path.join(config.logs_dir, "quality_check.log"))
            return 2
        if code != 0:
            warn("quality_check reported problems - continuing anyway.")

    # --- 3. training -------------------------------------------------------
    banner("run_pipe 3/4  train_%s" % denoiser)
    if denoiser == "cryocare":
        code = cryocare.train(config, force=force, dry_run=dry_run)
    else:
        code = isonet.train(config, force_ctf=force_ctf, force=force,
                            dry_run=dry_run)
    if code != 0:
        warn("training failed - stopping.")
        return code

    # --- 4. prediction -----------------------------------------------------
    banner("run_pipe 4/4  predict_%s" % denoiser)
    if denoiser == "cryocare":
        code = cryocare.predict(config, force=force, dry_run=dry_run)
    else:
        code = isonet.predict(config, force_ctf=force_ctf, force=force,
                              dry_run=dry_run)
    if code != 0:
        warn("prediction failed.")
        return code

    banner("run_pipe finished")
    info("logs:    %s" % config.logs_dir)
    info("results: %s" % config.processing_dir)
    return 0
