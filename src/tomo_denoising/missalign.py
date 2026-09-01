"""Step: MissAlignment-refined reconstruction (``run_missalignment``).

Runs after ``reconstruct_tomograms``, refining the SAME per-tomogram
alignment ``align_tiltseries`` (native ``WarpTools ts_aretomo3``) already
produced in ``warp_tiltseries/<tomo>.xml``. MissAlignment's own per-tilt
local deformation model is something neither AreTomo nor plain ts_aretomo3
can represent, so it trains directly against that project and modifies each
XML IN PLACE:

  1. Back up every ``warp_tiltseries/<tomo>.xml`` to
     ``warp_tiltseries_prealign/<tomo>.xml`` - this is what lets --force
     restart cleanly from the true pre-MissAlignment state.
  2. Patch every ``ImageDimensionsAngstrom``/``VolumeDimensionsAngstrom`` on
     the TiltSeries XMLs - real, native ``ts_aretomo3`` output can leave these
     at ``[0.0, 0.0]``, which MissAlignment refuses to train on. Uses the
     same ``tomo_dimensions`` ``align_tiltseries`` already resolved, unless
     ``missalign.training_volume_dimensions`` is set to deliberately tighten
     the volume (see MissAlignment's own docs: "the tomogram shape should
     tightly fit your sample ... to avoid training on empty regions").
  3. Write ``warp_tiltseries/missalign_config.yaml`` and run
     ``miss-alignment train``.
  4. Reconstruct the refined project with genuine ``WarpTools
     ts_reconstruct`` (the shared :func:`warptools.run_ts_reconstruct`) - the
     local grid is only actually applied by Warp's own reconstruction
     engine, not by AreTomo. The full tomogram AND both frame-split (for
     IsoNet) and tilt-split (for RESOLVE) half maps are all produced here -
     this is the ONLY reconstruction that happens when MissAlignment is part
     of the pipeline (unless ``--produce-intermediates`` also reconstructed
     the pre-MissAlignment ones); see :mod:`pipeline`. Written into its own
     ``missalignment/reconstruction/`` scratch location
     (``config.missalignment_reconstruction_dir``, passed to
     ``ts_reconstruct`` as ``--output_processing``) - deliberately NOT the
     same ``warp_tiltseries/reconstruction/`` ``reconstruct_tomograms``
     itself uses, since sharing one used to mean this step's reconstruction
     silently overwrote the physical files raw_halfMaps's symlinks already
     pointed to.

Everything this step writes lives under
``<processing_directory>/missalignment/`` and never touches
``warp_tiltseries/`` - the uncorrected ``tomograms``/``raw_halfMaps``/
``raw_halfMaps_tiltSplit`` there stay exactly as ``reconstruct_tomograms``
left them (or are never populated at all, when MissAlignment runs without
``--produce-intermediates`` - see :mod:`pipeline`). ``missalignment/`` gets
the SAME three named directories (``tomograms``, ``raw_halfMaps``,
``raw_halfMaps_tiltSplit``) - a deliberate analogue of ``warp_tiltseries/``'s
own layout, just MissAlignment-corrected. The two half-map ones are what
``halfmap_sets.resolve_frame_halfmaps_dir``/``resolve_tiltsplit_halfmaps_dir``
prefer by default for denoising and quality checking.

NOTE on the ``ImageDimensionsAngstrom``/``VolumeDimensionsAngstrom`` patch:
MissAlignment's own reference material (``update_warp_xml.py``) patches
these two attributes on the ``<TiltSeries>`` root element directly with
plain torch/warpylib code, because real, native ``ts_aretomo3`` output can
leave them at ``[0.0, 0.0]`` (confirmed directly: MissAlignment raises
``ValueError: ... has zero values in 'ImageDimensionsAngstrom'`` on such a
project otherwise) - this is NOT a rare edge case to guard against, it is
routine and this patch is not optional. :func:`patch_training_dimensions`
below patches the same two attributes with a small, dependency-free text
substitution, so no warpylib dependency is needed.
"""

import os
import re
import shutil

import yaml

from .mdoc import collect_tilt_series
from .steplog import StepLog
from .utils import (
    banner, ensure_dir, info, require_executable, run as run_command,
    warn, wrap_conda_run,
)
from .warptools import (
    aligned_series_names, dataset_apix, resolve_recon_angpix,
    resolve_tomo_dimensions, run_ts_reconstruct,
)

STEP = "run_missalignment"

#: The exact iteration schedule from MissAlignment's own reference config
#: (MissAlignment.txt / config_template.yaml): three coarse-to-fine global
#: passes, then four local 3x3-grid refinement passes.
_ITERATION_SETTINGS = [
    {"downsample": 3, "alignment": "anchoring"},
    {"downsample": 2, "alignment": "anchoring"},
    {"downsample": 1, "alignment": "global"},
    {"downsample": 1, "alignment": "global"},
    {"downsample": 1, "alignment": [3, 3]},
    {"downsample": 1, "alignment": [3, 3]},
    {"downsample": 1, "alignment": [3, 3]},
    {"downsample": 1, "alignment": [3, 3]},
]

_ITER_DIR_RE = re.compile(r"^iter(\d+)$")


# --------------------------------------------------------- pre-training backup


def backup_alignment(config, series_names, log, force=False, dry_run=False):
    """Copy every ``warp_tiltseries/<tomo>.xml`` to ``warp_tiltseries_prealign/``.

    Never overwrites an existing backup unless *force* - that backup is the
    true pre-MissAlignment state, and MissAlignment modifies the originals in
    place, so backing up again once training has already started would
    capture the WRONG state.
    """
    backup_dir = ensure_dir(config.warp_tiltseries_prealign_dir)
    copied = 0
    for name in series_names:
        source = os.path.join(config.warp_tiltseries_dir, name + ".xml")
        target = os.path.join(backup_dir, name + ".xml")
        if not os.path.isfile(source):
            continue
        if os.path.isfile(target) and not force:
            continue
        if not dry_run:
            shutil.copy2(source, target)
        copied += 1

    if copied:
        info("backed up %d alignment XML(s) to %s" % (copied, backup_dir))
        log.note("backed up %d alignment XML(s) to %s" % (copied, backup_dir))
    else:
        log.note("alignment XMLs already backed up in %s" % backup_dir)


# --------------------------------------------------- training-volume patch


#: The <TiltSeries ...> opening tag, wherever it ends (real native ts_aretomo3
#: output writes it all on one line, but [^>] already spans newlines too).
_ROOT_TAG_RE = re.compile(r"<TiltSeries\b[^>]*>")


def _patch_dimensions_attr(text, attr_name, value):
    """Set ``attr_name="value"`` on the ``<TiltSeries>`` root element.

    Replaces the attribute if already present (even empty/zero) - but real,
    native ``ts_aretomo3`` output confirmed NOT to write
    ImageDimensionsAngstrom/VolumeDimensionsAngstrom at all (checked against
    real project XMLs - the attribute is simply absent, not present-and-
    zero), so the common case is inserting it fresh into the root tag.
    """
    pattern = re.compile(r'(\s%s=")[^"]*(")' % re.escape(attr_name))
    new_text, count = pattern.subn(lambda m: m.group(1) + value + m.group(2),
                                   text, count=1)
    if count:
        return new_text, True

    match = _ROOT_TAG_RE.search(text)
    if not match:
        return text, False
    tag = match.group(0)
    new_tag = tag[:-1] + ' %s="%s">' % (attr_name, value)
    return text[:match.start()] + new_tag + text[match.end():], True


def resolve_training_dimensions(config, series_list, log, dry_run=False):
    """(X, Y, Z) unbinned pixels to patch every XML's dimensions to.

    ``missalign.training_volume_dimensions`` if set explicitly (to
    deliberately tighten the volume MissAlignment trains against - see
    MissAlignment's own docs: "the tomogram shape should tightly fit your
    sample ... to avoid training on empty regions"); otherwise the same
    ``tomo_dimensions`` ``align_tiltseries`` already resolved for
    ``--tomo_dimensions``, so every XML ends up consistent with what the
    project was actually created with.
    """
    raw = str(config["missalign"]["training_volume_dimensions"] or "").strip()
    if not raw:
        return resolve_tomo_dimensions(config, series_list, log, dry_run=dry_run)
    try:
        x_str, y_str, z_str = raw.lower().split("x")
        return int(x_str), int(y_str), int(z_str)
    except ValueError:
        raise SystemExit(
            "ERROR: missalign.training_volume_dimensions must be 'XxYxZ', "
            "got %r" % raw
        )


def patch_training_dimensions(config, series_names, apix, x, y, z, log, dry_run=False):
    """Patch every tomogram's ``ImageDimensionsAngstrom``/
    ``VolumeDimensionsAngstrom`` to *x*, *y*, *z* (unbinned pixels) x *apix*.

    Always run, unconditionally, before MissAlignment trains - NOT a no-op:
    real, native ``ts_aretomo3`` output can leave these at ``[0.0, 0.0]``
    (confirmed against real WarpTools output, not just MissAlignment's own
    reference material assuming the worst case), which MissAlignment itself
    refuses to train on (``ValueError: ... has zero values in
    'ImageDimensionsAngstrom'``). Always overwritten with the same computed
    value on every run, regardless of what was there before - this is
    project-wide geometry, not something MissAlignment's own training
    refines, so re-patching to the same value is idempotent and safe even on
    a resumed run.
    """
    image_val = "%.6f, %.6f" % (x * apix, y * apix)
    volume_val = "%.6f, %.6f, %.6f" % (x * apix, y * apix, z * apix)

    patched = 0
    for name in series_names:
        path = os.path.join(config.warp_tiltseries_dir, name + ".xml")
        if not os.path.isfile(path) or dry_run:
            continue
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        text, hit1 = _patch_dimensions_attr(text, "ImageDimensionsAngstrom", image_val)
        text, hit2 = _patch_dimensions_attr(text, "VolumeDimensionsAngstrom", volume_val)
        if hit1 and hit2:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            patched += 1
        else:
            warn("%s: no ImageDimensionsAngstrom/VolumeDimensionsAngstrom "
                 "attribute found on the TiltSeries element - dimensions not "
                 "patched for this tomogram." % name)

    if patched:
        log.note("dimensions patched to %dx%dx%d (unbinned px) x %.4f A on "
                 "%d XML(s)" % (x, y, z, apix, patched))


# --------------------------------------------------------------- MissAlignment


def _partition_gpus(gpus):
    """training_devices / reconstruction_devices, matching the pattern in
    every runMiss_align_*.sh example: roughly half the GPUs train, the rest
    reconstruct (each repeated 3x). A single GPU shares both roles.
    """
    gpus = list(gpus) or [0]
    if len(gpus) == 1:
        return list(gpus), list(gpus) * 3
    n_train = max(1, len(gpus) // 2)
    training = gpus[:n_train]
    remaining = gpus[n_train:] or gpus[:1]
    reconstruction = []
    for gpu in remaining:
        reconstruction.extend([gpu, gpu, gpu])
    return training, reconstruction


def _batch_size(config):
    explicit = config["missalign"]["batch_size"]
    if explicit:
        return int(explicit)
    gpu_memory_gb = float(config["missalign"]["gpu_memory_gb"])
    # MissAlignment's own documented reference: batch_size 32 for 24 GB cards.
    return max(4, round(32 * gpu_memory_gb / 24.0))


def write_missalign_config(config, dry_run=False):
    """Write ``warp_tiltseries/missalign_config.yaml``.

    A cheap, pure-Python file write with no GPU/compute cost - but still
    skipped under ``dry_run`` so a dry run never touches the project.
    """
    config_path = os.path.join(config.warp_tiltseries_dir, "missalign_config.yaml")
    batch_size = _batch_size(config)
    payload = {
        "general": {
            "training_directory": config.warp_tiltseries_dir,
            # Hardcoded, not configurable: CTF correction happens once, in
            # WarpTools' own reconstruction (ts_ctf/ts_reconstruct);
            # enabling it here too only doubles MissAlignment's own
            # processing time for no benefit to alignment.
            "apply_ctf": False,
            "iteration_settings": _ITERATION_SETTINGS,
            "seed": 45132,
        },
        "model_training": {
            "model_architecture": "default",
            "model_checkpoint": None,
            "loss_margin": 0.5,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "max_epochs_per_iteration": 30,
            "warmup_steps": 500,
            "multistep_lr_scheduler": {"milestones": [5, 15], "gamma": 0.5},
        },
        "data_loading": {
            "batch_size": batch_size,
            "patch_size": 96,
            "steps_per_epoch": 1000,
        },
        "shift_generation": {
            "trajectory_probability": 0.5,
            "trajectory_max_shift": 10.0,
            "jitter_probability": 0.5,
            "jitter_max_std": 2.0,
            "outlier_probability": 0.5,
            "outlier_max_shift": 20.0,
            "fracture_probability": 0.5,
            "fracture_max_shift": 20.0,
        },
        "tilt_series_alignment": {
            "patch_size": 96,
            "patch_overlap": 0.1,
            "batch_size": batch_size,
        },
    }
    if not dry_run:
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        info("wrote %s" % config_path)
    return config_path


def _resume_iteration(config):
    """The value to pass as ``--start-at-iteration``, or 0 if never started.

    NOT ``highest_existing_iterN + 1``: miss-alignment's own resume logic
    (``sync_start_iteration_xmls``) requires ``iter<N>/`` to already exist
    for ``--start-at-iteration N`` - that directory IS the snapshot it
    resumes from, not the last one it finished. iter0/ is the baseline
    written before any training happens, so the highest ``iterN/`` actually
    on disk is itself the correct resume point.
    """
    highest = -1
    for entry in os.listdir(config.warp_tiltseries_dir):
        match = _ITER_DIR_RE.match(entry)
        if match:
            highest = max(highest, int(match.group(1)))
    return max(highest, 0)


def run_missalign_train(config, missalign_config_path, log, force=False,
                        dry_run=False):
    gpus = config.gpus
    settings = config["missalign"]
    training_devices = settings["training_devices"]
    reconstruction_devices = settings["reconstruction_devices"]
    if not training_devices or not reconstruction_devices:
        training_devices, reconstruction_devices = _partition_gpus(gpus)
    dataloaders = int(settings["dataloaders_per_trainer"] or 5)
    start_at = 0 if force else _resume_iteration(config)
    if start_at >= len(_ITERATION_SETTINGS):
        info("MissAlignment: all %d iterations already completed - skipping "
             "training. Use --force to retrain from scratch." % len(_ITERATION_SETTINGS))
        return True

    command = [
        config.exe("missalign"), "train",
        "--config-file", missalign_config_path,
        "--training-devices", ",".join(str(g) for g in training_devices),
        "--reconstruction-devices", ",".join(str(g) for g in reconstruction_devices),
        "--dataloaders-per-trainer", str(dataloaders),
        "--start-at-iteration", str(start_at),
        "--prepare-stacks", str(float(settings["prepare_stacks_gb"])),
    ]
    conda_env = str(settings["conda_env"] or "").strip()
    if conda_env:
        command = wrap_conda_run(command, config.exe("conda"), conda_env)
        info("running miss-alignment inside conda environment %r" % conda_env)
    env = {
        "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        # Without this, every process that hits a torch.compile'd region
        # spins up its OWN TorchInductor compile-worker pool sized to the
        # node's full CPU count. With N training processes x
        # dataloaders_per_trainer dataloader workers each potentially
        # triggering a compile, that is many such pools launching at once,
        # oversubscribing the node badly enough that no single compile ever
        # finishes - which then means the dataloader depending on it never
        # produces a batch, and training looks permanently "stuck" at near-
        # zero GPU memory with 0% CPU everywhere (confirmed via py-spy: the
        # main thread blocks forever in dataloader._try_get_data). Capping
        # this keeps compilation serialized/small per process instead.
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    logs = ensure_dir(config.missalignment_logs_dir)
    code = run_command(command, cwd=config.warp_tiltseries_dir, env=env,
              log_file=os.path.join(logs, "missalign_train.log"),
              check=False, dry_run=dry_run)

    if code != 0:
        log.fail("miss-alignment train",
                 "exit status %d - see %s" % (code, os.path.join(logs, "missalign_train.log")))
        return False
    log.task("miss-alignment train", True, "-")
    return True


# ------------------------------------------------------------------ entrypoint


def run(config, force=None, dry_run=False):
    """``tomo_denoising run_missalignment``"""
    force = config.overwrite if force is None else force
    with StepLog(config, STEP, force=force) as log:
        return _run(config, log, force=force, dry_run=dry_run)


def _run(config, log, force=False, dry_run=False):
    banner("MissAlignment-refined reconstruction")
    if not config["missalign"]["enabled"]:
        log.note("skipped: missalign.enabled is false")
        return 0

    require_executable(config.exe("warptools"), "WarpTools")
    if str(config["missalign"]["conda_env"] or "").strip():
        # miss-alignment lives inside that conda env, not necessarily on the
        # PATH tomo_denoising itself is running with - checking for it here
        # would just be a false warning. Check for conda/mamba instead, since
        # that's what actually needs to be reachable from here.
        require_executable(config.exe("conda"), "conda/mamba")
    else:
        require_executable(config.exe("missalign"), "miss-alignment")

    if not dry_run and not os.path.isfile(config.warp_tiltseries_settings_path):
        log.fail("run_missalignment", "no %s - run 'tomo_denoising "
                 "align_tiltseries' first" % config.warp_tiltseries_settings_path)
        return 1

    series_names = aligned_series_names(config)
    if not series_names and not dry_run:
        log.fail("run_missalignment", "no aligned tilt series (*.xml) in %s "
                 "- run 'tomo_denoising align_tiltseries' first"
                 % config.warp_tiltseries_dir)
        return 1
    info("%d tilt series considered for MissAlignment" % len(series_names))

    ensure_dir(config.missalignment_dir)
    ensure_dir(config.missalignment_logs_dir)

    banner("MissAlignment 1/4  back up the pre-MissAlignment alignment")
    backup_alignment(config, series_names, log, force=force, dry_run=dry_run)

    banner("MissAlignment 2/4  patch TiltSeries XML dimensions")
    preproc = collect_tilt_series(config.mdoc_dir("preprocessing"), "preprocessing")
    apix = dataset_apix(config, preproc)
    dim_x, dim_y, dim_z = resolve_training_dimensions(config, preproc, log,
                                                       dry_run=dry_run)
    patch_training_dimensions(config, series_names, apix, dim_x, dim_y, dim_z,
                              log, dry_run=dry_run)

    banner("MissAlignment 3/4  miss-alignment train")
    missalign_config_path = write_missalign_config(config, dry_run=dry_run)
    ok = run_missalign_train(config, missalign_config_path, log, force=force,
                             dry_run=dry_run)
    if not ok:
        return 1

    banner("MissAlignment 4/4  ts_reconstruct")
    recon_angpix = resolve_recon_angpix(config, log=log)
    ok = run_ts_reconstruct(
        config, config.warp_tiltseries_settings_path,
        config.missalignment_halfmaps_dir, config.missalignment_tiltsplit_halfmaps_dir,
        config.missalignment_logs_dir, series_names, recon_angpix,
        config.missalignment_reconstruction_dir,
        full_dir=config.missalignment_tomograms_dir, do_full=True,
        do_tiltsplit=config.tilt_split, log=log, force=force, dry_run=dry_run,
    )
    if not ok:
        return 1

    info("MissAlignment half maps: %s" % config.missalignment_halfmaps_dir)
    log.note("full tomograms: %s" % config.missalignment_tomograms_dir)
    log.note("half maps: %s" % config.missalignment_halfmaps_dir)
    return 0 if not log.failed else 1
