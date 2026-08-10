"""Steps 2 and 3: cryoCARE training and prediction.

The three cryoCARE .json configuration files are generated from the pipeline
configuration into ``<processing_directory>/cryocare_config`` immediately
before they are used, so they always match the current settings and the
current set of half maps.
"""

import json
import os

from .halfmap_sets import collect_half_maps
from .steplog import StepLog, step_completed
from .utils import (
    banner,
    ensure_dir,
    info,
    require_executable,
    run,
)


def _write_json(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
    info("wrote %s" % path)
    return path


def _train_data_dir(config):
    return os.path.join(config.cryocare_dir, "train_data")


def _model_dir(config):
    return os.path.join(config.cryocare_dir, "model")


def _model_tarball(config):
    return os.path.join(
        _model_dir(config), "%s.tar.gz" % config["cryocare"]["model_name"]
    )


def write_train_configs(config, pairs):
    """Generate train_data_config.json and train_config.json."""
    settings = config["cryocare"]
    config_dir = ensure_dir(config.cryocare_config_dir)

    train_data_config = _write_json(
        os.path.join(config_dir, "train_data_config.json"),
        {
            "even": [even for _, even, _ in pairs],
            "odd": [odd for _, _, odd in pairs],
            "patch_shape": list(settings["patch_shape"]),
            "num_slices": int(settings["num_slices"]),
            "split": float(settings["split"]),
            "tilt_axis": settings["tilt_axis"],
            "n_normalization_samples": int(settings["n_normalization_samples"]),
            "path": _train_data_dir(config),
            # Without this, re-running training aborts once the directory
            # exists - after the normalization pass has already been done.
            "overwrite": bool(settings["overwrite"]),
        },
    )

    train_config = _write_json(
        os.path.join(config_dir, "train_config.json"),
        {
            "train_data": _train_data_dir(config),
            "epochs": int(settings["epochs"]),
            "steps_per_epoch": int(settings["steps_per_epoch"]),
            "batch_size": int(settings["batch_size"]),
            "unet_kern_size": int(settings["unet_kern_size"]),
            "unet_n_depth": int(settings["unet_n_depth"]),
            "unet_n_first": int(settings["unet_n_first"]),
            "learning_rate": float(settings["learning_rate"]),
            "model_name": settings["model_name"],
            "path": _model_dir(config),
            "gpu_id": config.gpus,
        },
    )
    return train_data_config, train_config


def write_predict_config(config, pairs):
    """Generate predict_config.json."""
    settings = config["cryocare"]
    config_dir = ensure_dir(config.cryocare_config_dir)
    return _write_json(
        os.path.join(config_dir, "predict_config.json"),
        {
            "path": _model_tarball(config),
            "even": [even for _, even, _ in pairs],
            "odd": [odd for _, _, odd in pairs],
            "n_tiles": list(settings["n_tiles"]),
            "output": os.path.join(config.processing_dir, "denoised_cryoCARE"),
            "overwrite": bool(settings["overwrite"]),
            "gpu_id": config.gpus,
        },
    )


# ------------------------------------------------------------------ entrypoints


def train(config, force=None, dry_run=False):
    """``tomo_denoising train_cryoCARE``"""
    force = config.overwrite if force is None else force
    with StepLog(config, "train_cryoCARE", force=force) as log:
        return _train(config, log, force=force, dry_run=dry_run)


def _train(config, log, force=False, dry_run=False):
    banner("cryoCARE training")
    require_executable(config.exe("cryocare_extract"), "cryoCARE extraction")
    require_executable(config.exe("cryocare_train"), "cryoCARE training")

    model = _model_tarball(config)
    if not force and os.path.exists(model) and step_completed(
            config, "train_cryoCARE"):
        info("cryoCARE model already trained: %s" % model)
        log.note("skipped: %s already exists and the previous training "
                 "completed. Use --force to retrain." % model)
        return 0

    pairs = collect_half_maps(config, "train")
    log.note("training on %d tomograms:" % len(pairs))
    for name, _, _ in pairs:
        info("  training on %s" % name)
        log.quiet("    %s" % name)

    # Only the parent is created here. cryoCARE owns train_data/ and model/
    # and refuses to write into a directory that already exists unless
    # 'overwrite' is set, so pre-creating them would abort the run after the
    # normalization pass has already finished.
    ensure_dir(config.cryocare_dir)
    logs = ensure_dir(config.cryocare_dir, "logs")

    train_data_config, train_config = write_train_configs(config, pairs)

    run([config.exe("cryocare_extract"), "--conf", train_data_config],
        cwd=config.cryocare_dir,
        log_file=os.path.join(logs, "extract_train_data.log"),
        dry_run=dry_run)

    run([config.exe("cryocare_train"), "--conf", train_config],
        cwd=config.cryocare_dir,
        log_file=os.path.join(logs, "train.log"),
        dry_run=dry_run)

    if not dry_run and not os.path.exists(_model_tarball(config)):
        log.fail("cryoCARE training", "no model at %s" % _model_tarball(config))
        return 1
    info("model: %s" % _model_tarball(config))
    log.note("model: %s" % _model_tarball(config))
    log.ok("cryoCARE training")
    return 0


def predict(config, force=None, dry_run=False):
    """``tomo_denoising predict_cryoCARE``"""
    force = config.overwrite if force is None else force
    with StepLog(config, "predict_cryoCARE", force=force) as log:
        return _predict(config, log, force=force, dry_run=dry_run)


def _predict(config, log, force=False, dry_run=False):
    banner("cryoCARE prediction")
    require_executable(config.exe("cryocare_predict"), "cryoCARE prediction")

    model = _model_tarball(config)
    if not os.path.exists(model) and not dry_run:
        raise SystemExit(
            "ERROR: no trained cryoCARE model at %s\n"
            "Run 'tomo_denoising train_cryoCARE' first." % model
        )

    pairs = collect_half_maps(config, "predict")
    output = ensure_dir(config.processing_dir, "denoised_cryoCARE")
    logs = ensure_dir(config.cryocare_dir, "logs")

    if not force and step_completed(config, "predict_cryoCARE") and os.listdir(
            output):
        info("cryoCARE prediction already complete: %s" % output)
        log.note("skipped: %s already holds a completed prediction. "
                 "Use --force to redo it." % output)
    else:
        predict_config = write_predict_config(config, pairs)
        run([config.exe("cryocare_predict"), "--conf", predict_config],
            cwd=config.cryocare_dir,
            log_file=os.path.join(logs, "predict.log"),
            dry_run=dry_run)

    info("denoised tomograms: %s" % output)
    log.note("denoised tomograms: %s" % output)
    for name, _, _ in pairs:
        log.ok(name)

    if not dry_run:
        from . import report
        report.refresh_after_prediction(config, "cryoCARE", log=log)
    return 0
