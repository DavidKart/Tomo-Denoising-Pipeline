"""CryoCARE training and prediction - the alternative denoiser to IsoNet2.

Selected with ``denoiser: "cryocare"`` for ``run_pipe``; the standalone
commands ``train_cryocare`` / ``predict_cryocare`` run regardless.

CryoCARE is noise2noise denoising of an even/odd pair, with no missing-wedge
correction. It is given the SAME half maps IsoNet2 gets - the contrast-
inverted copies (dark features) of the preferred frame-split half maps,
MissAlignment-corrected when they exist (see
``isonet._linked_half_maps`` / ``halfmap_sets``), linked per tomogram as
``<tomo>.mrc`` so the denoised output is named after the tomogram.

Layout, all under ``<processing_directory>/cryocare/``::

    train/{even,odd}/<tomo>.mrc     links to the training half maps
    predict/{even,odd}/<tomo>.mrc   links to the half maps to denoise
    train_data_config.json          cryoCARE_extract_train_data.py
    train_config.json               cryoCARE_train.py
    predict_config.json             cryoCARE_predict.py
    train_data/                     extracted training patches
    model/cryocare_model.tar.gz     the trained model
    denoised/<tomo>.mrc             THE deliverable

The fixed parameters are the values the previous (pre-WarpTools)
Tomography-Processing-Pipeline used; ``cryocare`` in the config exposes
epochs, patch_shape, num_slices and n_tiles.
"""

import json
import os

from . import tiltfilter
from .isonet import _linked_half_maps
from .mdoc import collect_tilt_series
from .steplog import StepLog, step_completed
from .utils import (
    banner, ensure_dir, gpu_env, gpu_id_note, gpu_ids_for, info, run, warn,
)

MODEL_NAME = "cryocare_model"
TRAIN_DATA_DIR = "train_data"
MODEL_DIR = "model"
DENOISED_DIR = "denoised"

#: Fixed settings, from the earlier pipeline's blueprint configs.
EXTRACT_FIXED = {
    "split": 0.9,
    "tilt_axis": "Y",
    "n_normalization_samples": 500,
}
TRAIN_FIXED = {
    "steps_per_epoch": 200,
    "batch_size": 16,
    "unet_kern_size": 3,
    "unet_n_depth": 3,
    "unet_n_first": 32,
    "learning_rate": 0.0004,
}


def model_path(config):
    """The archive cryoCARE_train.py writes: <path>/<model_name>.tar.gz."""
    return os.path.join(config.cryocare_dir, MODEL_DIR, MODEL_NAME + ".tar.gz")


def denoised_dir(config):
    return os.path.join(config.cryocare_dir, DENOISED_DIR)


def _int_list(value, key, length=3):
    if isinstance(value, (int, float)):
        value = [value] * length
    try:
        values = [int(v) for v in value]
    except (TypeError, ValueError):
        values = []
    if len(values) != length or min(values) <= 0:
        raise SystemExit("ERROR: cryocare.%s must be %d positive whole numbers, "
                         "e.g. [72, 72, 72] - got %r" % (key, length, value))
    return values


def _gpu_ids(config):
    return gpu_ids_for("cryocare", config.gpus)


def _write_json(path, payload, dry_run):
    if dry_run:
        info("would write %s" % path)
        return path
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
    info("wrote %s" % path)
    return path


def _half_map_lists(config, series_list, subset, log, force, dry_run):
    """Link the half maps and return (even paths, odd paths, usable
    series), in the same (name-sorted) order for both halves."""
    even_dir, odd_dir, usable = _linked_half_maps(
        config, series_list, subset, log=log, force=force, dry_run=dry_run,
        base_dir=config.cryocare_dir)
    even = [os.path.join(even_dir, s.name + ".mrc") for s in usable]
    odd = [os.path.join(odd_dir, s.name + ".mrc") for s in usable]
    return even, odd, usable


# ------------------------------------------------------------------ training


def train(config, force=None, dry_run=False):
    """``tomo_denoising train_cryocare``"""
    force = config.overwrite if force is None else force
    with StepLog(config, "train_cryocare", force=force) as log:
        return _train(config, log, force=force, dry_run=dry_run)


def _train(config, log, force=False, dry_run=False):
    banner("CryoCARE training")
    for name, purpose in (("cryocare_extract", "CryoCARE (training data)"),
                          ("cryocare_train", "CryoCARE (training)")):
        config.require(name, purpose)

    work = ensure_dir(config.cryocare_dir)
    logs = ensure_dir(work, "logs")
    settings = config["cryocare"]

    if (not force and step_completed(config, "train_cryocare")
            and os.path.isfile(model_path(config))):
        info("CryoCARE is already trained: %s" % model_path(config))
        log.note("skipped: the previous training completed and %s exists. "
                 "Use --force to retrain." % model_path(config))
        return 0

    series_list = tiltfilter.drop_excluded(
        config, collect_tilt_series(config.mdoc_dir("train"), "train"),
        "CryoCARE training", log=log)
    if not series_list:
        log.fail("train_cryocare", "every tilt series in %s is excluded (see "
                 "%s)" % (config.mdoc_dir("train"), tiltfilter.record_path(config)))
        return 1
    log.note("training on %d tilt series from %s"
             % (len(series_list), config.mdoc_dir("train")))
    log.note(gpu_id_note("cryocare", config.gpus))

    banner("CryoCARE 1/2  extract training data")
    even, odd, usable = _half_map_lists(config, series_list, "train", log,
                                        force, dry_run)
    extract_conf = dict(EXTRACT_FIXED)
    extract_conf.update({
        "even": even,
        "odd": odd,
        "patch_shape": _int_list(settings["patch_shape"], "patch_shape"),
        "num_slices": int(settings["num_slices"]),
        "path": os.path.join(work, TRAIN_DATA_DIR),
        # Re-running replaces the previous extraction instead of stopping.
        "overwrite": True,
    })
    conf = _write_json(os.path.join(work, "train_data_config.json"),
                       extract_conf, dry_run)
    run(config.command("cryocare_extract", "--conf", conf),
        cwd=work, env=gpu_env(config.gpus),
        log_file=os.path.join(logs, "extract_train_data.log"), dry_run=dry_run)

    banner("CryoCARE 2/2  train (%d epochs)" % int(settings["epochs"]))
    train_conf = dict(TRAIN_FIXED)
    train_conf.update({
        "train_data": os.path.join(work, TRAIN_DATA_DIR),
        "epochs": int(settings["epochs"]),
        "model_name": MODEL_NAME,
        "path": ensure_dir(work, MODEL_DIR),
        "gpu_id": _gpu_ids(config),
    })
    conf = _write_json(os.path.join(work, "train_config.json"), train_conf,
                       dry_run)
    run(config.command("cryocare_train", "--conf", conf),
        cwd=work, env=gpu_env(config.gpus),
        log_file=os.path.join(logs, "train.log"), dry_run=dry_run)

    if not dry_run and not os.path.isfile(model_path(config)):
        log.fail("cryoCARE_train", "produced no %s - see %s"
                 % (model_path(config), os.path.join(logs, "train.log")))
        return 1

    log.note("model: %s" % model_path(config))
    for entry in usable:
        log.ok(entry.name)
    return 0


# ---------------------------------------------------------------- prediction


def predict(config, force=None, dry_run=False):
    """``tomo_denoising predict_cryocare``"""
    force = config.overwrite if force is None else force
    with StepLog(config, "predict_cryocare", force=force) as log:
        return _predict(config, log, force=force, dry_run=dry_run)


def _predict(config, log, force=False, dry_run=False):
    banner("CryoCARE prediction")
    config.require("cryocare_predict", "CryoCARE (prediction)")

    work = ensure_dir(config.cryocare_dir)
    logs = ensure_dir(work, "logs")
    output = denoised_dir(config)

    series_list = tiltfilter.drop_excluded(
        config, collect_tilt_series(config.mdoc_dir("predict"), "predict"),
        "CryoCARE prediction", log=log)
    if not series_list:
        log.fail("predict_cryocare", "every tilt series in %s is excluded (see "
                 "%s)" % (config.mdoc_dir("predict"), tiltfilter.record_path(config)))
        return 1
    log.note("predicting %d tilt series from %s"
             % (len(series_list), config.mdoc_dir("predict")))

    if not force and step_completed(config, "predict_cryocare"):
        info("CryoCARE prediction already complete.")
        log.note("skipped: the previous prediction completed. "
                 "Use --force to redo it.")
    else:
        if not dry_run and not os.path.isfile(model_path(config)):
            log.fail("predict_cryocare", "no trained model at %s - run "
                     "'tomo_denoising train_cryocare' first" % model_path(config))
            return 1
        log.note(gpu_id_note("cryocare", config.gpus))
        even, odd, usable = _half_map_lists(config, series_list, "predict",
                                            log, force, dry_run)
        predict_conf = {
            "path": model_path(config),
            "even": even,
            "odd": odd,
            "n_tiles": _int_list(config["cryocare"]["n_tiles"], "n_tiles"),
            "output": output,
            # The output folder exists after the first run; cryoCARE stops
            # on that unless told to overwrite.
            "overwrite": True,
            "gpu_id": _gpu_ids(config),
        }
        conf = _write_json(os.path.join(work, "predict_config.json"),
                           predict_conf, dry_run)
        run(config.command("cryocare_predict", "--conf", conf),
            cwd=work, env=gpu_env(config.gpus),
            log_file=os.path.join(logs, "predict.log"), dry_run=dry_run)
        if not dry_run:
            missing = [s.name for s in usable
                       if not os.path.isfile(os.path.join(output, s.name + ".mrc"))]
            if missing:
                warn("CryoCARE wrote no denoised tomogram for: %s - see %s"
                     % (", ".join(missing), os.path.join(logs, "predict.log")))
                for name in missing:
                    log.fail(name, "no %s.mrc in %s" % (name, output))

    info("CryoCARE prediction finished. Denoised tomograms: %s" % output)
    log.note("denoised tomograms: %s" % output)
    for entry in series_list:
        if entry.name not in {name for name, _ in log.failed}:
            log.ok(entry.name)

    if not dry_run:
        from . import report
        report.refresh_after_prediction(config, "cryocare", log=log)
    return 0 if not log.failed else 1
