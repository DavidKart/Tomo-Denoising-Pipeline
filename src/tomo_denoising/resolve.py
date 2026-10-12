"""``tomo_denoising quality_check``: resolution estimation with RESOLVE.

RESOLVE (https://github.com/DavidKart/RESOLVE) is run in batch mode over a
directory of even/odd half maps, which is what it needs - two genuinely
independent reconstructions of the same volume. It writes ``summary.tsv`` with
one row per tomogram: ``id``, ``median_resolution`` and ``signal_ratio``.

WHICH half maps is ``quality_check.enable_tilt_split``:

* ``true``  - the dedicated TILT-split reconstruction, made at the coarse
  ``quality_check.tilt_split_target_pixel_size`` purely for this measurement.
  The two halves differ substantially, so the number is conservative.
* ``false`` - the FRAME-split half maps that are reconstructed anyway for
  IsoNet, with no extra reconstruction at all. The two halves differ much
  less, so the number is more optimistic. Nothing else changes: the ranking
  it produces is what the training-set choice needs, and the figure says
  "Frame-split resolution" instead of "Tilt-split resolution".

Within that, MissAlignment's own half maps are preferred over the plain
WarpTools ones whenever they exist - see
``halfmap_sets.resolve_quality_check_halfmaps_dir``. If the tilt-split half
maps are wanted but were never reconstructed (the normal state when
``enable_tilt_split`` was off during ``run_pipe``, then switched on later),
``halfmap_sets.ensure_halfmaps_available`` reconstructs them at this point.

Those numbers are used for two things: choosing a training set when none has
been picked yet, and captioning the results figure (see :mod:`report`).
"""

import csv
import json
import os

from .halfmap_sets import (
    ensure_halfmaps_available, resolve_quality_check_halfmaps_dir,
)
from . import tiltfilter
from .mdoc import find_mdocs, match_tomo_name, tomo_name_from_mdoc
from .steplog import StepLog
from .utils import (
    banner,
    ensure_dir,
    gpu_env,
    gpu_id_note,
    gpu_ids_for,
    info,
    link_file,
    run,
    warn,
)

STEP = "quality_check"

SUMMARY_NAME = "summary.tsv"

#: Prefixes the half maps carry in raw_halfMaps and raw_halfMaps_tiltSplit
#: alike - the same naming convention is used for both, which is why RESOLVE
#: can be pointed at either without any other change.
EVEN_ID = "even_"
ODD_ID = "odd_"


class Measurement(object):
    """One row of RESOLVE's summary.tsv."""

    def __init__(self, tomo, resolution, signal_ratio):
        self.tomo = tomo
        self.resolution = resolution
        self.signal_ratio = signal_ratio

    def __repr__(self):  # pragma: no cover - debugging aid
        return "<%s %.1fA sig=%.2f>" % (self.tomo, self.resolution,
                                        self.signal_ratio)


def summary_path(config, output_dir=None):
    return os.path.join(output_dir or config.resolve_dir, SUMMARY_NAME)


def known_tomo_names(config):
    """Every tomogram name the pipeline knows about, from the mdocs."""
    return [tomo_name_from_mdoc(p)
            for p in find_mdocs(config.mdoc_dir("preprocessing"))]


def read_summary(config, tomo_names=None, output_dir=None):
    """Parse summary.tsv into ``{tomo_name: Measurement}``.

    RESOLVE ids carry its own affixes, e.g.
    ``odd_Position_1_rec_tomo_Tomograms_locRes``. Rather than trying to strip
    every possible one, each id is matched against the tomogram names taken
    from the mdocs, as a whole underscore-delimited run - so Position_1 never
    picks up Position_10's row.

    *output_dir* defaults to ``config.resolve_dir``, the only place
    ``quality_check`` ever writes ``summary.tsv`` - whichever half maps were
    actually measured (tilt-split or frame-split, MissAlignment's or the
    plain ones - see ``halfmap_sets.resolve_quality_check_halfmaps_dir``).
    """
    path = summary_path(config, output_dir)
    if not os.path.isfile(path):
        return {}

    if tomo_names is None:
        tomo_names = known_tomo_names(config)

    results = {}
    unmatched = []
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            return {}
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        id_key = fields.get("id")
        res_key = fields.get("median_resolution")
        sig_key = fields.get("signal_ratio")
        if not (id_key and res_key):
            warn("%s has unexpected columns: %s"
                 % (path, ", ".join(reader.fieldnames)))
            return {}

        for row in reader:
            try:
                resolution = float(row[res_key])
            except (TypeError, ValueError):
                continue
            try:
                signal = float(row[sig_key]) if sig_key else float("nan")
            except (TypeError, ValueError):
                signal = float("nan")

            identifier = str(row[id_key]).strip()
            name = match_tomo_name(identifier, tomo_names)
            if name is None:
                unmatched.append(identifier)
                continue
            results[name] = Measurement(name, resolution, signal)

    # Matched against EVERY known name above (an excluded "Position_21_2"
    # must not be mistaken for "Position_21"), dropped only now: an excluded
    # tilt series' old result must never pick it for training.
    for name in tiltfilter.excluded_names(config):
        results.pop(name, None)

    if unmatched:
        warn("%d row(s) in %s could not be matched to a tomogram: %s\n"
             "  known tomograms: %s"
             % (len(unmatched), path, ", ".join(unmatched[:5]),
                ", ".join(sorted(tomo_names)[:10]) or "(none)"))
    return results


# ----------------------------------------------------------- what was measured


MEASURED_MARKER = ".measured.json"


def _marker_path(output_dir):
    return os.path.join(output_dir, MEASURED_MARKER)


def _read_marker(output_dir):
    try:
        with open(_marker_path(output_dir), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _write_marker(output_dir, inputs, tilt_split):
    with open(_marker_path(output_dir), "w", encoding="utf-8") as handle:
        json.dump({"input_dir": os.path.abspath(inputs),
                   "tilt_split": bool(tilt_split)}, handle, indent=2)


def _summary_is_stale(output_dir, inputs, tilt_split):
    """True when an existing summary.tsv measured something ELSE than what is
    being asked for now.

    Without this, flipping ``quality_check.enable_tilt_split`` (or
    MissAlignment's half maps appearing after a plain run) would silently
    reuse the previous summary.tsv and report the OLD measurement under the
    new label - the one way the resume logic could actively mislead rather
    than just save time. A marker older than this check (no marker at all)
    is treated as matching, so an existing project is not re-measured just
    for upgrading.
    """
    marker = _read_marker(output_dir)
    if not marker:
        return False
    if bool(marker.get("tilt_split")) != bool(tilt_split):
        return True
    recorded = marker.get("input_dir")
    return bool(recorded) and os.path.abspath(recorded) != os.path.abspath(inputs)


# ------------------------------------------------------------------- running


def run_resolve(config, log, force=False, dry_run=False, input_dir=None,
                output_dir=None, log_name="resolve.log"):
    """Run RESOLVE in batch mode over a directory of even/odd half maps.

    Defaults to the plain-WarpTools ``config.tiltsplit_halfmaps_dir`` ->
    ``config.resolve_dir``; ``quality_check`` itself always passes
    *input_dir* explicitly (see
    ``halfmap_sets.resolve_quality_check_halfmaps_dir``).
    """
    inputs = input_dir or config.tiltsplit_halfmaps_dir
    if not os.path.isdir(inputs) or not os.listdir(inputs):
        raise SystemExit(
            "ERROR: no half maps to measure in %s\n"
            "quality_check measures the %s half maps "
            "(quality_check.enable_tilt_split is %s). Run 'tomo_denoising "
            "reconstruct_tomograms' first."
            % (inputs, "tilt-split" if config.tilt_split else "frame-split",
               str(config.tilt_split).lower())
        )

    output = ensure_dir(output_dir or config.resolve_dir)
    settings = config["quality_check"]

    stale = _summary_is_stale(output, inputs, config.tilt_split)
    if not force and not stale and os.path.isfile(summary_path(config, output)):
        info("reusing the existing %s (use --force to re-measure)"
             % summary_path(config, output))
        log.note("RESOLVE: reused the existing summary.tsv")
        return True
    if stale and os.path.isfile(summary_path(config, output)):
        previous = _read_marker(output)
        message = ("re-measuring: the existing summary.tsv is from the %s half "
                   "maps in %s, this run wants the %s ones in %s"
                   % ("tilt-split" if previous.get("tilt_split") else "frame-split",
                      previous.get("input_dir"),
                      "tilt-split" if config.tilt_split else "frame-split",
                      inputs))
        info(message)
        log.note(message)

    command = config.command(
        "resolve",
        "--mode", "batch",
        "--config", "Tomograms",
        "--inputDir", inputs,
        "--even_id", EVEN_ID,
        "--odd_id", ODD_ID,
        "--outputDir", output)
    if settings["fast"]:
        command.append("--fast")
    if settings["gpu_enabled"]:
        command.append("--gpu_enabled")
        # --gpu_settings takes CUDA device indices, used as
        # torch.device(f"cuda:{gpu_id}") (RESOLVE's src/resolve.py) - so they
        # are the ids INSIDE the CUDA_VISIBLE_DEVICES-restricted process, not
        # the real ones. Passed explicitly rather than left to RESOLVE's own
        # default: the default is every visible device, which happens to be
        # the right set here, but relying on that would silently depend on us
        # never loosening the environment.
        command += ["--gpu_settings",
                    ",".join(str(g) for g in gpu_ids_for("resolve", config.gpus))]
        log.note(gpu_id_note("resolve", config.gpus))
    if int(settings["cpu_threads"]) > 0:
        command += ["--cpu_threads", str(int(settings["cpu_threads"]))]

    # Both halves matter: --gpu_settings above says which devices to put the
    # per-tomogram jobs on, CUDA_VISIBLE_DEVICES makes the rest unreachable
    # for anything in the process that does not consult that setting.
    run(command, cwd=output, env=gpu_env(config.gpus),
        log_file=os.path.join(ensure_dir(config.logs_dir), log_name),
        check=False, dry_run=dry_run)

    if dry_run:
        return True
    if not os.path.isfile(summary_path(config, output)):
        log.fail("RESOLVE", "produced no %s in %s" % (SUMMARY_NAME, output))
        return False
    _write_marker(output, inputs, config.tilt_split)
    return True


# ------------------------------------------------------ training set selection


def training_set_size(total):
    """How many tomograms to train on, given how many there are."""
    if total >= 20:
        return 5
    if total >= 10:
        return 4
    return 3


def select_training_tomograms(config, measurements, log):
    """Pick the training tomograms from the RESOLVE ranking.

    Best resolution first, never anything below ``min_signal_ratio``, and any
    pick whose resolution is worse than ``outlier_factor`` times the best is
    dropped - one bad tomogram in a small training set skews the model.
    """
    settings = config["quality_check"]
    min_signal = float(settings["min_signal_ratio"])
    factor = float(settings["outlier_factor"])

    total = len(measurements)
    eligible = [m for m in measurements.values()
                if not (m.signal_ratio != m.signal_ratio)   # not NaN
                and m.signal_ratio >= min_signal]
    rejected = total - len(eligible)
    if rejected:
        log.note("  %d of %d tomograms rejected for signal_ratio < %.2f"
                 % (rejected, total, min_signal))

    if not eligible:
        return [], ("no tomogram reached the minimum signal ratio of %.2f"
                    % min_signal)

    ranked = sorted(eligible, key=lambda m: m.resolution)
    wanted = training_set_size(total)
    picked = ranked[:wanted]

    best = picked[0].resolution
    kept = [m for m in picked if m.resolution <= best * factor]
    dropped = [m for m in picked if m not in kept]
    for measurement in dropped:
        log.note("  dropped %s (%.1f A, worse than %.1fx the best %.1f A)"
                 % (measurement.tomo, measurement.resolution, factor, best))

    if not kept:
        return [], "every candidate was rejected as a resolution outlier"
    return kept, ""


def _is_self_created(mdoc_path, source_dir):
    """True when *mdoc_path* in mdocs.train is one WE put there.

    :func:`populate_training_mdocs` links the chosen mdocs out of
    mdocs.preprocessing, so anything that is a symlink - or, where symlinks
    are unavailable and ``link_file`` fell back to a hard link, anything
    sharing an inode with the preprocessing mdoc of the same name - came from
    a previous automatic selection. Anything else is a file a human put in
    that directory, and is never touched.
    """
    if os.path.islink(mdoc_path):
        return True
    original = os.path.join(source_dir, os.path.basename(mdoc_path))
    try:
        return os.path.isfile(original) and os.path.samefile(mdoc_path, original)
    except OSError:
        return False


def populate_training_mdocs(config, measurements, log, force=False,
                            dry_run=False):
    """Fill the mdocs.train directory from the RESOLVE ranking, if it is empty.

    Returns True when training may proceed.

    An existing selection is normally never touched - that directory is one
    you may have curated by hand. *force* (``--force``/``overwrite: true``)
    re-picks from THIS run's fresh RESOLVE ranking, but only by discarding
    the links a previous automatic selection made (see
    :func:`_is_self_created`): a curated mdoc that was copied or moved in by
    hand still survives, and if any such file is present the selection is
    left alone entirely, forced or not.
    """
    if not config["quality_check"]["choose_training_set"]:
        return True

    train_dir = config.mdoc_dir("train")
    source_dir = config.mdoc_dir("preprocessing")
    existing = find_mdocs(train_dir)

    if force and existing:
        own = [path for path in existing
               if _is_self_created(path, source_dir)]
        curated = [path for path in existing if path not in own]
        if own and not curated:
            for path in own:
                if not dry_run:
                    os.unlink(path)
            info("discarded the %d previously auto-selected mdoc(s) in %s - "
                 "re-choosing from this run's RESOLVE ranking" % (len(own), train_dir))
            log.note("training set re-chosen (--force): discarded %d "
                     "previously auto-selected mdoc(s) in %s"
                     % (len(own), train_dir))
            existing = []
        elif curated:
            log.note("training set kept despite --force: %d of the %d mdoc(s) "
                     "in %s are not links this pipeline made, so the "
                     "selection is treated as hand-curated (%s)"
                     % (len(curated), len(existing), train_dir,
                        ", ".join(sorted(os.path.basename(p) for p in curated))))

    if existing:
        info("training set already chosen: %d mdoc(s) in %s"
             % (len(existing), train_dir))
        log.note("training set untouched (%d mdoc(s) already in %s)"
                 % (len(existing), train_dir))
        return True

    if not measurements:
        log.fail("training set", "no RESOLVE measurements to choose from")
        return False

    banner("Choosing a training set from the RESOLVE ranking")
    picked, problem = select_training_tomograms(config, measurements, log)

    if not picked:
        message = (
            "Could not choose a training set automatically: %s.\n"
            "Populate %s with the mdocs of the tomograms you want to train "
            "on, then run the training step again." % (problem, train_dir)
        )
        log.fail("training set", problem)
        log.note(message)
        warn(message)
        return False

    by_name = {tomo_name_from_mdoc(p): p for p in find_mdocs(source_dir)}

    ensure_dir(train_dir)
    linked = []
    for measurement in picked:
        mdoc = by_name.get(measurement.tomo)
        if mdoc is None:
            warn("no mdoc found for %s in %s" % (measurement.tomo, source_dir))
            continue
        target = os.path.join(train_dir, os.path.basename(mdoc))
        if not dry_run:
            link_file(mdoc, target, force=force)
        linked.append(measurement)
        info("  %-28s %7.1f A   signal %.2f"
             % (measurement.tomo, measurement.resolution,
                measurement.signal_ratio))

    if not linked:
        log.fail("training set", "none of the chosen tomograms had an mdoc")
        return False

    log.note("training set chosen (%d tomograms, linked into %s):"
             % (len(linked), train_dir))
    for measurement in linked:
        log.note("    %-28s %7.1f A   signal %.2f"
                 % (measurement.tomo, measurement.resolution,
                    measurement.signal_ratio))
    return True


# ------------------------------------------------------------------ entrypoint


def quality_check(config, force=None, dry_run=False):
    """``tomo_denoising quality_check``"""
    force = config.overwrite if force is None else force

    with StepLog(config, STEP, force=force) as log:
        config.require("resolve", "RESOLVE")
        banner("RESOLVE quality check")

        ensure_halfmaps_available(config, log=log, dry_run=dry_run,
                                  need_tiltsplit=config.tilt_split)
        inputs = resolve_quality_check_halfmaps_dir(config, log=log)
        log.note("measuring the %s half maps in %s"
                 % ("tilt-split" if config.tilt_split else "frame-split",
                    inputs))

        if not run_resolve(config, log, force=force, dry_run=dry_run,
                           input_dir=inputs):
            return 1

        measurements = read_summary(config)
        if dry_run:
            log.note("dry run: no measurements read")
            return 0

        if not measurements:
            log.fail("RESOLVE", "summary.tsv contained no usable rows")
            return 1

        log.note("%d tomograms measured:" % len(measurements))
        for measurement in sorted(measurements.values(),
                                  key=lambda m: m.resolution):
            log.note("    %-28s %7.1f A   signal %.2f"
                     % (measurement.tomo, measurement.resolution,
                        measurement.signal_ratio))
            log.ok(measurement.tomo, "%.1f A" % measurement.resolution)

        ok = populate_training_mdocs(config, measurements, log, force=force,
                                     dry_run=dry_run)

        # The figure needs both halves of the story; build whatever is ready.
        from . import report
        produced = report.build_all_reports(config, measurements, log)
        if not produced:
            log.note("No denoised tomograms yet - the results figure will be "
                     "written once a prediction step has run.")

        return 0 if ok else 2
