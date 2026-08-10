"""``tomo_denoising quality_check``: resolution estimation with RESOLVE.

RESOLVE (https://github.com/DavidKart/RESOLVE) is run in batch mode over the
tilt-split half maps, which is what it needs - two genuinely independent
reconstructions of the same volume. It writes ``summary.tsv`` with one row per
tomogram: ``id``, ``median_resolution`` and ``signal_ratio``.

Those numbers are used for two things: choosing a training set when none has
been picked yet, and captioning the results figure (see :mod:`report`).
"""

import csv
import os

from .mdoc import find_mdocs, match_tomo_name, tomo_name_from_mdoc
from .steplog import StepLog
from .utils import (
    banner,
    ensure_dir,
    info,
    link_file,
    require_executable,
    run,
    warn,
)

STEP = "quality_check"

SUMMARY_NAME = "summary.tsv"

#: Prefixes the half maps carry in raw_halfMaps_tiltSplit.
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


def summary_path(config):
    return os.path.join(config.resolve_dir, SUMMARY_NAME)


def known_tomo_names(config):
    """Every tomogram name the pipeline knows about, from the mdocs."""
    return [tomo_name_from_mdoc(p)
            for p in find_mdocs(config.mdoc_dir("preprocessing"))]


def read_summary(config, tomo_names=None):
    """Parse summary.tsv into ``{tomo_name: Measurement}``.

    RESOLVE ids carry its own affixes, e.g.
    ``odd_Position_1_rec_tomo_Tomograms_locRes``. Rather than trying to strip
    every possible one, each id is matched against the tomogram names taken
    from the mdocs, as a whole underscore-delimited run - so Position_1 never
    picks up Position_10's row.
    """
    path = summary_path(config)
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

    if unmatched:
        warn("%d row(s) in %s could not be matched to a tomogram: %s\n"
             "  known tomograms: %s"
             % (len(unmatched), path, ", ".join(unmatched[:5]),
                ", ".join(sorted(tomo_names)[:10]) or "(none)"))
    return results


# ------------------------------------------------------------------- running


def run_resolve(config, log, force=False, dry_run=False):
    """Run RESOLVE in batch mode over the tilt-split half maps."""
    inputs = config.tiltsplit_halfmaps_dir
    if not os.path.isdir(inputs) or not os.listdir(inputs):
        raise SystemExit(
            "ERROR: no tilt-split half maps in %s\n"
            "quality_check measures the tilt-split reconstructions. Set\n"
            "  tilt_split:\n    enabled: true\n"
            "and run 'tomo_denoising create_halfMaps' first." % inputs
        )

    output = ensure_dir(config.resolve_dir)
    settings = config["quality_check"]

    if not force and os.path.isfile(summary_path(config)):
        info("reusing the existing %s (use --force to re-measure)"
             % summary_path(config))
        log.note("RESOLVE: reused the existing summary.tsv")
        return True

    command = [config.exe("resolve"),
               "--mode", "batch",
               "--config", "Tomograms",
               "--inputDir", inputs,
               "--even_id", EVEN_ID,
               "--odd_id", ODD_ID,
               "--outputDir", output]
    if settings["fast"]:
        command.append("--fast")
    if settings["gpu_enabled"]:
        command.append("--gpu_enabled")
    if int(settings["cpu_threads"]) > 0:
        command += ["--cpu_threads", str(int(settings["cpu_threads"]))]

    run(command, cwd=output,
        log_file=os.path.join(ensure_dir(config.logs_dir), "resolve.log"),
        check=False, dry_run=dry_run)

    if dry_run:
        return True
    if not os.path.isfile(summary_path(config)):
        log.fail("RESOLVE", "produced no %s in %s" % (SUMMARY_NAME, output))
        return False
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


def populate_training_mdocs(config, measurements, log, dry_run=False):
    """Fill the mdocs.train directory from the RESOLVE ranking, if it is empty.

    An existing selection is never touched. Returns True when training may
    proceed.
    """
    if not config["quality_check"]["choose_training_set"]:
        return True

    train_dir = config.mdoc_dir("train")
    existing = find_mdocs(train_dir)
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

    source_dir = config.mdoc_dir("preprocessing")
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
            link_file(mdoc, target)
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
        require_executable(config.exe("resolve"), "RESOLVE")
        banner("RESOLVE quality check")

        if not run_resolve(config, log, force=force, dry_run=dry_run):
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

        ok = populate_training_mdocs(config, measurements, log, dry_run=dry_run)

        # The figure needs both halves of the story; build whatever is ready.
        from . import report
        produced = report.build_all_reports(config, measurements, log)
        if not produced:
            log.note("No denoised tomograms yet - the results figure will be "
                     "written once a prediction step has run.")

        return 0 if ok else 2
