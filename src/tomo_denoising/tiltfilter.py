"""Flagging flawed tilt series, and the record of which ones are excluded.

A tilt series with far fewer tilts than the rest of the dataset, a narrow
angular range or a hole in the middle (an aborted or broken acquisition) costs
hours downstream and then fails there - AreTomo3 cannot align it, or
MissAlignment's workers crash on it. :func:`evaluate` flags such series from
their tilt angles alone, against three criteria (``tiltseries_filter`` in the
config):

  - count : fewer than ``min_tilts``, or than ``min_tilts_fraction`` of the
            dataset's median tilt count
  - span  : an angular range below ``min_span_fraction`` of the dataset's
            median range
  - gap   : a gap between neighbouring tilts over ``max_gap_factor`` times
            the series' OWN median step

The relative criteria compare against the dataset itself, so nothing here
assumes a tilt scheme, step size or range.

Excluded series are recorded in ``<processing_directory>/
excluded_tiltseries.json`` - by stage:

  - ``tilt_check``  : flagged from the mdoc, before ts_import. Never handed
                      to ts_import at all (left out of mdocs_filtered/), so
                      WarpTools never processes them. Re-evaluated on every
                      run.
  - a WarpTools command (``ts_stack``, ``ts_aretomo3``, ``ts_ctf``,
    ``ts_reconstruct``) : WarpTools gave up on them ("Failed to process",
                      marked unselected) - or, for ts_aretomo3, AreTomo3
                      produced no alignment. The step goes on without them.
                      Kept until the next ``--force`` of align_tiltseries,
                      which runs WarpTools on them again - but WarpTools
                      itself marked them unselected in their XML, and may
                      skip them for that until they are re-selected
                      (WarpTools change_selection) or their XML is removed.

Nothing is ever moved or deleted here: exclusion is only ever "not passed
on", and files that would have to go are listed for the user to remove.
"""

import json
import os
import re
import statistics

from .mdoc import unique_angles
from .utils import warn

MODES = ("exclude", "stop", "off")

STAGE_TILT_CHECK = "tilt_check"
STAGE_ARETOMO = "ts_aretomo3"

#: Below this many tilt series, a dataset median says too little about what
#: a "normal" series looks like - only the absolute floor and the gap test
#: apply.
MIN_SERIES_FOR_RELATIVE = 5


# ------------------------------------------------------------------ settings


def settings(config):
    """The validated ``tiltseries_filter`` section."""
    raw = config["tiltseries_filter"]
    mode = str(raw["mode"]).strip().lower()
    if mode not in MODES:
        raise SystemExit("ERROR: tiltseries_filter.mode must be one of %s, "
                         "got %r" % ("/".join(MODES), raw["mode"]))
    try:
        values = {
            "mode": mode,
            "min_tilts": int(raw["min_tilts"]),
            "min_tilts_fraction": float(raw["min_tilts_fraction"]),
            "min_span_fraction": float(raw["min_span_fraction"]),
            "max_gap_factor": float(raw["max_gap_factor"]),
        }
    except (TypeError, ValueError) as exc:
        raise SystemExit("ERROR: tiltseries_filter has a non-numeric "
                         "threshold: %s" % exc)
    return values


# ------------------------------------------------------------------ criteria


def describe(angles):
    """(count, span, median step, largest gap) of a tilt series' angles,
    retakes counted once."""
    angles = unique_angles(angles)
    if not angles:
        return 0, 0.0, None, None
    steps = [b - a for a, b in zip(angles, angles[1:])]
    return (len(angles), angles[-1] - angles[0],
            statistics.median(steps) if steps else None,
            max(steps) if steps else None)


def evaluate(angles_by_name, filter_settings):
    """``{name: [reason, ...]}`` for every series that fails a criterion.

    *angles_by_name* maps each tilt series of the dataset to its tilt angles
    (any order, any sign convention); the dataset medians are taken over all
    of them.
    """
    stats = {name: describe(angles) for name, angles in angles_by_name.items()}
    populated = [s for s in stats.values() if s[0] > 0]
    relative = len(populated) >= MIN_SERIES_FOR_RELATIVE
    median_count = statistics.median(s[0] for s in populated) if populated else 0
    median_span = statistics.median(s[1] for s in populated) if populated else 0

    s = filter_settings
    flagged = {}
    for name in sorted(stats):
        count, span, step, gap = stats[name]
        reasons = []
        if count == 0:
            reasons.append("no usable tilts")
        else:
            if count < s["min_tilts"]:
                reasons.append("%d tilts < min_tilts %d" % (count, s["min_tilts"]))
            elif relative and count < s["min_tilts_fraction"] * median_count:
                reasons.append("%d tilts < %g x dataset median %g"
                               % (count, s["min_tilts_fraction"], median_count))
            if relative and span < s["min_span_fraction"] * median_span:
                reasons.append("range %.1f deg < %g x dataset median %.1f deg"
                               % (span, s["min_span_fraction"], median_span))
            if step and gap > s["max_gap_factor"] * step:
                reasons.append("gap of %.1f deg > %g x its median step %.1f deg"
                               % (gap, s["max_gap_factor"], step))
        if reasons:
            flagged[name] = reasons
    return flagged


# -------------------------------------------------------------------- tomostar


_LABEL_RE = re.compile(r"^\s*(_\S+)")
_TOKEN_RE = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")


def read_tomostar_angles(path):
    """Tilt angles (``_wrpAngleTilt``) of every row of a WarpTools .tomostar.

    A real STAR reader, not a fixed column layout: the column is found by its
    label, quoted values are honoured, and comment lines skipped. Raises
    ValueError if the file has no loop or no ``_wrpAngleTilt`` column.
    """
    labels = []
    rows = []
    in_loop = False
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                if in_loop and rows and not stripped:
                    break       # a blank line after the rows ends the loop
                continue
            if stripped.startswith("data_"):
                if rows:
                    break
                continue
            if stripped == "loop_":
                in_loop, labels, rows = True, [], []
                continue
            if not in_loop:
                continue
            label = _LABEL_RE.match(stripped)
            if label and not rows:
                labels.append(label.group(1))
                continue
            rows.append(_TOKEN_RE.findall(stripped))

    if "_wrpAngleTilt" not in labels:
        raise ValueError("%s: no _wrpAngleTilt column (columns: %s)"
                         % (path, ", ".join(labels) or "none"))
    column = labels.index("_wrpAngleTilt")
    angles = []
    for number, row in enumerate(rows, 1):
        try:
            angles.append(float(row[column].strip("'\"")))
        except (IndexError, ValueError):
            raise ValueError("%s: row %d has no valid _wrpAngleTilt value: %s"
                             % (path, number, " ".join(row)))
    return angles


# --------------------------------------------------------------------- record


def record_path(config):
    return config.sub("excluded_tiltseries.json")


def read_record(config):
    """``{name: {"stage": ..., "reason": ...}}``; empty if none yet."""
    path = record_path(config)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        warn("could not read %s (%s) - treating it as empty" % (path, exc))
        return {}
    return data if isinstance(data, dict) else {}


def write_record(config, record):
    """Write *record*. An empty record is written too, as ``{}``, so the
    file always says what the last run decided."""
    with open(record_path(config), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def excluded_names(config):
    return set(read_record(config))


def record_failures(config, stage, failures):
    """Add ``{name: reason}`` to the record under *stage* (a WarpTools
    command - see the module docstring)."""
    record = read_record(config)
    for name, reason in failures.items():
        record[name] = {"stage": stage, "reason": reason}
    write_record(config, record)


def format_record(record):
    """Report lines, one per excluded tilt series."""
    width = max([len(n) for n in record] + [10])
    return ["  %-*s  [%s] %s" % (width, name, entry.get("stage", "?"),
                                 entry.get("reason", ""))
            for name, entry in sorted(record.items())]


def drop_excluded(config, series_list, purpose, log=None):
    """*series_list* without the excluded tilt series, with a note naming
    any that were dropped (e.g. an mdoc the user put in mdocs.train)."""
    excluded = excluded_names(config)
    kept = [s for s in series_list if s.name not in excluded]
    dropped = sorted(s.name for s in series_list if s.name in excluded)
    if dropped:
        message = ("%s: skipping %d excluded tilt series (see %s): %s"
                   % (purpose, len(dropped), record_path(config),
                      ", ".join(dropped)))
        if log is not None:
            log.note(message)
        else:
            warn(message)
    return kept
