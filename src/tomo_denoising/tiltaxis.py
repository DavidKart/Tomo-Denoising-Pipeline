"""Dataset-wide tilt axis refinement.

The tilt axis is a property of the microscope, not of an individual tilt
series, so estimating it once from several series and then holding it fixed is
both more robust and more consistent than letting AreTomo2 solve it
independently every time. This mirrors what WarpTools does.

Each iteration aligns a random subset of tilt series, combines their tilt axis
angles into one robust estimate, and hands that to the next iteration as the
starting point. AreTomo2 refines an supplied angle only within +/-3 degrees, so
successive iterations converge rather than wander.
"""

import os
import random
import re
import time

import numpy as np

from .utils import ensure_dir, info, warn

#: Name of the human readable record written to the processing directory.
REPORT_NAME = "tilt_axis_refinement.txt"

_ALN_ROT_COLUMN = 1

#: Patterns AreTomo2 has used to report the solved tilt axis in its log.
_LOG_PATTERNS = [
    re.compile(r"[Nn]ew tilt axis angle[^-\d]*(-?\d+\.?\d*)"),
    re.compile(r"[Tt]ilt axis angle[^-\d]*(-?\d+\.?\d*)"),
    re.compile(r"[Bb]est tilt axis[^-\d]*(-?\d+\.?\d*)"),
    re.compile(r"[Rr]otation angle[^-\d]*(-?\d+\.?\d*)"),
]


# --------------------------------------------------------------- angle maths


def _reference_angle(angles):
    """An unambiguous reference for a set of tilt axis angles.

    A tilt axis is a line, not a direction: -84 and +96 describe the same axis.
    Doubling the angles removes that 180 degree ambiguity, so the circular mean
    of the doubled angles - halved again - gives a reference every measurement
    can be folded onto without a branch choice.
    """
    doubled = np.deg2rad(np.asarray(angles, dtype=float) * 2.0)
    mean = np.arctan2(np.sin(doubled).mean(), np.cos(doubled).mean())
    return np.rad2deg(mean) / 2.0


def fold_angles(angles):
    """Shift each angle by multiples of 180 to sit within +/-90 of the reference.

    After this the values are mutually consistent, so ordinary vector
    statistics on (cos, sin) are meaningful and the +/-180 wraparound cannot
    split what is really one cluster into two.
    """
    reference = _reference_angle(angles)
    folded = []
    for angle in angles:
        shifted = float(angle)
        while shifted - reference > 90.0:
            shifted -= 180.0
        while shifted - reference < -90.0:
            shifted += 180.0
        folded.append(shifted)
    return folded


def geometric_median(points, eps=1e-9, max_iterations=256):
    """Weiszfeld's algorithm: the point minimising the sum of distances.

    Unlike the mean this is not dragged around by a single bad measurement,
    which is the point of using it here - one tilt series whose alignment went
    wrong should not move the dataset estimate.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) == 1:
        return pts[0]

    guess = pts.mean(axis=0)
    for _ in range(max_iterations):
        offsets = pts - guess
        distances = np.sqrt((offsets ** 2).sum(axis=1))

        # A candidate sitting exactly on a data point has an undefined weight;
        # that point is already the optimum for its own term, so drop it.
        near = distances < eps
        if near.all():
            return guess
        weights = 1.0 / distances[~near]
        nxt = (pts[~near] * weights[:, None]).sum(axis=0) / weights.sum()

        if np.sqrt(((nxt - guess) ** 2).sum()) < eps:
            return nxt
        guess = nxt
    return guess


def combine_tilt_axes(angles):
    """Robustly combine per-series tilt axis angles into one dataset estimate.

    Returns ``(angle, method)``. Up to three measurements the mean is used -
    a geometric median of so few points is not more robust, just noisier.
    Beyond that the geometric median of the unit vectors is taken and converted
    back with atan2.
    """
    values = [float(a) for a in angles if a is not None]
    if not values:
        return None, "no measurements"

    folded = fold_angles(values)
    vectors = np.column_stack([
        np.cos(np.deg2rad(folded)),
        np.sin(np.deg2rad(folded)),
    ])

    if len(folded) <= 3:
        centre = vectors.mean(axis=0)
        method = "mean of %d" % len(folded)
    else:
        centre = geometric_median(vectors)
        method = "geometric median of %d" % len(folded)

    if np.sqrt((centre ** 2).sum()) < 1e-12:
        warn("tilt axis measurements cancel out exactly - using their mean.")
        return float(np.mean(folded)), "degenerate, mean of %d" % len(folded)

    return float(np.rad2deg(np.arctan2(centre[1], centre[0]))), method


# ------------------------------------------------------------------- parsing


def tilt_axis_from_aln(aln_path):
    """Median of the ROT column of an AreTomo .aln file."""
    if not aln_path or not os.path.isfile(aln_path):
        return None
    rotations = []
    with open(aln_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                for pattern in _LOG_PATTERNS:
                    match = pattern.search(stripped)
                    if match:
                        return float(match.group(1))
                continue
            parts = stripped.split()
            if len(parts) > _ALN_ROT_COLUMN:
                try:
                    rotations.append(float(parts[_ALN_ROT_COLUMN]))
                except ValueError:
                    continue
    if not rotations:
        return None
    return float(np.median(rotations))


def tilt_axis_from_log(log_path):
    """Solved tilt axis as reported in an AreTomo log, if it says so."""
    if not log_path or not os.path.isfile(log_path):
        return None
    found = None
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for pattern in _LOG_PATTERNS:
                match = pattern.search(line)
                if match:
                    try:
                        found = float(match.group(1))   # keep the last one
                    except ValueError:
                        pass
    return found


def read_tilt_axis(aln_path, log_path):
    """Best available tilt axis for one aligned tilt series."""
    value = tilt_axis_from_log(log_path)
    if value is None:
        value = tilt_axis_from_aln(aln_path)
    return value


# -------------------------------------------------------------------- subsets


def choose_subset(names, size, seed, iteration):
    """A reproducible random subset, different for each iteration."""
    if len(names) <= size:
        return list(names)
    rng = random.Random("%s-%d" % (seed, iteration))
    return sorted(rng.sample(list(names), size))


# --------------------------------------------------------------------- report


def report_path(config):
    return os.path.join(config.processing_dir, REPORT_NAME)


def write_report(config, rows, final_angle, note=""):
    """Write the human readable record of the refinement."""
    path = report_path(config)
    ensure_dir(os.path.dirname(path))

    lines = [
        "=" * 78,
        "AreTomo2 tilt axis refinement",
        "=" * 78,
        "written    : %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
        "processing : %s" % config.processing_dir,
        "iterations : %d" % int(config["aretomo"]["tilt_axis_iterations"]),
        "subset size: %d tilt series per iteration"
        % int(config["aretomo"]["tilt_axis_subset_size"]),
        "",
        "Angles are combined with the geometric median of the unit vectors",
        "(the mean for three or fewer), after folding onto a common branch -",
        "a tilt axis is defined modulo 180 degrees, so -84 and +96 are the",
        "same axis and must not be averaged naively.",
        "",
    ]

    for row in rows:
        lines.append("-" * 78)
        lines.append("Iteration %s   (%s)" % (row["iteration"], row["method"]))
        lines.append("-" * 78)
        if row.get("initial") is None:
            lines.append("  started from : free search over the full range")
        else:
            lines.append("  started from : %+.3f deg (refined within +/-3 deg)"
                         % row["initial"])
        for name, angle in sorted(row["measurements"].items()):
            if angle is None:
                lines.append("    %-28s  no tilt axis reported" % name)
            else:
                lines.append("    %-28s  %+8.3f deg" % (name, angle))
        if row["angle"] is None:
            lines.append("  --> no usable measurements")
        else:
            lines.append("  --> combined   : %+.3f deg" % row["angle"])
            if row.get("spread") is not None:
                lines.append("      spread     : %.3f deg (max - min)"
                             % row["spread"])
        lines.append("")

    lines.append("=" * 78)
    if final_angle is None:
        lines.append("NO TILT AXIS DETERMINED - every tilt series was aligned "
                     "independently.")
    else:
        lines.append("CONVERGED TILT AXIS: %+.4f deg" % final_angle)
        lines.append("")
        lines.append("All tilt series were then reconstructed once with")
        lines.append("  -TiltAxis %.4f -1   (angle fixed, no further refinement)"
                     % final_angle)
    if note:
        lines.append("")
        lines.append(note)
    lines.append("=" * 78)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    info("tilt axis record: %s" % path)
    return path


def read_converged_angle(config):
    """Recover a previously converged tilt axis, for resuming a run."""
    path = report_path(config)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = re.search(r"CONVERGED TILT AXIS:\s*(-?\d+\.?\d*)", line)
            if match:
                return float(match.group(1))
    return None
