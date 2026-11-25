"""IMOD patch tracking as the tilt-series aligner (``aligner: "imod"``).

The alignment itself is WarpTools' own ``ts_etomo_patches``, which runs
IMOD's ``batchruntomo`` (and refuses to start unless that is on its PATH).
This module supplies that environment, reads what the tilt axis search
rounds left behind, and keeps the IMOD section of
``tilt_axis_refinement.txt``. The calls themselves are made in
:mod:`warptools`, next to ts_aretomo3.

``ts_etomo_patches --do_axis_search`` (WarpTools source,
EtomoPatchTrackTiltseries.cs) aligns every series it is given with IMOD
fitting the axis, takes the geometric median over them, then aligns them all
again with that median fixed. One search round here is one such call on a
random subset; the final alignment is one call on every series with the
converged axis fixed (no --do_axis_search).
"""

import os
import re
import time

from . import tiltaxis
from .utils import capture, require_executable, warn, wrap_conda_run

#: A round whose result moved less than this from its start ends the search.
CONVERGED_CHANGE_DEG = 0.1

#: Above this, the IMOD axis is worth a warning against its starting axis.
LARGE_CHANGE_DEG = 2.0

#: WarpTools' own printout of the dataset median after the search pass.
_AVERAGE_AXIS_RE = re.compile(
    r"Average tilt axis angle from patch tracking:\s*([-+]?\d+(?:[.,]\d+)?)")

#: Starts the IMOD section of tilt_axis_refinement.txt.
_SECTION_RULE = "#" * 78
_SECTION_TITLE = "IMOD tilt axis search (ts_etomo_patches --do_axis_search)"
_FINAL_RE = re.compile(r"^IMOD CONVERGED TILT AXIS:\s*([+-]?\d+\.?\d*)")
_KEY_RE = re.compile(r"^(starting axis|rounds)\s*:\s*([+-]?\d+\.?\d*)")

_CONDA_CACHE = {}


# --------------------------------------------------------------- environment


def _conda_env_variables(config, env_name):
    """The environment variables ``conda run`` sets up inside *env_name*
    (read once per environment), or ``{}`` if that could not be run."""
    if env_name not in _CONDA_CACHE:
        output = capture(wrap_conda_run(["env"], config.exe("conda"), env_name))
        variables = {}
        for line in (output or "").splitlines():
            key, sep, value = line.partition("=")
            if sep:
                variables[key] = value
        _CONDA_CACHE[env_name] = variables
    return _CONDA_CACHE[env_name]


def imod_env(config):
    """Environment additions that make IMOD reachable for ts_etomo_patches:
    its ``bin/`` first on the PATH, and ``IMOD_DIR``.

    - ``conda_env_imod`` set: that environment's ``bin/`` (and its own
      ``IMOD_DIR``'s ``bin/``, if its activation sets one). WarpTools itself
      may run in another environment, so IMOD is put on the PATH rather
      than run through ``conda run``.
    - ``executables.imod`` set: that install folder.
    - neither: ``{}`` - whatever IMOD the PATH already has.
    """
    bins, imod_dir = [], ""
    env_name = config.conda_env("imod")
    if env_name:
        variables = _conda_env_variables(config, env_name)
        imod_dir = variables.get("IMOD_DIR", "").strip()
        prefix = variables.get("CONDA_PREFIX", "").strip()
        if imod_dir:
            bins.append(os.path.join(imod_dir, "bin"))
        if prefix:
            bins.append(os.path.join(prefix, "bin"))
        if not bins:
            warn("could not read the environment of conda_env_imod %r (via "
                 "'%s run') - ts_etomo_patches gets the PATH as it is."
                 % (env_name, config.exe("conda")))
    elif config.imod_dir:
        imod_dir = config.imod_dir
        bins.append(os.path.join(imod_dir, "bin"))
    if not bins:
        return {}
    env = {"PATH": os.pathsep.join(bins + [os.environ.get("PATH", "")])}
    if imod_dir:
        env["IMOD_DIR"] = imod_dir
    return env


def require_imod(config):
    """Warn (not stop) if IMOD's batchruntomo will not be found - WarpTools
    stops with its own error then."""
    env_name = config.conda_env("imod")
    if env_name:
        return require_executable(
            config.exe("conda"), "conda/mamba, to find IMOD in %r" % env_name)
    if config.imod_dir:
        path = os.path.join(config.imod_dir, "bin", "batchruntomo")
        if not os.path.isfile(path):
            warn("%s does not exist - is executables.imod (%s) IMOD's install "
                 "folder?" % (path, config.imod_dir))
            return False
        return True
    return require_executable(
        "batchruntomo", "IMOD, for ts_etomo_patches (aligner: imod) - or set "
                        "executables.imod / conda_env_imod")


# ------------------------------------------------------------------- reading


def xml_tilt_axis(xml_path):
    """The tilt axis a WarpTools XML holds (``<AxisAngle>``, one value per
    tilt, combined), or ``None``. (``TiltAxisAngles`` is only the C# field's
    name - the XML element WarpTools writes is ``AxisAngle``.)"""
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError):
        return None
    element = root.find("AxisAngle")
    if element is None or not (element.text or "").strip():
        return None
    try:
        values = [float(x) for x in element.text.split()]
    except ValueError:
        return None
    angle, _ = tiltaxis.combine_tilt_axes(values)
    return angle


def log_average_axis(log_path):
    """WarpTools' own "Average tilt axis angle from patch tracking" in a
    ts_etomo_patches log (the last one), or ``None``."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            found = _AVERAGE_AXIS_RE.findall(handle.read())
    except OSError:
        return None
    return float(found[-1].replace(",", ".")) if found else None


def axis_change(a, b):
    """Difference between two tilt axis angles, in degrees (modulo 180)."""
    folded = tiltaxis.fold_angles([a, b])
    return abs(folded[1] - folded[0])


# -------------------------------------------------------------------- report


def write_report(config, seed, seed_source, rows, final_angle, note=""):
    """Write the IMOD section of tilt_axis_refinement.txt.

    Placed after the AreTomo record when the starting axis came from the
    AreTomo search (replacing an earlier IMOD section); the file holds only
    this section when the starting axis was configured.
    """
    path = tiltaxis.report_path(config)
    kept = ""
    if seed_source == "aretomo" and os.path.isfile(path):
        with open(path, encoding="utf-8", errors="replace") as handle:
            kept = handle.read().split(_SECTION_RULE)[0].rstrip() + "\n\n"
    source_text = {
        "aretomo": "the AreTomo search above",
        "config": "imod.initial_tilt_axis",
    }.get(seed_source, seed_source)
    settings = config["imod"]

    lines = [
        _SECTION_RULE,
        _SECTION_TITLE,
        _SECTION_RULE,
        "written      : %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
        "starting axis: %+.4f" % seed,
        "taken from   : %s" % source_text,
        "rounds       : %d" % int(settings["tilt_axis_iterations"]),
        "subset size  : %d tilt series per round"
        % int(settings["tilt_axis_subset_size"]),
        "",
        "Each round: IMOD fits each series' axis, WarpTools takes the median",
        "and aligns the round's series again with it fixed - so the values",
        "read back from their XMLs mostly repeat that median.",
        "",
    ]
    for row in rows:
        lines.append("-" * 78)
        lines.append("Round %d   (started from %+.3f deg)"
                     % (row["round"], row["initial"]))
        lines.append("-" * 78)
        for name, angle in sorted(row["measurements"].items()):
            if angle is None:
                lines.append("    %-28s  no tilt axis read back" % name)
            else:
                lines.append("    %-28s  %+8.3f deg" % (name, angle))
        if row["angle"] is None:
            lines.append("  --> no usable result")
        else:
            lines.append("  --> result     : %+.3f deg (%s)"
                         % (row["angle"], row["method"]))
            lines.append("      change     : %.3f deg from the round's start"
                         % row["change"])
        lines.append("")

    lines.append(_SECTION_RULE)
    if final_angle is None:
        lines.append("NO IMOD TILT AXIS DETERMINED")
    else:
        lines.append("IMOD CONVERGED TILT AXIS: %+.4f deg" % final_angle)
        lines.append("")
        lines.append("Every tilt series was then aligned once by WarpTools")
        lines.append("  ts_etomo_patches --initial_axis %.4f   (no axis search)"
                     % final_angle)
    if note:
        lines.append("")
        lines.append(note)
    lines.append(_SECTION_RULE)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(kept + "\n".join(lines) + "\n")
    return path


def read_converged_angle(config, seed):
    """The IMOD axis from an earlier run, for resuming - only if that run
    started from the same axis (to 1e-3 deg) with the same number of
    rounds; ``None`` otherwise."""
    path = tiltaxis.report_path(config)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as handle:
        parts = handle.read().split(_SECTION_RULE, 1)
    if len(parts) < 2:
        return None
    found, final = {}, None
    for line in parts[1].splitlines():
        line = line.strip()
        key = _KEY_RE.match(line)
        if key:
            found[key.group(1)] = float(key.group(2))
        match = _FINAL_RE.match(line)
        if match:
            final = float(match.group(1))
    rounds = int(config["imod"]["tilt_axis_iterations"])
    if (final is None or "starting axis" not in found
            or abs(found["starting axis"] - seed) > 1e-3
            or int(found.get("rounds", -1)) != rounds):
        return None
    return final
