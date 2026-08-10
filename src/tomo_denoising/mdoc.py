"""Parsing SerialEM .mdoc files and resolving tomogram names."""

import glob
import os
import re

from .utils import capture, warn

_TILT_ANGLE_RE = re.compile(r"TiltAngle\s*=\s*([-+]?[0-9]*\.?[0-9]+)")
_PIXEL_SPACING_RE = re.compile(r"PixelSpacing\s*=\s*([0-9]*\.?[0-9]+)")
_SUBFRAME_RE = re.compile(r"SubFramePath\s*=\s*(.+)")

#: Extensions that may sit between the tomogram name and ".mdoc",
#: e.g. "TS_01.mrc.mdoc" or "TS_01.st.mdoc".
_STACK_SUFFIXES = (".mrc", ".st", ".tif", ".tiff", ".eer", ".ali", ".rec")


def tomo_name_from_mdoc(mdoc_path):
    """Derive the tomogram name from an mdoc file name.

    ``mdocs/TS_01.mrc.mdoc`` -> ``TS_01``  (directory ``rec_TS_01``)
    """
    base = os.path.basename(mdoc_path)
    if base.lower().endswith(".mdoc"):
        base = base[: -len(".mdoc")]
    lowered = base.lower()
    for suffix in _STACK_SUFFIXES:
        if lowered.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


class TiltSeries(object):
    """One tilt series as described by its mdoc file.

    ``frames`` and ``angles`` are kept in ACQUISITION order, which is the order
    the records appear in the mdoc. ``frames[0]`` is therefore the first
    collected tilt - the one CTFFIND4 is run on for the IsoNet defocus estimate.
    """

    def __init__(self, mdoc_path, name, frames, angles, pixel_spacing):
        self.mdoc_path = mdoc_path
        self.name = name
        self.frames = frames
        self.angles = angles
        self.pixel_spacing = pixel_spacing

    def __repr__(self):  # pragma: no cover - debugging aid
        return "<TiltSeries %s: %d tilts>" % (self.name, len(self.frames))

    def sorted_by_angle(self):
        """Return (frames, angles) sorted by increasing tilt angle."""
        order = sorted(range(len(self.angles)), key=lambda k: self.angles[k])
        return (
            [self.frames[i] for i in order],
            [self.angles[i] for i in order],
        )


def parse_mdoc(mdoc_path):
    """Parse an mdoc into a :class:`TiltSeries`."""
    frames = []
    angles = []
    pixel_spacing = None

    with open(mdoc_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _PIXEL_SPACING_RE.search(line)
            if match and pixel_spacing is None:
                pixel_spacing = float(match.group(1))
                continue

            match = _TILT_ANGLE_RE.search(line)
            if match:
                angles.append(float(match.group(1)))
                continue

            match = _SUBFRAME_RE.search(line)
            if match:
                # SerialEM writes Windows paths: C:\data\frames\foo_001.tif
                raw = match.group(1).strip()
                frames.append(raw.replace("\\", "/").rstrip("/").split("/")[-1])

    if len(frames) != len(angles):
        warn(
            "%s: %d SubFramePath entries but %d TiltAngle entries - "
            "using the first %d of each."
            % (
                os.path.basename(mdoc_path),
                len(frames),
                len(angles),
                min(len(frames), len(angles)),
            )
        )
        count = min(len(frames), len(angles))
        frames, angles = frames[:count], angles[:count]

    return TiltSeries(
        mdoc_path=os.path.abspath(mdoc_path),
        name=tomo_name_from_mdoc(mdoc_path),
        frames=frames,
        angles=angles,
        pixel_spacing=pixel_spacing,
    )


def matches_tomo(text, tomo_name):
    """True when *text* refers to *tomo_name*, and only to that tomogram.

    Other programs wrap the tomogram name in prefixes and suffixes of their
    own - RESOLVE reports ``odd_Position_1_rec_tomo_Tomograms_locRes``, cryoCARE
    writes ``denoised_Position_1_rec_tomo.mrc`` - so the name is matched as a
    whole underscore-delimited run rather than by stripping known affixes.

    The boundaries matter: a plain substring test would let ``Position_1`` match
    ``Position_10``.
    """
    stem = os.path.splitext(os.path.basename(str(text)))[0]
    pattern = r"(?:^|_)%s(?:_|$)" % re.escape(str(tomo_name))
    return re.search(pattern, stem) is not None


def match_tomo_name(text, tomo_names):
    """Which of *tomo_names* does *text* refer to? Longest match wins."""
    for name in sorted(tomo_names, key=len, reverse=True):
        if matches_tomo(text, name):
            return name
    return None


def find_mdocs(directory):
    """Return every ``*.mdoc`` inside *directory*, sorted by name."""
    if not os.path.isdir(directory):
        return []
    found = glob.glob(os.path.join(directory, "*.mdoc"))
    found += glob.glob(os.path.join(directory, "*.MDOC"))
    return sorted(set(found))


def collect_tilt_series(directory, label):
    """Parse every mdoc in *directory*; returns a list of :class:`TiltSeries`."""
    mdocs = find_mdocs(directory)
    if not mdocs:
        raise SystemExit(
            "ERROR: no .mdoc files found for %s in:\n  %s\n"
            "Check the 'mdocs' section of your configuration."
            % (label, directory)
        )
    series = [parse_mdoc(path) for path in mdocs]
    for entry in series:
        if not entry.frames:
            warn("%s contains no SubFramePath entries." % entry.mdoc_path)
    return series


def resolve_pixel_size(config, series, first_frame_path=None):
    """Determine the unbinned pixel size in Angstrom for one tilt series.

    ``acquisition.pixel_size: read`` -> PixelSpacing from the mdoc, falling
    back to IMOD ``header`` on the first raw frame. Anything else is taken
    as a literal number.
    """
    setting = config["acquisition"]["pixel_size"]
    if not (isinstance(setting, str) and setting.strip().lower() == "read"):
        return float(setting)

    if series.pixel_spacing:
        return float(series.pixel_spacing)

    if first_frame_path and os.path.exists(first_frame_path):
        output = capture([config.exe("header"), first_frame_path])
        if output:
            for line in output.splitlines():
                if "Pixel spacing" in line:
                    parts = line.split()
                    try:
                        return float(parts[3])
                    except (IndexError, ValueError):
                        pass

    raise SystemExit(
        "ERROR: could not determine the pixel size for %s.\n"
        "The mdoc has no PixelSpacing entry and IMOD 'header' did not report "
        "one either. Set acquisition.pixel_size to a number in your config."
        % series.name
    )
