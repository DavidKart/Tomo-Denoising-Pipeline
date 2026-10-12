"""Parsing SerialEM .mdoc files and resolving tomogram names."""

import glob
import os
import re

from .utils import capture, executable_available, gpu_env, info, warn

#: "2*pixel_size" or "pixel_size*2.5" - see resolve_angpix().
_ANGPIX_MULT_RE = re.compile(
    r"^\s*(?:(?P<a>[0-9]*\.?[0-9]+)\s*\*\s*pixel_size"
    r"|pixel_size\s*\*\s*(?P<b>[0-9]*\.?[0-9]+))\s*$",
    re.IGNORECASE,
)

#: What to say when the pixel size has to be read from the data, the mdoc
#: does not carry it, and IMOD is not installed. IMOD is an OPTIONAL
#: dependency of this pipeline - "header" is used for nothing else anywhere -
#: so a missing one is not a broken installation to warn about, it just means
#: the pixel size has to be given explicitly.
HEADER_MISSING_MESSAGE = """\
ERROR: could not determine the pixel size for %(series)s.

  acquisition.pixel_size is "read", but %(mdoc)s has no PixelSpacing entry,
  and the only fallback - IMOD's 'header' (%(header)r) - is not installed
  here. IMOD is OPTIONAL for this pipeline; 'header' is used for nothing
  else, so there is nothing to install unless you want that fallback.

  Set the pixel size explicitly instead, in %(config)s:

      acquisition:
        pixel_size: <unbinned pixel size of the raw frames, in Angstrom>

  then run again."""

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


class Tilt(object):
    """One ``[ZValue = N]`` block of an mdoc.

    *index* is the block's 0-based position in the file - the same index
    :func:`split_mdoc_blocks` gives it, which is what
    :func:`write_filtered_mdoc` selects blocks by. *angle*/*frame* are
    ``None`` when the block has no TiltAngle/SubFramePath.
    """

    def __init__(self, index, zvalue, angle, frame, datetime_raw=None):
        self.index = index
        self.zvalue = zvalue
        self.angle = angle
        self.frame = frame
        #: The block's DateTime as written (None if absent), and parsed.
        self.datetime_raw = datetime_raw
        self.datetime = parse_mdoc_datetime(datetime_raw)

    @property
    def complete(self):
        return self.angle is not None and self.frame is not None


class TiltSeries(object):
    """One tilt series as described by its mdoc file.

    ``tilts`` holds every ``[ZValue]`` block, complete or not, in file
    (= acquisition) order. ``frames`` and ``angles`` are the COMPLETE blocks
    only (both a TiltAngle and a SubFramePath), index for index - an
    incomplete block can never shift one list against the other.
    ``problems`` lists what was wrong with the file, for reporting.
    """

    def __init__(self, mdoc_path, name, tilts, pixel_spacing, problems=None):
        self.mdoc_path = mdoc_path
        self.name = name
        self.tilts = tilts
        self.pixel_spacing = pixel_spacing
        self.problems = problems or []

    @property
    def complete_tilts(self):
        return [t for t in self.tilts if t.complete]

    @property
    def frames(self):
        return [t.frame for t in self.complete_tilts]

    @property
    def angles(self):
        return [t.angle for t in self.complete_tilts]

    def __repr__(self):  # pragma: no cover - debugging aid
        return "<TiltSeries %s: %d tilts>" % (self.name, len(self.frames))


#: ``[Name = value]`` section headers. ``[ZValue = N]`` opens a tilt block;
#: any other (``[T = ...]`` titles, ``[FrameSet = N]`` in a per-movie frame
#: mdoc, ``[MontSection = N]``, ...) ends the current one.
_SECTION_RE = re.compile(r"^\s*\[\s*([A-Za-z]+)\s*=\s*([^\]]*)\]")
#: ``Key = value`` - the key at the START of the line, so e.g. the header's
#: ``TiltAxisAngle`` can never be read as a ``TiltAngle``.
_KEY_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def _first_float(value):
    try:
        return float(value.split()[0])
    except (IndexError, ValueError):
        return None


def parse_mdoc(mdoc_path):
    """Parse a SerialEM tilt-series mdoc into a :class:`TiltSeries`.

    Block-based: every ``[ZValue = N]`` section is read on its own, so a
    block missing its TiltAngle or SubFramePath is reported by name instead
    of silently pairing every later frame with the wrong angle. Nothing is
    assumed about naming, tilt scheme, angle order or step size.
    """
    tilts = []
    pixel_spacing = None        # from the header, before any section
    block_pixel_spacing = []    # per-block fallback, first one wins
    current = None              # dict of the open ZValue block's keys
    other_sections = set()

    def close_block():
        if current is None:
            return
        if "PixelSpacing" in current:
            block_pixel_spacing.append(_first_float(current["PixelSpacing"]))
        frame = current.get("SubFramePath")
        if frame is not None:
            # SerialEM writes Windows paths: C:\data\frames\foo_001.tif
            frame = frame.replace("\\", "/").rstrip("/").split("/")[-1] or None
        tilts.append(Tilt(index=len(tilts), zvalue=current["_zvalue"],
                          angle=_first_float(current.get("TiltAngle", "")),
                          frame=frame,
                          datetime_raw=current.get("DateTime") or None))

    with open(mdoc_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            section = _SECTION_RE.match(line)
            if section:
                close_block()
                current = None
                if section.group(1) == "ZValue":
                    try:
                        zvalue = int(section.group(2).strip())
                    except ValueError:
                        zvalue = None
                    current = {"_zvalue": zvalue}
                else:
                    other_sections.add(section.group(1))
                continue
            match = _KEY_RE.match(line)
            if not match:
                continue
            key, value = match.groups()
            if current is not None:
                current.setdefault(key, value)
            elif key == "PixelSpacing" and pixel_spacing is None:
                pixel_spacing = _first_float(value)
        close_block()

    if pixel_spacing is None:
        # Not in the header - fall back to the first block that has one.
        pixel_spacing = next((p for p in block_pixel_spacing if p is not None),
                             None)

    name = os.path.basename(mdoc_path)
    problems = []
    if not tilts:
        problems.append("no [ZValue] blocks%s - not a tilt-series mdoc?"
                        % (" (has %s sections instead)"
                           % "/".join(sorted(other_sections - {"T"}))
                           if other_sections - {"T"} else ""))
    for tilt in tilts:
        missing = [key for key, val in (("TiltAngle", tilt.angle),
                                        ("SubFramePath", tilt.frame))
                   if val is None]
        if missing:
            problems.append("[ZValue = %s] has no %s - tilt ignored"
                            % (tilt.zvalue, " or ".join(missing)))
    zvalues = [t.zvalue for t in tilts]
    if tilts and zvalues != list(range(len(tilts))):
        problems.append("ZValues are not 0..%d in order (%s)"
                        % (len(tilts) - 1, _abbreviate(zvalues)))
    problems += _datetime_problems(tilts)
    for problem in problems:
        warn("%s: %s" % (name, problem))

    return TiltSeries(
        mdoc_path=os.path.abspath(mdoc_path),
        name=tomo_name_from_mdoc(mdoc_path),
        tilts=tilts,
        pixel_spacing=pixel_spacing,
        problems=problems,
    )


#: SerialEM's DateTime, e.g. "08-Sep-2026  18:55:11" (2- or 4-digit year).
#: Months matched by name here rather than strptime's %b, which follows the
#: process locale.
_DATETIME_RE = re.compile(
    r"^\s*(\d{1,2})-([A-Za-z]{3})-(\d{2}|\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})\s*$")
_MONTHS = {name: number for number, name in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


def parse_mdoc_datetime(raw):
    """A block's DateTime as a ``datetime``, or None if absent/unreadable."""
    import datetime

    match = _DATETIME_RE.match(raw or "")
    if not match:
        return None
    day, month, year, hour, minute, second = match.groups()
    month = _MONTHS.get(month.lower())
    if month is None:
        return None
    year = int(year) + (2000 if len(year) == 2 else 0)
    try:
        return datetime.datetime(year, month, int(day), int(hour),
                                 int(minute), int(second))
    except ValueError:
        return None


def _datetime_problems(tilts):
    """Warnings (never errors) about the blocks' DateTime entries.

    WarpTools' ts_import has failed on mdocs with tilts lacking a DateTime,
    and acquisition order matters for dose weighting - but that is a
    property of today's WarpTools, not of the data, so this only ever WARNS:
    a future version may not care, and a hard stop here would then block
    perfectly good data.
    """
    problems = []
    missing = [t.zvalue for t in tilts if t.datetime_raw is None]
    unreadable = [t.zvalue for t in tilts
                  if t.datetime_raw is not None and t.datetime is None]
    if missing:
        problems.append("no DateTime in %d block(s) (ZValue %s) - WarpTools "
                        "has failed on such mdocs before; continuing anyway"
                        % (len(missing), _abbreviate(missing)))
    if unreadable:
        problems.append("DateTime not in SerialEM's usual format in %d "
                        "block(s) (ZValue %s) - continuing anyway"
                        % (len(unreadable), _abbreviate(unreadable)))
    dated = [t for t in tilts if t.datetime is not None]
    backwards = [later.zvalue for earlier, later in zip(dated, dated[1:])
                 if later.datetime < earlier.datetime]
    if backwards:
        problems.append("DateTime goes backwards at ZValue %s - acquisition "
                        "(dose) order may not match the block order; "
                        "continuing anyway" % _abbreviate(backwards))
    return problems


def _abbreviate(values, limit=8):
    shown = ", ".join(str(v) for v in values[:limit])
    return shown + (", ..." if len(values) > limit else "")


def unique_angles(angles, tolerance=0.1):
    """*angles* sorted, with near-duplicates (SerialEM retakes of the same
    tilt, within *tolerance* degrees) counted once."""
    result = []
    for angle in sorted(angles):
        if not result or angle - result[-1] > tolerance:
            result.append(angle)
    return result


_ZVALUE_RE = re.compile(r"^\s*\[\s*ZValue\s*=\s*(-?\d+)\s*\]", re.MULTILINE)


def split_mdoc_blocks(text):
    """Split raw mdoc text into ``(preamble, [block_text, ...])``.

    Each block is one ``[ZValue = N] ... `` section verbatim (up to the next
    ``[ZValue = ...]`` or end of file), in file order - which is the same
    order ``parse_mdoc`` builds ``frames``/``angles`` in, so block index i
    corresponds to ``frames[i]``/``angles[i]``.
    """
    matches = list(_ZVALUE_RE.finditer(text))
    if not matches:
        return text, []
    preamble = text[: matches[0].start()]
    blocks = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append(text[start:end])
    return preamble, blocks


def write_filtered_mdoc(source_path, dest_path, keep_indices):
    """Write a copy of *source_path* keeping only the ZValue blocks whose
    0-based file-order index is in *keep_indices* - everything else about
    the file (preamble, per-block content) is preserved verbatim.
    """
    with open(source_path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    preamble, blocks = split_mdoc_blocks(text)
    kept = [block for index, block in enumerate(blocks) if index in keep_indices]
    # Written beside *dest_path* and swapped in, never opened for writing in
    # place: an earlier run may have left a symlink or HARD link to the
    # user's original mdoc at *dest_path*, and writing through it would
    # silently rewrite the original. os.replace swaps the directory entry
    # only - whatever it pointed at is left exactly as it was.
    temp_path = dest_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.write(preamble)
        handle.write("".join(kept))
    os.replace(temp_path, dest_path)
    return dest_path


def matches_tomo(text, tomo_name):
    """True when *text* refers to *tomo_name*, and only to that tomogram.

    Other programs wrap the tomogram name in prefixes and suffixes of their
    own - RESOLVE reports ``odd_Position_1_rec_tomo_Tomograms_locRes``, IsoNet
    writes ``final_Position_1_rec_tomo.mrc`` - so the name is matched as a
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


def pixel_size_is_read(config):
    """True when ``acquisition.pixel_size`` says to READ the value from the
    data rather than use a number given in the config.

    Kept next to (and used by) :func:`resolve_pixel_size` so the two can
    never disagree about which source is actually in play - a log line that
    names the wrong source is worse than no log line at all.
    """
    setting = config["acquisition"]["pixel_size"]
    return isinstance(setting, str) and setting.strip().lower() == "read"


def resolve_pixel_size(config, series, first_frame_path=None):
    """Determine the unbinned pixel size in Angstrom for one tilt series.

    ``acquisition.pixel_size: read`` -> PixelSpacing from the mdoc, falling
    back to IMOD ``header`` on the first raw frame. Anything else is taken
    as a literal number.

    The ``header`` fallback is the ONLY thing IMOD is ever used for, which is
    why a missing IMOD stops the run with
    :data:`HEADER_MISSING_MESSAGE` - telling you to set
    ``acquisition.pixel_size`` - rather than being reported as a missing
    dependency.
    """
    if not pixel_size_is_read(config):
        return float(config["acquisition"]["pixel_size"])

    if series.pixel_spacing:
        return float(series.pixel_spacing)

    header = config.exe("header")
    if not executable_available(header):
        raise SystemExit(HEADER_MISSING_MESSAGE % {
            "series": series.name,
            "mdoc": os.path.basename(series.mdoc_path),
            "header": header,
            "config": config.source,
        })

    if first_frame_path and os.path.exists(first_frame_path):
        info("pixel size not in the mdoc - reading it with IMOD: %s %s"
             % (header, os.path.basename(first_frame_path)))
        output = capture([header, first_frame_path],
                         env=gpu_env(config.gpus))
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


def resolve_angpix(value, reference_apix, key="value"):
    """Resolve a WarpTools pixel-size config setting.

    *value* is either a plain number, or a simple expression relative to the
    dataset's own unbinned pixel size (``resolve_pixel_size``'s result,
    passed in as *reference_apix*): ``"pixel_size"``, ``"2*pixel_size"`` or
    ``"pixel_size*2.5"``. No general expression evaluation - just these two
    shapes, matched explicitly rather than via ``eval()``.
    """
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if text.lower() == "pixel_size":
        return float(reference_apix)

    match = _ANGPIX_MULT_RE.match(text)
    if match:
        multiplier = match.group("a") or match.group("b")
        return float(multiplier) * float(reference_apix)

    try:
        return float(text)
    except ValueError:
        raise SystemExit(
            "ERROR: %s must be a number, \"pixel_size\", or an expression "
            "like \"2*pixel_size\", got %r" % (key, value)
        )
