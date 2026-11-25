"""Steps 4 and 5: IsoNet training and prediction.

``train_isonet`` performs the whole preparation and training chain:

  1. ``isonet.py prepare_star``  + per-tomogram tilt range patching
  2. ``isonet.py denoise``       noise2noise denoiser
  3. ``isonet.py predict``       apply the denoiser (needed for the masks)
  4. ``isonet.py make_mask``     masks from the denoised volumes
  5. ``isonet.py refine``        missing-wedge network

CTF: WarpTools' ts_reconstruct phase-flips the half maps (corrects the
CTF's sign only). With ``isonet.ctf_mode: "None"`` IsoNet2 uses no CTF model
at all. With ``"network"`` (default) it models the remaining CTF amplitude
modulation; it then needs each tomogram's defocus, written into
``tomograms.star`` from the WarpTools XML (:func:`tomogram_defocus`), plus
voltage/Cs/amplitude contrast from ``acquisition``, and is told the maps are
already phase-flipped (``isonet.is_ctf_flipped``).

Tilt ranges: WarpTools' XML stores the tilt angles with the OPPOSITE sign to
IsoNet2's convention (confirmed on data: the empty wedge sits where IsoNet2
puts it for -46/52 when the XML says -52/46). Every per-tomogram range is
therefore converted on the way into the star file - see :func:`tilt_range`.

``predict_isonet`` is the final step: apply the refined network to the
prediction set. Its output - the only directory holding the finished,
deliverable tomograms - is ``isonet/corrected_tomos_final/`` (see the output
layout below ``DENOISE_DIR`` and friends).
"""

import glob
import os
import re
import time

from . import tiltfilter
from .halfmap_sets import ensure_halfmaps_available, resolve_frame_halfmaps_dir
from .mdoc import collect_tilt_series
from .steplog import StepLog, step_completed
from .warptools import level_angles, tiltseries_tilt_range
from .utils import (
    banner,
    ensure_dir,
    gpu_env,
    gpu_flag_isonet,
    gpu_id_note,
    gpu_ids_for,
    info,
    link_file,
    run,
    warn,
)

# ------------------------------------------------------ isonet output layout
#
# Passed explicitly as --output_dir / --output_prefix on every isonet.py call
# below, rather than left to isonet.py's own defaults: two different calls
# (the intermediate predict needed only for make_mask, and the final predict
# on the refined network) both invoke ``isonet.py predict``, and without an
# explicit, distinct --output_dir they would land in the very same directory
# under the same names - the final result silently indistinguishable from,
# and liable to mix with, the intermediate one.
#
#   <processing_directory>/isonet/
#     denoise/                    noise2noise networks         (train step 2)
#     predict_denoise_only/       denoise_*.mrc - intermediate, feeds
#                                  make_mask ONLY, not a deliverable
#     mask/                       masks from the denoised volumes (step 4)
#     isonet_maps/                missing-wedge networks        (step 5)
#     corrected_tomos_final/      final_*.mrc - THE deliverable: denoised AND
#                                  missing-wedge corrected
DENOISE_DIR = "denoise"
DENOISE_PREDICT_DIR = "predict_denoise_only"
DENOISE_PREDICT_PREFIX = "denoise_"
MASK_DIR = "mask"
REFINE_DIR = "isonet_maps"
FINAL_PREDICT_DIR = "corrected_tomos_final"
FINAL_PREDICT_PREFIX = "final_"


# ------------------------------------------------------------------ half maps


def _invert_mrc(source, target):
    """Write *source* multiplied by -1 to *target*: same data type, same
    pixel size and origin. Slice by slice through memory maps, so a whole
    tomogram is never held in RAM; written beside *target* and swapped in
    at the end, so an interrupted run never leaves a half-written file."""
    import mrcfile

    temp = target + ".tmp"
    with mrcfile.mmap(source, mode="r", permissive=True) as src:
        with mrcfile.new_mmap(temp, shape=src.data.shape,
                              mrc_mode=int(src.header.mode),
                              overwrite=True) as dst:
            for z in range(src.data.shape[0]):
                dst.data[z] = -src.data[z]
            dst.voxel_size = src.voxel_size
            dst.header.origin = src.header.origin
            # Statistics from the source header instead of recomputing them
            # over the whole volume: inverting swaps and negates min/max and
            # negates the mean; the RMS deviation is unchanged.
            dst.header.dmin = -src.header.dmax
            dst.header.dmax = -src.header.dmin
            dst.header.dmean = -src.header.dmean
            dst.header.rms = src.header.rms
    os.replace(temp, target)


def _inverted_half_maps(config, source_dir, series_list, log=None,
                        dry_run=False):
    """The directory of contrast-inverted copies of *source_dir*'s half
    maps, for IsoNet2 - made here, on first use, for *series_list* only.

    IsoNet2 expects features DARK: its mask keeps the darkest voxels
    (``maxmask`` filters ``-tomo``), training normalises as
    ``(mean - subvolume) / std``, and predict inverts in and back out.
    WarpTools' ts_reconstruct inverts contrast by default (features
    bright), so its half maps are inverted once (x -1) before IsoNet sees
    them. IsoNet's output keeps its input's convention - the final
    tomograms have dark features.

    One directory per source (MissAlignment-corrected or uncorrected), same
    file names. An inverted copy older than its half map (reconstructed
    again since) is recreated, with a message - checked here only, i.e. on
    every IsoNet run.
    """
    corrected = (os.path.abspath(source_dir)
                 == os.path.abspath(config.missalignment_halfmaps_dir))
    target_dir = (config.inverted_halfmaps_dir if corrected
                  else config.inverted_uncorrected_halfmaps_dir)
    if dry_run:
        info("contrast-inverted half maps for IsoNet2 would be made in %s"
             % target_dir)
        return target_dir
    ensure_dir(target_dir)
    made, renewed = 0, []
    for series in series_list:
        for half in ("even", "odd"):
            name = "%s_%s_rec_tomo.mrc" % (half, series.name)
            source = os.path.join(source_dir, name)
            target = os.path.join(target_dir, name)
            if not os.path.exists(source):
                continue
            if os.path.isfile(target):
                if os.path.getmtime(target) >= os.path.getmtime(source):
                    continue
                renewed.append(name)
            else:
                made += 1
            _invert_mrc(source, target)
    if made:
        message = ("inverted the contrast of %d half map(s) for IsoNet2 "
                   "(features dark) into %s" % (made, target_dir))
        info(message)
        if log is not None:
            log.note(message)
    if renewed:
        message = ("%d inverted half map(s) in %s were older than their half "
                   "maps (reconstructed again since) - recreated: %s"
                   % (len(renewed), target_dir, ", ".join(renewed)))
        warn(message)
        if log is not None:
            log.note(message)
    return target_dir


def _linked_half_maps(config, series_list, subset, log=None, force=False,
                      dry_run=False, base_dir=None):
    """Link the half maps of *series_list* into ``isonet/<subset>/{even,odd}``
    (or ``<base_dir>/<subset>/{even,odd}`` - CryoCARE uses the very same
    inverted half maps, linked into its own folder).

    Both halves are linked under the SAME base name (``<tomo>.mrc``) so that
    IsoNet pairs them unambiguously regardless of how it matches the two
    directories. What is linked are contrast-inverted copies of the half
    maps (:func:`_inverted_half_maps`): IsoNet2 expects dark features,
    WarpTools reconstructs them bright. The half maps themselves are never
    touched.

    Sourced from ``halfmap_sets.resolve_frame_halfmaps_dir`` - the
    MissAlignment-corrected half maps by default, falling back to
    ``raw_halfMaps`` (with a logged notice) if those are not available. If
    NEITHER exists yet (this step run on its own, before MissAlignment or
    reconstruct_tomograms ever produced any), ``ensure_halfmaps_available``
    reconstructs the plain ones on demand first.

    The returned list is sorted by tomogram name, which is the order
    ``prepare_star`` sees the linked files in.

    Existing links are always re-pointed if they resolve elsewhere: a
    previous run may have linked the plain half maps (before MissAlignment's
    appeared), or the uninverted ones (before the inversion existed). *force*
    no longer changes that.
    """
    ensure_halfmaps_available(config, log=log, dry_run=dry_run)

    even_dir = ensure_dir(base_dir or config.isonet_dir, subset, "even")
    odd_dir = ensure_dir(base_dir or config.isonet_dir, subset, "odd")

    source_dir = resolve_frame_halfmaps_dir(config, log=log)
    usable = []
    for series in sorted(series_list, key=lambda s: s.name):
        even = os.path.join(source_dir, "even_%s_rec_tomo.mrc" % series.name)
        odd = os.path.join(source_dir, "odd_%s_rec_tomo.mrc" % series.name)
        if not (os.path.exists(even) and os.path.exists(odd)):
            warn("no half maps for %s - excluded from denoising." % series.name)
            continue
        usable.append(series)

    inverted_dir = _inverted_half_maps(config, source_dir, usable, log=log,
                                       dry_run=dry_run)
    if not dry_run:
        for series in usable:
            for half, half_dir in (("even", even_dir), ("odd", odd_dir)):
                # Always re-pointed if it points elsewhere (samefile check in
                # link_file makes this a no-op otherwise): links from before
                # the inversion existed point at the bright-feature maps.
                link_file(os.path.join(inverted_dir, "%s_%s_rec_tomo.mrc"
                                       % (half, series.name)),
                          os.path.join(half_dir, series.name + ".mrc"),
                          force=True, within=config.processing_dir)

    if not usable:
        raise SystemExit(
            "ERROR: none of the selected tomograms have half maps in %s.\n"
            "Run 'tomo_denoising reconstruct_tomograms' first."
            % source_dir
        )
    return even_dir, odd_dir, usable


# ---------------------------------------------------------------- tilt range


def xml_to_isonet_range(xml_min, xml_max):
    """A tilt range from WarpTools' XML, in IsoNet2's convention.

    The XML's angles have the opposite sign to IsoNet2's (and the mdoc's),
    so the range is negated AND swapped: XML -52/46 -> IsoNet -46/52.
    Independent of the defocus handedness ts_defocus_hand finds - that
    (``AreAnglesInverted``) only changes how WarpTools computes the defocus
    across a tilt image, not the projection geometry.
    """
    return -xml_max, -xml_min


def tilt_range(config, tomo_name):
    """Minimum and maximum tilt angle actually used by *tomo_name*, in
    IsoNet2's convention (see :func:`xml_to_isonet_range`).

    Read through ``warptools.tiltseries_tilt_range`` - see that function for
    exactly where this comes from and why (the FINAL
    ``warp_tiltseries/<tomo>.xml``, never an AreTomo-generated file).
    """
    result = tiltseries_tilt_range(config, tomo_name)
    if result is None:
        warn("no usable tilt range for %s (no warp_tiltseries/%s.xml, or it "
             "has no angles)" % (tomo_name, tomo_name))
        return None
    level = level_angles(config, tomo_name) or (0.0, 0.0)
    if level == (0.0, 0.0):
        return xml_to_isonet_range(*result)
    # Leveled (autolevel.enabled): WarpTools adds LevelAngleY to every tilt
    # angle (it is not in <Angles>), so the range IsoNet gets must too.
    # LevelAngleX tilts the missing wedge itself - IsoNet2 cannot model it.
    level_x, level_y = level
    shifted = (result[0] + level_y, result[1] + level_y)
    isonet_range = xml_to_isonet_range(*shifted)
    warn("%s is LEVELED (ts_autolevel): its IsoNet2 tilt range %.2f/%.2f "
         "includes LevelAngleY %+.2f deg. LevelAngleX %+.2f deg tilts the "
         "missing wedge in a way IsoNet2 cannot model - its missing-wedge "
         "correction is off by that much."
         % ((tomo_name,) + isonet_range + (level_y, level_x)))
    return isonet_range


# ------------------------------------------------------------------- defocus


#: Of the (up to) three first-acquired tilts' defoci, one is dropped as an
#: outlier when they span more than this many micrometres.
DEFOCUS_OUTLIER_SPAN_UM = 0.5


def combine_defoci(values_um):
    """One defocus (micrometres) from the first-acquired tilts' values.

    Values <= 0 are ignored (an unfitted or failed tilt). Of three, if they
    span more than :data:`DEFOCUS_OUTLIER_SPAN_UM`, the one farthest from
    their median is dropped and the other two averaged; otherwise their
    median is used. Two are averaged, one is used as is. None if nothing
    usable is left.
    """
    import statistics

    values = [v for v in values_um if v > 0]
    if not values:
        return None
    if len(values) >= 3:
        values = values[:3]
        if max(values) - min(values) > DEFOCUS_OUTLIER_SPAN_UM:
            middle = statistics.median(values)
            values.remove(max(values, key=lambda v: abs(v - middle)))
            return sum(values) / len(values)
        return statistics.median(values)
    return sum(values) / len(values)


def tomogram_defocus(config, tomo_name, count=3):
    """The defocus of *tomo_name* in ANGSTROM for IsoNet2's rlnDefocus, or
    None.

    From the per-tilt ``<GridCTF>`` that ts_ctf writes into
    ``warp_tiltseries/<tomo>.xml`` (NOT the XML's ``<CTF>`` block): the
    *count* used tilts acquired FIRST - lowest ``<Dose>``, i.e. the 0-degree
    view of the acquisition (also with auto_zero / tilt_offset, which only
    shift the angles) and its neighbours in the dose-symmetric scheme -
    combined by :func:`combine_defoci`. GridCTF is in micrometres.
    """
    import xml.etree.ElementTree as ET

    xml_path = os.path.join(config.warp_tiltseries_dir, tomo_name + ".xml")
    try:
        root = ET.parse(xml_path).getroot()
        dose = [float(x) for x in root.find("Dose").text.split()]
        grid = root.find("GridCTF")
        per_tilt = {}
        for node in grid.findall("Node"):
            per_tilt.setdefault(int(node.get("Z")), []).append(float(node.get("Value")))
    except (OSError, ET.ParseError, AttributeError, TypeError, ValueError):
        return None
    if len(per_tilt) != len(dose):
        return None     # no per-tilt GridCTF yet (ts_ctf not run)

    use = root.find("UseTilt")
    flags = ([x.strip().lower() == "true" for x in use.text.split()]
             if use is not None and (use.text or "").strip() else [])
    if len(flags) != len(dose):
        flags = [True] * len(dose)

    first = sorted((i for i in range(len(dose)) if flags[i]),
                   key=lambda i: dose[i])[:count]
    # Mean over the image plane, should GridCTF ever be a spatial grid.
    values = [sum(per_tilt[i]) / len(per_tilt[i]) for i in first if i in per_tilt]
    defocus = combine_defoci(values)
    return None if defocus is None else defocus * 10000.0


def _ctf_mode_on(config):
    return str(config["isonet"]["ctf_mode"]).strip().lower() != "none"


def tilt_ranges(config, series_list):
    """``{tomo_name: (min, max) or None}`` for every series in *series_list*."""
    return {series.name: tilt_range(config, series.name) for series in series_list}


#: Tilt ranges closer than this are treated as identical (rounding in mdocs).
TILT_TOLERANCE = 2.0


def resolve_tilt_range(config, usable, ranges):
    """Pick the single tilt range ``prepare_star`` will be given - in
    IsoNet2's convention, like every range here (*ranges* comes from
    :func:`tilt_range`, ``isonet.fix_tilt_range`` is written in it).

    ``prepare_star`` accepts one ``--tilt_min``/``--tilt_max`` for the whole
    set, so tomograms whose ranges differ have to be reconciled. The WIDEST
    range is used, because of how the two errors fail:

    * declaring a range wider than a tomogram really has -> IsoNet believes
      the outer shells contain data, leaves them alone, and that tomogram is
      merely under-corrected;
    * declaring a range narrower than it really has -> IsoNet believes those
      shells are missing and synthesises them, overwriting measured data.

    The first is recoverable, the second destroys information, so the union
    is the safe default.

    This is only what ``prepare_star`` is called with - it is corrected
    afterwards, back to a per-tomogram range read from each tomogram's own
    ``warp_tiltseries/<tomo>.xml`` (see ``_edit_star_rows``), UNLESS
    ``isonet.fix_tilt_range`` overrides the whole calculation, in which case
    the fixed range returned here is what ends up in the star file too.

    *ranges* is ``tilt_ranges()``'s return value. Returns ``(tilt_min,
    tilt_max, is_override)``.
    """
    override = config["isonet"].get("fix_tilt_range")
    if override:
        low, high = float(override[0]), float(override[1])
        info("isonet.fix_tilt_range override: %.1f to %.1f deg "
             "(used for every tomogram, unedited)" % (low, high))
        return low, high, True

    ranges = {s.name: ranges[s.name] for s in usable if ranges.get(s.name) is not None}
    if not ranges:
        warn("no warp_tiltseries/<tomo>.xml tilt ranges found - falling back "
             "to -60 to 60 deg. Set isonet.fix_tilt_range in your configuration.")
        return -60.0, 60.0, False

    minima = [low for low, _ in ranges.values()]
    maxima = [high for _, high in ranges.values()]
    tilt_min, tilt_max = min(minima), max(maxima)

    spread_low = max(minima) - min(minima)
    spread_high = max(maxima) - min(maxima)
    if spread_low <= TILT_TOLERANCE and spread_high <= TILT_TOLERANCE:
        info("tilt range %.1f to %.1f deg (consistent across %d tomograms)"
             % (tilt_min, tilt_max, len(ranges)))
        return tilt_min, tilt_max, False

    # Not a problem worth a warning: this range only goes to prepare_star,
    # and tomograms.star gets each tomogram's own range right after.
    info("tilt ranges differ across %d tomograms - prepare_star is called "
         "with the widest, %.1f to %.1f deg; each tomogram then gets its own "
         "range in tomograms.star" % (len(ranges), tilt_min, tilt_max))
    return tilt_min, tilt_max, False


_STAR_COLUMN_RE = re.compile(r"^_(\S+)\s+#(\d+)")


def _edit_star_rows(star_path, usable, ranges=None, defoci=None):
    """Write each tomogram's own values into the rows of *star_path*:

    - *ranges* (``tilt_ranges()``'s return value, IsoNet2 convention): its
      own rlnTiltMin/rlnTiltMax instead of the dataset-wide range
      ``prepare_star`` wrote on every row (``resolve_tilt_range`` had to
      pick one value for the whole call). None leaves them as written
      (``isonet.fix_tilt_range``).
    - *defoci* (``{name: Angstrom or None}``, :func:`tomogram_defocus`): its
      own rlnDefocus. None leaves the column as written.

    Rows are matched to tomograms positionally, in the same order ``usable``
    was handed to ``prepare_star`` (sorted by name - which is also the
    order IsoNet2 lists the linked half maps in) - the star file has no
    other way to identify which row is which tomogram, since
    ``rlnTomoName`` is "None".
    """
    with open(star_path, encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    columns = {}
    header_end = None
    for index, line in enumerate(lines):
        match = _STAR_COLUMN_RE.match(line.strip())
        if match:
            columns[match.group(1)] = int(match.group(2)) - 1
            header_end = index

    if ranges is not None and not ("rlnTiltMin" in columns and "rlnTiltMax" in columns):
        warn("%s has no rlnTiltMin/rlnTiltMax column - per-tomogram tilt "
             "ranges were not applied." % star_path)
        ranges = None
    if defoci is not None and "rlnDefocus" not in columns:
        warn("%s has no rlnDefocus column - per-tomogram defocus was not "
             "applied." % star_path)
        defoci = None
    if header_end is None or (ranges is None and defoci is None):
        return

    data_rows = [index for index in range(header_end + 1, len(lines))
                 if lines[index].strip()]
    if len(data_rows) != len(usable):
        warn("%s has %d rows but %d tomograms were prepared - per-tomogram "
             "tilt ranges / defocus were not applied."
             % (star_path, len(data_rows), len(usable)))
        return

    for row_index, series in zip(data_rows, usable):
        fields = lines[row_index].split()
        if ranges is not None:
            own_range = ranges.get(series.name)
            if own_range is None:
                warn("no warp_tiltseries/%s.xml tilt range - that row keeps "
                     "the dataset-wide range prepare_star wrote." % series.name)
            else:
                fields[columns["rlnTiltMin"]] = "%.6f" % own_range[0]
                fields[columns["rlnTiltMax"]] = "%.6f" % own_range[1]
        if defoci is not None and defoci.get(series.name) is not None:
            fields[columns["rlnDefocus"]] = "%.1f" % defoci[series.name]
        lines[row_index] = "\t".join(fields) + "\n"

    with open(star_path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    applied = [part for part, on in (("tilt ranges", ranges is not None),
                                     ("defocus", defoci is not None)) if on]
    info("tomograms.star: per-tomogram %s applied from each "
         "warp_tiltseries/<tomo>.xml" % " and ".join(applied))


def _half_map_pixel_size(config, series):
    """The half maps' actual voxel size: ts_reconstruct is always called
    with an explicit --angpix (warptools.recon_angpix), so that is the
    ground truth directly - no need to re-derive it from independent binning
    factors the way the old MotionCor3+AreTomo2 pipeline had to.

    Resolved through ``warptools.resolve_recon_angpix`` rather than read
    verbatim, since recon_angpix may be written relative to the dataset's own
    pixel size ("2*pixel_size") - exactly as the ts_reconstruct call itself
    resolves it.
    """
    from .warptools import resolve_recon_angpix
    return resolve_recon_angpix(config)


# ------------------------------------------------------------------- star file


def prepare_star(config, series_list, subset, log=None, force=False,
                 dry_run=False):
    """Run ``isonet.py prepare_star``.

    Voltage, Cs and amplitude contrast come from ``acquisition``.
    ``--tilt_min``/``--tilt_max`` take a single value each, so the overall
    range across the selected tomograms is used for that call; the
    per-tomogram ranges are logged.

    Once IsoNet has written ``tomograms.star``, each row gets its tomogram's
    own values (see ``_edit_star_rows``): ``rlnTiltMin``/``rlnTiltMax`` from
    its ``warp_tiltseries/<tomo>.xml``, converted to IsoNet2's sign
    convention - unless ``isonet.fix_tilt_range`` is set, in which case the
    fixed range is left as IsoNet wrote it on every row - and ``rlnDefocus``
    (:func:`tomogram_defocus`). The defocus is written whatever
    ``isonet.ctf_mode`` is (IsoNet2 ignores it in "None" mode), but only a
    CTF mode REQUIRES it: then a tomogram without one stops the step.
    """
    even_dir, odd_dir, usable = _linked_half_maps(config, series_list, subset,
                                                  log=log, force=force,
                                                  dry_run=dry_run)

    pixel_size = _half_map_pixel_size(config, usable[0])
    ranges = tilt_ranges(config, usable)

    defoci = {} if dry_run else {s.name: tomogram_defocus(config, s.name)
                                 for s in usable}

    info("half-map pixel size %.4f A" % pixel_size)
    info("tomograms, in the order prepare_star receives them "
         "(tilt range in IsoNet2's convention):")
    for series in usable:
        own_range = ranges.get(series.name)
        tilts = ("%6.1f to %6.1f" % own_range if own_range is not None
                 else "     no warp_tiltseries/%s.xml" % series.name)
        defocus = defoci.get(series.name)
        info("    %-24s tilt %s   defocus %s"
             % (series.name, tilts,
                "%.0f A" % defocus if defocus is not None else "unknown"))

    missing = [s.name for s in usable if defoci.get(s.name) is None]
    if missing and not dry_run and _ctf_mode_on(config):
        raise SystemExit(
            "ERROR: isonet.ctf_mode is %r, which needs each tomogram's defocus, "
            "but none could be read for: %s\n"
            "It comes from <GridCTF> in warp_tiltseries/<tomo>.xml, written by "
            "ts_ctf (reconstruct_tomograms). Check that ts_ctf ran for these "
            "tilt series, or set isonet.ctf_mode to \"None\"."
            % (config["isonet"]["ctf_mode"], ", ".join(missing)))

    tilt_min, tilt_max, fixed_tilt_range = resolve_tilt_range(config, usable, ranges)

    acquisition = config["acquisition"]
    work_dir = ensure_dir(config.isonet_config_dir, "prepare_" + subset)
    run(config.command("isonet", "prepare_star",
                       "--even", even_dir,
                       "--odd", odd_dir,
                       "--pixel_size", "%.4f" % pixel_size,
                       "--tilt_min", "%.2f" % tilt_min,
                       "--tilt_max", "%.2f" % tilt_max,
                       "--voltage", "%g" % float(acquisition["voltage"]),
                       "--cs", "%g" % float(acquisition["spherical_aberration"]),
                       "--ac", "%g" % float(acquisition["amplitude_contrast"])),
        cwd=work_dir, env=_gpu_env(config),
        log_file=os.path.join(config.isonet_config_dir,
                              "prepare_star_%s.log" % subset),
        dry_run=dry_run)

    if dry_run:
        return os.path.join(work_dir, "tomograms.star"), usable

    produced = sorted(glob.glob(os.path.join(work_dir, "*.star")),
                      key=os.path.getmtime, reverse=True)
    if not produced:
        raise SystemExit(
            "ERROR: isonet.py prepare_star produced no star file in %s\n"
            "See %s"
            % (work_dir,
               os.path.join(config.isonet_config_dir,
                            "prepare_star_%s.log" % subset))
        )
    info("star file: %s" % produced[0])

    if fixed_tilt_range:
        info("isonet.fix_tilt_range is set - tomograms.star keeps that fixed "
             "range on every row.")
    _edit_star_rows(produced[0], usable,
                    ranges=None if fixed_tilt_range else ranges,
                    defoci=defoci)

    return produced[0], usable


# -------------------------------------------------------------------- networks


def resolve_network(config, key, default_dir):
    """Locate a trained network: an explicit setting, or the newest in a directory."""
    setting = str(config["isonet"][key] or "auto").strip()
    if setting and setting.lower() != "auto":
        path = setting if os.path.isabs(setting) else os.path.join(
            config.isonet_dir, setting)
        if not os.path.exists(path):
            raise SystemExit("ERROR: isonet.%s points at a missing file: %s"
                             % (key, path))
        return path

    search = os.path.join(config.isonet_dir, default_dir)
    found = sorted(glob.glob(os.path.join(search, "network_*.pt")),
                   key=os.path.getmtime, reverse=True)
    if not found:
        found = sorted(glob.glob(os.path.join(search, "*.pt")),
                       key=os.path.getmtime, reverse=True)
    if not found:
        raise SystemExit(
            "ERROR: no trained network found in %s\n"
            "Either training has not finished, or set isonet.%s to the "
            "network file explicitly." % (search, key)
        )
    info("using network %s" % found[0])
    return found[0]


def _gpu_args(config):
    """``--gpuID`` for IsoNet: the REAL GPU ids from the config, not remapped.

    This is the one program in the pipeline that must not be given the
    0..N-1 logical ids every other GPU argument uses, and getting it wrong
    fails silently on the wrong cards rather than erroring.

    IsoNet does not treat --gpuID as an index into the already-visible
    devices. It ASSIGNS the value straight to the environment variable,
    before importing its GPU backend - confirmed in IsoNet's own source
    (``bin/refine.py``, ``bin/predict.py``)::

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuID

    So whatever we pass here REPLACES the CUDA_VISIBLE_DEVICES we set, and
    is then interpreted against the full set of physical GPUs. Handing it
    remapped ids (``--gpuID 0,1`` for a config of ``gpus: [2, 3]``) would
    therefore put IsoNet on physical GPUs 0 and 1 - exactly the GPUs the
    config asked it to stay off.

    Contrast miss-alignment, which does the opposite: it uses its device
    arguments as ``cuda:<index>`` against the visible set and never touches
    the environment variable, so there the ids MUST be remapped (see
    ``missalign.resolve_devices``).
    """
    return ["--gpuID", gpu_flag_isonet(gpu_ids_for("isonet", config.gpus))]


def _gpu_env(config):
    """Restrict IsoNet to the configured GPUs.

    --gpuID alone is not quite enough: it only covers the steps we pass it
    to (denoise/predict/refine), while prepare_star and make_mask get no GPU
    argument at all, and any library in the process that initialises CUDA on
    its own would otherwise default to device 0.

    This and the real-id --gpuID above agree rather than fight: IsoNet
    overwrites CUDA_VISIBLE_DEVICES with the same real ids we set here, so
    the value is replaced by an identical one.
    """
    return gpu_env(config.gpus)


#: isonet.cube_size "auto": 128 from this much VRAM per GPU (gpu_memory_gb).
LARGE_CUBE_GPU_MEMORY_GB = 24.0


def resolve_cube_size(config):
    """``isonet.cube_size``: a number as given, or "auto" - 128 when
    ``gpu_memory_gb`` >= 24, otherwise 96 (IsoNet2's own default)."""
    raw = str(config["isonet"].get("cube_size") or "auto").strip().lower()
    if raw == "auto":
        return 128 if config.gpu_memory_gb >= LARGE_CUBE_GPU_MEMORY_GB else 96
    try:
        size = int(float(raw))
    except ValueError:
        raise SystemExit("ERROR: isonet.cube_size must be \"auto\" or a number, "
                         "got %r" % config["isonet"]["cube_size"])
    if size <= 0 or size % 16:
        raise SystemExit("ERROR: isonet.cube_size must be a positive multiple "
                         "of 16, got %d" % size)
    return size


def _ctf_flipped_args(config):
    """``--isCTFflipped`` - only with a CTF mode on; IsoNet2 ignores it in
    "None" mode, and leaving it out keeps those calls exactly as before."""
    if not _ctf_mode_on(config):
        return []
    return ["--isCTFflipped", str(bool(config["isonet"]["is_ctf_flipped"]))]


def _cube_args(config, log=None):
    """``--cube_size`` for denoise/refine - left out when isonet.extra_args
    already passes one (that wins, as it always did)."""
    if "--cube_size" in _extra_args(config):
        return []
    size = resolve_cube_size(config)
    if log is not None:
        log.note("IsoNet cube size %d (isonet.cube_size %r, gpu_memory_gb %g)"
                 % (size, config["isonet"].get("cube_size"), config.gpu_memory_gb))
    return ["--cube_size", str(size)]


def _extra_args(config):
    extra = str(config["isonet"]["extra_args"]).strip()
    return extra.split() if extra else []


# ------------------------------------------------------------------ entrypoints


def train(config, force=None, dry_run=False):
    """``tomo_denoising train_isonet``"""
    force = config.overwrite if force is None else force
    with StepLog(config, "train_isonet", force=force) as log:
        code = _train(config, log, force=force, dry_run=dry_run)
        return code


def _train(config, log, force=False, dry_run=False):
    banner("IsoNet training")
    config.require("isonet", "IsoNet")

    settings = config["isonet"]
    work = ensure_dir(config.isonet_dir)
    logs = ensure_dir(work, "logs")
    ensure_dir(config.isonet_config_dir)

    if not force and step_completed(config, "train_isonet"):
        try:
            network = resolve_network(config, "refine_network", REFINE_DIR)
        except SystemExit:
            network = None
        if network:
            info("IsoNet is already trained: %s" % network)
            log.note("skipped: the previous training completed and %s exists. "
                     "Use --force to retrain." % network)
            return 0

    series_list = tiltfilter.drop_excluded(
        config, collect_tilt_series(config.mdoc_dir("train"), "train"),
        "IsoNet training", log=log)
    if not series_list:
        log.fail("train_isonet", "every tilt series in %s is excluded (see "
                 "%s)" % (config.mdoc_dir("train"), tiltfilter.record_path(config)))
        return 1
    info("%d tilt series selected for IsoNet training" % len(series_list))
    log.note("training on %d tilt series from %s"
             % (len(series_list), config.mdoc_dir("train")))

    log.note(gpu_id_note("isonet", config.gpus))

    banner("IsoNet 1/5  prepare_star")
    star, _ = prepare_star(config, series_list, "train", log=log, force=force,
                           dry_run=dry_run)

    cube_args = _cube_args(config, log)

    banner("IsoNet 2/5  denoise (%d epochs)" % int(settings["denoise_epochs"]))
    run(config.command("isonet", "denoise", star,
                       "--CTF_mode", str(settings["ctf_mode"]),
                       _ctf_flipped_args(config), cube_args,
                       "--epochs", str(int(settings["denoise_epochs"])),
                       "--output_dir", DENOISE_DIR,
                       _gpu_args(config), _extra_args(config)),
        cwd=work, env=_gpu_env(config),
        log_file=os.path.join(logs, "denoise.log"), dry_run=dry_run)

    banner("IsoNet 3/5  apply the denoiser (intermediate - feeds make_mask only)")
    denoise_network = (
        os.path.join(work, DENOISE_DIR, "network.pt") if dry_run
        else resolve_network(config, "denoise_network", DENOISE_DIR)
    )
    run(config.command("isonet", "predict", star, denoise_network,
                       "--output_dir", DENOISE_PREDICT_DIR,
                       "--output_prefix", DENOISE_PREDICT_PREFIX,
                       _ctf_flipped_args(config),
                       _gpu_args(config), _extra_args(config)),
        cwd=work, env=_gpu_env(config),
        log_file=os.path.join(logs, "predict_denoise.log"),
        dry_run=dry_run)

    banner("IsoNet 4/5  make_mask")
    run(config.command("isonet", "make_mask", star,
                       "--density_percentage", str(settings["density_percentage"]),
                       "--std_percentage", str(settings["std_percentage"]),
                       "--input_column", "rlnDenoisedTomoName",
                       "--output_dir", MASK_DIR),
        cwd=work, env=_gpu_env(config),
        log_file=os.path.join(logs, "make_mask.log"), dry_run=dry_run)

    banner("IsoNet 5/5  refine (%d epochs)" % int(settings["refine_epochs"]))
    run(config.command("isonet", "refine", star,
                       "--CTF_mode", str(settings["ctf_mode"]),
                       _ctf_flipped_args(config), cube_args,
                       "--mw_weight", str(settings["mw_weight"]),
                       "--bfactor", str(settings["bfactor"]),
                       "--epochs", str(int(settings["refine_epochs"])),
                       "--output_dir", REFINE_DIR,
                       _gpu_args(config), _extra_args(config)),
        cwd=work, env=_gpu_env(config),
        log_file=os.path.join(logs, "refine.log"), dry_run=dry_run)

    info("IsoNet training finished. Networks are in %s"
         % os.path.join(work, REFINE_DIR))
    log.note("networks: %s" % os.path.join(work, REFINE_DIR))
    for entry in series_list:
        log.ok(entry.name)
    return 0


def predict(config, force=None, dry_run=False):
    """``tomo_denoising predict_isonet``"""
    force = config.overwrite if force is None else force
    with StepLog(config, "predict_isonet", force=force) as log:
        return _predict(config, log, force=force, dry_run=dry_run)


def _write_pixel_size(directory, pixel_size, since, log):
    """Write *pixel_size* into the header of every tomogram IsoNet predicted
    into *directory* in this call - IsoNet2 before its voxel-size fix writes
    1.0. Header only, in place: the volume data is neither read nor
    rewritten. Only files written since *since*, so results of an earlier
    run (possibly at another pixel size) are never relabelled."""
    import mrcfile

    written = 0
    names = sorted(os.listdir(directory)) if os.path.isdir(directory) else []
    for name in names:
        path = os.path.join(directory, name)
        if (name.endswith(".mrc") and not os.path.islink(path)
                and os.path.getmtime(path) >= since):
            try:
                with mrcfile.mmap(path, mode="r+") as mrc:
                    mrc.voxel_size = pixel_size
                written += 1
            except Exception as exc:
                warn("%s: pixel size not written to its header (%s)" % (name, exc))
    log.note("pixel size %.3f A written to the header of %d final tomogram(s)"
             % (pixel_size, written))


def _predict(config, log, force=False, dry_run=False):
    banner("IsoNet prediction")
    config.require("isonet", "IsoNet")

    work = ensure_dir(config.isonet_dir)
    logs = ensure_dir(work, "logs")
    ensure_dir(config.isonet_config_dir)

    series_list = tiltfilter.drop_excluded(
        config, collect_tilt_series(config.mdoc_dir("predict"), "predict"),
        "IsoNet prediction", log=log)
    if not series_list:
        log.fail("predict_isonet", "every tilt series in %s is excluded (see "
                 "%s)" % (config.mdoc_dir("predict"), tiltfilter.record_path(config)))
        return 1
    info("%d tilt series selected for IsoNet prediction" % len(series_list))
    log.note("predicting %d tilt series from %s"
             % (len(series_list), config.mdoc_dir("predict")))

    log.note(gpu_id_note("isonet", config.gpus))

    if not force and step_completed(config, "predict_isonet"):
        info("IsoNet prediction already complete.")
        log.note("skipped: the previous prediction completed. "
                 "Use --force to redo it.")
    else:
        star, _ = prepare_star(config, series_list, "predict", log=log,
                               force=force, dry_run=dry_run)

        network = (
            os.path.join(work, REFINE_DIR, "network.pt") if dry_run
            else resolve_network(config, "refine_network", REFINE_DIR)
        )

        started = time.time()
        run(config.command("isonet", "predict", star, network,
                           "--output_dir", FINAL_PREDICT_DIR,
                           "--output_prefix", FINAL_PREDICT_PREFIX,
                           _ctf_flipped_args(config),
                           _gpu_args(config), _extra_args(config)),
            cwd=work, env=_gpu_env(config),
            log_file=os.path.join(logs, "predict_refine.log"),
            dry_run=dry_run)
        if not dry_run:
            _write_pixel_size(os.path.join(work, FINAL_PREDICT_DIR),
                              _half_map_pixel_size(config, series_list[0]),
                              started, log)

    final_dir = os.path.join(work, FINAL_PREDICT_DIR)
    info("IsoNet prediction finished. Final denoised tomograms: %s" % final_dir)
    log.note("final denoised tomograms: %s" % final_dir)
    for entry in series_list:
        log.ok(entry.name)

    if not dry_run:
        from . import report
        report.refresh_after_prediction(config, "isonet", log=log)
    return 0
