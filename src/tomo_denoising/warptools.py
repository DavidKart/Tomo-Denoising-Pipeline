"""Steps 1-3: raw frames -> WarpTools project -> aligned, reconstructed tomograms.

Everything up to reconstruction runs through WarpTools natively:

  1. motion_correct        create_settings (frame series) + fs_motion_and_ctf
  2. align_tiltseries      ts_import -> create_settings (tilt series) ->
                           ts_stack -> OUR OWN tilt axis search (tiltaxis.py,
                           unchanged) -> ts_import --override_axis ->
                           native ts_aretomo3, axis fixed, no search
  3. reconstruct_tomograms ts_defocus_hand -> ts_ctf -> ts_reconstruct
                           (frame-split and, when quality_check.
                           enable_tilt_split is set, tilt-split)

WarpTools' own tilt-series alignment cannot average a tilt axis across a
dataset - that is exactly what this pipeline's tilt axis refinement already
does well, so it is kept as an external step: a few alignment-only AreTomo
runs (the SAME AreTomo build ``ts_aretomo3 --exe`` is later handed) on random
subsets of ``ts_stack``'s output, combined into one dataset-wide axis, fixed
with ``--override_axis``/``--axis ... --axis_iter 0``. The real per-tomogram
alignment (every per-tilt shift) is then computed by genuine, native
``ts_aretomo3`` - nothing of ours is imported back.

Directory layout under ``<processing_directory>``::

  warp_frameseries.settings
  warp_frameseries/            fs_motion_and_ctf output (averages, CTF, ...)
  tomostar/                    ts_import output
  warp_tiltseries.settings
  warp_tiltseries/
    <tomo>.xml                 native ts_aretomo3 alignment (later refined in
                                place by run_missalignment)
    tiltstack/<tomo>/<tomo>.mrc + .rawtlt   ts_stack output
    tiltaxis_search/<tomo>/    OUR OWN scratch alignment-only AreTomo runs -
                                never mistaken for real project state
    reconstruction/            ts_reconstruct output, one --output_processing
                                per call and pixel size (full_<px>/,
                                frames_<px>/, tilts_<px>/) - the physical
                                files the links below point at, left where
                                WarpTools wrote them
    tomograms/                 symlinks - full, non-split tomogram per series
    raw_halfMaps/               symlinks - exactly two entries per tomogram
    raw_halfMaps_tiltSplit/     symlinks - the same, tilt-split - made ONLY
                                when quality_check.enable_tilt_split is true,
                                since RESOLVE is the only thing that uses
                                them (it measures the frame-split half maps
                                directly otherwise)

``run_ts_reconstruct`` is also used by :mod:`missalign`, which reconstructs
the SAME ``warp_tiltseries`` project again after refining its alignment in
place, into a different destination.
"""

import collections
import glob
import json
import os
import re
import shlex
import time
import uuid

from . import tiltaxis, tiltfilter
from .mdoc import (
    collect_tilt_series, find_mdocs, match_tomo_name, pixel_size_is_read,
    resolve_angpix, resolve_pixel_size, tomo_name_from_mdoc, write_filtered_mdoc,
)
from .steplog import StepLog
from .utils import (
    banner, ensure_dir, gpu_env, gpu_id_note, gpu_ids_for, info, link_file,
    run as run_command, warn,
)

STEP_MOTION = "motion_correct"
STEP_ALIGN = "align_tiltseries"
STEP_RECONSTRUCT = "reconstruct_tomograms"


def _has_any_files(directory):
    if not os.path.isdir(directory):
        return False
    with os.scandir(directory) as entries:
        return any(entries)


def _gpu_env(config):
    """The environment EVERY call in this module gets - not only the GPU-heavy
    ones.

    Applied uniformly on purpose: WarpTools decides for itself which of its
    sub-steps touch a GPU, so restricting visibility only where we expect
    compute would leave gaps. CPU-only calls (create_settings, ts_import)
    simply ignore it.
    """
    return gpu_env(config.gpus)


#: WarpTools subcommands documented to accept ``--device_list`` (WarpTools
#: 2.0.0 API reference, frame_series and tilt_series pages). Passing it to a
#: subcommand that does NOT take it is a hard parse failure, so this is an
#: explicit allowlist rather than a guess - note that create_settings,
#: ts_import and ts_defocus_hand are all absent from it, and that
#: ``ts_aretomo3`` is deliberately absent too: the published reference
#: documents ``ts_aretomo`` (AreTomo2) only, and this pipeline runs the newer
#: ts_aretomo3, whose option list could not be verified. Leaving it off costs
#: nothing - ``--device_list`` defaults to every GPU the process can see,
#: which CUDA_VISIBLE_DEVICES has already narrowed to exactly the configured
#: ones. The flag is belt-and-braces, never the actual restriction.
_DEVICE_LIST_COMMANDS = frozenset((
    "fs_motion_and_ctf", "fs_motion", "fs_ctf", "fs_boxnet_infer",
    "fs_export_micrographs", "fs_export_particles",
    "ts_stack", "ts_aretomo", "ts_etomo_fiducials", "ts_etomo_patches",
    "ts_ctf", "ts_reconstruct", "ts_template_match", "ts_export_particles",
))


def _warp(config, subcommand, *args):
    """``WarpTools <subcommand> <args...>``, run inside conda_env_warptools
    when set, and told which GPUs to use when the subcommand supports it.

    Every WarpTools call in the package goes through this rather than
    ``config.exe("warptools")`` directly, so WarpTools living in its own
    environment - and the ``--device_list`` question - are each settled in a
    single place instead of at every call site.

    ``--device_list`` is placed immediately after the subcommand, before the
    caller's own arguments. That is deliberate: it takes a SPACE-separated
    list of ids, so whatever follows the last id must be an option token
    rather than a bare value. Every call site below starts its arguments with
    ``--settings``, which makes that guaranteed rather than incidental.
    """
    device = []
    if subcommand in _DEVICE_LIST_COMMANDS:
        ids = gpu_ids_for("warptools", config.gpus)
        if ids:
            device = ["--device_list"] + [str(g) for g in ids]
    return config.command("warptools", subcommand, device, *args)


# ------------------------------------------------- WarpTools command outcome


#: One per item a WarpTools command gives up on, e.g. "Failed to process
#: /…/warp_tiltseries/Position_21.tomostar, marked as unselected".
_WARP_FAILED_RE = re.compile(r"Failed to process\s+(.+?), marked as unselected")
#: The per-item progress lines, e.g. "48/48, 2 failed, 00:00 remaining".
#: Only these carry "remaining" - the "48/48, previous metadata found for
#: 48" of the earlier parsing phase does not count.
_WARP_PROGRESS_RE = re.compile(r"(?m)^\s*(\d+)/(\d+)\b[^\n]*remaining")
#: WarpTools' own bug: after the last item, writing its progress summary
#: (BaseCommand.WriteMiniJson) throws on a NaN/Infinity value in some item's
#: metadata, and the process aborts (exit 134). Observed after ts_aretomo3
#: and ts_ctf, on two datasets (WarpTools 2.0.0dev39).
_WARP_SUMMARY_CRASH = ("WriteMiniJson", "cannot be written as valid JSON")

WarpLog = collections.namedtuple("WarpLog", "failed finished summary_crash")


def read_warp_log(log_path):
    """What a WarpTools command's log says (it is rewritten on every run,
    so it only ever describes the latest one): the items it reported as
    failed (bare names), whether its progress reached the last item, and
    whether it then hit the summary crash."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return WarpLog(set(), False, False)
    failed = {os.path.splitext(os.path.basename(path.strip()))[0]
              for path in _WARP_FAILED_RE.findall(text)}
    finished = any(int(done) == int(total) > 0
                   for done, total in _WARP_PROGRESS_RE.findall(text))
    crash = all(marker in text for marker in _WARP_SUMMARY_CRASH)
    return WarpLog(failed, finished, crash)


def _warp_outcome(code, log_path, label, log):
    """Whether a WarpTools command got through its work: the parsed log
    (:func:`read_warp_log`) if it did, ``None`` - with the failure logged -
    if it did not.

    Exit 0, or a non-zero exit that is exactly WarpTools' summary crash
    after the last item, count as done (the latter with a warning: every
    item's result is already saved by then; only WarpTools' own summary is
    missing). Any other non-zero exit is a real failure. Items WarpTools
    gave up on along the way are reported either way.
    """
    outcome = read_warp_log(log_path)
    if code != 0:
        if not (outcome.finished and outcome.summary_crash):
            if log is not None:
                log.fail(label, "exit status %d - see %s" % (code, log_path))
            return None
        message = ("%s finished every item, then exited with status %d on "
                   "WarpTools' own bug writing its progress summary (a "
                   "NaN/Infinity value) - results are complete, continuing. "
                   "See %s" % (label, code, log_path))
        warn(message)
        if log is not None:
            log.quiet(message)
    if outcome.failed:
        message = ("%s: WarpTools gave up on %d item(s) and marked them "
                   "unselected: %s" % (label, len(outcome.failed),
                                       ", ".join(sorted(outcome.failed))))
        warn(message)
        if log is not None:
            log.quiet(message)
    return outcome


#: Every tilt series left is excluded - "%s" (imported/aligned), record path.
_ALL_EXCLUDED = ("every %s tilt series is excluded (see %s) - nothing left to "
                 "process. Fix the cause (warp_tiltseries/logs/), then run "
                 "'tomo_denoising align_tiltseries --force'. WarpTools marked "
                 "the ones it gave up on unselected in their XMLs and may skip "
                 "them for that - re-select them with WarpTools "
                 "change_selection first.")


def _exclude_failed(config, stage, outcome, series_names, log):
    """*series_names* without the ones WarpTools gave up on in *stage*
    (see :func:`_warp_outcome`), which are recorded as excluded - so the step
    and everything after it go on without them instead of failing on their
    missing output later."""
    hit = sorted(set(series_names) & outcome.failed)
    if hit:
        reason = ("%s failed - WarpTools marked it unselected; see %s"
                  % (stage, os.path.join(config.warp_tiltseries_dir, "logs")))
        tiltfilter.record_failures(config, stage, {name: reason for name in hit})
        if log is not None:
            for name in hit:
                log.exclude(name, "[%s] %s" % (stage, reason))
    return [name for name in series_names if name not in outcome.failed]


def _gain_orientation_args(acquisition):
    """--gain_flip_x/--gain_flip_y/--gain_transpose for create_settings -
    shared by the frame-series and tilt-series calls, which both need the
    same gain reference orientation applied."""
    args = []
    if acquisition["gain_flip_x"]:
        args.append("--gain_flip_x")
    if acquisition["gain_flip_y"]:
        args.append("--gain_flip_y")
    if acquisition["gain_transpose"]:
        args.append("--gain_transpose")
    return args


def dataset_apix(config, series_list):
    """Unbinned pixel size (Angstrom), assumed constant across the dataset -
    read from the first tilt series that has one (mdoc PixelSpacing, falling
    back to IMOD header - see mdoc.resolve_pixel_size)."""
    series = series_list[0]
    first_frame = (os.path.join(config.input_frames, series.frames[0])
                   if series.frames else None)
    return resolve_pixel_size(config, series, first_frame)


# ============================================================== motion_correct


def motion_correct(config, force=None, dry_run=False):
    """``tomo_denoising motion_correct``"""
    force = config.overwrite if force is None else force
    with StepLog(config, STEP_MOTION, force=force) as log:
        return _motion_correct(config, log, force=force, dry_run=dry_run)


def _motion_correct(config, log, force=False, dry_run=False):
    banner("WarpTools motion correction + CTF (fs_motion_and_ctf)")
    config.require("warptools", "WarpTools")

    settings_path = config.warp_frameseries_settings_path
    frameseries_dir = ensure_dir(config.warp_frameseries_dir)
    logs = ensure_dir(config.logs_dir)

    if not force and os.path.isfile(settings_path) and _has_any_files(frameseries_dir):
        log.note("skipped: %s already has WarpTools frame-series output. "
                 "Use --force to redo it." % frameseries_dir)
        return 0

    log.note(gpu_id_note("warptools", config.gpus))

    gain = config.resolve_gain()
    if gain:
        log.note("gain reference: %s" % gain)

    directory = config.mdoc_dir("preprocessing")
    series_list = collect_tilt_series(directory, "preprocessing")
    log.note("%d tilt series found in %s" % (len(series_list), directory))

    apix = dataset_apix(config, series_list)
    # Name the source the value ACTUALLY came from - crediting the first tilt
    # series unconditionally made an explicitly configured pixel size look
    # like it had been read from that mdoc.
    if pixel_size_is_read(config):
        log.note("unbinned pixel size: %.4f A (read from %s)"
                 % (apix, series_list[0].name))
    else:
        log.note("unbinned pixel size: %.4f A (acquisition.pixel_size, set "
                 "explicitly in %s)" % (apix, config.source))

    acquisition = config["acquisition"]
    extension = acquisition["frame_extension"]
    fs = config["frameseries"]

    command = _warp(
        config, "create_settings",
        "--output", settings_path,
        "--folder_data", config.input_frames,
        "--folder_processing", frameseries_dir,
        "--extension", "*.%s" % extension,
        "--angpix", "%.4f" % apix,
        "--exposure", "%.4f" % float(fs["exposure"]),
        "--bin", str(int(fs["bin"])),
    )
    if gain:
        command += ["--gain_path", gain]
        command += _gain_orientation_args(acquisition)
    if extension == "eer" and str(fs["eer_ngroups"]).strip():
        # EER grouping is a property of how the raw data is read, so it's
        # set once here on create_settings - not on fs_motion_and_ctf, which
        # rejects it.
        command += ["--eer_ngroups", str(fs["eer_ngroups"])]
    run_command(command, cwd=config.processing_dir, env=_gpu_env(config),
               log_file=os.path.join(logs, "create_settings_frameseries.log"),
               dry_run=dry_run)
    if not dry_run and not os.path.isfile(settings_path):
        raise SystemExit(
            "ERROR: WarpTools create_settings produced no %s\nSee %s"
            % (settings_path, os.path.join(logs, "create_settings_frameseries.log"))
        )
    log.task("create_settings (frameseries)", True, "-")

    command = _warp(
        config, "fs_motion_and_ctf",
        "--settings", settings_path,
        "--m_grid", str(fs["m_grid"]),
        "--c_grid", str(fs["c_grid"]),
        "--c_range_max", str(fs["c_range_max"]),
        "--c_defocus_max", str(fs["c_defocus_max"]),
        # --c_voltage is an integer field (WarpTools' own --help shows
        # "Default: 300." with no decimal, unlike --c_cs/--c_amplitude which
        # show "2.7."/"0.07.") - a decimal value fails its parser.
        "--c_voltage", str(int(round(float(acquisition["voltage"])))),
        "--c_cs", "%.4f" % float(acquisition["spherical_aberration"]),
        "--c_amplitude", "%.4f" % float(acquisition["amplitude_contrast"]),
        "--out_averages",
        "--out_average_halves",
        "--perdevice", str(int(fs["perdevice"])),
    )
    if fs["c_use_sum"]:
        command.append("--c_use_sum")
    extra = str(fs["extra_args"]).strip()
    if extra:
        command += extra.split()

    log_file = os.path.join(logs, "fs_motion_and_ctf.log")
    code = run_command(command, cwd=config.processing_dir, env=_gpu_env(config),
                       log_file=log_file, check=False, dry_run=dry_run)
    if not dry_run and _warp_outcome(code, log_file, "fs_motion_and_ctf", log) is None:
        return 1
    log.task("fs_motion_and_ctf", True, "-")

    if dry_run:
        return 0
    if not _has_any_files(frameseries_dir):
        log.fail("fs_motion_and_ctf", "produced nothing in %s" % frameseries_dir)
        return 1

    for series in series_list:
        log.ok(series.name)
    log.note("frame-series output: %s" % frameseries_dir)
    return 0


# ------------------------------------------------------- frame-series averages


def _prefix_boundary_ok(basename, prefix):
    """True if *basename* starts with *prefix* followed by a non-alphanumeric
    character, or nothing at all.

    Guards a "prefix*" style match from mistaking *prefix* for a strict
    prefix of some OTHER, unrelated name that merely happens to start the
    same way (e.g. "Position_7" is a literal prefix of "Position_7_4" and of
    "Position_70") - see the (confirmed, in production) bug this exact
    mistake caused in the reconstruction linking loop below.
    """
    if not basename.startswith(prefix):
        return False
    rest = basename[len(prefix):]
    return not rest or not rest[0].isalnum()


def frameseries_average_path(config, stem):
    """Location of the motion-corrected average WarpTools wrote for movie
    *stem* (used by ``tomo_dimensions.xy: "auto"`` - see
    ``_auto_tomo_dimensions_xy``).

    The real, confirmed layout is FLAT: ``warp_frameseries/average/<stem>.mrc``
    - not nested under a per-movie subdirectory. Movie stems commonly contain
    the tilt angle in square brackets (e.g. ``Position_02_001[-10.00]_EER``),
    which is why every lookup here uses ``glob.escape(stem)`` - plain
    ``glob.glob``/``fnmatch`` treat ``[...]`` as a wildcard character class,
    not literal text, and would otherwise silently match nothing for every
    such file.
    """
    candidate = os.path.join(config.warp_frameseries_dir, "average", stem + ".mrc")
    if os.path.isfile(candidate):
        return candidate
    # Fallback for a WarpTools layout that does nest by movie after all.
    pattern = os.path.join(config.warp_frameseries_dir, "**",
                           glob.escape(stem) + ".mrc")
    found = [f for f in glob.glob(pattern, recursive=True)
             if "average" in f.replace(os.sep, "/").split("/")]
    if found:
        return found[0]
    found = sorted(f for f in glob.glob(
        os.path.join(config.warp_frameseries_dir, "**", glob.escape(stem) + "*.mrc"),
        recursive=True)
        if _prefix_boundary_ok(os.path.basename(f), stem))
    return found[0] if found else None


# ================================================================ align_tiltseries


def resolve_tomo_dimensions(config, series_list, log, dry_run=False):
    xy = str(config["tomo_dimensions"]["xy"]).strip()
    z = int(config["tomo_dimensions"]["z"])
    if xy.lower() == "auto":
        x, y = _auto_tomo_dimensions_xy(config, series_list, log, dry_run=dry_run)
    else:
        try:
            x_str, y_str = xy.lower().split("x")
            x, y = int(x_str), int(y_str)
        except ValueError:
            raise SystemExit(
                "ERROR: tomo_dimensions.xy must be 'auto' or 'XxY', got %r" % xy
            )
    return x, y, z


def frame_image_xy(config, series_list):
    """(X, Y, path): the true camera-frame size in unbinned pixels, read off
    a frame-series average's own shape and scaled back up by
    ``2**frameseries.bin`` (WarpTools' own log2 Fourier binning convention).
    ``None`` if no frame-series average exists yet.

    This is always the physical tilt-image size, independent of whatever
    ``tomo_dimensions.xy`` is set to - an explicit, axis-swapped
    ``tomo_dimensions.xy`` changes the VOLUME, never the images.
    """
    import mrcfile

    fs_bin = int(config["frameseries"]["bin"])
    for series in series_list:
        if not series.frames:
            continue
        stem = os.path.splitext(series.frames[0])[0]
        path = frameseries_average_path(config, stem)
        if path is None:
            continue
        with mrcfile.open(path, mode="r", permissive=True) as mrc:
            shape = mrc.data.shape
        height, width = (shape[-2], shape[-1])
        return width * (2 ** fs_bin), height * (2 ** fs_bin), path
    return None


def _auto_tomo_dimensions_xy(config, series_list, log, dry_run=False):
    """(X, Y) unbinned pixels for ``tomo_dimensions.xy: "auto"`` - the
    camera-frame size from :func:`frame_image_xy`."""
    found = frame_image_xy(config, series_list)
    if found is not None:
        x, y, path = found
        log.note("tomo_dimensions.xy auto-detected: %dx%d (from %s, "
                 "frameseries.bin=%d)" % (x, y, os.path.basename(path),
                                          int(config["frameseries"]["bin"])))
        return x, y

    if dry_run:
        log.note("tomo_dimensions.xy: 'auto' cannot be resolved yet (no "
                 "frame-series average exists) - using a 0x0 placeholder for "
                 "this dry run. The real value is read once "
                 "'tomo_denoising motion_correct' has actually run.")
        return 0, 0

    raise SystemExit(
        "ERROR: tomo_dimensions.xy is 'auto' but no frame-series average "
        "could be found under %s to read the dimensions from.\n"
        "Run 'tomo_denoising motion_correct' first, or set tomo_dimensions.xy "
        "explicitly, e.g. \"4400x6000\"." % config.warp_frameseries_dir
    )


def aligned_series_names(config, include_excluded=False):
    """Tilt series names that already have a native ts_aretomo3 alignment -
    leaving out the ones in ``excluded_tiltseries.json`` (an AreTomo3
    failure keeps its XML, marked unselected by WarpTools) unless
    *include_excluded*."""
    pattern = os.path.join(config.warp_tiltseries_dir, "*.xml")
    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(pattern))
    if include_excluded:
        return names
    excluded = tiltfilter.excluded_names(config)
    return [name for name in names if name not in excluded]


def tiltseries_tilt_range(config, tomo_name):
    """Minimum and maximum tilt angle actually used, read from the FINAL
    ``warp_tiltseries/<tomo>.xml`` - never from an AreTomo-generated file
    (.aln) or a ts_stack ``.rawtlt``.

    This is deliberately the single source of truth used everywhere a tilt
    range is needed downstream (currently: IsoNet's per-tomogram
    ``rlnTiltMin``/``rlnTiltMax``): the XML reflects native ts_aretomo3's own
    alignment and, once ``run_missalignment`` has run, MissAlignment's
    in-place refinement of it too - so this always matches whatever was
    actually reconstructed, with no separate code path to fall out of sync.

    The XML's root ``<TiltSeries>`` element carries a whitespace-separated
    ``<Angles>`` list and a same-length ``<UseTilt>`` list of ``True``/
    ``False`` flags; only angles flagged used are considered. Returns
    ``None`` if the XML does not exist yet or has no usable angles.
    """
    angles = used_tilt_angles(config, tomo_name)
    if not angles:
        return None
    return min(angles), max(angles)


def used_tilt_angles(config, tomo_name):
    """The tilt angles ``warp_tiltseries/<tomo_name>.xml`` actually uses
    (``<Angles>`` filtered by ``<UseTilt>``), or ``None`` if the XML does not
    exist, cannot be parsed, or has no ``<Angles>``."""
    import xml.etree.ElementTree as ET

    xml_path = os.path.join(config.warp_tiltseries_dir, tomo_name + ".xml")
    if not os.path.isfile(xml_path):
        return None

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        warn("%s: could not parse %s: %s" % (tomo_name, xml_path, exc))
        return None

    angles_el = root.find("Angles")
    if angles_el is None or not (angles_el.text or "").strip():
        return None
    try:
        angles = [float(x) for x in angles_el.text.split()]
    except ValueError:
        warn("%s: non-numeric value in <Angles> of %s" % (tomo_name, xml_path))
        return None

    use_el = root.find("UseTilt")
    if use_el is not None and (use_el.text or "").strip():
        flags = [x.strip().lower() == "true" for x in use_el.text.split()]
        if len(flags) == len(angles):
            angles = [angle for angle, used in zip(angles, flags) if used]
        else:
            warn("%s: <UseTilt> has %d entries but <Angles> has %d in %s - "
                 "ignoring <UseTilt>, using every angle."
                 % (tomo_name, len(flags), len(angles), xml_path))
    return angles


# --------------------------------------------------- our own tilt axis search


def find_aln(stack_path, *search_dirs):
    """Locate the .aln AreTomo wrote - version dependent, so look around."""
    stem = os.path.splitext(os.path.basename(stack_path))[0]
    candidates = [os.path.join(os.path.dirname(stack_path), stem + ".aln")]
    for directory in search_dirs:
        candidates.append(os.path.join(directory, stem + ".aln"))
        candidates.extend(sorted(glob.glob(os.path.join(directory, "*.aln"))))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def resolve_alignz_angstrom(config, apix, log=None):
    """Physical alignment thickness in Angstrom from ``aretomo.alignz``, or
    ``None`` when it is left empty.

    ``aretomo.alignz`` is a SEPARATE value from ``tomo_dimensions.z`` on
    purpose - the volume worth aligning on is not the volume worth
    reconstructing - but shares its unit: UNBINNED pixels. It is resolved to
    a physical thickness here, once, because its two consumers want
    different units for the same quantity:

      * ``WarpTools ts_aretomo3 --alignz`` - ANGSTROM (an integer field).
      * ``AreTomo3 -AlignZ`` (this pipeline's own axis-search calls) -
        PIXELS at the input stack's own pixel size, i.e. ``align_angpix``.

    Empty (the default) means neither call is given the argument at all,
    leaving each tool at its own default.
    """
    raw = str(config["aretomo"]["alignz"] or "").strip()
    if not raw:
        return None
    try:
        unbinned_px = float(raw)
    except ValueError:
        raise SystemExit(
            "ERROR: aretomo.alignz must be a number of unbinned pixels (or "
            "empty to leave it unset), got %r" % raw
        )
    angstrom = unbinned_px * apix
    if log is not None:
        log.note("alignz: %g unbinned px -> %.1f A (unbinned pixel size "
                 "%.4f A)" % (unbinned_px, angstrom, apix))
    return angstrom


def _axis_search_align(config, name, stack, align_angpix, tilt_axis,
                       refine, iteration, alignz_px=None, dry_run=False):
    """One alignment-only (``-VolZ 0``) AreTomo3 run, for the tilt axis search.

    Writes into ``tiltaxis_search/<name>/`` (``-OutDir``) - never into
    ``tiltstack/``: with ``-VolZ 0`` AreTomo3 writes an aligned stack rather
    than a reconstruction, and that must never be mistaken for real project
    state. The ``.rawtlt`` is not passed explicitly - AreTomo3 discovers it
    itself next to *stack* by naming convention (``<stack-stem>.rawtlt``,
    exactly what ``ts_stack`` writes alongside ``<stack-stem>.st``).

    Restricted to the configured GPUs the same way every other call in this
    module is: ``CUDA_VISIBLE_DEVICES``, plus ``-Gpu`` built by
    ``utils.gpu_ids_for`` from that same config list. Without this the axis
    search was the one place in the pipeline that could run on a GPU nobody
    asked for: ts_aretomo3 gets the environment from WarpTools, but these
    calls are made by us directly.
    """
    aretomo = config["aretomo"]
    out_dir = ensure_dir(config.tiltaxis_search_dir, name)

    command = config.command(
        "aretomo",
        "-InPrefix", stack, "-OutDir", out_dir + os.sep,
        "-PixSize", "%.4f" % align_angpix,
        "-VolZ", "0", "-DarkTol", str(aretomo["dark_tolerance"]),
        "-Gpu", *[str(g) for g in gpu_ids_for("aretomo", config.gpus)])
    # -AlignZ is in PIXELS at this stack's own pixel size (align_angpix),
    # unlike ts_aretomo3's --alignz, which wants Angstrom - see
    # resolve_alignz_angstrom. Omitted entirely when aretomo.alignz is empty.
    if alignz_px is not None:
        command += ["-AlignZ", str(alignz_px)]
    if aretomo["wbp"]:
        command += ["-Wbp", "1"]
    if tilt_axis is not None:
        command += ["-TiltAxis", "%.4f" % tilt_axis, str(refine)]
    extra = str(aretomo["extra_args"]).strip()
    if extra:
        command += extra.split()

    logs = ensure_dir(config.logs_dir, "tiltaxis_search")
    log_file = os.path.join(logs, "%s_iter%d.log" % (name, iteration))
    run_command(command, cwd=out_dir, env=_gpu_env(config), log_file=log_file,
                check=False, dry_run=dry_run)
    return out_dir, log_file


def refine_tilt_axis(config, series_names, align_angpix, log,
                     alignz_angstrom=None, force=False, dry_run=False):
    """Determine one tilt axis for the whole dataset (see :mod:`tiltaxis`).

    Returns the converged angle, or ``None`` when refinement is disabled and
    no ``aretomo.initial_tilt_axis`` is set (or nothing could be measured) -
    in which case native ``ts_aretomo3`` searches each tomogram's axis
    independently, exactly as before this pipeline existed.

    ``aretomo.initial_tilt_axis``, if set, seeds iteration 1's starting point
    (refined within +/-3 deg, same window every later iteration already
    uses) instead of a free search; iteration 2 onward is unaffected, still
    taking the previous iteration's result as usual. With
    ``tilt_axis_iterations: 0`` it is instead used directly AS the converged
    angle, with no external search performed at all.

    *alignz_angstrom* is the already-resolved ``aretomo.alignz`` thickness
    (see :func:`resolve_alignz_angstrom`), converted here to PIXELS at the
    input stacks' own pixel size - which is what ``AreTomo3 -AlignZ`` wants,
    unlike ts_aretomo3's Angstrom-valued ``--alignz``. ``None`` leaves the
    argument off entirely.
    """
    iterations = int(config["aretomo"]["tilt_axis_iterations"])
    initial_raw = str(config["aretomo"]["initial_tilt_axis"] or "").strip()
    initial_angle = float(initial_raw) if initial_raw else None

    if iterations <= 0:
        if initial_angle is not None:
            info("aretomo.initial_tilt_axis is set and "
                 "aretomo.tilt_axis_iterations is 0 - using %+.4f deg "
                 "directly as the converged tilt axis; no external search "
                 "performed." % initial_angle)
            log.note("tilt axis: %+.4f deg, taken directly from "
                     "aretomo.initial_tilt_axis (tilt_axis_iterations: 0)"
                     % initial_angle)
            if not dry_run:
                tiltaxis.write_report(
                    config, [], initial_angle,
                    note="Taken directly from aretomo.initial_tilt_axis "
                         "(aretomo.tilt_axis_iterations: 0) - no external "
                         "AreTomo3 search was performed.")
            return initial_angle
        info("tilt axis refinement disabled (aretomo.tilt_axis_iterations: 0) "
             "- ts_aretomo3 will search each tomogram's axis independently.")
        return None

    if not force:
        previous = tiltaxis.read_converged_angle(config)
        if previous is not None:
            info("reusing the converged tilt axis %+.4f deg from %s"
                 % (previous, tiltaxis.report_path(config)))
            log.note("tilt axis: reused %+.4f deg from a previous run" % previous)
            return previous

    subset_size = int(config["aretomo"]["tilt_axis_subset_size"])
    seed = config["aretomo"]["tilt_axis_seed"]

    # Fail loudly, immediately, for EVERY tilt series before running any
    # AreTomo at all - never silently skip a series with missing ts_stack
    # input and let the search quietly degrade. A ts_stack call that
    # succeeded writes <name>.st (IMOD-style stack), not <name>.mrc.
    if not dry_run:
        missing = [name for name in series_names
                  if not (os.path.isfile(os.path.join(config.tiltstack_dir, name, name + ".st"))
                         and os.path.isfile(os.path.join(config.tiltstack_dir, name, name + ".rawtlt")))]
        if missing:
            example = os.path.join(config.tiltstack_dir, series_names[0],
                                   series_names[0] + ".st")
            raise SystemExit(
                "ERROR: ts_stack output missing for %d of %d tilt series - "
                "cannot run the tilt axis search.\n"
                "Expected e.g. %s\n"
                "Missing: %s\n"
                "Check that ts_stack actually produced output for these "
                "tilt series, and that its real file naming still matches "
                "what this pipeline expects." % (
                    len(missing), len(series_names), example,
                    ", ".join(sorted(missing)))
            )

    banner("Tilt axis refinement (%d iterations, %d tilt series each)"
           % (iterations, min(subset_size, len(series_names))))

    log.note(gpu_id_note("aretomo", config.gpus))
    alignz_px = (None if alignz_angstrom is None
                 else int(round(alignz_angstrom / align_angpix)))
    if alignz_px is not None:
        info("AreTomo3 -AlignZ %d px (at the %.4f A stack pixel size)"
             % (alignz_px, align_angpix))

    rows = []
    angle = initial_angle
    measured = False
    if angle is not None:
        info("starting the tilt axis search from the configured %+.4f deg "
             "(aretomo.initial_tilt_axis)" % angle)
    for iteration in range(1, iterations + 1):
        names = tiltaxis.choose_subset(sorted(series_names), subset_size, seed,
                                       iteration)
        info("iteration %d/%d on: %s" % (iteration, iterations, ", ".join(names)))

        measurements, sources = {}, {}
        for name in names:
            stack = os.path.join(config.tiltstack_dir, name, name + ".st")
            _, log_file = _axis_search_align(
                config, name, stack, align_angpix,
                tilt_axis=angle, refine=0 if angle is not None else -1,
                iteration=iteration, alignz_px=alignz_px, dry_run=dry_run)
            if dry_run:
                measurements[name] = None
                continue

            aln = find_aln(stack, os.path.join(config.tiltaxis_search_dir, name))
            value, source = tiltaxis.read_tilt_axis(aln, log_file)
            measurements[name] = value
            sources[name] = source

            if value is not None and angle is not None:
                drift = abs(tiltaxis.fold_angles([angle, value])[1]
                            - tiltaxis.fold_angles([angle, value])[0])
                if drift > tiltaxis.SUSPICIOUS_DEVIATION:
                    warn("%s: reported %+.3f deg after starting from %+.3f - "
                         "that is %.1f deg, outside the +/-%.0f deg AreTomo "
                         "searches. The value was read from %s and is probably "
                         "not the solved tilt axis."
                         % (name, value, angle, drift, tiltaxis.REFINE_WINDOW,
                            source))

        usable = [v for v in measurements.values() if v is not None]
        combined, method = tiltaxis.combine_tilt_axes(usable)
        spread = None
        if len(usable) > 1:
            folded = tiltaxis.fold_angles(usable)
            spread = max(folded) - min(folded)

        rows.append({
            "iteration": iteration, "initial": angle, "measurements": measurements,
            "sources": sources, "angle": combined, "method": method, "spread": spread,
        })

        if spread is not None and spread > 10.0:
            warn("iteration %d: the tilt axis measurements span %.1f deg. "
                 "For one microscope they should agree to within a couple of "
                 "degrees - check %s before trusting the result."
                 % (iteration, spread, tiltaxis.report_path(config)))

        if combined is None:
            warn("iteration %d produced no usable tilt axis." % iteration)
            continue
        info("iteration %d: %+.3f deg (%s)" % (iteration, combined, method))
        angle = combined
        measured = True

    if angle is None:
        if dry_run:
            # Nothing really ran under a dry run - not a real failure to
            # stop on, just an informational note about what a real run
            # would need to determine.
            warn("No tilt axis could be measured (dry run - nothing "
                 "actually ran).")
            return None

        tiltaxis.write_report(config, rows, angle,
                              note="No tilt axis could be measured.")
        search_logs = ensure_dir(config.logs_dir, "tiltaxis_search")
        raise SystemExit(
            "ERROR: the external tilt axis search "
            "(aretomo.tilt_axis_iterations: %d) produced no usable "
            "measurement from ANY tilt series, in ANY iteration.\n"
            "AreTomo itself most likely failed for every attempt - see %s "
            "for what was attempted, and the per-series logs under %s for "
            "what AreTomo actually reported.\n"
            "This is a hard failure rather than a silent fallback to native "
            "ts_aretomo3's own per-tomogram axis search: a configured, "
            "nonzero aretomo.tilt_axis_iterations means the external, "
            "dataset-wide search is expected to actually work. Set it to 0 "
            "only if you deliberately want the per-tomogram fallback "
            "instead." % (iterations, tiltaxis.report_path(config), search_logs)
        )

    if dry_run and not measured:
        # angle is only non-None here because it was seeded from
        # aretomo.initial_tilt_axis - nothing was actually measured under
        # this dry run, so it is not really a "converged" result yet.
        warn("No tilt axis was measured (dry run - nothing actually ran). "
             "A real run would start from the configured %+.4f deg "
             "(aretomo.initial_tilt_axis) and refine from there." % angle)
        return angle

    if not dry_run:
        tiltaxis.write_report(config, rows, angle)
    log.note("tilt axis converged to %+.4f deg after %d iteration(s)"
             % (angle, iterations))
    return angle


# ------------------------------------------------------- mdoc preparation


def _prepare_mdocs(config, series_list, log, force=False, dry_run=False):
    """Decide which tilt series - and which of their tilts - ts_import gets.

    1. Tilts are dropped when their mdoc block is incomplete (no TiltAngle or
       SubFramePath - see :func:`mdoc.parse_mdoc`) or their raw frame has no
       completed WarpTools frame-series data. A single missing/failed tilt
       should not take out an otherwise good tilt series - and in some
       WarpTools builds, a single tilt referencing a missing frame series
       makes ts_import fail to parse that WHOLE mdoc (logged, tolerated) and
       then crash the entire batch outright when it later tries to re-open
       that mdoc's never-written .tomostar.
    2. What remains of every series goes through the ``tiltseries_filter``
       check (:mod:`tiltfilter`): "exclude" leaves flagged series out,
       "stop" stops here, before anything is imported, "off" skips it.
    3. ``excluded_tiltseries.json`` is updated: ``tilt_check`` entries are
       decided afresh on every run, those of a WarpTools step are kept until
       *force* (see :mod:`tiltfilter` for what that does and does not
       retry).

    Returns ``(mdocs_dir, import_series, record)``. *mdocs_dir* is what
    ts_import reads: ``mdocs_filtered/`` whenever a tilt was dropped or a
    series excluded - filtered copies and links built here, the user's own
    mdocs.preprocessing never touched - and mdocs.preprocessing itself
    otherwise. *import_series* still includes series a WarpTools step
    excluded: they were imported before and stay so.

    Never deletes anything: when files from an earlier run would bring a
    no-longer-imported series back (an extra mdoc in mdocs_filtered/, which
    ts_import reads as a whole, or a .tomostar/XML WarpTools already made
    for it), this stops and lists them for the user to remove.

    Under ``dry_run`` frame availability is not checked (nothing has
    necessarily been motion corrected yet, so every tilt would look
    missing), and nothing is written or stopped on.
    """
    original_dir = config.mdoc_dir("preprocessing")
    filtered_dir = config.mdocs_filtered_dir
    filter_settings = tiltfilter.settings(config)
    mode = filter_settings["mode"]

    keep = {}
    usable = []
    any_dropped = False
    for series in series_list:
        # Already printed by parse_mdoc; kept in the report too.
        for problem in series.problems:
            log.quiet("WARNING %s: %s" % (os.path.basename(series.mdoc_path),
                                          problem))
        indices, missing = [], []
        for tilt in series.tilts:
            if not tilt.complete:
                missing.append("[ZValue = %s] (incomplete)" % tilt.zvalue)
            elif dry_run or frameseries_average_path(
                    config, os.path.splitext(tilt.frame)[0]) is not None:
                indices.append(tilt.index)
            else:
                missing.append(tilt.frame)
        any_dropped = any_dropped or bool(missing)

        if not indices and mode == "off":
            warn("%s: none of its %d tilt(s) are usable - left out of "
                 "align_tiltseries. Run 'tomo_denoising motion_correct' "
                 "first, or check whether it failed for this series."
                 % (series.name, len(series.tilts)))
            log.fail(series.name, "no usable tilts (missing frame-series "
                     "data or incomplete mdoc blocks: %s)" % ", ".join(missing))
            continue
        if missing and indices:
            warn("%s: %d of %d tilt(s) unusable and dropped (no WarpTools "
                 "frame-series data, or an incomplete mdoc block) - the rest "
                 "is still imported: %s"
                 % (series.name, len(missing), len(series.tilts),
                    ", ".join(missing)))
            log.note("%s: dropped %d of %d tilt(s): %s"
                     % (series.name, len(missing), len(series.tilts),
                        ", ".join(missing)))
        keep[series.name] = indices
        usable.append(series)

    flagged = {}
    if mode != "off":
        angles = {}
        for series in usable:
            wanted = set(keep[series.name])
            angles[series.name] = [t.angle for t in series.tilts
                                   if t.index in wanted]
        flagged = tiltfilter.evaluate(angles, filter_settings)

    if flagged and mode == "stop" and not dry_run:
        lines = ["tiltseries_filter flagged %d tilt series as likely flawed "
                 "(mode: stop):" % len(flagged), ""]
        lines += ["    %-30s %s" % (name, "; ".join(reasons))
                  for name, reasons in sorted(flagged.items())]
        lines += [
            "",
            "Nothing has been imported yet. Either:",
            "  - take their mdocs out of mdocs.preprocessing (%s) if you "
            "agree they are flawed," % original_dir,
            "  - set tiltseries_filter.mode to \"exclude\" to leave them out "
            "automatically, or",
            "  - relax the thresholds, or set tiltseries_filter.mode to "
            "\"off\", to keep them.",
            "Then re-run.",
        ]
        raise SystemExit("ERROR: " + "\n".join(lines))

    for name, reasons in sorted(flagged.items()):
        log.note("%s: %s by tiltseries_filter - %s"
                 % (name, "would be excluded" if dry_run else "excluded",
                    "; ".join(reasons)))
    if flagged and dry_run:
        log.note("dry run: the %d series above would be excluded"
                 % len(flagged))

    record = tiltfilter.read_record(config)
    names_now = {s.name for s in series_list}
    record = {name: entry for name, entry in record.items()
              if entry.get("stage") != tiltfilter.STAGE_TILT_CHECK
              and not force and name in names_now}
    for name, reasons in flagged.items():
        record[name] = {"stage": tiltfilter.STAGE_TILT_CHECK,
                        "reason": "; ".join(reasons)}
    if not dry_run:
        tiltfilter.write_record(config, record)

    import_series = [s for s in usable if s.name not in flagged]
    import_names = {s.name for s in import_series}
    use_filtered = any_dropped or bool(flagged)

    if not dry_run:
        stale = []
        if use_filtered:
            stale += [mdoc for mdoc in find_mdocs(filtered_dir)
                      if tomo_name_from_mdoc(mdoc) not in import_names]
        for directory, pattern in ((config.tomostar_dir, "*.tomostar"),
                                   (config.warp_tiltseries_dir, "*.xml")):
            stale += [path for path in sorted(glob.glob(os.path.join(directory, pattern)))
                      if os.path.splitext(os.path.basename(path))[0]
                      not in import_names]
        if stale:
            names = sorted({tomo_name_from_mdoc(p) if p.lower().endswith(".mdoc")
                            else os.path.splitext(os.path.basename(p))[0]
                            for p in stale})
            raise SystemExit(
                "ERROR: files from an earlier run would bring back %d tilt "
                "series that are no longer imported (excluded by "
                "tiltseries_filter, or their mdoc is no longer in "
                "mdocs.preprocessing): %s\n\n"
                "This pipeline never deletes files itself. Remove these, then "
                "re-run - nothing else needs to be redone:\n\n%s"
                % (len(names), ", ".join(names),
                   "\n".join("    rm %s" % shlex.quote(p) for p in stale)))

    if not use_filtered:
        return original_dir, import_series, record

    if not dry_run:
        ensure_dir(filtered_dir)
        for series in import_series:
            target = os.path.join(filtered_dir, os.path.basename(series.mdoc_path))
            if len(keep[series.name]) < len(series.tilts):
                write_filtered_mdoc(series.mdoc_path, target,
                                    set(keep[series.name]))
            else:
                # Unchanged series are linked through, not re-written. force:
                # an earlier run may have left a filtered COPY here.
                link_file(series.mdoc_path, target, force=True)
    return filtered_dir, import_series, record


# --------------------------------------------------- ts_aretomo3 completion


def _ts_aretomo_marker_path(config):
    return os.path.join(config.warp_tiltseries_dir, ".ts_aretomo_done.json")


def _read_ts_aretomo_marker(config):
    """Series names native ts_aretomo3 has actually completed, or ``None`` if
    there is no marker (or it cannot be read).

    NOT derived from ``warp_tiltseries/*.xml`` existence: ``create_settings``
    itself already creates one per tomogram immediately, with raw, mdoc-
    derived geometry and no real alignment, well before ts_aretomo3 ever
    runs - so XML existence is not a valid "already aligned" signal (this
    was a real bug: it made align_tiltseries skip the tilt axis search AND
    native ts_aretomo3 entirely, even on a first run, leaving every tomogram
    on whatever axis ts_import read from the mdoc). This marker is written
    by us, after a ts_aretomo3 call, and lists only the series that call
    actually aligned (see :func:`_judge_aretomo`).
    """
    path = _ts_aretomo_marker_path(config)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return set(data.get("series_names") or [])
    except (ValueError, OSError):
        return None


def _check_tomostars(config, series_names, log):
    """Run the ``tiltseries_filter`` criteria again on what ts_import
    actually wrote - REPORT ONLY.

    The exclusion itself is decided from the mdocs, before ts_import (see
    :func:`_prepare_mdocs`), because that is the only point where a series
    can be left out without removing files. ts_import can still drop tilts
    of its own (e.g. ``--min_intensity``), so a series can pass on its mdoc
    and fail here: under "stop" this stops, otherwise it warns.
    """
    settings = tiltfilter.settings(config)
    if settings["mode"] == "off":
        return
    angles = {}
    for name in series_names:
        path = os.path.join(config.tomostar_dir, name + ".tomostar")
        if not os.path.isfile(path):
            warn("%s: ts_import wrote no %s" % (name, path))
            continue
        try:
            angles[name] = tiltfilter.read_tomostar_angles(path)
        except ValueError as exc:
            warn("could not check %s: %s" % (name, exc))
    flagged = tiltfilter.evaluate(angles, settings)
    if not flagged:
        log.note("tiltseries_filter: all %d imported tilt series pass on "
                 "their .tomostar too" % len(angles))
        return
    lines = ["    %-30s %s" % (name, "; ".join(reasons))
             for name, reasons in sorted(flagged.items())]
    message = ("%d tilt series passed on their mdoc but fail tiltseries_filter "
               "on what ts_import actually imported (it dropped tilts of its "
               "own, e.g. via --min_intensity):\n%s"
               % (len(flagged), "\n".join(lines)))
    if settings["mode"] == "stop":
        raise SystemExit(
            "ERROR: %s\n\nTo leave them out, take their mdocs out of "
            "mdocs.preprocessing and remove their .tomostar files from %s; "
            "or set tiltseries_filter.mode to \"exclude\" to only warn about "
            "this. Then re-run." % (message, config.tomostar_dir))
    warn(message + "\nThey are NOT excluded automatically (that is only "
         "possible before ts_import) - take their mdocs out of "
         "mdocs.preprocessing if they should not be processed.")
    log.quiet(message)


def _xml_root_attr(config, name, attr):
    """The value of *attr* on ``warp_tiltseries/<name>.xml``'s root
    ``<TiltSeries>`` element, or ``None`` if absent."""
    path = os.path.join(config.warp_tiltseries_dir, name + ".xml")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as handle:
        head = handle.read(4096)
    root = re.search(r"<TiltSeries\b[^>]*>", head)
    value = root and re.search(r'\s%s="([^"]*)"' % re.escape(attr), root.group(0))
    return value.group(1) if value else None


def _xml_unselected(config, name):
    """Whether WarpTools marked ``warp_tiltseries/<name>.xml`` unselected
    (``UnselectManual="True"`` - what it does to a series a command failed
    on: "marked as unselected")."""
    return (_xml_root_attr(config, name, "UnselectManual") or "").lower() == "true"


def _judge_aretomo(config, series_names, logged_failed):
    """``(aligned, {name: reason})`` after a ts_aretomo3 run that got
    through (see :func:`_warp_outcome`): a series failed if WarpTools
    reported it ("Failed to process", *logged_failed*) or left it marked
    unselected - which also covers one skipped for having been unselected
    by an earlier run. Every other series was aligned."""
    logs_dir = os.path.join(config.warp_tiltseries_dir, "logs")
    aligned, failed = [], {}
    for name in series_names:
        if name in logged_failed or _xml_unselected(config, name):
            failed[name] = ("AreTomo3 produced no alignment (WarpTools marked "
                            "it unselected) - see %s" % logs_dir)
        else:
            aligned.append(name)
    return aligned, failed


def _write_ts_aretomo_marker(config, series_names, converged):
    path = _ts_aretomo_marker_path(config)
    data = {
        "series_names": sorted(series_names),
        "converged_axis": converged,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        # Identifies THIS alignment - see alignment_id().
        "alignment_id": uuid.uuid4().hex,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def alignment_id(config):
    """An id that changes whenever ts_aretomo3 writes a new alignment, or
    ``None`` if none was written yet. MissAlignment's pre-training backups
    are tagged with it, so a backup of an earlier alignment can never be
    mistaken for one of the current alignment (see missalign.py)."""
    try:
        with open(_ts_aretomo_marker_path(config), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    # Markers from before the id existed: their timestamp serves the same.
    return data.get("alignment_id") or data.get("written")


# -------------------------------------------------------------- entrypoint


def align_tiltseries(config, force=None, dry_run=False):
    """``tomo_denoising align_tiltseries``"""
    force = config.overwrite if force is None else force
    with StepLog(config, STEP_ALIGN, force=force) as log:
        return _align_tiltseries(config, log, force=force, dry_run=dry_run)


def _align_tiltseries(config, log, force=False, dry_run=False):
    banner("WarpTools tilt-series import, stacking and alignment")
    for key, purpose in (("warptools", "WarpTools"), ("aretomo", "AreTomo")):
        config.require(key, purpose)

    if not dry_run and not (os.path.isfile(config.warp_frameseries_settings_path)
                            and _has_any_files(config.warp_frameseries_dir)):
        log.fail("align_tiltseries", "no WarpTools frame-series output in %s "
                 "- run 'tomo_denoising motion_correct' first"
                 % config.warp_frameseries_dir)
        return 1

    original_dir = config.mdoc_dir("preprocessing")
    series_list = collect_tilt_series(original_dir, "preprocessing")
    log.note("%d tilt series found in %s" % (len(series_list), original_dir))

    directory, series_list, record = _prepare_mdocs(config, series_list, log,
                                                    force=force, dry_run=dry_run)
    if not series_list and not dry_run:
        log.fail("align_tiltseries", "no tilt series left to import - every "
                 "one is unusable or excluded (see %s)"
                 % tiltfilter.record_path(config))
        return 1
    # Series a WarpTools step gave up on in an earlier run stay imported, but
    # are left out until --force clears them from the record (WarpTools' own
    # "unselected" mark on their XML is a separate matter - see tiltfilter).
    # AreTomo3 failures are still judged again after ts_aretomo3, which runs
    # on every imported series anyway.
    warp_excluded = {name for name, entry in record.items()
                     if entry.get("stage") != tiltfilter.STAGE_TILT_CHECK}
    aretomo_failed = {name for name, entry in record.items()
                      if entry.get("stage") == tiltfilter.STAGE_ARETOMO}
    if warp_excluded:
        log.note("a WarpTools step gave up on these before - left out until "
                 "align_tiltseries --force: %s" % ", ".join(sorted(warp_excluded)))
    series_names = sorted(s.name for s in series_list
                          if s.name not in warp_excluded)
    if not series_names and not dry_run:
        log.fail("align_tiltseries", _ALL_EXCLUDED
                 % ("imported", tiltfilter.record_path(config)))
        return 1

    acquisition = config["acquisition"]
    fs = config["frameseries"]
    wt = config["warptools"]
    gain = config.resolve_gain()
    apix = dataset_apix(config, series_list)
    align_angpix = resolve_angpix(wt["align_angpix"], apix, "warptools.align_angpix")
    log.note("align_angpix: %.4f A (from %r, unbinned pixel size %.4f A)"
             % (align_angpix, wt["align_angpix"], apix))
    env = _gpu_env(config)
    logs = ensure_dir(config.logs_dir)
    log.note(gpu_id_note("warptools", config.gpus))

    # --- 1. ts_import --------------------------------------------------
    banner("align_tiltseries 1/5  ts_import")
    tomostar_dir = ensure_dir(config.tomostar_dir)
    ts_import_command = _warp(
        config, "ts_import",
        "--mdocs", directory,
        # The frame-series PROCESSING FOLDER (holding each movie's own XML),
        # not the .settings file - WarpTools looks each tilt's motion
        # correction up by scanning this folder.
        "--frameseries", config.warp_frameseries_dir,
        "--tilt_exposure", "%.4f" % float(fs["exposure"]),
        "--min_intensity", "%.4f" % float(wt["min_intensity"]),
        "--output", tomostar_dir,
    )
    if wt["dont_invert"]:
        ts_import_command.append("--dont_invert")

    # Tilt-ANGLE offset (the 0-tilt zero point / specimen pre-tilt), not the
    # tilt AXIS - see the config comments. Both default to off, so ts_import
    # keeps the mdoc's own tilt angles unless asked otherwise.
    if wt["auto_zero"]:
        ts_import_command.append("--auto_zero")
        log.note("ts_import --auto_zero: the brightest tilt is taken as the "
                 "0-tilt and every angle shifted accordingly")
    tilt_offset = str(wt["tilt_offset"]).strip()
    if tilt_offset:
        try:
            float(tilt_offset)
        except ValueError:
            raise SystemExit(
                "ERROR: warptools.tilt_offset must be a number of degrees (or "
                "empty to leave the tilt angles alone), got %r" % tilt_offset
            )
        ts_import_command += ["--tilt_offset", tilt_offset]
        log.note("ts_import --tilt_offset %s: subtracted from every tilt angle"
                 % tilt_offset)
    if wt["auto_zero"] and tilt_offset:
        warn("warptools.auto_zero and warptools.tilt_offset are BOTH set. "
             "They are two different answers to the same question - auto_zero "
             "estimates the 0-tilt from image intensity, tilt_offset applies a "
             "fixed shift you supply - and WarpTools will apply whatever it "
             "makes of the combination. Set only one unless you know you want "
             "both.")

    if force or not _has_any_files(tomostar_dir):
        code = run_command(ts_import_command, cwd=config.processing_dir,
                           env=env,
                           log_file=os.path.join(logs, "ts_import.log"),
                           check=False, dry_run=dry_run)
        if code != 0 and not dry_run:
            log.fail("ts_import", "exit status %d" % code)
            return 1
        log.task("ts_import", True, "-")
    else:
        log.note("%s already populated - skipping ts_import (the "
                 "axis-override re-import below still runs)" % tomostar_dir)

    if not dry_run:
        _check_tomostars(config, series_names, log)

    # --- 2. create_settings (tilt series) -------------------------------
    banner("align_tiltseries 2/5  create_settings (tilt series)")
    settings_path = config.warp_tiltseries_settings_path

    if force or not os.path.isfile(settings_path):
        x, y, z = resolve_tomo_dimensions(config, series_list, log,
                                          dry_run=dry_run)
        command = _warp(
            config, "create_settings",
            "--output", settings_path,
            "--folder_data", tomostar_dir,
            "--folder_processing", config.warp_tiltseries_dir,
            "--extension", "*.tomostar",
            "--angpix", "%.4f" % apix,
            "--exposure", "%.4f" % float(fs["exposure"]),
            "--tomo_dimensions", "%dx%dx%d" % (x, y, z),
        )
        if gain:
            command += ["--gain_path", gain]
            command += _gain_orientation_args(acquisition)
        run_command(command, cwd=config.processing_dir, env=env,
                   log_file=os.path.join(logs, "create_settings_tiltseries.log"),
                   dry_run=dry_run)
        if not dry_run and not os.path.isfile(settings_path):
            raise SystemExit(
                "ERROR: WarpTools create_settings produced no %s\nSee %s"
                % (settings_path, os.path.join(logs, "create_settings_tiltseries.log"))
            )
        log.task("create_settings (tiltseries)", True, "-")
        log.note("tomo_dimensions: %dx%dx%d (unbinned px)" % (x, y, z))
    else:
        log.note("%s already exists - skipping create_settings" % settings_path)

    # --- 3. ts_stack ------------------------------------------------------
    banner("align_tiltseries 3/5  ts_stack")
    if force or not _has_any_files(config.tiltstack_dir):
        command = _warp(config, "ts_stack",
                        "--settings", settings_path,
                        "--angpix", "%.4f" % align_angpix)
        log_file = os.path.join(logs, "ts_stack.log")
        code = run_command(command, cwd=config.processing_dir, env=env,
                           log_file=log_file, check=False, dry_run=dry_run)
        if not dry_run:
            outcome = _warp_outcome(code, log_file, "ts_stack", log)
            if outcome is None:
                return 1
            series_names = _exclude_failed(config, "ts_stack", outcome,
                                           series_names, log)
            if not series_names:
                log.fail("ts_stack", _ALL_EXCLUDED
                         % ("imported", tiltfilter.record_path(config)))
                return 1
        log.task("ts_stack", True, "-")
    else:
        log.note("%s already populated - skipping ts_stack" % config.tiltstack_dir)

    # Whether native ts_aretomo3 (step 5) needs to run at all - checked before
    # the tilt axis search (step 4) so a fully-resumed project does not pay
    # for the (real, external AreTomo) tilt axis search just to throw the
    # result away. Based on OUR OWN completion marker, not bare .xml
    # existence - see _read_ts_aretomo_marker's docstring for why that
    # would be wrong.
    marker_series = None if force else _read_ts_aretomo_marker(config)
    need_alignment = marker_series is None or not (set(series_names) <= marker_series)

    if need_alignment:
        # A new alignment invalidates any MissAlignment state built on the
        # old one. Checked BEFORE the tilt axis search, so a stop over
        # leftover training state costs nothing. Local import: missalign
        # itself imports this module.
        from .missalign import reset_for_realignment
        reset_for_realignment(config, log, dry_run=dry_run)

    # --- 4. tilt axis refinement (tiltaxis.py, unchanged algorithm) -------
    banner("align_tiltseries 4/5  tilt axis refinement")
    alignz_angstrom = resolve_alignz_angstrom(config, apix, log=log)

    if need_alignment:
        converged = refine_tilt_axis(config, series_names, align_angpix, log,
                                     alignz_angstrom=alignz_angstrom,
                                     force=force, dry_run=dry_run)
    else:
        converged = None
        log.note("ts_aretomo3 already completed for these tilt series (marker: "
                 "%s) - skipping the tilt axis search and ts_aretomo3"
                 % _ts_aretomo_marker_path(config))

    # --- 5. native ts_aretomo3, axis fixed, no search -----------------------
    banner("align_tiltseries 5/5  ts_aretomo3 (native)")

    if need_alignment:
        if converged is not None:
            reimport =list(ts_import_command) + ["--override_axis", "%.4f" % converged]
            run_command(reimport, cwd=config.processing_dir, env=env,
                       log_file=os.path.join(logs, "ts_import_override_axis.log"),
                       check=False, dry_run=dry_run)
            log.note("re-imported with OVERRIDE_AXIS %+.4f deg" % converged)

        command = _warp(
            config, "ts_aretomo3",
            "--settings", settings_path,
            "--exe", config.exe("aretomo"),
            "--angpix", "%.4f" % align_angpix,
            "--min_fov", str(int(wt["min_fov"])),
            "--perdevice", str(int(wt["perdevice"])),
        )
        # --alignz wants the thickness in ANGSTROM (unlike AreTomo3's own
        # -AlignZ, which is in stack pixels - see resolve_alignz_angstrom),
        # and it is an INTEGER field (same pattern as --c_voltage; a decimal
        # value fails its parser). Omitted entirely when aretomo.alignz is
        # empty, leaving ts_aretomo3 at its own default.
        if alignz_angstrom is not None:
            command += ["--alignz", str(int(round(alignz_angstrom)))]
        if converged is not None:
            command += ["--axis", "%.4f" % converged, "--axis_iter", "0"]
        log_file = os.path.join(logs, "ts_aretomo3.log")
        code = run_command(command, cwd=config.processing_dir, env=env,
                           log_file=log_file, check=False, dry_run=dry_run)
        if not dry_run:
            outcome = _warp_outcome(code, log_file, "ts_aretomo3", log)
            if outcome is None:
                return 1
            # WarpTools runs every imported .tomostar, including series that
            # failed in an earlier run - judge them all, so one that now
            # succeeds leaves the record again.
            judged = sorted(set(series_names) | aretomo_failed)
            aligned, failed = _judge_aretomo(config, judged, outcome.failed)
            if not aligned:
                log.fail("ts_aretomo3", "no tilt series aligned - see %s" % log_file)
                return 1
            record = tiltfilter.read_record(config)
            for name in aligned:
                if record.get(name, {}).get("stage") == tiltfilter.STAGE_ARETOMO:
                    del record[name]
            tiltfilter.write_record(config, record)
            tiltfilter.record_failures(config, tiltfilter.STAGE_ARETOMO, failed)
            series_names = sorted(aligned)
            _write_ts_aretomo_marker(config, series_names, converged)
        log.task("ts_aretomo3", True, "-")

    if not dry_run:
        aligned = set(aligned_series_names(config))
        log.note("%d of %d tilt series aligned" % (len(aligned & set(series_names)),
                                                    len(series_names)))
        for name in series_names:
            if name in aligned:
                log.ok(name)
            else:
                log.fail(name, "no warp_tiltseries/%s.xml after ts_aretomo3" % name)
        already = {name for name, _ in log.excluded}
        for name, entry in sorted(tiltfilter.read_record(config).items()):
            if name not in already:
                log.exclude(name, "[%s] %s" % (entry.get("stage", "?"),
                                               entry.get("reason", "")))

    return 0 if not log.failed else 1


# ============================================================ reconstruct_tomograms


def _ts_reconstruct_call(config, settings_path, angpix, half_flag, output_processing_dir,
                         logs_dir, log_name, dry_run=False):
    command = _warp(config, "ts_reconstruct",
                    "--settings", settings_path,
                    "--angpix", "%.4f" % angpix,
                    "--output_processing", output_processing_dir,
                    "--perdevice", str(int(config["warptools"]["perdevice"])),
                    [half_flag] if half_flag else [])
    return run_command(command, cwd=config.processing_dir, env=_gpu_env(config),
                       log_file=os.path.join(logs_dir, log_name),
                       check=False, dry_run=dry_run)


def _warp_reconstruction_output_dir(output_processing_dir):
    """Where ts_reconstruct's own output ACTUALLY lands, given the directory
    passed to it as ``--output_processing``.

    CONFIRMED against WarpTools' own source (``TiltSeries.cs``:
    ``ReconstructionDir => Combine(ProcessingDirectoryName, "reconstruction")``,
    and ``--output_processing`` sets ``ProcessingDirectoryName``) - it always
    writes into a ``reconstruction`` subdirectory of what it is given, never
    directly into it. This is exactly what produced the real, observed
    ``missalignment/reconstruction/reconstruction`` path that first raised
    the question.
    """
    return os.path.join(output_processing_dir, "reconstruction")


#: WarpTools' own reconstruction file name, "<name>_<angpix>Apx.mrc".
_WARP_RECON_NAME_RE = re.compile(r"^(.+)_\d+(?:\.\d+)?Apx\.mrc$", re.IGNORECASE)


def _angpix_tag(angpix):
    """Pixel-size part of a call's folder name, e.g. ``9.000Apx``."""
    return "%.3fApx" % float(angpix)


def _call_dir(reconstruction_dir, kind, angpix):
    """The ``--output_processing`` of ONE kind of ts_reconstruct call at one
    pixel size, e.g. ``<reconstruction_dir>/frames_9.000Apx``.

    One folder per call and pixel size is what lets every result stay
    exactly where WarpTools wrote it: no call can ever see another call's
    files (every call writes a full tomogram at the top level of its output,
    and the tilt-split call's is at the coarse pixel size), so nothing has
    to be moved out or cleaned up between calls, and a re-run of the same
    call simply overwrites its own files - including any a crashed attempt
    left half-written.
    """
    return os.path.join(reconstruction_dir, "%s_%s" % (kind, _angpix_tag(angpix)))


def _full_tomograms(warp_output_dir):
    if not os.path.isdir(warp_output_dir):
        return []
    return [name for name in os.listdir(warp_output_dir)
            if name.lower().endswith(".mrc")
            and os.path.isfile(os.path.join(warp_output_dir, name))]


def _check_call_output(warp_output_dir, halves, logs_dir, log_name):
    """Stop with a pointer to the log if a successful ts_reconstruct call did
    not leave the layout this pipeline links from.

    CONFIRMED against WarpTools' own source: the full tomogram is written to
    ``<warp_output_dir>/<name>_<angpix>Apx.mrc`` on EVERY call
    (``TiltSeries.ReconstructFull.cs``); ``--halfmap_frames``/
    ``--halfmap_tilts`` only ADD ``even/`` and ``odd/`` beside it
    (``TiltSeries.ReconstructionEvenDir``/``ReconstructionOddDir``).
    """
    problems = []
    if not _full_tomograms(warp_output_dir):
        problems.append("no full tomogram (*.mrc) directly in it")
    if halves:
        for half in ("even", "odd"):
            if not os.path.isdir(os.path.join(warp_output_dir, half)):
                problems.append("no %s/ subdirectory" % half)
    if problems:
        found = sorted(glob.glob(os.path.join(warp_output_dir, "*")))
        raise SystemExit(
            "ERROR: after ts_reconstruct, %s has %s (found: %s).\n"
            "ts_reconstruct's real output layout differs from what this "
            "pipeline assumed - check %s."
            % (warp_output_dir, " and ".join(problems), found or "nothing",
               os.path.join(logs_dir, log_name)))


def _first_populated(*directories):
    """The first of *directories* holding any file, else the first one."""
    for directory in directories:
        if os.path.isdir(directory) and any(
                os.path.isfile(os.path.join(directory, name))
                for name in os.listdir(directory)):
            return directory
    return directories[0]


def resolve_recon_angpix(config, log=None):
    """Resolve warptools.recon_angpix against the dataset's own unbinned
    pixel size (it may be a plain number or an expression like
    "2*pixel_size" - see mdoc.resolve_angpix)."""
    directory = config.mdoc_dir("preprocessing")
    series_list = collect_tilt_series(directory, "preprocessing")
    apix = dataset_apix(config, series_list)
    recon_angpix = resolve_angpix(config["warptools"]["recon_angpix"], apix,
                                  "warptools.recon_angpix")
    if log is not None:
        log.note("recon_angpix: %.4f A (from %r, unbinned pixel size %.4f A)"
                 % (recon_angpix, config["warptools"]["recon_angpix"], apix))
    return recon_angpix


def run_ts_reconstruct(config, settings_path, halfmaps_dir, tiltsplit_halfmaps_dir,
                       logs_dir, series_names, angpix, reconstruction_dir,
                       full_dir=None, do_full=False,
                       do_halfmaps=True, do_tiltsplit=True, log=None,
                       force=False, dry_run=False):
    """Reconstruct the full, non-split tomogram (when *do_full*), frame-split
    half maps (when *do_halfmaps*) and tilt-split half maps (when
    *do_halfmaps* and *do_tiltsplit*), symlinking results into *full_dir* /
    *halfmaps_dir* / *tiltsplit_halfmaps_dir* with the raw_halfMaps naming
    convention (``<tomo>_rec_tomo.mrc`` for the full tomogram,
    ``even_<tomo>_rec_tomo.mrc`` / ``odd_<tomo>_rec_tomo.mrc`` for the
    halves).

    *angpix* is the already-resolved reconstruction pixel size (see
    :func:`resolve_recon_angpix`) - resolved once by the caller rather than
    read from config here, since warptools.recon_angpix may be an
    expression relative to the dataset's own pixel size. The full tomogram
    and the frame-split half maps both use it; the tilt-split half maps use
    the coarser ``quality_check.tilt_split_target_pixel_size`` instead (see
    below).

    Every call gets its OWN ``--output_processing`` under
    *reconstruction_dir* (:func:`_call_dir`: ``full_<px>``, ``frames_<px>``,
    ``tilts_<px>``), and the links point straight at the files where
    WarpTools wrote them - nothing is moved, nothing deleted. Byproducts
    nobody links (the tilt-split call's coarse full tomogram, a duplicate
    full tomogram, WarpTools' own extras) simply stay, as do the
    reconstructions of an earlier pixel size, which are never linked again.

    Shared by :func:`reconstruct_tomograms` (``do_full=True``, *do_halfmaps*
    controlled by its own ``halfmaps``/``--produce-intermediates`` logic -
    see :mod:`pipeline`) and :mod:`missalign` (also ``do_full=True``,
    *do_halfmaps* always true), which reconstruct the SAME
    ``warp_tiltseries`` project (before and after MissAlignment refines its
    alignment in place). The two callers MUST pass DIFFERENT
    *reconstruction_dir* values (``config.reconstruction_dir`` vs
    ``config.missalignment_reconstruction_dir``) - they used to share one,
    which meant MissAlignment's later reconstruction silently overwrote the
    files reconstruct_tomograms's own raw_halfMaps symlinks already pointed
    to, corrupting them in place. Resume is decided from the DESTINATION
    (link) directories.

    The full tomogram is written on EVERY ts_reconstruct call (see
    :func:`_check_call_output`), so when *do_full* and *do_halfmaps* are
    BOTH wanted it is linked from the ``--halfmap_frames`` call (same
    ``angpix``) instead of paying for a second, separate reconstruction of
    the identical volume. A dedicated plain call (no half-map flag at all)
    is only made when *do_full* is wanted WITHOUT *do_halfmaps* - there is
    then no halfmap call to take it from, and a plain call skips the
    (otherwise unwanted) even/odd compute entirely.
    """
    reconstruction_dir = ensure_dir(reconstruction_dir)
    ensure_dir(logs_dir)
    if do_full:
        ensure_dir(full_dir)
    if do_halfmaps:
        ensure_dir(halfmaps_dir)
        if do_tiltsplit:
            ensure_dir(tiltsplit_halfmaps_dir)

    tiltsplit_angpix = config.tilt_split_angpix
    full_out = _warp_reconstruction_output_dir(
        _call_dir(reconstruction_dir, "full", angpix))
    frames_out = _warp_reconstruction_output_dir(
        _call_dir(reconstruction_dir, "frames", angpix))
    tilts_out = _warp_reconstruction_output_dir(
        _call_dir(reconstruction_dir, "tilts", tiltsplit_angpix))

    plain_full_call = do_full and not do_halfmaps

    have_full = do_full and _has_any_files(full_dir)
    if plain_full_call and (force or not have_full):
        code = _ts_reconstruct_call(config, settings_path, angpix, None,
                                    os.path.dirname(full_out), logs_dir,
                                    "ts_reconstruct_full.log", dry_run=dry_run)
        if not dry_run:
            outcome = _warp_outcome(code, os.path.join(logs_dir, "ts_reconstruct_full.log"),
                                    "ts_reconstruct (full tomogram)", log)
            if outcome is None:
                return False
            series_names = _exclude_failed(config, "ts_reconstruct", outcome,
                                           series_names, log)
            _check_call_output(full_out, False, logs_dir, "ts_reconstruct_full.log")
        if log:
            log.task("ts_reconstruct (full tomogram)", True, "-")
    elif plain_full_call and log:
        log.note("full tomograms already present in %s - skipping" % full_dir)

    have_frame = do_halfmaps and _has_any_files(halfmaps_dir)
    if do_halfmaps and (force or not have_frame):
        code = _ts_reconstruct_call(config, settings_path, angpix,
                                    "--halfmap_frames", os.path.dirname(frames_out),
                                    logs_dir, "ts_reconstruct_frame.log",
                                    dry_run=dry_run)
        if not dry_run:
            outcome = _warp_outcome(code, os.path.join(logs_dir, "ts_reconstruct_frame.log"),
                                    "ts_reconstruct (frame halves)", log)
            if outcome is None:
                return False
            series_names = _exclude_failed(config, "ts_reconstruct", outcome,
                                           series_names, log)
            _check_call_output(frames_out, True, logs_dir, "ts_reconstruct_frame.log")
            if do_full and log:
                log.note("full tomogram taken from the --halfmap_frames call "
                         "(same reconstruction, no separate pass)")
        if log:
            log.task("ts_reconstruct (frame halves)", True, "-")
    elif do_halfmaps and log:
        log.note("frame-split half maps already present in %s - skipping"
                 % halfmaps_dir)
    elif log:
        log.note("half-map reconstruction skipped (see reconstruct_tomograms's "
                 "'halfmaps' / run_pipe's --produce-intermediates)")

    have_tilt = do_halfmaps and do_tiltsplit and _has_any_files(tiltsplit_halfmaps_dir)
    if do_halfmaps and do_tiltsplit and (force or not have_tilt):
        # Deliberately coarser than the frame-split reconstruction: tilt-
        # split half-tomos differ from each other far more than frame-split
        # ones do, so RESOLVE's resolution estimate needs a coarser pixel
        # size to be meaningful (see quality_check.
        # tilt_split_target_pixel_size). This is requested directly as
        # --angpix - ts_reconstruct's own Fourier resampling takes an
        # arbitrary target, no achievable-binning-step calculation needed
        # here (unlike the old AreTomo-binning design). Its own full
        # tomogram is at that coarse size too, and is never linked.
        if log:
            log.note("tilt-split recon_angpix: %.4f A "
                     "(quality_check.tilt_split_target_pixel_size)"
                     % tiltsplit_angpix)
        code = _ts_reconstruct_call(config, settings_path, tiltsplit_angpix,
                                    "--halfmap_tilts", os.path.dirname(tilts_out),
                                    logs_dir, "ts_reconstruct_tilt.log",
                                    dry_run=dry_run)
        if not dry_run:
            outcome = _warp_outcome(code, os.path.join(logs_dir, "ts_reconstruct_tilt.log"),
                                    "ts_reconstruct (tilt halves)", log)
            if outcome is None:
                return False
            series_names = _exclude_failed(config, "ts_reconstruct", outcome,
                                           series_names, log)
            _check_call_output(tilts_out, True, logs_dir, "ts_reconstruct_tilt.log")
        if log:
            log.task("ts_reconstruct (tilt halves)", True, "-")
    elif do_halfmaps and do_tiltsplit and log:
        log.note("tilt-split half maps already present in %s - skipping"
                 % tiltsplit_halfmaps_dir)

    if dry_run:
        return True

    # Link sources. The full tomogram comes from whichever call made it in
    # this mode, falling back to the other one (the same volume, from an
    # earlier run in the other mode). Each source also falls back to the
    # flat directory a project reconstructed before per-call folders existed
    # holds its files in (the old code moved them there, one pixel size
    # only, as it emptied them on --force).
    legacy = lambda name: os.path.join(reconstruction_dir, name)
    sources = []
    if do_full:
        order = ((frames_out, full_out) if do_halfmaps else (full_out, frames_out))
        sources.append((_first_populated(*(order + (legacy("full"),))),
                        None, full_dir))
    if do_halfmaps:
        sources += [
            (_first_populated(os.path.join(frames_out, "even"), legacy("frame_even")),
             "even", halfmaps_dir),
            (_first_populated(os.path.join(frames_out, "odd"), legacy("frame_odd")),
             "odd", halfmaps_dir),
        ]
        if do_tiltsplit:
            sources += [
                (_first_populated(os.path.join(tilts_out, "even"), legacy("tilt_even")),
                 "even", tiltsplit_halfmaps_dir),
                (_first_populated(os.path.join(tilts_out, "odd"), legacy("tilt_odd")),
                 "odd", tiltsplit_halfmaps_dir),
            ]

    # Ownership is resolved with mdoc.match_tomo_name (whole underscore-
    # delimited run, longest name wins) rather than a "*name*" substring
    # glob - a real production bug (confirmed: "Position_7"'s glob matched
    # "Position_7_4"'s reconstructed file too, and glob's unordered listing
    # meant whichever of the two happened to sort first silently became BOTH
    # tomograms' symlink target) showed a plain substring test is not safe
    # whenever one tilt series' name is a literal prefix of another's - a
    # routine occurrence with numeric position naming (Position_7 vs
    # Position_7_4 vs Position_70-79, ...), not a rare edge case.
    #
    # Even that is not enough once a directory can hold files of tilt series
    # that are NOT in this run (excluded ones, kept since nothing is
    # deleted): "Position_21_2_10.000Apx.mrc" matches "Position_21" as a
    # whole run too. So the exact name is read off WarpTools' own
    # "<name>_<angpix>Apx.mrc" first, and the run-matching above is only the
    # fallback - against EVERY name this project knows, so a longer excluded
    # name still wins over its prefix - and linked only if it is this run's.
    run_names = set(series_names)
    known_names = (run_names | set(aligned_series_names(config, include_excluded=True))
                   | tiltfilter.excluded_names(config))
    linked = 0
    matched_names = set()
    for src_dir, half, dst_dir in sources:
        if not os.path.isdir(src_dir):
            continue
        for entry in sorted(os.listdir(src_dir)):
            source = os.path.join(src_dir, entry)
            # Only volumes: these are WarpTools' own output folders, which
            # also hold e.g. thumbnails named after the same tomogram.
            if not (os.path.isfile(source) and entry.lower().endswith(".mrc")):
                continue
            exact = _WARP_RECON_NAME_RE.match(entry)
            owner = (exact.group(1) if exact and exact.group(1) in known_names
                     else match_tomo_name(entry, known_names))
            if owner is not None and owner not in run_names:
                continue        # another (excluded) tilt series' file
            if owner is None:
                warn("%s: %r does not match any of this run's tilt series "
                     "by name - not linked. If this is a real result, check "
                     "why its tilt series is missing from series_names."
                     % (src_dir, entry))
                continue
            target_name = ("%s_%s_rec_tomo.mrc" % (half, owner) if half
                           else "%s_rec_tomo.mrc" % owner)
            target = os.path.join(dst_dir, target_name)
            # force matters here even though the target NAME never changes:
            # what it points at does, as soon as a pixel size changed.
            link_file(source, target, force=force)
            linked += 1
            matched_names.add(owner)

    if log:
        for name in series_names:
            if name in matched_names:
                log.ok(name)
            else:
                log.fail(name, "no reconstructed output matched this series "
                         "in any of: %s"
                         % ", ".join(sorted({s for s, _, _ in sources})))
        log.note("linked %d file(s)" % linked)
    return True


#: ts_defocus_hand --check's result, e.g. "Average correlation: -0.350".
_DEFOCUS_CORRELATION_RE = re.compile(
    r"Average correlation:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


def _defocus_correlation(log_path):
    """The average correlation ``ts_defocus_hand --check`` reported, or
    ``None`` if its log has none (failed before getting there, or NaN)."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            found = _DEFOCUS_CORRELATION_RE.findall(handle.read())
    except OSError:
        return None
    return float(found[-1]) if found else None


def _angles_inverted(config, name):
    """The handedness WarpTools stored for a series: ``AreAnglesInverted``,
    which ``--set_flip`` sets to True (confirmed on real data). Absent
    counts as False, WarpTools' default."""
    return (_xml_root_attr(config, name, "AreAnglesInverted") or "").lower() == "true"


def _defocus_hand(config, settings_path, series_names, env, logs, log,
                  dry_run=False):
    """Measure the defocus handedness and make every series' setting match
    it. Returns False if that could not be done.

    Judged by results, never by exit codes or wording:

      - ``--check`` measures and changes nothing. Its result is the
        "Average correlation" number - no number, no answer, and the step
        stops (a wrong handedness would silently spoil every later step).
        The decision is the number's SIGN alone, WarpTools' own rule
        (negative -> flip). Never a text search for "negative": that word
        is in WarpTools' own crash message ("positive and negative
        infinity"), which used to trigger a flip by itself.
      - The measurement ignores the current setting (confirmed: still
        -0.350 after --set_flip), so it is compared with what each series'
        XML says (``AreAnglesInverted``) - and ``--set_flip``/
        ``--set_noflip`` run only if some series differ, so a re-run never
        flips anything again. (Both SET a state; neither toggles.)
      - Whether that worked is read back from the XMLs, not from the
        command's exit code.
    """
    check_log = os.path.join(logs, "ts_defocus_hand_check.log")
    code = run_command(_warp(config, "ts_defocus_hand",
                             "--settings", settings_path, "--check"),
                       cwd=config.processing_dir, env=env,
                       log_file=check_log, check=False, dry_run=dry_run)
    if dry_run:
        info("ts_defocus_hand: the flip decision depends on --check's real "
             "output - skipped for this dry run")
        return True

    correlation = _defocus_correlation(check_log)
    if correlation is None:
        log.fail("ts_defocus_hand (check)",
                 "no 'Average correlation' result (exit status %d) - the "
                 "defocus handedness is unknown, so nothing is reconstructed. "
                 "See %s" % (code, check_log))
        return False
    if code != 0:
        log.note("ts_defocus_hand --check exited with status %d after "
                 "reporting its result - the result is used" % code)
    log.task("ts_defocus_hand (check)", True, "-")

    flip = correlation < 0
    wanted = "flip" if flip else "no flip"
    differ = [name for name in series_names
              if _angles_inverted(config, name) != flip]
    log.note("ts_defocus_hand: average correlation %.3f -> handedness should "
             "be '%s'; %d of %d tilt series are set otherwise"
             % (correlation, wanted, len(differ), len(series_names)))
    if not differ:
        return True
    if not config["warptools"]["auto_fix_defocus_hand"]:
        warn("ts_defocus_hand: %d tilt series are not set to '%s', but "
             "warptools.auto_fix_defocus_hand is false - left unchanged: %s"
             % (len(differ), wanted, ", ".join(differ)))
        return True

    option = "--set_flip" if flip else "--set_noflip"
    set_log = os.path.join(logs, "ts_defocus_hand_%s.log" % option.lstrip("-"))
    code = run_command(_warp(config, "ts_defocus_hand",
                             "--settings", settings_path, option),
                       cwd=config.processing_dir, env=env,
                       log_file=set_log, check=False)
    still = [name for name in series_names
             if _angles_inverted(config, name) != flip]
    if still:
        log.fail("ts_defocus_hand (%s)" % option.lstrip("-"),
                 "%d tilt series still not set to '%s' (exit status %d): %s - "
                 "see %s" % (len(still), wanted, code, ", ".join(still), set_log))
        return False
    log.task("ts_defocus_hand (%s)" % option.lstrip("-"), True, "-")
    log.note("ts_defocus_hand: %s applied to %d tilt series"
             % (option, len(differ)))
    return True


def reconstruct_tomograms(config, force=None, dry_run=False, halfmaps=True):
    """``tomo_denoising reconstruct_tomograms``

    The full, non-split tomogram is ALWAYS reconstructed, regardless of
    *halfmaps* - it is a standalone deliverable nothing downstream depends
    on, not an intermediate. ``ts_defocus_hand``/``ts_ctf`` also always run
    - they are prerequisites for ANY reconstruction, MissAlignment-refined
    or not.

    *halfmaps* controls only whether the frame-split/tilt-split half-map
    calls run. ``run_pipe`` passes ``halfmaps=False`` by default when
    MissAlignment is about to run right after: there is no reason to
    reconstruct the pre-MissAlignment half maps in that case, since
    MissAlignment reconstructs its own from the refined alignment (see
    :mod:`missalign`) - unless ``run_pipe --produce-intermediates`` asks for
    them anyway, alongside MissAlignment's later corrected ones.
    """
    force = config.overwrite if force is None else force
    with StepLog(config, STEP_RECONSTRUCT, force=force) as log:
        return _reconstruct_tomograms(config, log, force=force, dry_run=dry_run,
                                      halfmaps=halfmaps)


def _reconstruct_tomograms(config, log, force=False, dry_run=False, halfmaps=True):
    banner("WarpTools reconstruction (defocus hand, CTF, ts_reconstruct)")
    config.require("warptools", "WarpTools")

    settings_path = config.warp_tiltseries_settings_path
    if not dry_run and not os.path.isfile(settings_path):
        log.fail("reconstruct_tomograms", "no %s - run 'tomo_denoising "
                 "align_tiltseries' first" % settings_path)
        return 1

    series_names = aligned_series_names(config)
    if not series_names and not dry_run:
        log.fail("reconstruct_tomograms", "no aligned tilt series (*.xml) in "
                 "%s - run 'tomo_denoising align_tiltseries' first"
                 % config.warp_tiltseries_dir)
        return 1
    log.note("%d aligned tilt series" % len(series_names))

    wt = config["warptools"]
    env = _gpu_env(config)
    logs = ensure_dir(config.logs_dir)
    log.note(gpu_id_note("warptools", config.gpus))

    banner("reconstruct_tomograms 1/3  ts_defocus_hand")
    if not _defocus_hand(config, settings_path, series_names, env, logs, log,
                         dry_run=dry_run):
        return 1

    banner("reconstruct_tomograms 2/3  ts_ctf")
    command = _warp(config, "ts_ctf",
                    "--settings", settings_path,
                    "--range_high", str(float(wt["ts_ctf_range_high"])),
                    "--defocus_max", str(float(wt["ts_ctf_defocus_max"])),
                    "--perdevice", str(int(wt["perdevice"])))
    log_file = os.path.join(logs, "ts_ctf.log")
    code = run_command(command, cwd=config.processing_dir, env=env,
                       log_file=log_file, check=False, dry_run=dry_run)
    if not dry_run:
        outcome = _warp_outcome(code, log_file, "ts_ctf", log)
        if outcome is None:
            return 1
        series_names = _exclude_failed(config, "ts_ctf", outcome, series_names, log)
        if not series_names:
            log.fail("ts_ctf", _ALL_EXCLUDED
                     % ("aligned", tiltfilter.record_path(config)))
            return 1
    log.task("ts_ctf", True, "-")

    banner("reconstruct_tomograms 3/3  ts_reconstruct")
    if not halfmaps:
        log.note("half-map reconstruction skipped: MissAlignment will "
                 "reconstruct the half maps itself from its refined "
                 "alignment (see run_missalignment) - the full tomogram is "
                 "still reconstructed below")

    recon_angpix = resolve_recon_angpix(config, log=log)
    ok = run_ts_reconstruct(
        config, settings_path, config.raw_halfmaps_dir,
        config.tiltsplit_halfmaps_dir, logs, series_names, recon_angpix,
        config.reconstruction_dir,
        full_dir=config.tomograms_dir, do_full=True,
        do_halfmaps=halfmaps, do_tiltsplit=config.tilt_split,
        log=log, force=force, dry_run=dry_run,
    )
    if not ok:
        return 1

    log.note("")
    log.note("full tomograms:        %s" % config.tomograms_dir)
    if halfmaps:
        log.note("frame-split half maps: %s" % config.raw_halfmaps_dir)
        if config.tilt_split:
            log.note("tilt-split half maps:  %s" % config.tiltsplit_halfmaps_dir)
    return 0 if dry_run or not log.failed else 1
