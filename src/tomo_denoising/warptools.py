"""Steps 1-3: raw frames -> WarpTools project -> aligned, reconstructed tomograms.

Everything up to reconstruction runs through WarpTools natively:

  1. motion_correct        create_settings (frame series) + fs_motion_and_ctf
  2. align_tiltseries      ts_import -> create_settings (tilt series) ->
                           ts_stack -> OUR OWN tilt axis search (tiltaxis.py,
                           unchanged) -> ts_import --override_axis ->
                           native ts_aretomo3, axis fixed, no search
  3. reconstruct_tomograms ts_defocus_hand -> ts_ctf -> ts_reconstruct
                           (frame-split and, optionally, tilt-split)

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
    reconstruction/            ts_reconstruct scratch (--output_processing;
                                relocated out as each call finishes)
    tomograms/                 symlinks - full, non-split tomogram per series
    raw_halfMaps/               symlinks - exactly two entries per tomogram
    raw_halfMaps_tiltSplit/     symlinks - the same, tilt-split (for RESOLVE)

``run_ts_reconstruct`` is also used by :mod:`missalign`, which reconstructs
the SAME ``warp_tiltseries`` project again after refining its alignment in
place, into a different destination.
"""

import glob
import json
import os
import shutil
import time

from . import tiltaxis
from .mdoc import (
    collect_tilt_series, match_tomo_name, pixel_size_is_read, resolve_angpix,
    resolve_pixel_size, write_filtered_mdoc,
)
from .steplog import StepLog
from .utils import (
    banner, ensure_dir, info, link_file, require_executable, run as run_command, warn,
)

STEP_MOTION = "motion_correct"
STEP_ALIGN = "align_tiltseries"
STEP_RECONSTRUCT = "reconstruct_tomograms"


def _has_any_files(directory):
    return os.path.isdir(directory) and any(os.scandir(directory))


def _gpu_env(config):
    return {"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in config.gpus)}


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
    require_executable(config.exe("warptools"), "WarpTools")

    settings_path = config.warp_frameseries_settings_path
    frameseries_dir = ensure_dir(config.warp_frameseries_dir)
    logs = ensure_dir(config.logs_dir)

    if not force and os.path.isfile(settings_path) and _has_any_files(frameseries_dir):
        log.note("skipped: %s already has WarpTools frame-series output. "
                 "Use --force to redo it." % frameseries_dir)
        return 0

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

    command = [
        config.exe("warptools"), "create_settings",
        "--output", settings_path,
        "--folder_data", config.input_frames,
        "--folder_processing", frameseries_dir,
        "--extension", "*.%s" % extension,
        "--angpix", "%.4f" % apix,
        "--exposure", "%.4f" % float(fs["exposure"]),
        "--bin", str(int(fs["bin"])),
    ]
    if gain:
        command += ["--gain_path", gain]
        command += _gain_orientation_args(acquisition)
    if extension == "eer" and str(fs["eer_ngroups"]).strip():
        # EER grouping is a property of how the raw data is read, so it's
        # set once here on create_settings - not on fs_motion_and_ctf, which
        # rejects it.
        command += ["--eer_ngroups", str(fs["eer_ngroups"])]
    run_command(command, cwd=config.processing_dir,
               log_file=os.path.join(logs, "create_settings_frameseries.log"),
               dry_run=dry_run)
    if not dry_run and not os.path.isfile(settings_path):
        raise SystemExit(
            "ERROR: WarpTools create_settings produced no %s\nSee %s"
            % (settings_path, os.path.join(logs, "create_settings_frameseries.log"))
        )
    log.task("create_settings (frameseries)", True, "-")

    command = [
        config.exe("warptools"), "fs_motion_and_ctf",
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
    ]
    if fs["c_use_sum"]:
        command.append("--c_use_sum")
    extra = str(fs["extra_args"]).strip()
    if extra:
        command += extra.split()

    code = run_command(command, cwd=config.processing_dir, env=_gpu_env(config),
                       log_file=os.path.join(logs, "fs_motion_and_ctf.log"),
                       check=False, dry_run=dry_run)
    if code != 0 and not dry_run:
        log.fail("fs_motion_and_ctf", "exit status %d - see %s"
                 % (code, os.path.join(logs, "fs_motion_and_ctf.log")))
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


def _auto_tomo_dimensions_xy(config, series_list, log, dry_run=False):
    """(X, Y) unbinned pixels, read off a frame-series average's own shape
    and scaled back up by ``2**frameseries.bin`` (WarpTools' own log2 Fourier
    binning convention)."""
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
        x, y = width * (2 ** fs_bin), height * (2 ** fs_bin)
        log.note("tomo_dimensions.xy auto-detected: %dx%d (from %s, "
                 "frameseries.bin=%d)" % (x, y, os.path.basename(path), fs_bin))
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


def aligned_series_names(config):
    """Tilt series names that already have a native ts_aretomo3 alignment."""
    pattern = os.path.join(config.warp_tiltseries_dir, "*.xml")
    return sorted(os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(pattern))


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

    if not angles:
        return None
    return min(angles), max(angles)


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
    """
    aretomo = config["aretomo"]
    out_dir = ensure_dir(config.tiltaxis_search_dir, name)

    command = [config.exe("aretomo"),
               "-InPrefix", stack, "-OutDir", out_dir + os.sep,
               "-PixSize", "%.4f" % align_angpix,
               "-VolZ", "0", "-DarkTol", str(aretomo["dark_tolerance"])]
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
    run_command(command, cwd=out_dir, log_file=log_file, check=False, dry_run=dry_run)
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


# ------------------------------------------------------- mdoc sanitization


def _sanitize_mdocs(config, series_list, log, dry_run=False):
    """Drop tilts (or, if none remain, whole tilt series) whose raw frame has
    no completed WarpTools frame-series data, before ts_import ever sees
    them.

    A single missing/failed tilt should not take out an otherwise good tilt
    series - and in some WarpTools builds, a single tilt referencing a
    missing frame series makes ts_import fail to parse that WHOLE mdoc
    (logged, tolerated) and then crash the entire batch outright when it
    later tries to re-open that mdoc's never-written .tomostar. Filtering
    the offending tilt out here means ts_import never encounters the
    reference at all, for every tilt series in one pass, not just the one
    that happened to trip it.

    Returns ``(mdocs_dir, series_list)``: *mdocs_dir* is where ts_import
    should read from - a directory of filtered copies if anything needed
    dropping, or ``config.mdoc_dir("preprocessing")`` unchanged otherwise;
    *series_list* is narrowed to the tilt series that still have at least
    one usable tilt.

    A no-op under ``dry_run``: nothing has necessarily been motion corrected
    yet at that point, so every tilt would otherwise look "missing" and this
    would (wrongly) exclude everything from the printed preview.
    """
    original_dir = config.mdoc_dir("preprocessing")
    if dry_run:
        return original_dir, series_list

    filtered_dir = config.mdocs_filtered_dir
    kept_series = []
    any_filtered = False

    for series in series_list:
        missing = []
        keep_indices = set()
        for index, frame in enumerate(series.frames):
            stem = os.path.splitext(frame)[0]
            if frameseries_average_path(config, stem) is not None:
                keep_indices.add(index)
            else:
                missing.append(frame)

        if not missing:
            kept_series.append(series)
            continue

        any_filtered = True
        if not keep_indices:
            warn("%s: none of its %d tilt(s) have WarpTools frame-series "
                 "data - excluded entirely from align_tiltseries. Run "
                 "'tomo_denoising motion_correct' first, or check whether "
                 "it failed for this series." % (series.name, len(series.frames)))
            log.fail(series.name, "no usable tilts (missing frame-series "
                     "data for: %s)" % ", ".join(missing))
            continue

        warn("%s: %d of %d tilt(s) have no WarpTools frame-series data and "
             "will be dropped - the rest of the series is still imported: %s"
             % (series.name, len(missing), len(series.frames),
                ", ".join(missing)))
        log.note("%s: dropped %d of %d tilt(s), missing frame-series data: "
                 "%s" % (series.name, len(missing), len(series.frames),
                        ", ".join(missing)))

        ensure_dir(filtered_dir)
        target = os.path.join(filtered_dir, os.path.basename(series.mdoc_path))
        write_filtered_mdoc(series.mdoc_path, target, keep_indices)
        kept_series.append(series)

    if not any_filtered:
        return original_dir, kept_series

    # Every OTHER mdoc (nothing dropped from it) still needs to be present
    # in filtered_dir too, since ts_import reads the whole directory as one
    # --mdocs argument - link it through unchanged rather than re-parsing it.
    ensure_dir(filtered_dir)
    for series in kept_series:
        target = os.path.join(filtered_dir, os.path.basename(series.mdoc_path))
        if not os.path.exists(target):
            link_file(series.mdoc_path, target)

    return filtered_dir, kept_series


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
    by us, only after a real, successful ts_aretomo3 call.
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


def _write_ts_aretomo_marker(config, series_names, converged):
    path = _ts_aretomo_marker_path(config)
    data = {
        "series_names": sorted(series_names),
        "converged_axis": converged,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


# -------------------------------------------------------------- entrypoint


def align_tiltseries(config, force=None, dry_run=False):
    """``tomo_denoising align_tiltseries``"""
    force = config.overwrite if force is None else force
    with StepLog(config, STEP_ALIGN, force=force) as log:
        return _align_tiltseries(config, log, force=force, dry_run=dry_run)


def _align_tiltseries(config, log, force=False, dry_run=False):
    banner("WarpTools tilt-series import, stacking and alignment")
    for key, purpose in (("warptools", "WarpTools"), ("aretomo", "AreTomo")):
        require_executable(config.exe(key), purpose)

    if not dry_run and not (os.path.isfile(config.warp_frameseries_settings_path)
                            and _has_any_files(config.warp_frameseries_dir)):
        log.fail("align_tiltseries", "no WarpTools frame-series output in %s "
                 "- run 'tomo_denoising motion_correct' first"
                 % config.warp_frameseries_dir)
        return 1

    original_dir = config.mdoc_dir("preprocessing")
    series_list = collect_tilt_series(original_dir, "preprocessing")
    log.note("%d tilt series found in %s" % (len(series_list), original_dir))

    directory, series_list = _sanitize_mdocs(config, series_list, log,
                                             dry_run=dry_run)
    if not series_list and not dry_run:
        log.fail("align_tiltseries", "no tilt series has any usable tilt "
                 "left after removing ones missing frame-series data")
        return 1
    series_names = sorted(s.name for s in series_list)

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

    # --- 1. ts_import --------------------------------------------------
    banner("align_tiltseries 1/5  ts_import")
    tomostar_dir = ensure_dir(config.tomostar_dir)
    ts_import_command = [
        config.exe("warptools"), "ts_import",
        "--mdocs", directory,
        # The frame-series PROCESSING FOLDER (holding each movie's own XML),
        # not the .settings file - WarpTools looks each tilt's motion
        # correction up by scanning this folder.
        "--frameseries", config.warp_frameseries_dir,
        "--tilt_exposure", "%.4f" % float(fs["exposure"]),
        "--min_intensity", "%.4f" % float(wt["min_intensity"]),
        "--output", tomostar_dir,
    ]
    if wt["dont_invert"]:
        ts_import_command.append("--dont_invert")

    if force or not _has_any_files(tomostar_dir):
        code = run_command(ts_import_command, cwd=config.processing_dir,
                           log_file=os.path.join(logs, "ts_import.log"),
                           check=False, dry_run=dry_run)
        if code != 0 and not dry_run:
            log.fail("ts_import", "exit status %d" % code)
            return 1
        log.task("ts_import", True, "-")
    else:
        log.note("%s already populated - skipping ts_import (the "
                 "axis-override re-import below still runs)" % tomostar_dir)

    # --- 2. create_settings (tilt series) -------------------------------
    banner("align_tiltseries 2/5  create_settings (tilt series)")
    settings_path = config.warp_tiltseries_settings_path

    if force or not os.path.isfile(settings_path):
        x, y, z = resolve_tomo_dimensions(config, series_list, log,
                                          dry_run=dry_run)
        command = [
            config.exe("warptools"), "create_settings",
            "--output", settings_path,
            "--folder_data", tomostar_dir,
            "--folder_processing", config.warp_tiltseries_dir,
            "--extension", "*.tomostar",
            "--angpix", "%.4f" % apix,
            "--exposure", "%.4f" % float(fs["exposure"]),
            "--tomo_dimensions", "%dx%dx%d" % (x, y, z),
        ]
        if gain:
            command += ["--gain_path", gain]
            command += _gain_orientation_args(acquisition)
        run_command(command, cwd=config.processing_dir,
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
        command = [config.exe("warptools"), "ts_stack",
                   "--settings", settings_path,
                   "--angpix", "%.4f" % align_angpix]
        code = run_command(command, cwd=config.processing_dir, env=env,
                           log_file=os.path.join(logs, "ts_stack.log"),
                           check=False, dry_run=dry_run)
        if code != 0 and not dry_run:
            log.fail("ts_stack", "exit status %d" % code)
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
            reimport = list(ts_import_command) + ["--override_axis", "%.4f" % converged]
            run_command(reimport, cwd=config.processing_dir,
                       log_file=os.path.join(logs, "ts_import_override_axis.log"),
                       check=False, dry_run=dry_run)
            log.note("re-imported with OVERRIDE_AXIS %+.4f deg" % converged)

        command = [
            config.exe("warptools"), "ts_aretomo3",
            "--settings", settings_path,
            "--exe", config.exe("aretomo"),
            "--angpix", "%.4f" % align_angpix,
            "--min_fov", str(int(wt["min_fov"])),
            "--perdevice", str(int(wt["perdevice"])),
        ]
        # --alignz wants the thickness in ANGSTROM (unlike AreTomo3's own
        # -AlignZ, which is in stack pixels - see resolve_alignz_angstrom),
        # and it is an INTEGER field (same pattern as --c_voltage; a decimal
        # value fails its parser). Omitted entirely when aretomo.alignz is
        # empty, leaving ts_aretomo3 at its own default.
        if alignz_angstrom is not None:
            command += ["--alignz", str(int(round(alignz_angstrom)))]
        if converged is not None:
            command += ["--axis", "%.4f" % converged, "--axis_iter", "0"]
        code = run_command(command, cwd=config.processing_dir, env=env,
                           log_file=os.path.join(logs, "ts_aretomo3.log"),
                           check=False, dry_run=dry_run)
        if code != 0 and not dry_run:
            log.fail("ts_aretomo3", "exit status %d" % code)
            return 1
        log.task("ts_aretomo3", True, "-")
        if not dry_run:
            _write_ts_aretomo_marker(config, series_names, converged)

    if not dry_run:
        aligned = set(aligned_series_names(config))
        log.note("%d of %d tilt series aligned" % (len(aligned & set(series_names)),
                                                    len(series_names)))
        for name in series_names:
            if name in aligned:
                log.ok(name)
            else:
                log.fail(name, "no warp_tiltseries/%s.xml after ts_aretomo3" % name)

    return 0 if not log.failed else 1


# ============================================================ reconstruct_tomograms


def _ts_reconstruct_call(config, settings_path, angpix, half_flag, output_processing_dir,
                         logs_dir, log_name, dry_run=False):
    command = [config.exe("warptools"), "ts_reconstruct",
               "--settings", settings_path,
               "--angpix", "%.4f" % angpix,
               "--output_processing", output_processing_dir,
               "--perdevice", str(int(config["warptools"]["perdevice"]))]
    if half_flag:
        command.append(half_flag)
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


def _relocate_reconstruction_halves(warp_output_dir, dest_even, dest_odd,
                                    half_flag, logs_dir, log_name):
    """Move ts_reconstruct's ``even``/``odd`` output (confirmed location:
    directly under *warp_output_dir* - see ``TiltSeries.ReconstructionEvenDir``
    /``ReconstructionOddDir`` in WarpTools' own source, only populated
    alongside ``--halfmap_frames``/``--halfmap_tilts``) into *dest_even*/
    *dest_odd*, before the next ts_reconstruct call into the same
    ``--output_processing`` directory can overwrite them.
    """
    even_src = os.path.join(warp_output_dir, "even")
    odd_src = os.path.join(warp_output_dir, "odd")
    if not os.path.isdir(even_src) or not os.path.isdir(odd_src):
        found = sorted(glob.glob(os.path.join(warp_output_dir, "*")))
        raise SystemExit(
            "ERROR: expected %s to hold even/ and odd/ after ts_reconstruct "
            "%s, found: %s\n"
            "ts_reconstruct's real output layout differs from what this "
            "pipeline assumed - check %s and adjust warptools.py's "
            "_relocate_reconstruction_halves."
            % (warp_output_dir, half_flag, found,
               os.path.join(logs_dir, log_name))
        )
    ensure_dir(dest_even)
    ensure_dir(dest_odd)
    for src_dir, dst_dir in ((even_src, dest_even), (odd_src, dest_odd)):
        for name in os.listdir(src_dir):
            os.replace(os.path.join(src_dir, name), os.path.join(dst_dir, name))
        os.rmdir(src_dir)


def _discard_reconstruction_byproducts(warp_output_dir):
    """Remove whatever ts_reconstruct left directly in *warp_output_dir*
    (a leftover full-tomogram .mrc no one asked to keep, its .png thumbnail)
    and its ``ctf``/``deconv`` subdirectories - none of these are pipeline
    deliverables, and leaving them behind risks colliding with (or being
    mistaken for output of) the NEXT ts_reconstruct call into the same
    ``--output_processing`` directory.
    """
    if not os.path.isdir(warp_output_dir):
        return
    for name in os.listdir(warp_output_dir):
        path = os.path.join(warp_output_dir, name)
        if os.path.isfile(path):
            os.remove(path)
    for sub in ("ctf", "deconv"):
        sub_dir = os.path.join(warp_output_dir, sub)
        if os.path.isdir(sub_dir):
            shutil.rmtree(sub_dir)


def _relocate_full_reconstruction(warp_output_dir, dest_dir, logs_dir, log_name):
    """Move the full, non-split tomogram(s) ts_reconstruct wrote directly
    into *warp_output_dir* into *dest_dir*, then discard everything else
    left there (see :func:`_discard_reconstruction_byproducts`).

    CONFIRMED against WarpTools' own source (``TiltSeries.
    ReconstructFull.cs``): the full tomogram is written to
    ``<warp_output_dir>/<name>_<angpix>Apx.mrc`` on EVERY ts_reconstruct
    call, unconditionally - with or without ``--halfmap_frames``/
    ``--halfmap_tilts``. Those flags only ADD ``even``/``odd`` output
    alongside it; they never replace the full tomogram. This is why
    :func:`run_ts_reconstruct` harvests the full tomogram from the
    ``--halfmap_frames`` call for free instead of paying for a second,
    separate reconstruction of the identical volume.
    """
    tomograms = [name for name in os.listdir(warp_output_dir)
                if name.lower().endswith(".mrc")
                and os.path.isfile(os.path.join(warp_output_dir, name))]
    if not tomograms:
        raise SystemExit(
            "ERROR: expected %s to hold the reconstructed tomogram(s) "
            "directly (ts_reconstruct always writes the full tomogram "
            "there), found nothing.\n"
            "ts_reconstruct's real output layout differs from what this "
            "pipeline assumed - check %s and adjust warptools.py's "
            "_relocate_full_reconstruction."
            % (warp_output_dir, os.path.join(logs_dir, log_name))
        )
    ensure_dir(dest_dir)
    for name in tomograms:
        os.replace(os.path.join(warp_output_dir, name), os.path.join(dest_dir, name))
    _discard_reconstruction_byproducts(warp_output_dir)


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
    the coarser ``tilt_split.target_pixel_size`` instead (see below).

    *reconstruction_dir* is passed to every ``ts_reconstruct`` call as
    ``--output_processing`` - the ephemeral scratch location its raw output
    lands in (one level deeper, in ``<reconstruction_dir>/reconstruction/``
    - see :func:`_warp_reconstruction_output_dir`) before relocation into
    *full_dir*/*halfmaps_dir*/*tiltsplit_halfmaps_dir*. Shared by
    :func:`reconstruct_tomograms` (``do_full=True``, *do_halfmaps*
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
    directories, not this ephemeral intermediate, since that is emptied by
    the relocation step right after every call.

    CONFIRMED against WarpTools' own source (``TiltSeries.
    ReconstructFull.cs``): the full, non-split tomogram is written
    unconditionally on EVERY ts_reconstruct call, with or without
    ``--halfmap_frames``/``--halfmap_tilts`` - those flags only ADD
    even/odd output alongside it, at no extra cost to the full tomogram.
    So when *do_full* and *do_halfmaps* are BOTH wanted, the full tomogram
    is harvested as a free byproduct of the ``--halfmap_frames`` call
    (same ``angpix``) instead of paying for a second, separate
    reconstruction of the identical volume. A dedicated plain call (no
    half-map flag at all) is only made when *do_full* is wanted WITHOUT
    *do_halfmaps* - there is then no halfmap call to harvest it from, and
    a plain call skips the (otherwise unwanted) even/odd compute entirely.
    """
    reconstruction_dir = ensure_dir(reconstruction_dir)
    warp_output_dir = _warp_reconstruction_output_dir(reconstruction_dir)
    ensure_dir(logs_dir)
    if do_full:
        ensure_dir(full_dir)
    if do_halfmaps:
        ensure_dir(halfmaps_dir)
        if do_tiltsplit:
            ensure_dir(tiltsplit_halfmaps_dir)

    full_scratch = os.path.join(reconstruction_dir, "full")
    frame_even_dst = os.path.join(reconstruction_dir, "frame_even")
    frame_odd_dst = os.path.join(reconstruction_dir, "frame_odd")
    tilt_even_dst = os.path.join(reconstruction_dir, "tilt_even")
    tilt_odd_dst = os.path.join(reconstruction_dir, "tilt_odd")

    plain_full_call = do_full and not do_halfmaps
    harvest_full_from_frame_call = do_full and do_halfmaps

    have_full = do_full and _has_any_files(full_dir)
    if plain_full_call and (force or not have_full):
        code = _ts_reconstruct_call(config, settings_path, angpix, None,
                                    reconstruction_dir, logs_dir,
                                    "ts_reconstruct_full.log", dry_run=dry_run)
        if not dry_run:
            if code != 0:
                if log:
                    log.fail("ts_reconstruct (full tomogram)",
                            "exit status %d" % code)
                return False
            _relocate_full_reconstruction(
                warp_output_dir, full_scratch, logs_dir,
                "ts_reconstruct_full.log")
        if log:
            log.task("ts_reconstruct (full tomogram)", True, "-")
    elif plain_full_call and log:
        log.note("full tomograms already present in %s - skipping" % full_dir)

    have_frame = do_halfmaps and _has_any_files(halfmaps_dir)
    if do_halfmaps and (force or not have_frame):
        code = _ts_reconstruct_call(config, settings_path, angpix,
                                    "--halfmap_frames", reconstruction_dir,
                                    logs_dir, "ts_reconstruct_frame.log",
                                    dry_run=dry_run)
        if not dry_run:
            if code != 0:
                if log:
                    log.fail("ts_reconstruct (frame halves)",
                            "exit status %d" % code)
                return False
            _relocate_reconstruction_halves(
                warp_output_dir, frame_even_dst, frame_odd_dst,
                "--halfmap_frames", logs_dir, "ts_reconstruct_frame.log")
            if harvest_full_from_frame_call and (force or not have_full):
                _relocate_full_reconstruction(
                    warp_output_dir, full_scratch, logs_dir,
                    "ts_reconstruct_frame.log")
                if log:
                    log.note("full tomogram harvested from the "
                             "--halfmap_frames call (same reconstruction, "
                             "no separate pass)")
            else:
                _discard_reconstruction_byproducts(warp_output_dir)
                if harvest_full_from_frame_call and log:
                    log.note("full tomograms already present in %s - "
                             "discarded this call's redundant duplicate"
                             % full_dir)
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
        # size to be meaningful (see tilt_split.target_pixel_size). This is
        # requested directly as --angpix - ts_reconstruct's own Fourier
        # resampling takes an arbitrary target, no achievable-binning-step
        # calculation needed here (unlike the old AreTomo-binning design).
        tiltsplit_angpix = float(config["tilt_split"]["target_pixel_size"])
        if log:
            log.note("tilt-split recon_angpix: %.4f A (tilt_split.target_pixel_size)"
                     % tiltsplit_angpix)
        code = _ts_reconstruct_call(config, settings_path, tiltsplit_angpix,
                                    "--halfmap_tilts", reconstruction_dir,
                                    logs_dir, "ts_reconstruct_tilt.log",
                                    dry_run=dry_run)
        if not dry_run:
            if code != 0:
                if log:
                    log.fail("ts_reconstruct (tilt halves)",
                            "exit status %d" % code)
                return False
            _relocate_reconstruction_halves(
                warp_output_dir, tilt_even_dst, tilt_odd_dst,
                "--halfmap_tilts", logs_dir, "ts_reconstruct_tilt.log")
            # This call's own full-tomogram byproduct is at the coarser
            # tilt_split angpix, not the fine recon_angpix tomograms_dir is
            # meant to hold - discard rather than keep a second, wrong one.
            _discard_reconstruction_byproducts(warp_output_dir)
        if log:
            log.task("ts_reconstruct (tilt halves)", True, "-")
    elif do_halfmaps and do_tiltsplit and log:
        log.note("tilt-split half maps already present in %s - skipping"
                 % tiltsplit_halfmaps_dir)

    if dry_run:
        return True

    sources = []
    if do_full:
        sources.append((full_scratch, None, full_dir))
    if do_halfmaps:
        sources += [(frame_even_dst, "even", halfmaps_dir),
                   (frame_odd_dst, "odd", halfmaps_dir)]
        if do_tiltsplit:
            sources += [(tilt_even_dst, "even", tiltsplit_halfmaps_dir),
                       (tilt_odd_dst, "odd", tiltsplit_halfmaps_dir)]

    # Ownership is resolved with mdoc.match_tomo_name (whole underscore-
    # delimited run, longest name wins) rather than a "*name*" substring
    # glob - a real production bug (confirmed: "Position_7"'s glob matched
    # "Position_7_4"'s reconstructed file too, and glob's unordered listing
    # meant whichever of the two happened to sort first silently became BOTH
    # tomograms' symlink target) showed a plain substring test is not safe
    # whenever one tilt series' name is a literal prefix of another's - a
    # routine occurrence with numeric position naming (Position_7 vs
    # Position_7_4 vs Position_70-79, ...), not a rare edge case.
    linked = 0
    matched_names = set()
    for src_dir, half, dst_dir in sources:
        if not os.path.isdir(src_dir):
            continue
        for entry in sorted(os.listdir(src_dir)):
            source = os.path.join(src_dir, entry)
            if not os.path.isfile(source):
                continue
            owner = match_tomo_name(entry, series_names)
            if owner is None:
                warn("%s: %r does not match any of this run's tilt series "
                     "by name - not linked. If this is a real result, check "
                     "why its tilt series is missing from series_names."
                     % (src_dir, entry))
                continue
            target_name = ("%s_%s_rec_tomo.mrc" % (half, owner) if half
                           else "%s_rec_tomo.mrc" % owner)
            target = os.path.join(dst_dir, target_name)
            link_file(source, target)
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
    require_executable(config.exe("warptools"), "WarpTools")

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

    banner("reconstruct_tomograms 1/3  ts_defocus_hand")
    # --check only measures and reports a recommendation - it does not
    # change anything. --set_flip is a SEPARATE, later call, and only when
    # --check actually reported a negative correlation: unconditionally
    # combining --check and --set_flip in one call would flip the
    # handedness even for a dataset that was already correct.
    check_log = os.path.join(logs, "ts_defocus_hand_check.log")
    command = [config.exe("warptools"), "ts_defocus_hand",
               "--settings", settings_path, "--check"]
    run_command(command, cwd=config.processing_dir, env=env,
               log_file=check_log, check=False, dry_run=dry_run)
    log.task("ts_defocus_hand (check)", True, "-")

    if dry_run:
        info("ts_defocus_hand: --set_flip decision skipped for this dry run "
             "(it depends on --check's real output)")
    elif not wt["auto_fix_defocus_hand"]:
        log.note("ts_defocus_hand: auto_fix_defocus_hand is false - check "
                 "only, no changes made")
    else:
        negative = False
        if os.path.isfile(check_log):
            with open(check_log, encoding="utf-8", errors="replace") as handle:
                negative = "negative" in handle.read().lower()
        if negative:
            info("ts_defocus_hand: negative correlation detected - "
                 "applying --set_flip")
            log.note("ts_defocus_hand: negative correlation detected - "
                     "applied --set_flip")
            flip_command = [config.exe("warptools"), "ts_defocus_hand",
                            "--settings", settings_path, "--set_flip"]
            run_command(flip_command, cwd=config.processing_dir, env=env,
                       log_file=os.path.join(logs, "ts_defocus_hand_set_flip.log"),
                       check=False, dry_run=dry_run)
            log.task("ts_defocus_hand (set_flip)", True, "-")
        else:
            info("ts_defocus_hand: handedness OK - no flip needed")
            log.note("ts_defocus_hand: handedness OK - no flip needed")

    banner("reconstruct_tomograms 2/3  ts_ctf")
    command = [config.exe("warptools"), "ts_ctf",
               "--settings", settings_path,
               "--range_high", str(float(wt["ts_ctf_range_high"])),
               "--defocus_max", str(float(wt["ts_ctf_defocus_max"])),
               "--perdevice", str(int(wt["perdevice"]))]
    code = run_command(command, cwd=config.processing_dir, env=env,
                       log_file=os.path.join(logs, "ts_ctf.log"),
                       check=False, dry_run=dry_run)
    if code != 0 and not dry_run:
        log.fail("ts_ctf", "exit status %d" % code)
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
