"""Step 1: raw frames -> aligned tilt series -> tomogram + half maps.

MotionCor3 -> IMOD newstack -> AreTomo2, in four stages:

  1. prepare    motion correction, angle file and the full stack, per series
  2. tilt axis  a few AreTomo2 alignment-only runs on random subsets, combined
                into one dataset-wide tilt axis (see :mod:`tiltaxis`)
  3. frame      the full reconstruction with that axis fixed, plus the
                even/odd FRAME half maps -> raw_halfMaps
  4. tilt       optionally, the same alignment reused to reconstruct even/odd
                TILT half maps -> raw_halfMaps_tiltSplit

Everything a tilt series produces lives in ``rec_<tomoName>``; the half maps
are symlinked into the two shared directories. Nothing is moved or deleted.

Resuming (``overwrite: false``, the default) is layered to match: a missing
converged tilt axis means stages 3 and 4 are redone for EVERY tilt series,
since they all depend on it, while a known tilt axis makes stages 3 and 4
resume per tomogram and per half from whatever ``rec_<tomoName>/aretomo_*``
already holds - see ``full_reconstruction_present``, ``frame_half_maps_present``
and ``tilt_half_maps_present``. Either way, a final pass fills in any shared
directory symlink still missing for a reconstruction that already exists.
"""

import glob
import os
import re

import numpy as np

from . import tiltaxis
from .mdoc import collect_tilt_series, resolve_pixel_size
from .steplog import StepLog
from .utils import (
    banner,
    ensure_dir,
    info,
    link_file,
    require_executable,
    run,
    warn,
    gpu_flag_motioncor,
    CommandError,
)

STEP = "create_halfMaps"


class TomoDirs(object):
    """The directory layout of a single ``rec_<tomoName>`` directory."""

    def __init__(self, config, tomo_name):
        self.root = ensure_dir(config.rec_dir(tomo_name))
        self.motioncor = ensure_dir(self.root, "motioncor")
        self.stacks = ensure_dir(self.root, "stacks")
        self.aretomo_full = ensure_dir(self.root, "aretomo_full")
        self.aretomo_even = ensure_dir(self.root, "aretomo_even")
        self.aretomo_odd = ensure_dir(self.root, "aretomo_odd")
        # Throwaway alignment-only runs from the tilt axis search (stage 2)
        # land here, never in aretomo_full - see align_full()'s output_dir.
        self.aretomo_tiltaxis = ensure_dir(self.root, "aretomo_tiltaxis")
        self.logs = ensure_dir(self.root, "logs")

    def tiltsplit(self, half):
        return ensure_dir(self.root, "aretomo_tiltSplit_" + half)


def resolve_input_file(value, key):
    """Resolve an auxiliary input file the same way as the gain reference.

    Absolute as given, relative against the current working directory, never
    joined to input_frames. Returns "" when nothing is configured, and raises
    when a value is set but missing.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""

    path = os.path.abspath(raw)
    if os.path.isfile(path):
        return path

    raise SystemExit(
        "ERROR: %s %r not found at:\n  %s\n"
        "Relative paths are resolved against the directory you run "
        "tomo_denoising from. Use an absolute path, or leave it empty to omit "
        "the option." % (key, raw, path)
    )


# --------------------------------------------------------------- MotionCor3


def _motioncor_input_flag(extension):
    if extension == "eer":
        return "-InEer"
    if extension in ("tif", "tiff"):
        return "-InTiff"
    return "-InMrc"


def run_motioncor(config, series, dirs, log, force=False, dry_run=False):
    """Motion correct every frame of *series*; returns the averaged file names.

    ``-SplitSum 1`` also writes ``*_averaged_EVN.mrc`` and ``*_averaged_ODD.mrc``
    next to each average - the per-tilt half images the frame-split stacks are
    built from. This always runs, whether or not tilt splitting is enabled.
    """
    acquisition = config["acquisition"]
    mc = config["motioncor"]
    extension = acquisition["frame_extension"]
    gain = config.resolve_gain()
    gain_name = config.gain_name

    produced = []
    failures = []
    for frame in series.frames:
        if gain_name and os.path.basename(frame) == gain_name:
            continue
        if "_averaged" in frame:
            continue

        source = os.path.join(config.input_frames, frame)
        if not os.path.exists(source):
            failures.append(os.path.basename(frame))
            warn("frame not found, skipping: %s" % source)
            continue

        stem = os.path.splitext(frame)[0]
        output = os.path.join(dirs.motioncor, stem + "_averaged.mrc")
        produced.append(os.path.basename(output))

        if os.path.exists(output) and not force and not dry_run:
            continue

        command = [
            config.exe("motioncor"),
            _motioncor_input_flag(extension), source,
            "-OutMrc", output,
        ]
        if gain:
            command += ["-Gain", gain]
        if int(mc["flip_gain"]):
            command += ["-FlipGain", str(int(mc["flip_gain"]))]
        if extension == "eer":
            command += ["-EerSampling", str(mc["eer_sampling"])]
            fm_int = resolve_input_file(mc["fm_int_file"],
                                        "motioncor.fm_int_file")
            if fm_int:
                command += ["-FmIntFile", fm_int]
        binning = int(mc["binning"])
        if binning not in (0, 1):
            command += ["-FtBin", str(binning)]
        command += ["-Iter", str(int(mc["iterations"]))]
        command += ["-Tol", str(mc["tolerance"])]
        command += ["-SplitSum", "1"]
        command += gpu_flag_motioncor(config.gpus).split()
        command += ["-LogFile", os.path.join(dirs.logs, stem + "_mc.log")]
        if str(mc["extra_args"]).strip():
            command += str(mc["extra_args"]).split()

        run(command, cwd=dirs.motioncor,
            log_file=os.path.join(dirs.logs, stem + "_motioncor.out"),
            check=False, dry_run=dry_run)

        if not dry_run and not os.path.exists(output):
            failures.append(os.path.basename(frame))
            produced.remove(os.path.basename(output))

    ok = len(produced)
    log.task("MotionCor3 (tilts)", ok > 0, series.name)
    if failures:
        warn("%s: MotionCor3 produced nothing for %d tilt(s): %s"
             % (series.name, len(failures), ", ".join(failures[:6])))
        log.quiet("  %s: MotionCor3 failed on %d tilt(s): %s"
                  % (series.name, len(failures), ", ".join(failures)))
    return produced, failures


# --------------------------------------------------------------------- .rawtlt


def write_rawtlt(dirs, series, averaged_names, tomo_name, dry_run=False):
    """Write ``<tomoName>.rawtlt`` and return (rawtlt_path, ordered_averages).

    Tilts whose motion correction produced nothing are dropped from both the
    angle file and the stack list, so the two stay in step.
    """
    order = sorted(range(len(series.angles)), key=lambda k: series.angles[k])

    ordered_names = []
    ordered_angles = []
    for index in order:
        if index >= len(series.frames):
            continue
        candidate = os.path.splitext(series.frames[index])[0] + "_averaged.mrc"
        if candidate not in averaged_names:
            continue
        if not dry_run and not os.path.exists(
                os.path.join(dirs.motioncor, candidate)):
            continue
        ordered_names.append(candidate)
        ordered_angles.append(series.angles[index])

    rawtlt = os.path.join(dirs.root, tomo_name + ".rawtlt")
    if not dry_run:
        with open(rawtlt, "w", encoding="utf-8") as handle:
            for angle in ordered_angles:
                handle.write("%s\n" % angle)
    if ordered_angles:
        info("%d tilts, range %.2f to %.2f deg"
             % (len(ordered_angles), min(ordered_angles), max(ordered_angles)))
    return rawtlt, ordered_names


def split_rawtlt(rawtlt_path, even_out, odd_out,
                 even_sections=None, odd_sections=None):
    """Split an angle file to match the stack split.

    The sections come from the alignment, so dark tilts are dropped here too
    and the angle file stays in step with its stack.
    """
    with open(rawtlt_path, encoding="utf-8", errors="replace") as handle:
        angles = [line.strip() for line in handle if line.strip()]

    if even_sections is None:
        even_sections = list(range(0, len(angles), 2))
    if odd_sections is None:
        odd_sections = list(range(1, len(angles), 2))

    for sections, out in ((even_sections, even_out), (odd_sections, odd_out)):
        chosen = [angles[s] for s in sections if 0 <= s < len(angles)]
        if len(chosen) != len(sections):
            warn("%s has %d angles but the alignment refers to section %d"
                 % (os.path.basename(rawtlt_path), len(angles),
                    max(sections) if sections else -1))
        with open(out, "w", encoding="utf-8") as handle:
            handle.write("\n".join(chosen) + "\n")
    return even_out, odd_out


# ------------------------------------------------------------------- newstack


def build_stack(config, dirs, members, output, dry_run=False):
    """Stack *members* (names relative to the motioncor directory) into *output*."""
    if not members:
        raise CommandError("no tilts available to stack into %s" % output)
    run([config.exe("newstack")] + list(members) + [output],
        cwd=dirs.motioncor,
        log_file=os.path.join(dirs.logs,
                              os.path.basename(output) + ".newstack.log"),
        dry_run=dry_run)
    return output


# ------------------------------------------------------------------- AreTomo2


def _aretomo_common(config, apix_binned, vol_z=None, binning=None):
    aretomo = config["aretomo"]
    command = [
        "-VolZ", str(int(aretomo["vol_z"] if vol_z is None else vol_z)),
        "-OutBin", str(int(aretomo["binning"] if binning is None else binning)),
        "-DarkTol", str(aretomo["dark_tolerance"]),
    ]
    if apix_binned:
        command += ["-PixSize", "%.4f" % apix_binned]
    if aretomo["wbp"]:
        command += ["-Wbp", "1"]
    # Always flip the reconstructed volume (-FlipVol 1), full map included:
    # someone may want to open that full reconstruction directly to inspect
    # it, and an unflipped one would be inconsistent with the half maps.
    command += ["-FlipVol", "1"]
    local = str(aretomo["local_align"]).strip()
    if local:
        command += local.split()
    extra = str(aretomo["extra_args"]).strip()
    if extra:
        command += extra.split()
    return command


def _tilt_axis_args(angle, refine):
    """``-TiltAxis <angle> <refine>``.

    refine: -1 no refinement at all, 0 one tilt axis for the whole series,
    1 a variable tilt axis. AreTomo2 only searches +/-3 degrees around a
    supplied angle, so passing one both constrains and speeds up the search.
    """
    if angle is None:
        return []
    return ["-TiltAxis", "%.4f" % angle, str(refine)]


def _tilt_cor_args(config):
    return ["-TiltCor", "1" if config["aretomo"]["tilt_axis_correction"] else "0"]


def find_aln(stack_path, *search_dirs):
    """Locate the .aln AreTomo2 wrote - version dependent, so look around."""
    stem = os.path.splitext(os.path.basename(stack_path))[0]
    candidates = [os.path.join(os.path.dirname(stack_path), stem + ".aln")]
    for directory in search_dirs:
        candidates.append(os.path.join(directory, stem + ".aln"))
        candidates.extend(sorted(glob.glob(os.path.join(directory, "*.aln"))))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def align_full(config, dirs, stack, rawtlt, tomo_name, apix_binned,
               tilt_axis=None, refine=-1, vol_z=None, log_suffix="full",
               output_dir=None, dry_run=False):
    """Reconstruct (or, with vol_z 0, only align) the full tilt series.

    *output_dir* defaults to ``dirs.aretomo_full``, the final reconstruction's
    home. Callers that only align (``vol_z=0`` - the tilt axis search) MUST
    pass a different directory: with ``-VolZ 0`` AreTomo2 does not reconstruct
    anything, it writes the aligned tilt STACK to -OutMrc instead. Landing
    that under the same name the real reconstruction uses would make the
    resume logic mistake the stack for a finished reconstruction and skip it.
    """
    output_dir = output_dir or dirs.aretomo_full
    output = os.path.join(output_dir, tomo_name + "_rec_tomo.mrc")
    command = [config.exe("aretomo"),
               "-InMrc", stack,
               "-OutMrc", output,
               "-AngFile", rawtlt,
               "-OutXf", "1"]
    command += _aretomo_common(config, apix_binned, vol_z=vol_z)
    command += _tilt_axis_args(tilt_axis, refine)
    command += _tilt_cor_args(config)

    log_file = os.path.join(dirs.logs, "%s_aretomo_%s.log"
                            % (tomo_name, log_suffix))
    run(command, cwd=output_dir, log_file=log_file,
        check=False, dry_run=dry_run)
    return output, log_file


def reconstruct_half(config, dirs, stack, aln, output, log_name, apix_binned,
                     work_dir, binning=None, dry_run=False):
    """Half-map reconstruction reusing the alignment solved on the full stack."""
    command = [config.exe("aretomo"),
               "-InMrc", stack,
               "-OutMrc", output,
               "-AlnFile", aln]
    command += _aretomo_common(config, apix_binned, binning=binning)
    command += _tilt_cor_args(config)
    run(command, cwd=work_dir,
        log_file=os.path.join(dirs.logs, log_name),
        check=False, dry_run=dry_run)
    return output


# ---------------------------------------------------------------- tilt splitting


_DARK_FRAME_RE = re.compile(r"DarkFrame\s*=\s*(\d+)")


def parse_aln(aln_path):
    """Read an AreTomo .aln into ``(header_lines, dark_sections, rows)``.

    ``rows`` is ``[(section_index, line), ...]`` for the sections AreTomo
    actually aligned. Sections it rejected appear only as ``# DarkFrame``
    headers and are returned separately - they have no alignment, so they
    cannot be carried into a split.
    """
    header, dark, rows = [], [], []
    with open(aln_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if stripped.startswith("#"):
                match = _DARK_FRAME_RE.search(stripped)
                if match:
                    dark.append(int(match.group(1)))
                else:
                    header.append(line)
                continue
            if not stripped:
                continue
            try:
                section = int(float(stripped.split()[0]))
            except (ValueError, IndexError):
                continue
            rows.append((section, line))
    return header, dark, rows


def _set_raw_size_z(line, sections):
    """Rewrite the third number of a ``# RawSize = x y z`` header."""
    numbers = list(re.finditer(r"\d+", line))
    if len(numbers) < 3:
        return line
    third = numbers[2]
    return line[:third.start()] + str(sections) + line[third.end():]


def write_split_aln(header, rows, out_path):
    """Write one half of a split alignment, renumbered from zero.

    No ``# DarkFrame`` headers are emitted: the dark sections are left out of
    the split stack entirely, so every section the half stack contains is
    described by a row here.
    """
    with open(out_path, "w", encoding="utf-8") as handle:
        for line in header:
            if "RawSize" in line:
                handle.write(_set_raw_size_z(line, len(rows)) + "\n")
            else:
                handle.write(line + "\n")
        for index, (_, line) in enumerate(rows):
            handle.write(
                re.sub(r"^(\s*)\d+",
                       lambda m, i=index: "%s%d" % (m.group(1), i),
                       line) + "\n")
    return out_path


def split_alignment(aln_path, even_out, odd_out):
    """Split an AreTomo .aln by tilt number.

    Returns ``(even_sections, odd_sections)`` - the ORIGINAL section indices
    that went into each half, so the stack and the angle file can be split to
    match exactly.

    Sections AreTomo marked dark are dropped rather than split. They carry no
    alignment, so a half stack that still contained them would have more
    sections than its .aln describes, and AreTomo would be handed an
    inconsistent pair.
    """
    header, dark, rows = parse_aln(aln_path)
    if not rows:
        raise CommandError("no alignment rows in %s" % aln_path)

    # Alternate over the sections that survived, not over raw section numbers:
    # with leading sections dropped as dark, raw parity no longer says which
    # half a tilt belongs to.
    even_rows = rows[0::2]
    odd_rows = rows[1::2]

    write_split_aln(header, even_rows, even_out)
    write_split_aln(header, odd_rows, odd_out)

    if dark:
        info("  %d dark section(s) excluded from the tilt split: %s"
             % (len(dark), ", ".join(str(d) for d in sorted(dark))))
    return [s for s, _ in even_rows], [s for s, _ in odd_rows]


def split_stack_file(stack_path, even_out, odd_out,
                     even_sections=None, odd_sections=None):
    """Write two half stacks holding the given sections of *stack_path*.

    Without explicit sections this falls back to plain alternation, which is
    the same thing when AreTomo rejected nothing.
    """
    import mrcfile

    with mrcfile.open(stack_path, permissive=True) as source:
        data = source.data
        voxel_size = source.voxel_size
        total = data.shape[0]

        if even_sections is None:
            even_sections = list(range(0, total, 2))
        if odd_sections is None:
            odd_sections = list(range(1, total, 2))

        for sections, out in ((even_sections, even_out),
                              (odd_sections, odd_out)):
            usable = [s for s in sections if 0 <= s < total]
            if len(usable) != len(sections):
                warn("%s: the alignment refers to sections outside the stack "
                     "(%d of %d usable)"
                     % (os.path.basename(stack_path), len(usable),
                        len(sections)))
            if not usable:
                raise CommandError("no sections left for %s" % out)
            with mrcfile.new(out, overwrite=True) as target:
                target.set_data(np.float32(data[usable, :, :]))
                target.voxel_size = voxel_size
    return even_out, odd_out


def tiltsplit_binning(config, apix_binned):
    """AreTomo binning that lands closest to tilt_split.target_pixel_size.

    Returns ``(binning, achieved_pixel_size)``. Binning is an integer, so the
    target is approached rather than hit.
    """
    target = float(config["tilt_split"]["target_pixel_size"])
    if apix_binned <= 0:
        return int(config["aretomo"]["binning"]), None
    binning = max(1, int(round(target / apix_binned)))
    return binning, binning * apix_binned


# ------------------------------------------------------------------ resuming


def half_map_targets(config, tomo_name, tilt_split=False):
    """The two shared-directory paths for one tomogram."""
    directory = (config.tiltsplit_halfmaps_dir if tilt_split
                 else config.raw_halfmaps_dir)
    return (os.path.join(directory, "even_%s_rec_tomo.mrc" % tomo_name),
            os.path.join(directory, "odd_%s_rec_tomo.mrc" % tomo_name))


def half_maps_present(config, tomo_name, tilt_split=False):
    return all(os.path.exists(p)
               for p in half_map_targets(config, tomo_name, tilt_split))


# ---------------------------------------------- resuming: per-tomogram outputs
#
# ``half_maps_present`` above looks at the SHARED raw_halfMaps directories -
# cheap, and a reliable "this tomogram needs nothing at all" fast path, since
# a symlink only ever appears there once AreTomo2 has actually produced the
# file it points at. But the reverse is not true: a missing symlink does not
# mean the reconstruction is missing too - a run can be interrupted after
# AreTomo2 finishes and before the file is linked in. The checks below look
# directly in rec_<tomo>/aretomo_* - the source of truth - so that case is
# recognised as "just needs linking" rather than "needs AreTomo2 again".


def full_reconstruction_present(dirs, full_stack, name):
    """``rec_<tomo>/aretomo_full`` already has the reconstruction AND its .aln.

    Both are required: an .aln without its reconstruction means AreTomo2 was
    interrupted mid-run, and neither is any use for the half maps on its own.
    """
    reconstruction = os.path.join(dirs.aretomo_full, name + "_rec_tomo.mrc")
    aln = find_aln(full_stack, dirs.aretomo_full, dirs.stacks)
    return aln is not None and os.path.exists(reconstruction)


def frame_half_maps_present(dirs, name):
    """``rec_<tomo>/aretomo_even`` and ``aretomo_odd`` already hold both halves."""
    even = os.path.join(dirs.aretomo_even, "even_%s_rec_tomo.mrc" % name)
    odd = os.path.join(dirs.aretomo_odd, "odd_%s_rec_tomo.mrc" % name)
    return os.path.exists(even) and os.path.exists(odd)


def tilt_half_maps_present(dirs, name):
    """``rec_<tomo>/aretomo_tiltSplit_even`` and ``_odd`` already hold both halves."""
    even = os.path.join(dirs.tiltsplit("even"), "even_%s_rec_tomo.mrc" % name)
    odd = os.path.join(dirs.tiltsplit("odd"), "odd_%s_rec_tomo.mrc" % name)
    return os.path.exists(even) and os.path.exists(odd)


def relink_existing_half_maps(config, tomo_name, do_tilt_split, log):
    """Fill in any shared-directory symlink whose reconstruction already exists.

    Safety net for a run that crashed between AreTomo2 finishing and the file
    being linked in - nothing is recomputed, the existing reconstruction is
    just picked up. A no-op for a tilt series that was never reconstructed at
    all (``rec_<tomo>`` does not exist yet).
    """
    root = config.rec_dir(tomo_name)
    if not os.path.isdir(root):
        return

    pairs = [(
        (os.path.join(root, "aretomo_even", "even_%s_rec_tomo.mrc" % tomo_name),
         os.path.join(root, "aretomo_odd", "odd_%s_rec_tomo.mrc" % tomo_name)),
        half_map_targets(config, tomo_name),
        "raw_halfMaps",
    )]
    if do_tilt_split:
        pairs.append((
            (os.path.join(root, "aretomo_tiltSplit_even",
                          "even_%s_rec_tomo.mrc" % tomo_name),
             os.path.join(root, "aretomo_tiltSplit_odd",
                          "odd_%s_rec_tomo.mrc" % tomo_name)),
            half_map_targets(config, tomo_name, tilt_split=True),
            "raw_halfMaps_tiltSplit",
        ))

    for (even_source, odd_source), (even_target, odd_target), label in pairs:
        for source, target in ((even_source, even_target),
                               (odd_source, odd_target)):
            if os.path.exists(source) and not os.path.lexists(target):
                link_file(source, target)
                log.quiet("  %s: relinked into %s (was missing)"
                          % (tomo_name, label))


# ------------------------------------------------------------ stage 1: prepare


class Prepared(object):
    """What a tilt series needs before AreTomo2 can be run on it."""

    def __init__(self, series, dirs, apix_binned, rawtlt, ordered, full_stack):
        self.series = series
        self.name = series.name
        self.dirs = dirs
        self.apix_binned = apix_binned
        self.rawtlt = rawtlt
        self.ordered = ordered
        self.full_stack = full_stack


def prepare_series(config, series, log, force=False, dry_run=False):
    """Motion correct, write the angle file and build the full stack."""
    dirs = TomoDirs(config, series.name)

    first_frame = (os.path.join(config.input_frames, series.frames[0])
                   if series.frames else None)
    apix = resolve_pixel_size(config, series, first_frame)
    apix_binned = apix * int(config["motioncor"]["binning"])

    averaged, _ = run_motioncor(config, series, dirs, log,
                                force=force, dry_run=dry_run)
    if not averaged:
        log.fail(series.name, "MotionCor3 produced no averages")
        return None

    rawtlt, ordered = write_rawtlt(dirs, series, averaged, series.name,
                                   dry_run=dry_run)
    if not ordered:
        log.fail(series.name, "no usable tilts after motion correction")
        return None

    full_stack = os.path.join(dirs.stacks, series.name + "_stacked.mrc")
    if force or dry_run or not os.path.exists(full_stack):
        try:
            build_stack(config, dirs, ordered, full_stack, dry_run=dry_run)
        except CommandError as exc:
            log.fail(series.name, "newstack failed: %s" % exc)
            return None
    if not dry_run and not os.path.exists(full_stack):
        log.fail(series.name, "newstack produced no stack")
        return None

    log.task("newstack (full stack)", True, series.name)
    return Prepared(series, dirs, apix_binned, rawtlt, ordered, full_stack)


# -------------------------------------------------- stage 2: tilt axis search


def refine_tilt_axis(config, prepared, log, force=False, dry_run=False):
    """Determine one tilt axis for the whole dataset.

    Alignment-only AreTomo2 runs (``-VolZ 0``) on a random subset per
    iteration; the resulting angles are combined with the geometric median of
    their unit vectors. Returns the converged angle, or None when refinement
    is switched off or nothing could be measured.
    """
    iterations = int(config["aretomo"]["tilt_axis_iterations"])
    if iterations <= 0:
        info("tilt axis refinement disabled (aretomo.tilt_axis_iterations: 0)")
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
    by_name = {prep.name: prep for prep in prepared}

    banner("Tilt axis refinement (%d iterations, %d tilt series each)"
           % (iterations, min(subset_size, len(prepared))))

    rows = []
    angle = None
    for iteration in range(1, iterations + 1):
        names = tiltaxis.choose_subset(sorted(by_name), subset_size, seed,
                                       iteration)
        info("iteration %d/%d on: %s" % (iteration, iterations, ", ".join(names)))

        measurements = {}
        sources = {}
        for name in names:
            prep = by_name[name]
            # -VolZ 0 aligns without reconstructing, which is all we need here.
            # output_dir keeps this out of aretomo_full: with -VolZ 0 AreTomo2
            # writes the aligned STACK to -OutMrc rather than a reconstruction,
            # and that must never be mistaken for the real one.
            _, log_file = align_full(
                config, prep.dirs, prep.full_stack, prep.rawtlt, name,
                prep.apix_binned,
                tilt_axis=angle,
                refine=0 if angle is not None else -1,
                vol_z=0,
                log_suffix="tiltaxis_iter%d" % iteration,
                output_dir=prep.dirs.aretomo_tiltaxis,
                dry_run=dry_run,
            )
            if dry_run:
                measurements[name] = None
                continue
            aln = find_aln(prep.full_stack, prep.dirs.aretomo_tiltaxis,
                           prep.dirs.stacks)
            value, source = tiltaxis.read_tilt_axis(aln, log_file)
            measurements[name] = value
            sources[name] = source

            # A starting angle restricts AreTomo to a +/-3 degree search, so a
            # result far outside that window did not come from the alignment.
            if value is not None and angle is not None:
                drift = abs(tiltaxis.fold_angles([angle, value])[1]
                            - tiltaxis.fold_angles([angle, value])[0])
                if drift > tiltaxis.SUSPICIOUS_DEVIATION:
                    warn("%s: reported %+.3f deg after starting from %+.3f - "
                         "that is %.1f deg, outside the +/-%.0f deg AreTomo "
                         "searches. The value was read from %s and is probably "
                         "not the solved tilt axis."
                         % (name, value, angle, drift,
                            tiltaxis.REFINE_WINDOW, source))

        usable = [v for v in measurements.values() if v is not None]
        combined, method = tiltaxis.combine_tilt_axes(usable)
        spread = None
        if len(usable) > 1:
            folded = tiltaxis.fold_angles(usable)
            spread = max(folded) - min(folded)

        rows.append({
            "iteration": iteration,
            "initial": angle,
            "measurements": measurements,
            "sources": sources,
            "angle": combined,
            "method": method,
            "spread": spread,
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

    note = ""
    if angle is None:
        note = ("No tilt axis could be measured. Every tilt series was aligned "
                "independently instead.")
        warn(note)
    if not dry_run:
        tiltaxis.write_report(config, rows, angle, note=note)

    if angle is not None:
        log.note("tilt axis converged to %+.4f deg after %d iteration(s)"
                 % (angle, iterations))
    return angle


# ------------------------------------------------- stage 3: frame-split halves


def ensure_full_alignment(config, prep, tilt_axis, log, force=False,
                          dry_run=False):
    """The final full reconstruction, with the tilt axis fixed. Returns the .aln."""
    if (not force and not dry_run
            and full_reconstruction_present(prep.dirs, prep.full_stack, prep.name)):
        return find_aln(prep.full_stack, prep.dirs.aretomo_full, prep.dirs.stacks)

    align_full(config, prep.dirs, prep.full_stack, prep.rawtlt, prep.name,
               prep.apix_binned,
               tilt_axis=tilt_axis,
               refine=-1,          # angle fixed: it was solved dataset-wide
               log_suffix="full",
               dry_run=dry_run)

    aln = find_aln(prep.full_stack, prep.dirs.aretomo_full, prep.dirs.stacks)
    log.task("AreTomo2 (full)", aln is not None or dry_run, prep.name)
    return aln


def frame_split_half_maps(config, prep, aln, log, force=False, dry_run=False):
    """Even/odd FRAME half maps -> raw_halfMaps.

    Skips straight to linking if ``rec_<tomo>/aretomo_even``/``aretomo_odd``
    already hold both reconstructions - AreTomo2 is expensive, and a missing
    shared-directory symlink alone (the only reason this is called at all) is
    not a reason to redo it.
    """
    dirs, name = prep.dirs, prep.name
    even_out = os.path.join(dirs.aretomo_even, "even_%s_rec_tomo.mrc" % name)
    odd_out = os.path.join(dirs.aretomo_odd, "odd_%s_rec_tomo.mrc" % name)

    if force or dry_run or not frame_half_maps_present(dirs, name):
        even_stack = os.path.join(dirs.stacks, "even_%s_stacked.mrc" % name)
        odd_stack = os.path.join(dirs.stacks, "odd_%s_stacked.mrc" % name)

        for stack, marker in ((even_stack, "_averaged_EVN.mrc"),
                              (odd_stack, "_averaged_ODD.mrc")):
            if force or dry_run or not os.path.exists(stack):
                members = [n.replace("_averaged.mrc", marker) for n in prep.ordered]
                try:
                    build_stack(config, dirs, members, stack, dry_run=dry_run)
                except CommandError as exc:
                    log.fail(name, "newstack failed on the %s half: %s"
                             % ("even" if marker.endswith("EVN.mrc") else "odd", exc))
                    return False

        reconstruct_half(config, dirs, even_stack, aln, even_out,
                         name + "_aretomo_even.log", prep.apix_binned,
                         dirs.aretomo_even, dry_run=dry_run)
        reconstruct_half(config, dirs, odd_stack, aln, odd_out,
                         name + "_aretomo_odd.log", prep.apix_binned,
                         dirs.aretomo_odd, dry_run=dry_run)
    else:
        log.quiet("  %s: frame-split reconstructions already present in %s"
                  % (name, dirs.root))

    if dry_run:
        return True

    even_target, odd_target = half_map_targets(config, name)
    for source, target in ((even_out, even_target), (odd_out, odd_target)):
        if not os.path.exists(source):
            log.fail(name, "AreTomo2 produced no frame-split half map (%s)"
                     % os.path.basename(source))
            log.task("AreTomo2 (frame halves)", False, name)
            return False
        link_file(source, target)
    log.task("AreTomo2 (frame halves)", True, name)
    return True


# -------------------------------------------------- stage 4: tilt-split halves


def tilt_split_half_maps(config, prep, aln, log, force=False, dry_run=False):
    """Even/odd TILT half maps -> raw_halfMaps_tiltSplit.

    Reuses the motion corrected averages and the alignment from the frame-split
    run: only the stack, the angle file and the .aln are split by tilt number.

    Skips straight to linking if ``rec_<tomo>/aretomo_tiltSplit_even``/``_odd``
    already hold both reconstructions - see ``frame_split_half_maps``.
    """
    dirs, name = prep.dirs, prep.name
    even_dir = dirs.tiltsplit("even")
    odd_dir = dirs.tiltsplit("odd")
    even_out = os.path.join(even_dir, "even_%s_rec_tomo.mrc" % name)
    odd_out = os.path.join(odd_dir, "odd_%s_rec_tomo.mrc" % name)

    if force or dry_run or not tilt_half_maps_present(dirs, name):
        binning, achieved = tiltsplit_binning(config, prep.apix_binned)

        even_stack = os.path.join(dirs.stacks, "tiltSplit_even_%s.mrc" % name)
        odd_stack = os.path.join(dirs.stacks, "tiltSplit_odd_%s.mrc" % name)
        even_aln = os.path.join(dirs.stacks, "tiltSplit_even_%s.aln" % name)
        odd_aln = os.path.join(dirs.stacks, "tiltSplit_odd_%s.aln" % name)
        even_tlt = os.path.join(dirs.stacks, "tiltSplit_even_%s.rawtlt" % name)
        odd_tlt = os.path.join(dirs.stacks, "tiltSplit_odd_%s.rawtlt" % name)

        info("%s: tilt split at bin %d -> %.2f A/px (target %.1f)"
             % (name, binning, achieved or 0.0,
                float(config["tilt_split"]["target_pixel_size"])))

        if not dry_run:
            try:
                # The alignment decides the split: it knows which sections
                # AreTomo actually solved, so the stack and the angle file
                # follow from it rather than from raw section parity.
                even_sections, odd_sections = split_alignment(
                    aln, even_aln, odd_aln)
                split_stack_file(prep.full_stack, even_stack, odd_stack,
                                 even_sections, odd_sections)
                split_rawtlt(prep.rawtlt, even_tlt, odd_tlt,
                             even_sections, odd_sections)
                info("  %s: tilt split into %d + %d sections"
                     % (name, len(even_sections), len(odd_sections)))
            except Exception as exc:                      # noqa: BLE001
                log.fail(name, "could not split the tilt series: %s" % exc)
                log.task("AreTomo2 (tilt halves)", False, name)
                return False

        reconstruct_half(config, dirs, even_stack, even_aln, even_out,
                         name + "_aretomo_tiltSplit_even.log", prep.apix_binned,
                         even_dir, binning=binning, dry_run=dry_run)
        reconstruct_half(config, dirs, odd_stack, odd_aln, odd_out,
                         name + "_aretomo_tiltSplit_odd.log", prep.apix_binned,
                         odd_dir, binning=binning, dry_run=dry_run)
    else:
        log.quiet("  %s: tilt-split reconstructions already present in %s"
                  % (name, dirs.root))

    if dry_run:
        return True

    even_target, odd_target = half_map_targets(config, name, tilt_split=True)
    for source, target in ((even_out, even_target), (odd_out, odd_target)):
        if not os.path.exists(source):
            log.fail(name, "AreTomo2 produced no tilt-split half map (%s)"
                     % os.path.basename(source))
            log.task("AreTomo2 (tilt halves)", False, name)
            return False
        link_file(source, target)
    log.task("AreTomo2 (tilt halves)", True, name)
    return True


# ------------------------------------------------------------------ entrypoint


def create_half_maps(config, force=None, dry_run=False):
    """``tomo_denoising create_halfMaps``"""
    force = config.overwrite if force is None else force
    do_tilt_split = bool(config["tilt_split"]["enabled"])

    with StepLog(config, STEP, force=force) as log:
        for key, purpose in (("motioncor", "MotionCor3"),
                             ("newstack", "IMOD newstack"),
                             ("aretomo", "AreTomo2")):
            require_executable(config.exe(key), purpose)

        ensure_dir(config.processing_dir)
        ensure_dir(config.raw_halfmaps_dir)
        if do_tilt_split:
            ensure_dir(config.tiltsplit_halfmaps_dir)

        gain = config.resolve_gain()
        if gain:
            log.note("gain reference: %s" % gain)

        directory = config.mdoc_dir("preprocessing")
        all_series = collect_tilt_series(directory, "preprocessing")
        log.note("%d tilt series found in %s" % (len(all_series), directory))
        log.note("frame splitting: always; tilt splitting: %s"
                 % ("yes" if do_tilt_split else "no"))

        # Is the dataset-wide tilt axis already settled? Every AreTomo2
        # reconstruction depends on it, so if it still needs solving (no
        # converged value on record, and refinement is not disabled), nothing
        # already built - even for a tilt series that otherwise looks
        # complete - can be trusted until it is rebuilt with the new value.
        tilt_axis_iterations = int(config["aretomo"]["tilt_axis_iterations"])
        tilt_axis_known = (
            tilt_axis_iterations <= 0
            or (not force and tiltaxis.read_converged_angle(config) is not None)
        )
        force_aretomo = force or not tilt_axis_known
        if force_aretomo and not force:
            info("no converged tilt axis on record (%s) - solving it and "
                 "redoing every AreTomo2 reconstruction to match."
                 % tiltaxis.report_path(config))

        # Which tilt series still need work? Motion correction and stacking
        # are unaffected by the tilt axis, so this stays the cheap
        # shared-directory check even when force_aretomo forces every series
        # through the AreTomo2 stages below.
        todo = []
        for series in all_series:
            if force_aretomo:
                todo.append(series)
                continue
            frame_done = half_maps_present(config, series.name)
            tilt_done = (not do_tilt_split
                         or half_maps_present(config, series.name,
                                              tilt_split=True))
            if frame_done and tilt_done:
                log.skip(series.name, "half maps already present")
                continue
            todo.append(series)

        if not todo:
            log.note("Everything is already reconstructed. "
                     "Use --force (or overwrite: true) to redo it.")
            return 0
        log.note("%d tilt series to process, %d already complete"
                 % (len(todo), len(all_series) - len(todo)))

        # --- stage 1 -------------------------------------------------------
        banner("Motion correction and stacking")
        prepared = []
        for series in todo:
            banner("Preparing %s" % series.name)
            try:
                prep = prepare_series(config, series, log, force=force,
                                      dry_run=dry_run)
            except CommandError as exc:
                log.fail(series.name, str(exc))
                prep = None
            if prep is not None:
                prepared.append(prep)

        if not prepared:
            log.note("No tilt series could be prepared.")
            return 1

        # --- stage 2 -------------------------------------------------------
        tilt_axis = refine_tilt_axis(config, prepared, log, force=force_aretomo,
                                     dry_run=dry_run)

        # --- stages 3 and 4 ------------------------------------------------
        # ensure_full_alignment / frame_split_half_maps / tilt_split_half_maps
        # each check their own rec_<tomo>/aretomo_* outputs before doing any
        # work, so force_aretomo aside, calling them is always safe - a
        # tilt series only in ``todo`` because ONE half was missing does not
        # pay to redo the other.
        for prep in prepared:
            banner("Reconstructing %s" % prep.name)
            try:
                aln = ensure_full_alignment(config, prep, tilt_axis, log,
                                            force=force_aretomo, dry_run=dry_run)
                if aln is None:
                    if not dry_run:
                        log.fail(prep.name, "AreTomo2 produced no .aln - "
                                            "alignment failed")
                        continue
                    # Nothing ran, so name the file AreTomo2 would have written.
                    aln = os.path.join(prep.dirs.stacks,
                                       prep.name + "_stacked.aln")
                info("using alignment %s" % aln)

                ok = frame_split_half_maps(config, prep, aln, log,
                                           force=force_aretomo, dry_run=dry_run)

                if ok and do_tilt_split:
                    ok = tilt_split_half_maps(config, prep, aln, log,
                                              force=force_aretomo, dry_run=dry_run)

                if ok:
                    log.ok(prep.name)
            except CommandError as exc:
                log.fail(prep.name, str(exc))

        # --- final check -----------------------------------------------------
        # Fill in any shared-directory symlink still missing for a
        # reconstruction that already exists - covers tilt series this run
        # did not touch too (e.g. a previous run crashed after AreTomo2
        # finished but before linking).
        if not dry_run:
            for series in all_series:
                relink_existing_half_maps(config, series.name, do_tilt_split, log)

        log.note("")
        log.note("frame-split half maps: %s" % config.raw_halfmaps_dir)
        if do_tilt_split:
            log.note("tilt-split half maps:  %s"
                     % config.tiltsplit_halfmaps_dir)
        return 0 if log.succeeded else 1
