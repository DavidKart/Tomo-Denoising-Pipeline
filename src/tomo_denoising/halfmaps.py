"""Step 1: raw frames -> aligned tilt series -> tomogram + even/odd half maps.

MotionCor3 -> IMOD newstack -> AreTomo2.

Everything a tilt series produces lives in a single directory
``<processing_directory>/rec_<tomoName>``; the even/odd half maps are written
straight into ``<processing_directory>/raw_halfMaps``. Nothing is moved or
deleted afterwards - each program is either given an explicit output path, or
run with its working directory set to where its output belongs.
"""

import glob
import os
import re

import numpy as np

from .mdoc import collect_tilt_series, resolve_pixel_size
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


class TomoDirs(object):
    """The directory layout of a single ``rec_<tomoName>`` directory."""

    def __init__(self, config, tomo_name):
        self.root = ensure_dir(config.rec_dir(tomo_name))
        self.motioncor = ensure_dir(self.root, "motioncor")
        self.stacks = ensure_dir(self.root, "stacks")
        self.aretomo_full = ensure_dir(self.root, "aretomo_full")
        self.aretomo_even = ensure_dir(self.root, "aretomo_even")
        self.aretomo_odd = ensure_dir(self.root, "aretomo_odd")
        self.logs = ensure_dir(self.root, "logs")


# --------------------------------------------------------------- MotionCor3


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


def _motioncor_input_flag(extension):
    if extension == "eer":
        return "-InEer"
    if extension in ("tif", "tiff"):
        return "-InTiff"
    return "-InMrc"


def run_motioncor(config, series, dirs, dry_run=False):
    """Motion correct every frame of *series*; returns the averaged file names.

    Without ``tilt_split`` MotionCor3 also writes ``*_averaged_EVN.mrc`` and
    ``*_averaged_ODD.mrc`` next to each average (``-SplitSum 1``); those are the
    per-tilt half images the even/odd stacks are built from.
    """
    acquisition = config["acquisition"]
    mc = config["motioncor"]
    extension = acquisition["frame_extension"]
    gain = config.resolve_gain()
    gain_name = config.gain_name

    produced = []
    for frame in series.frames:
        if gain_name and os.path.basename(frame) == gain_name:
            continue
        if "_averaged" in frame:
            continue

        source = os.path.join(config.input_frames, frame)
        if not os.path.exists(source):
            warn("frame not found, skipping: %s" % source)
            continue

        stem = os.path.splitext(frame)[0]
        output = os.path.join(dirs.motioncor, stem + "_averaged.mrc")
        produced.append(os.path.basename(output))

        if os.path.exists(output) and not dry_run:
            info("already motion corrected: %s" % os.path.basename(output))
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
        if not config.tilt_split:
            command += ["-SplitSum", "1"]
        command += gpu_flag_motioncor(config.gpus).split()
        command += ["-LogFile", os.path.join(dirs.logs, stem + "_mc.log")]
        if str(mc["extra_args"]).strip():
            command += str(mc["extra_args"]).split()

        run(
            command,
            cwd=dirs.motioncor,
            log_file=os.path.join(dirs.logs, stem + "_motioncor.out"),
            check=False,
            dry_run=dry_run,
        )

    return produced


# --------------------------------------------------------------------- .rawtlt


def write_rawtlt(dirs, series, averaged_names, tomo_name, dry_run=False):
    """Write ``<tomoName>.rawtlt`` and return (rawtlt_path, ordered_averages).

    Tilts whose motion correction did not produce an average are dropped from
    both the angle file and the stack list, so the two stay in step.
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
        if not dry_run and not os.path.exists(os.path.join(dirs.motioncor, candidate)):
            warn("missing motion corrected tilt, excluded: %s" % candidate)
            continue
        ordered_names.append(candidate)
        ordered_angles.append(series.angles[index])

    rawtlt = os.path.join(dirs.root, tomo_name + ".rawtlt")
    if not dry_run:
        with open(rawtlt, "w", encoding="utf-8") as handle:
            for angle in ordered_angles:
                handle.write("%s\n" % angle)
    info("%d tilts, range %.2f to %.2f deg"
         % (len(ordered_angles),
            min(ordered_angles) if ordered_angles else 0.0,
            max(ordered_angles) if ordered_angles else 0.0))
    return rawtlt, ordered_names


# ------------------------------------------------------------------- newstack


def build_stack(config, dirs, members, output, dry_run=False):
    """Stack *members* (names relative to the motioncor directory) into *output*."""
    if not members:
        raise CommandError("no tilts available to stack into %s" % output)
    command = [config.exe("newstack")] + list(members) + [output]
    run(
        command,
        cwd=dirs.motioncor,
        log_file=os.path.join(dirs.logs, os.path.basename(output) + ".newstack.log"),
        dry_run=dry_run,
    )
    return output


# ------------------------------------------------------------------- AreTomo2


def _aretomo_common(config, apix_binned):
    aretomo = config["aretomo"]
    command = ["-VolZ", str(int(aretomo["vol_z"])),
               "-OutBin", str(int(aretomo["binning"])),
               "-DarkTol", str(aretomo["dark_tolerance"])]
    if apix_binned:
        command += ["-PixSize", "%.4f" % apix_binned]
    if aretomo["wbp"]:
        command += ["-Wbp", "1"]
    if aretomo["flip_vol"]:
        command += ["-FlipVol", "1"]
    local = str(aretomo["local_align"]).strip()
    if local:
        command += local.split()
    extra = str(aretomo["extra_args"]).strip()
    if extra:
        command += extra.split()
    return command


def _tilt_axis_flag(config):
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


def reconstruct_full(config, dirs, stack, rawtlt, tomo_name, apix_binned,
                     dry_run=False):
    """Full (non-split) reconstruction. This is the run that solves the alignment."""
    output = os.path.join(dirs.aretomo_full, tomo_name + "_rec_tomo.mrc")
    command = [config.exe("aretomo"),
               "-InMrc", stack,
               "-OutMrc", output,
               "-AngFile", rawtlt,
               "-OutXf", "1"]
    command += _aretomo_common(config, apix_binned)
    command += _tilt_axis_flag(config)
    run(command,
        cwd=dirs.aretomo_full,
        log_file=os.path.join(dirs.logs, tomo_name + "_aretomo_full.log"),
        dry_run=dry_run)
    return output


def reconstruct_half(config, dirs, stack, aln, output, log_name, apix_binned,
                     work_dir, dry_run=False):
    """Half-map reconstruction reusing the alignment solved on the full stack."""
    command = [config.exe("aretomo"),
               "-InMrc", stack,
               "-OutMrc", output,
               "-AlnFile", aln]
    command += _aretomo_common(config, apix_binned)
    command += _tilt_axis_flag(config)
    run(command,
        cwd=work_dir,
        log_file=os.path.join(dirs.logs, log_name),
        dry_run=dry_run)
    return output


# ---------------------------------------------------------------- tilt splitting


def split_aln_file(aln_path, even_out, odd_out):
    """Split an AreTomo .aln into an even-tilt and an odd-tilt half.

    Section indices and the RawSize header are renumbered so each half is a
    consistent description of half the sections.
    """
    with open(aln_path, encoding="utf-8", errors="replace") as source, \
            open(even_out, "w", encoding="utf-8") as even, \
            open(odd_out, "w", encoding="utf-8") as odd:
        count = 0
        for line in source:
            if line.startswith("#"):
                if "RawSize" in line:
                    match = re.findall(r"\d+", line)[2]
                    even.write(line.replace(match, str(int(np.ceil(int(match) / 2)))))
                    odd.write(line.replace(match, str(int(np.floor(int(match) / 2)))))
                    continue
                even.write(line)
                odd.write(line)
                continue
            match = re.search(r"\d+", line)
            if match is None:
                even.write(line)
                odd.write(line)
                continue
            token = match.group()
            if count % 2 == 0:
                even.write(line.replace(token, str(int(np.ceil(int(token) / 2)))))
            else:
                odd.write(line.replace(token, str(int(np.floor(int(token) / 2)))))
            count += 1


def split_stack_file(stack_path, even_out, odd_out):
    """Split a tilt series stack into even-numbered and odd-numbered tilts."""
    import mrcfile

    with mrcfile.open(stack_path, permissive=True) as source:
        data = source.data
        voxel_size = source.voxel_size

        with mrcfile.new(even_out, overwrite=True) as target:
            target.set_data(np.float32(data[::2, :, :]))
            target.voxel_size = voxel_size
        with mrcfile.new(odd_out, overwrite=True) as target:
            target.set_data(np.float32(data[1::2, :, :]))
            target.voxel_size = voxel_size


# ------------------------------------------------------------------ per series


def process_series(config, series, force=False, dry_run=False):
    """Run the full reconstruction for one tilt series."""
    tomo_name = series.name
    banner("Reconstructing %s" % tomo_name)

    raw_halfmaps = ensure_dir(config.raw_halfmaps_dir)
    even_map = os.path.join(raw_halfmaps, "even_%s_rec_tomo.mrc" % tomo_name)
    odd_map = os.path.join(raw_halfmaps, "odd_%s_rec_tomo.mrc" % tomo_name)

    if not force and os.path.exists(even_map) and os.path.exists(odd_map):
        info("half maps already exist, skipping (use --force to redo).")
        return True

    dirs = TomoDirs(config, tomo_name)

    first_frame = (
        os.path.join(config.input_frames, series.frames[0])
        if series.frames else None
    )
    apix = resolve_pixel_size(config, series, first_frame)
    apix_binned = apix * int(config["motioncor"]["binning"])
    info("pixel size %.4f A (unbinned) -> %.4f A after MotionCor3 binning"
         % (apix, apix_binned))

    # 1. motion correction ---------------------------------------------------
    averaged = run_motioncor(config, series, dirs, dry_run=dry_run)
    if not averaged:
        warn("no frames were motion corrected for %s - skipping." % tomo_name)
        return False

    # 2. angle file ----------------------------------------------------------
    rawtlt, ordered = write_rawtlt(dirs, series, averaged, tomo_name, dry_run=dry_run)
    if not ordered:
        warn("no usable tilts for %s - skipping." % tomo_name)
        return False

    # 3. full stack ----------------------------------------------------------
    full_stack = build_stack(
        config, dirs, ordered,
        os.path.join(dirs.stacks, tomo_name + "_stacked.mrc"),
        dry_run=dry_run,
    )

    # 4. full reconstruction, solves the alignment ---------------------------
    reconstruct_full(config, dirs, full_stack, rawtlt, tomo_name, apix_binned,
                     dry_run=dry_run)

    aln = find_aln(full_stack, dirs.aretomo_full, dirs.stacks)
    if aln is None and not dry_run:
        warn("no .aln produced for %s - AreTomo2 alignment failed, skipping "
             "half maps." % tomo_name)
        return False
    if aln is None:
        aln = os.path.join(dirs.stacks, tomo_name + "_stacked.aln")
    info("using alignment %s" % aln)

    # 5. half stacks ---------------------------------------------------------
    even_stack = os.path.join(dirs.stacks, "even_%s_stacked.mrc" % tomo_name)
    odd_stack = os.path.join(dirs.stacks, "odd_%s_stacked.mrc" % tomo_name)

    if config.tilt_split:
        info("tilt_split: splitting the aligned stack by tilt number")
        even_aln = os.path.join(dirs.stacks, "even_%s_stacked.aln" % tomo_name)
        odd_aln = os.path.join(dirs.stacks, "odd_%s_stacked.aln" % tomo_name)
        if not dry_run:
            split_aln_file(aln, even_aln, odd_aln)
            split_stack_file(full_stack, even_stack, odd_stack)
    else:
        even_aln = odd_aln = aln
        build_stack(config, dirs,
                    [name.replace("_averaged.mrc", "_averaged_EVN.mrc")
                     for name in ordered],
                    even_stack, dry_run=dry_run)
        build_stack(config, dirs,
                    [name.replace("_averaged.mrc", "_averaged_ODD.mrc")
                     for name in ordered],
                    odd_stack, dry_run=dry_run)

    # 6. half reconstructions ------------------------------------------------
    # AreTomo2 derives its side products (_projXY.mrc, _projXZ.mrc) from
    # -OutMrc, not from the working directory, so the reconstruction is written
    # inside rec_<tomo> and only the half map itself is linked into
    # raw_halfMaps. That keeps raw_halfMaps to exactly two files per tomogram
    # without moving or deleting anything.
    even_out = os.path.join(dirs.aretomo_even, "even_%s_rec_tomo.mrc" % tomo_name)
    odd_out = os.path.join(dirs.aretomo_odd, "odd_%s_rec_tomo.mrc" % tomo_name)

    reconstruct_half(config, dirs, even_stack, even_aln, even_out,
                     tomo_name + "_aretomo_even.log", apix_binned,
                     dirs.aretomo_even, dry_run=dry_run)
    reconstruct_half(config, dirs, odd_stack, odd_aln, odd_out,
                     tomo_name + "_aretomo_odd.log", apix_binned,
                     dirs.aretomo_odd, dry_run=dry_run)

    if not dry_run:
        for source, target in ((even_out, even_map), (odd_out, odd_map)):
            if not os.path.exists(source):
                warn("expected half map was not produced: %s" % source)
                return False
            link_file(source, target)
        info("linked into %s" % config.raw_halfmaps_dir)

    info("done: %s" % dirs.root)
    return True


# ------------------------------------------------------------------ entrypoint


def create_half_maps(config, force=False, dry_run=False):
    """``tomo_denoising create_halfMaps``"""
    for key, purpose in (("motioncor", "MotionCor3"),
                         ("newstack", "IMOD newstack"),
                         ("aretomo", "AreTomo2")):
        require_executable(config.exe(key), purpose)

    ensure_dir(config.processing_dir)
    ensure_dir(config.raw_halfmaps_dir)

    # Resolve up front: a missing gain would otherwise surface as a MotionCor3
    # failure on every single tilt.
    gain = config.resolve_gain()
    if gain:
        info("gain reference: %s" % gain)

    directory = config.mdoc_dir("preprocessing")
    all_series = collect_tilt_series(directory, "preprocessing")
    info("found %d tilt series in %s" % (len(all_series), directory))

    succeeded, failed = [], []
    for series in all_series:
        try:
            ok = process_series(config, series, force=force, dry_run=dry_run)
        except CommandError as exc:
            warn(str(exc))
            ok = False
        (succeeded if ok else failed).append(series.name)

    banner("create_halfMaps summary")
    info("%d of %d tilt series reconstructed" % (len(succeeded), len(all_series)))
    if failed:
        warn("failed or skipped: %s" % ", ".join(failed))
    info("half maps: %s" % config.raw_halfmaps_dir)
    return 0 if succeeded else 1
