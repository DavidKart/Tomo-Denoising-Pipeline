"""Shared helpers: directory creation, command execution and file linking.

Design rules for the whole package:

* No ``rm`` and no ``mv``. Every directory is created up front and every
  program is told where to write. Where a program insists on writing next to
  its input or into the current working directory, we ``cd`` into the
  destination instead of moving files afterwards.
* Half maps are made available to IsoNet by linking, never by copying or
  moving.
"""

import os
import shlex
import shutil
import subprocess
import sys
import threading
import time

# --------------------------------------------------------------------- output


def banner(text):
    line = "=" * 78
    print("\n%s\n%s\n%s" % (line, text, line), flush=True)


def info(text):
    print("[tomo_denoising] %s" % text, flush=True)


def warn(text):
    print("[tomo_denoising] WARNING: %s" % text, file=sys.stderr, flush=True)


# ----------------------------------------------------------------- filesystem


def ensure_dir(*parts):
    """Create a directory (and parents) and return its absolute path."""
    path = os.path.abspath(os.path.join(*parts))
    os.makedirs(path, exist_ok=True)
    return path


def _replace_stale_link(destination, source):
    """Remove *destination* so :func:`link_file` can re-point it at *source*.

    Returns False when nothing needs doing - either *destination* already
    refers to exactly *source*, or it is something we refuse to replace.

    This is the one place the package's "no rm" rule is relaxed, and only
    for a link/file WE created in an earlier run: the destination names in
    raw_halfMaps/ (``even_<tomo>_rec_tomo.mrc``) are fixed, while the
    physical reconstruction they point at is named by WarpTools after its
    own pixel size (``<tomo>_<angpix>Apx.mrc``). So a forced re-run at a
    different warptools.recon_angpix (or quality_check.
    tilt_split_target_pixel_size) writes a NEW file, and keeping the old
    link would silently leave every downstream step reading the previous
    pixel size's reconstruction - confirmed in production, and the reason
    *force* exists here at all.
    """
    try:
        if os.path.exists(destination) and os.path.samefile(destination, source):
            return False
    except OSError:
        # A broken symlink (exists() is False) or a filesystem that cannot
        # compare the two - fall through and replace it.
        pass
    if os.path.isdir(destination) and not os.path.islink(destination):
        warn("%s is a real directory, not a link this pipeline made - "
             "left untouched." % destination)
        return False
    try:
        os.unlink(destination)
    except OSError as exc:
        warn("could not replace %s: %s" % (destination, exc))
        return False
    return True


def link_file(source, destination, force=False):
    """Make *source* reachable at *destination* without moving or copying.

    Tries a symlink first, then a hard link, and only copies as a last resort
    (e.g. on filesystems or platforms where neither is available).

    *force* (the step's own ``--force``/``overwrite``) re-points an existing
    *destination* at *source* instead of keeping it - see
    :func:`_replace_stale_link` for why that is not optional.
    """
    source = os.path.abspath(source)
    if os.path.lexists(destination):
        if not force:
            # Already linked from a previous run - nothing to do.
            return destination
        if not _replace_stale_link(destination, source):
            return destination
    ensure_dir(os.path.dirname(destination))
    try:
        os.symlink(source, destination)
        return destination
    except (OSError, NotImplementedError, AttributeError):
        pass
    try:
        os.link(source, destination)
        return destination
    except OSError:
        pass
    shutil.copy2(source, destination)
    return destination


def executable_available(name):
    """True when *name* can actually be run, without warning about it.

    The silent counterpart of :func:`require_executable`, for the genuinely
    OPTIONAL dependencies - currently only IMOD's ``header`` (see
    ``mdoc.resolve_pixel_size``), which is never needed at all unless the
    pixel size has to be read from the data AND the mdocs do not carry it.
    """
    if os.path.sep in name or (os.path.altsep and os.path.altsep in name):
        return os.path.isfile(name)
    return shutil.which(name) is not None


def require_executable(name, purpose):
    """Warn (but do not abort) if an executable cannot be found on PATH."""
    if os.path.sep in name or (os.path.altsep and os.path.altsep in name):
        if not os.path.isfile(name):
            warn("%s (%s) does not exist." % (name, purpose))
            return False
        return True
    if shutil.which(name) is None:
        warn(
            "%s (%s) was not found on PATH. Activate the right environment "
            "or set an absolute path in executables." % (name, purpose)
        )
        return False
    return True


# -------------------------------------------------------------------- running


def human_duration(seconds):
    """``"3h 07m 21s"`` / ``"7m 21s"`` / ``"21s"``."""
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return "%dh %02dm %02ds" % (hours, minutes, secs)
    if minutes:
        return "%dm %02ds" % (minutes, secs)
    return "%ds" % seconds


#: How long a command may run without printing anything before :func:`run`
#: mentions it, in seconds.
#:
#: Deliberately long. The steps this pipeline runs are genuinely quiet for long
#: stretches - a miss-alignment training iteration or a large ts_reconstruct
#: can work for a long time between messages - so a short threshold would mean
#: a stream of notices about perfectly healthy runs, which is how people learn
#: to ignore them. At an hour, a notice means "this has been quiet longer than
#: usual", which is worth reading.
QUIET_NOTICE_SECONDS = 3600.0

#: How often the watcher wakes up to compare timestamps. Only affects how
#: promptly a notice appears, never whether one does.
_QUIET_TICK_SECONDS = 30.0


def _command_label(argv):
    """A readable name for the program in *argv*, for the quiet notice.

    Looks past a ``conda run -n <env> --no-capture-output`` wrapper so the
    notice names the program that is actually quiet rather than "conda".
    """
    if not argv:
        return "the command"
    if "--no-capture-output" in argv:
        index = argv.index("--no-capture-output")
        if index + 1 < len(argv):
            return os.path.basename(argv[index + 1])
    return os.path.basename(argv[0])


def _watch_for_quiet(state, stop, label, pid, log_file, quiet_after, lock, handle):
    """Say so when a running command has printed nothing for a long time.

    Purely informational - this NEVER kills, signals or interferes with the
    process. The point is that a stalled run and a busy one look identical
    from outside when both are silent, and the stalled one can otherwise
    burn a whole night before anyone notices.
    """
    while not stop.wait(min(_QUIET_TICK_SECONDS, quiet_after)):
        silent = time.monotonic() - state["last"]
        if silent < quiet_after or silent - state["notified"] < quiet_after:
            continue
        state["notified"] = silent
        message = (
            "%s has printed nothing for %s. This is NOT in itself a sign that "
            "anything is wrong - these steps are often legitimately quiet for "
            "long stretches, and the run is still going. It is worth a look if "
            "the gap keeps growing: the process is PID %d%s. "
            "'nvidia-smi' shows whether the GPUs are still busy, and "
            "'py-spy dump --pid %d' shows what it is currently doing."
            % (label, human_duration(silent), pid,
               ", its log is %s" % log_file if log_file else "", pid))
        info(message)
        if handle is not None:
            with lock:
                # Re-checked under the lock: run()'s finally sets *stop* and
                # then closes the handle while holding this same lock, so
                # without it a notice landing at exactly the wrong moment
                # would write to a closed file and leave a thread traceback
                # at the end of an otherwise successful run.
                if not stop.is_set() and not handle.closed:
                    handle.write("[tomo_denoising] %s\n" % message)
                    handle.flush()


class CommandError(Exception):
    """A external program returned a non-zero exit status."""


def run(command, cwd=None, log_file=None, check=True, dry_run=False, env=None,
        quiet_after=None):
    """Run an external command.

    Parameters
    ----------
    command : str or list
        Command line. A string is split with :func:`shlex.split`.
    cwd : str, optional
        Working directory. Used to control where programs drop the auxiliary
        files they insist on writing next to their inputs.
    log_file : str, optional
        stdout+stderr are written here in addition to being streamed to the
        console.
    check : bool
        Raise :class:`CommandError` on a non-zero exit status.
    env : dict, optional
        Extra/overriding environment variables (e.g. ``CUDA_VISIBLE_DEVICES``),
        merged on top of the current process environment. The subprocess
        still inherits everything else (PATH, activated conda env, ...).
    quiet_after : float, optional
        Seconds of no output after which a notice is printed, repeating at
        that interval for as long as the silence lasts (default
        :data:`QUIET_NOTICE_SECONDS`; 0 or negative disables it). Purely
        informational - nothing is ever killed on account of being quiet.
    """
    if isinstance(command, (list, tuple)):
        argv = [str(part) for part in command if str(part) != ""]
    else:
        argv = shlex.split(command)

    printable = " ".join(shlex.quote(part) for part in argv)
    if env:
        printable = " ".join("%s=%s" % (k, shlex.quote(str(v)))
                             for k, v in env.items()) + " " + printable
    where = " (in %s)" % cwd if cwd else ""
    info("$ %s%s" % (printable, where))

    if dry_run:
        return 0

    if cwd:
        ensure_dir(cwd)
    if log_file:
        ensure_dir(os.path.dirname(log_file))

    # PYTHONUNBUFFERED=1 unconditionally: when a child process's stdout is a
    # pipe (always true here - doubly so through a conda run wrapper),
    # CPython block-buffers it by default, so a Python tool's own progress
    # output can sit unflushed for a long time - looking exactly like a
    # hang, with an empty log file, even while real work is happening. This
    # is a no-op for non-Python executables (WarpTools, AreTomo, ...), which
    # simply ignore an environment variable they don't recognize.
    full_env = dict(os.environ)
    full_env["PYTHONUNBUFFERED"] = "1"
    if env:
        full_env.update({k: str(v) for k, v in env.items()})

    handle = open(log_file, "w", encoding="utf-8", errors="replace") if log_file else None
    quiet_after = QUIET_NOTICE_SECONDS if quiet_after is None else float(quiet_after)
    stop_watching = threading.Event()
    write_lock = threading.Lock()
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        if quiet_after > 0:
            state = {"last": time.monotonic(), "notified": 0.0}
            threading.Thread(
                target=_watch_for_quiet,
                args=(state, stop_watching, _command_label(argv), process.pid,
                      log_file, quiet_after, write_lock, handle),
                daemon=True,
            ).start()
        else:
            state = None
        for line in process.stdout:
            if state is not None:
                state["last"] = time.monotonic()
                state["notified"] = 0.0
            sys.stdout.write(line)
            if handle:
                with write_lock:
                    handle.write(line)
        process.stdout.close()
        code = process.wait()
    except FileNotFoundError:
        raise CommandError("Executable not found: %s" % argv[0])
    except PermissionError:
        # Something exists on PATH under this name but the OS refused to
        # execute it directly - the classic case is "conda", which on many
        # systems is only a shell FUNCTION defined by sourcing conda.sh
        # (e.g. in .bashrc), not a real executable file; a plain subprocess
        # call bypasses that shell integration entirely and finds nothing
        # runnable at that name.
        raise CommandError(
            "Permission denied executing: %s\n"
            "  It exists on PATH but is not directly runnable - if this is "
            "'conda', it is very likely a shell FUNCTION on this system "
            "(defined by sourcing conda.sh), not a real executable. Point "
            "the relevant 'executables' entry in your config at the actual "
            "conda binary instead, e.g. <conda_install_dir>/condabin/conda "
            "or <conda_install_dir>/bin/conda." % argv[0]
        )
    finally:
        # Before the handle is closed, so the watcher can never write into a
        # closed file.
        stop_watching.set()
        if handle:
            with write_lock:
                handle.close()

    if code != 0:
        message = "Command failed with exit status %d: %s" % (code, printable)
        if log_file:
            message += "\n  see %s" % log_file
        if check:
            raise CommandError(message)
        warn(message)
    return code


def capture(command, cwd=None, env=None):
    """Run a command and return its stdout, or None if it failed.

    Takes *env* for the same reason :func:`run` does - so that "every
    subprocess this pipeline starts is restricted to the configured GPUs"
    holds without an exception to remember. Its only current caller runs
    IMOD's ``header``, which never touches a GPU.
    """
    argv = command if isinstance(command, (list, tuple)) else shlex.split(command)
    full_env = None
    if env:
        full_env = dict(os.environ)
        full_env.update({k: str(v) for k, v in env.items()})
    try:
        return subprocess.check_output(
            [str(part) for part in argv],
            cwd=cwd,
            env=full_env,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


# ------------------------------------------------------------------ gpu flags
#
# EVERY external program this pipeline runs is launched with
# CUDA_VISIBLE_DEVICES set from the config's single ``gpus`` list (see
# :func:`gpu_env`) - WarpTools, our own AreTomo3 axis-search calls, native
# ts_aretomo3, RESOLVE, IsoNet and miss-alignment alike. That is the one
# mechanism every CUDA program honours, whether or not it has a GPU option
# of its own, so nothing can wander onto a GPU someone else is using.
#
# The consequence to keep in mind everywhere below: inside such a process the
# GPUs are RENUMBERED 0..N-1, in the order they appear in ``gpus``. A program
# that also takes GPU ids as arguments (IsoNet's --gpuID, miss-alignment's
# --training-devices/--reconstruction-devices, AreTomo3's -Gpu) therefore has
# to be given those LOGICAL ids, not the real ones from the config - see
# :func:`logical_gpu_ids`.


def gpu_env(gpus):
    """The environment every external call gets: only *gpus* are visible.

    CUDA_DEVICE_ORDER is pinned to PCI_BUS_ID so that the numbers written in
    the config mean the same cards everywhere, and the same cards nvidia-smi
    shows. Without it, CUDA's default ordering (fastest device first) can
    number a mixed set of GPUs differently from nvidia-smi - and differently
    from IsoNet, which sets PCI_BUS_ID for itself regardless (confirmed in
    its own source), so "GPU 2" could quietly mean one card in WarpTools and
    another in IsoNet on the same machine.
    """
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus),
    }


#: How each program interprets the GPU ids it is handed on its command line.
#: This is the ONE place that knowledge lives - every GPU argument anywhere in
#: the package is produced by :func:`gpu_ids_for` from this table and the
#: config's own ``gpus`` list, so no call site invents a numbering of its own.
#:
#: "logical" - the id is used as a CUDA device index INSIDE the process, i.e.
#:             after CUDA_VISIBLE_DEVICES has already renumbered the devices
#:             0..N-1. Verified in each program's source:
#:               AreTomo3        cudaSetDevice on the visible set
#:               miss-alignment  torch.device(f"cuda:{device}")  (preprocessing.py)
#:               RESOLVE         torch.device(f"cuda:{gpu_id}")  (src/resolve.py)
#:               WarpTools       --device_list, documented as "Space-separated
#:                               list of GPU IDs to use for processing. Default:
#:                               all GPUs in the system" - and "the system" for a
#:                               process is whatever CUDA_VISIBLE_DEVICES left
#:                               visible, so these are indices into that set.
#:             None of these ever writes CUDA_VISIBLE_DEVICES itself.
#:
#: "real"    - the id is the real, machine-wide GPU number, because the program
#:             ASSIGNS it to CUDA_VISIBLE_DEVICES itself rather than reading it
#:             as an index. Only IsoNet does this (bin/refine.py, bin/predict.py:
#:             os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuID), and handing it
#:             logical ids puts it on the WRONG physical GPUs, silently.
#:
#: None      - the program has no GPU argument at all; CUDA_VISIBLE_DEVICES is
#:             the only control. Nothing is in this state at the moment - it is
#:             kept because it is a real possibility for a program added later,
#:             and because assuming "no GPU option" without checking is exactly
#:             the mistake that left WarpTools relying on its own default here
#:             for longer than it should have.
GPU_ID_CONVENTION = {
    "aretomo": "logical",
    "missalign": "logical",
    "resolve": "logical",
    "warptools": "logical",
    "isonet": "real",
}


def gpu_ids_for(program, gpus):
    """The GPU ids to put on *program*'s command line, derived from *gpus*.

    *gpus* is always the config's own list of real GPU numbers - that is the
    single source, for every program. What differs is only how each one reads
    the number it is given; see :data:`GPU_ID_CONVENTION` for which does what
    and why. Returns [] for a program that takes no GPU argument.
    """
    convention = GPU_ID_CONVENTION.get(program)
    if convention is None:
        return []
    if convention == "real":
        return [int(g) for g in gpus]
    return logical_gpu_ids(gpus)


def gpu_id_note(program, gpus):
    """One log line explaining what *program* was given, and why.

    Printed by every step that passes a GPU argument, so the translation is
    never something to reconstruct from the command line after the fact: a
    config of ``gpus: [2, 3]`` legitimately appears as ``-Gpu 0 1`` for
    AreTomo3 and ``--gpuID 2,3`` for IsoNet, and that difference should be
    visible and explained rather than surprising.
    """
    gpus = [int(g) for g in gpus]
    convention = GPU_ID_CONVENTION.get(program)
    if convention is None:
        return ("GPUs %s (from the config) - %s has no GPU option; restricted "
                "by CUDA_VISIBLE_DEVICES alone"
                % (",".join(str(g) for g in gpus), program))
    ids = ",".join(str(g) for g in gpu_ids_for(program, gpus))
    if convention == "real":
        return ("GPUs %s (from the config) -> %s gets %s: it assigns its GPU "
                "argument to CUDA_VISIBLE_DEVICES itself, so it needs the real "
                "numbers" % (",".join(str(g) for g in gpus), program, ids))
    return ("GPUs %s (from the config) -> %s gets %s: CUDA_VISIBLE_DEVICES "
            "renumbers them 0..N-1 inside the process"
            % (",".join(str(g) for g in gpus), program, ids))


def logical_gpu_ids(gpus):
    """The ids *gpus* have INSIDE a process started with :func:`gpu_env`.

    Always ``[0, 1, ... len(gpus)-1]``: CUDA_VISIBLE_DEVICES renumbers the
    devices it exposes, so a config of ``gpus: [2, 3]`` means a child process
    sees exactly two GPUs, called 0 and 1.
    """
    return list(range(len(list(gpus))))


def to_logical_gpu_ids(gpus, requested, setting="value"):
    """Translate real GPU ids from the config into their logical ids.

    Used for the settings a user writes as real GPU ids (missalign.
    training_devices / reconstruction_devices) but which are consumed inside
    a CUDA_VISIBLE_DEVICES-restricted process. Repetitions are preserved -
    miss-alignment uses them to ask for several processes on one GPU.
    """
    gpus = [int(g) for g in gpus]
    index = {gpu: position for position, gpu in enumerate(gpus)}
    out = []
    for value in requested:
        gpu = int(value)
        if gpu not in index:
            raise SystemExit(
                "ERROR: %s names GPU %d, which is not in the 'gpus' list "
                "(%s).\n"
                "Only the GPUs listed there are made visible to any program "
                "this pipeline runs." % (setting, gpu, gpus)
            )
        out.append(index[gpu])
    return out


def gpu_flag_isonet(gpus):
    return ",".join(str(g) for g in gpus)


# ------------------------------------------------------------- conda env wrap


def wrap_conda_run(command, conda_exe, env):
    """Prefix *command* with ``conda run`` so it executes inside *env*,
    without needing that environment activated in the current shell first.

    *env* may be a name (``-n``) or a path to an env prefix (``-p``) -
    whichever it looks like is used. Returns *command* unchanged if *env* is
    empty, so this is safe to call unconditionally.

    ``--no-capture-output`` matters here: without it, ``conda run`` buffers
    the whole subprocess and only prints it once the process exits, which
    would silently break :func:`run`'s live, line-by-line streaming/logging.
    """
    if not env:
        return list(command)
    flag = "-p" if (os.sep in env or (os.altsep and os.altsep in env)
                   or os.path.isdir(env)) else "-n"
    return [conda_exe, "run", flag, env, "--no-capture-output"] + list(command)
