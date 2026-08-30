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


def link_file(source, destination):
    """Make *source* reachable at *destination* without moving or copying.

    Tries a symlink first, then a hard link, and only copies as a last resort
    (e.g. on filesystems or platforms where neither is available).
    """
    source = os.path.abspath(source)
    if os.path.lexists(destination):
        # Already linked from a previous run - nothing to do.
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


class CommandError(Exception):
    """A external program returned a non-zero exit status."""


def run(command, cwd=None, log_file=None, check=True, dry_run=False, env=None):
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
        for line in process.stdout:
            sys.stdout.write(line)
            if handle:
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
        if handle:
            handle.close()

    if code != 0:
        message = "Command failed with exit status %d: %s" % (code, printable)
        if log_file:
            message += "\n  see %s" % log_file
        if check:
            raise CommandError(message)
        warn(message)
    return code


def capture(command, cwd=None):
    """Run a command and return its stdout, or None if it failed."""
    argv = command if isinstance(command, (list, tuple)) else shlex.split(command)
    try:
        return subprocess.check_output(
            [str(part) for part in argv],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


# ------------------------------------------------------------------ gpu flags


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
