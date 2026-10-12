"""Per-step reports and completion markers.

Every pipeline step writes one human readable report to
``<processing_directory>/logs/<step>.log`` saying what worked, what did not,
and which tilt series were affected. The same directory holds the markers the
resume logic reads when ``overwrite`` is false.
"""

import getpass
import os
import platform
import socket
import time

from .utils import ensure_dir, human_duration, info, warn

#: Written next to a step's log once it has finished without fatal errors.
_MARKER_SUFFIX = ".done"


def _timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class StepLog(object):
    """Collects the outcome of one pipeline step and writes it out.

    Used as a context manager so the report is written even when the step
    raises - an interrupted run leaves a log saying where it stopped, and no
    completion marker, so the resume logic knows to redo it.
    """

    def __init__(self, config, step, force=False):
        self.config = config
        self.step = step
        self.force = force
        self.path = os.path.join(ensure_dir(config.logs_dir), "%s.log" % step)
        self.lines = []
        self.succeeded = []
        self.failed = []
        self.skipped = []
        self.excluded = []
        self.tasks = {}
        self.fatal = None
        self._started = None

    # ------------------------------------------------------------- recording
    def note(self, text):
        """Add a free-form line to the report (also printed)."""
        self.lines.append(text)
        info(text)

    def quiet(self, text):
        """Add a line to the report without printing it."""
        self.lines.append(text)

    def task(self, name, ok, detail=""):
        """Record the outcome of a sub-task, e.g. ts_reconstruct on one tilt series."""
        bucket = self.tasks.setdefault(name, {"ok": [], "failed": {}})
        if ok:
            bucket["ok"].append(detail or "")
        else:
            bucket["failed"][detail or "?"] = True

    def ok(self, name, detail=""):
        self.succeeded.append(name)
        if detail:
            self.quiet("  OK    %s  %s" % (name, detail))
        else:
            self.quiet("  OK    %s" % name)

    def fail(self, name, reason):
        self.failed.append((name, reason))
        warn("%s: %s" % (name, reason))

    def skip(self, name, reason="already done"):
        self.skipped.append((name, reason))

    def exclude(self, name, reason):
        """A tilt series deliberately left out (see tiltfilter) - reported
        on its own, neither a failure nor a success."""
        self.excluded.append((name, reason))

    # --------------------------------------------------------------- context
    def __enter__(self):
        self._started = time.time()
        return self

    def __exit__(self, exc_type, exc, _tb):
        if exc is not None and exc_type is not KeyboardInterrupt:
            self.fatal = "%s: %s" % (exc_type.__name__, exc)
        elif exc_type is KeyboardInterrupt:
            self.fatal = "interrupted by the user"
        self.write()
        return False

    # ----------------------------------------------------------------- write
    def _summary_lines(self):
        out = []
        total = len(self.succeeded) + len(self.failed) + len(self.skipped)
        if total:
            out.append("")
            out.append("Tilt series: %d succeeded, %d failed, %d skipped "
                       "(%d total)"
                       % (len(self.succeeded), len(self.failed),
                          len(self.skipped), total))
        for name in sorted(self.tasks):
            bucket = self.tasks[name]
            done = len(bucket["ok"])
            bad = sorted(bucket["failed"])
            out.append("  %-24s %d of %d succeeded"
                       % (name, done, done + len(bad)))
            if bad:
                out.append("      failed for: %s" % ", ".join(bad))
        if self.skipped:
            out.append("")
            out.append("Skipped (already complete):")
            for name, reason in self.skipped:
                out.append("  %-24s %s" % (name, reason))
        if self.excluded:
            out.append("")
            out.append("EXCLUDED tilt series (not processed further):")
            for name, reason in self.excluded:
                out.append("  %-24s %s" % (name, reason))
        if self.failed:
            out.append("")
            out.append("FAILED:")
            for name, reason in self.failed:
                out.append("  %-24s %s" % (name, reason))
        return out

    def write(self):
        elapsed = time.time() - (self._started or time.time())
        header = [
            "=" * 78,
            "tomo_denoising  %s" % self.step,
            "=" * 78,
            "finished   : %s" % _timestamp(),
            "duration   : %s" % human_duration(elapsed),
            "host       : %s" % _host(),
            "config     : %s" % self.config.source,
            "processing : %s" % self.config.processing_dir,
            "overwrite  : %s" % ("forced on the command line" if self.force
                                 else self.config.overwrite),
            "",
        ]
        body = list(self.lines) + self._summary_lines()

        status = "OK"
        if self.fatal:
            status = "ABORTED - %s" % self.fatal
        elif self.failed:
            status = "COMPLETED WITH %d FAILURE(S)" % len(self.failed)
        footer = ["", "-" * 78, "RESULT: %s" % status, "-" * 78]

        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(header + body + footer) + "\n")

        marker = os.path.join(ensure_dir(self.config.state_dir),
                              self.step + _MARKER_SUFFIX)
        if self.fatal is None:
            with open(marker, "w", encoding="utf-8") as handle:
                handle.write("%s\n%s\n" % (_timestamp(), status))
        elif os.path.exists(marker):
            # A previously complete step was re-run and failed: the marker no
            # longer reflects reality, so make it say so rather than deleting it.
            with open(marker, "w", encoding="utf-8") as handle:
                handle.write("%s\nINVALIDATED: %s\n" % (_timestamp(), status))

        info("report: %s" % self.path)
        if self.failed:
            # Entries are tilt series AND whole sub-steps (e.g. "ts_aretomo3"),
            # so this is a count of problems, not of tilt series.
            warn("%d problem(s) reported - see the report." % len(self.failed))
        if self.excluded:
            warn("%d tilt series excluded: %s - see the report."
                 % (len(self.excluded), ", ".join(n for n, _ in self.excluded)))


def _host():
    try:
        return "%s@%s (%s)" % (getpass.getuser(), socket.gethostname(),
                               platform.platform())
    except Exception:  # pragma: no cover - never worth failing a run over
        return "unknown"


def step_completed(config, step):
    """True when *step* finished successfully and was not invalidated since."""
    marker = os.path.join(config.state_dir, step + _MARKER_SUFFIX)
    if not os.path.isfile(marker):
        return False
    with open(marker, encoding="utf-8", errors="replace") as handle:
        return "INVALIDATED" not in handle.read()
