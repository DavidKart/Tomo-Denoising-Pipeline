"""Step: MissAlignment-refined reconstruction (``run_missalignment``).

Runs after ``reconstruct_tomograms``, refining the SAME per-tomogram
alignment ``align_tiltseries`` (native ``WarpTools ts_aretomo3``) already
produced in ``warp_tiltseries/<tomo>.xml``. MissAlignment's own per-tilt
local deformation model is something neither AreTomo nor plain ts_aretomo3
can represent, so it trains directly against that project and modifies each
XML IN PLACE:

  1. Back up every ``warp_tiltseries/<tomo>.xml`` that has no backup yet
     to ``warp_tiltseries_prealign/<tomo>.xml`` (tagged with the alignment
     it belongs to); with --force, restore every backed-up one FROM it, so a
     forced run restarts from the true pre-MissAlignment state (see
     :func:`backup_alignment`). Whenever ``align_tiltseries`` re-aligns, it
     first restores the XMLs from the backups and stops if ``iterN/``
     training state is left over (:func:`reset_for_realignment`); the next
     run retakes the backups from the new alignment - nothing is ever
     deleted.
  2. Patch every ``ImageDimensionsAngstrom``/``VolumeDimensionsAngstrom`` on
     the TiltSeries XMLs - real, native ``ts_aretomo3`` output can leave these
     at ``[0.0, 0.0]``, which MissAlignment refuses to train on. The IMAGE
     dimensions always come from the real camera frame (a frame-series
     average); the VOLUME dimensions use the same ``tomo_dimensions``
     ``align_tiltseries`` already resolved, unless
     ``missalign.training_volume_dimensions`` is set to deliberately tighten
     the volume (see MissAlignment's own docs: "the tomogram shape should
     tightly fit your sample ... to avoid training on empty regions").
  3. Write ``warp_tiltseries/missalign_config.yaml`` and run
     ``miss-alignment train``.
  4. Reconstruct the refined project with genuine ``WarpTools
     ts_reconstruct`` (the shared :func:`warptools.run_ts_reconstruct`) - the
     local grid is only actually applied by Warp's own reconstruction
     engine, not by AreTomo. The full tomogram and the frame-split (for
     IsoNet) half maps are produced here, plus the tilt-split ones when
     ``quality_check.enable_tilt_split`` asks for them -
     this is the ONLY reconstruction that happens when MissAlignment is part
     of the pipeline (unless ``--produce-intermediates`` also reconstructed
     the pre-MissAlignment ones); see :mod:`pipeline`. Written into its own
     ``missalignment/reconstruction/`` (``config.
     missalignment_reconstruction_dir``; one ``--output_processing`` per
     ts_reconstruct call under it - see
     :func:`warptools.run_ts_reconstruct`) - deliberately NOT the
     same ``warp_tiltseries/reconstruction_uncorrected/``
     ``reconstruct_tomograms`` itself uses, since sharing one used to mean
     this step's reconstruction silently overwrote the physical files
     raw_halfMaps's symlinks already pointed to.

Its reconstructions and half maps live under
``<processing_directory>/missalignment/``; in ``warp_tiltseries/`` it only
refines the XMLs in place (the point of the step) and adds links: the full
tomograms in ``warp_tiltseries/tomos_MissAlignment/``, and the same under
WarpTools' own names in ``warp_tiltseries/reconstruction/``, where its later
commands look for them. The uncorrected ``tomos_uncorrected``/
``raw_halfMaps``/``raw_halfMaps_tiltSplit`` there stay exactly as
``reconstruct_tomograms`` left them (or are never populated at all, when
MissAlignment runs without ``--produce-intermediates`` - see
:mod:`pipeline`). ``missalignment/`` gets ``raw_halfMaps`` and
``raw_halfMaps_tiltSplit`` - the analogue of ``warp_tiltseries/``'s own,
just MissAlignment-corrected (``raw_halfMaps_tiltSplit`` only when
``quality_check.enable_tilt_split`` is set). Those half maps are what
``halfmap_sets.resolve_frame_halfmaps_dir``/
``resolve_quality_check_halfmaps_dir`` prefer by default for denoising and
quality checking.

NOTE on the ``ImageDimensionsAngstrom``/``VolumeDimensionsAngstrom`` patch:
MissAlignment's own reference material (``update_warp_xml.py``) patches
these two attributes on the ``<TiltSeries>`` root element directly with
plain torch/warpylib code, because real, native ``ts_aretomo3`` output can
leave them at ``[0.0, 0.0]`` (confirmed directly: MissAlignment raises
``ValueError: ... has zero values in 'ImageDimensionsAngstrom'`` on such a
project otherwise) - this is NOT a rare edge case to guard against, it is
routine and this patch is not optional. :func:`patch_training_dimensions`
below patches the same two attributes with a small, dependency-free text
substitution, so no warpylib dependency is needed.
"""

import os
import re
import shlex
import shutil
import sys

import yaml

from . import tiltfilter
from .mdoc import collect_tilt_series
from .steplog import StepLog
from .utils import (
    banner, ensure_dir, gpu_env, gpu_id_note, gpu_ids_for, info,
    run as run_command, to_logical_gpu_ids, warn,
)
from .warptools import (
    aligned_series_names, alignment_id, dataset_apix, frame_image_xy,
    resolve_recon_angpix, resolve_tomo_dimensions, run_ts_reconstruct,
    used_tilt_angles,
)

STEP = "run_missalignment"

#: The exact iteration schedule from MissAlignment's own reference config
#: (MissAlignment.txt / config_template.yaml): three coarse-to-fine global
#: passes, then four local 3x3-grid refinement passes.
_ITERATION_SETTINGS = [
    {"downsample": 3, "alignment": "anchoring"},
    {"downsample": 2, "alignment": "anchoring"},
    {"downsample": 1, "alignment": "global"},
    {"downsample": 1, "alignment": "global"},
    {"downsample": 1, "alignment": [3, 3]},
    {"downsample": 1, "alignment": [3, 3]},
    {"downsample": 1, "alignment": [3, 3]},
    {"downsample": 1, "alignment": [3, 3]},
]

_ITER_DIR_RE = re.compile(r"^iter(\d+)$")


# ================================================================ NCCL transport
#
# Multi-GPU training runs DDP over NCCL, and on servers whose GPUs sit behind
# different PCIe switches NCCL's peer-to-peer transport can stall instead of
# failing - miss-alignment documents this itself ("NCCL hang with multiple GPUs
# on HPC servers"). Observed in production here: both training GPUs holding a
# few GB with 0% utilisation, the reconstruction devices never allocating
# anything, iter0/ written but not one loss value, and nothing in the log, for
# hours.
#
# miss-alignment's own fix is NCCL_P2P_DISABLE=1. That works, but it is blunter
# than the problem: NVIDIA's documentation is explicit that P2P is the
# transport "which uses CUDA direct access between GPUs, using NVLink or PCI",
# so disabling it gives up NVLink too, on machines where NVLink is present and
# working fine.
#
# NCCL_P2P_LEVEL is the targeted form - "the maximum distance between GPUs
# where NCCL will use the P2P transport":
#
#   LOC  never use P2P                      (same as NCCL_P2P_DISABLE=1)
#   NVL  only between NVLink-connected GPUs
#   PIX  only within one PCI switch         <- the default here
#   PXB  across PCI switches, several hops
#
# PIX keeps the fast paths that work and drops P2P exactly where it has to
# cross the CPU's root complex, which is the case that hangs. Use LOC if PIX
# turns out not to be enough on a given machine, or "" to leave NCCL alone.
#
#: Values accepted for missalign.nccl_p2p, mapped to what they actually set.
_NCCL_P2P_LEVELS = ("LOC", "NVL", "PIX", "PXB", "PHB", "SYS")
_NCCL_P2P_OFF = ("OFF", "DISABLE", "DISABLED", "FALSE", "NO")


def resolve_nccl_env(config, log=None):
    """Environment variables for ``missalign.nccl_p2p``, or ``{}``.

    Only consulted for multi-GPU training - with a single training device
    there is no DDP and no NCCL, so the setting is irrelevant either way (it
    is still exported; NCCL simply never reads it).
    """
    raw = str(config["missalign"].get("nccl_p2p") or "").strip()
    if not raw:
        if log is not None:
            log.note("NCCL: missalign.nccl_p2p is empty - NCCL's own defaults "
                     "are left alone")
        return {}

    upper = raw.upper()
    if upper in _NCCL_P2P_OFF:
        env = {"NCCL_P2P_DISABLE": "1"}
        detail = ("NCCL_P2P_DISABLE=1 - peer-to-peer transport off entirely, "
                  "NVLink included")
    elif upper in _NCCL_P2P_LEVELS or upper.isdigit():
        env = {"NCCL_P2P_LEVEL": upper}
        detail = ("NCCL_P2P_LEVEL=%s - peer-to-peer transport used only up to "
                  "that distance between GPUs" % upper)
    else:
        raise SystemExit(
            "ERROR: missalign.nccl_p2p is %r, which is neither a "
            "NCCL_P2P_LEVEL value (%s, or a number), nor one of %s to disable "
            "peer-to-peer entirely, nor empty to leave NCCL alone."
            % (raw, ", ".join(_NCCL_P2P_LEVELS),
               "/".join(v.lower() for v in _NCCL_P2P_OFF[:2]))
        )

    if log is not None:
        log.note("NCCL: %s (missalign.nccl_p2p: %r)" % (detail, raw))
    return env


# ============================================================== memory profile
#
# MissAlignment's own reference settings assume a 24 GB card. Everything below
# that needs a second, smaller profile - and WHICH settings to shrink is not
# obvious, so here is what was actually checked against MissAlignment's own
# documented config:
#
#   data_loading.batch_size (32 @ 24 GB)
#       Patches held on the TRAINING device at once. The dominant term, and
#       the one MissAlignment itself says to change ("reduce it for smaller
#       cards"). Scaled below.
#
#   tilt_series_alignment.batch_size (32 @ 24 GB)
#       "amount of patches simultaneously reconstructed in memory" on the
#       RECONSTRUCTION devices - a separate pool from the training one, and
#       genuinely just throughput ("the more the merrier"), with no effect on
#       the result. Scaled below, now as its own setting rather than silently
#       sharing the training value.
#
#   how many reconstruction PROCESSES share one GPU (3 @ 24 GB, from every
#   runMiss_align_*.sh example: "--reconstruction-devices 2,2,2,3,3,3")
#       Easy to miss and, on a small card, decisive: each process carries its
#       own CUDA context and reconstruction buffers, so three of them is a
#       fixed cost that does NOT shrink with batch_size. On a SINGLE 12 GB GPU
#       the reference partitioning puts one trainer plus three reconstruction
#       processes on the same card - which is the most likely reason 12 GB did
#       not run even with the batch size already halved.
#
#   dataloaders_per_trainer (5)
#       Host-side workers, but each one drives reconstruction work, so fewer
#       of them also means less concurrent GPU demand.
#
# Deliberately NOT scaled:
#
#   patch_size (96, both sections)
#       Memory scales with its CUBE, so this is by far the most tempting knob
#       - and the wrong one. It is the model's input size ("model_architecture:
#       default"), and MissAlignment requires the alignment patch_size to
#       match the training one; changing it changes what the network is,
#       not just how much of it fits.
#
#   steps_per_epoch / max_epochs_per_iteration / the iteration schedule
#       Runtime, not memory.
#
#   prepare_stacks_gb
#       Host-side scratch for the pre-extracted stacks, nothing to do with
#       VRAM - left as its own config setting.

#: VRAM per GPU (GB) MissAlignment's own reference settings are written for.
REFERENCE_GPU_MEMORY_GB = 24.0

#: Below this there is nothing sensible left to shrink, so a smaller value is
#: treated as 12 and warned about rather than scaled into uselessness.
MIN_GPU_MEMORY_GB = 12.0

#: MissAlignment's own documented values, used unchanged at >= 24 GB.
REFERENCE_PROFILE = {
    "batch_size": 32,
    "alignment_batch_size": 32,
    "reconstruction_processes_per_gpu": 3,
    "dataloaders_per_trainer": 5,
}


def _memory_profile(config, log=None):
    """The memory-dependent settings for this machine.

    At ``gpu_memory_gb >= 24`` this is MissAlignment's own reference
    configuration, byte for byte - the settings that are known to work are not
    second-guessed. Below it, the batch sizes are interpolated linearly down to
    8 at 12 GB (rather than scaled proportionally from 32, which ignores the
    1-2 GB of per-process overhead that does not shrink with the card), and the
    number of reconstruction processes sharing each GPU is reduced too.

    Any of these can be pinned explicitly in the config, which always wins.
    """
    settings = config["missalign"]
    requested = float(settings["gpu_memory_gb"])
    effective = max(requested, MIN_GPU_MEMORY_GB)

    if effective >= REFERENCE_GPU_MEMORY_GB:
        profile = dict(REFERENCE_PROFILE)
        profile["name"] = ("reference (>= %g GB per GPU - MissAlignment's own "
                           "documented settings, unchanged)"
                           % REFERENCE_GPU_MEMORY_GB)
    else:
        if requested < MIN_GPU_MEMORY_GB:
            warn("missalign.gpu_memory_gb is %g, below the %g GB this "
                 "pipeline's low-memory profile is designed for - treating it "
                 "as %g GB. MissAlignment may still run out of memory."
                 % (requested, MIN_GPU_MEMORY_GB, MIN_GPU_MEMORY_GB))
        # 8 at 12 GB, 32 at 24 GB. Anchored at both ends so there is no jump
        # where the two profiles meet.
        batch = int(round(8 + (effective - MIN_GPU_MEMORY_GB) * 2.0))
        batch = max(4, min(batch, REFERENCE_PROFILE["batch_size"]))
        if effective < 16.0:
            per_gpu, dataloaders = 1, 3
        elif effective < 22.0:
            per_gpu, dataloaders = 2, 4
        else:
            per_gpu, dataloaders = 3, 5
        profile = {
            "batch_size": batch,
            "alignment_batch_size": batch,
            "reconstruction_processes_per_gpu": per_gpu,
            "dataloaders_per_trainer": dataloaders,
            "name": "low memory (%g GB per GPU)" % requested,
        }

    overridden = []
    for key in ("batch_size", "alignment_batch_size",
                "reconstruction_processes_per_gpu", "dataloaders_per_trainer"):
        value = settings.get(key)
        if value:
            profile[key] = int(value)
            overridden.append("%s=%d" % (key, int(value)))

    if log is not None:
        log.note("MissAlignment memory profile: %s" % profile["name"])
        log.note("  data_loading.batch_size              %d" % profile["batch_size"])
        log.note("  tilt_series_alignment.batch_size     %d"
                 % profile["alignment_batch_size"])
        log.note("  reconstruction processes per GPU     %d"
                 % profile["reconstruction_processes_per_gpu"])
        log.note("  dataloaders per trainer             %d"
                 % profile["dataloaders_per_trainer"])
        if overridden:
            log.note("  pinned explicitly in the config: %s"
                     % ", ".join(overridden))
    return profile


# ------------------------------------------------------ tilt-count pre-flight


#: miss-alignment's own floor: its reconstruction workers generate a shift
#: trajectory with one point per tilt, and ``generate_shift_trajectory``
#: raises ``ValueError: num_points must be at least 4`` below this. Worse
#: than the error itself, the dying workers do not take training down with
#: them - it waits for their reconstructions forever, silently (seen in
#: production: 8+ hours, stuck at step 110 of iteration 1).
_MIN_TILTS = 4


def check_excluded_xmls(config, log):
    """Stop if XMLs of excluded tilt series are still in warp_tiltseries/.

    MissAlignment trains on EVERY XML in its training directory, not on a
    list - and a series a WarpTools step gave up on keeps its XML there
    (WarpTools only marks it unselected; whether MissAlignment honours that
    is not known). This
    pipeline never deletes files itself, so the files are listed for the
    user to remove. Returns True if there are none.
    """
    included = set(aligned_series_names(config))
    leftover = [name for name in aligned_series_names(config, include_excluded=True)
                if name not in included]
    if not leftover:
        return True
    record = tiltfilter.read_record(config)
    paths = []
    for name in leftover:
        paths.append(os.path.join(config.warp_tiltseries_dir, name + ".xml"))
        tomostar = os.path.join(config.tomostar_dir, name + ".tomostar")
        if os.path.isfile(tomostar):
            paths.append(tomostar)
    lines = [
        "%d excluded tilt series still have their XML in %s, and "
        "MissAlignment trains on every XML there:"
        % (len(leftover), config.warp_tiltseries_dir),
        "",
    ]
    lines += ["    %-30s %s" % (name, record.get(name, {}).get("reason", ""))
              for name in leftover]
    lines += [
        "",
        "This pipeline never deletes files itself. Remove these (the "
        ".tomostar too, so WarpTools' later steps cannot pick the series up "
        "again without its alignment), then re-run:",
        "",
    ]
    lines += ["    rm %s" % shlex.quote(p) for p in paths]
    lines += [
        "",
        "With their XML removed, a later 'align_tiltseries --force' imports "
        "them afresh and AreTomo3 tries them again.",
    ]
    message = "\n".join(lines)
    print("\nERROR: " + message + "\n", file=sys.stderr)
    log.fail("excluded XMLs", "%d excluded tilt series still in %s: %s"
             % (len(leftover), config.warp_tiltseries_dir, ", ".join(leftover)))
    log.quiet(message)
    return False


def check_tilt_counts(config, series_names, log):
    """Stop before anything is modified if a tilt series has too few used
    tilts for MissAlignment (:data:`_MIN_TILTS`). Returns True if every
    series is fine.

    Usually ``tiltseries_filter`` has already excluded such a series from
    its mdoc; this catches the rest - tilts WarpTools/AreTomo3 dropped
    later, or the filter switched off.
    """
    too_few = []
    for name in series_names:
        angles = used_tilt_angles(config, name)
        count = len(angles) if angles else 0
        if count < _MIN_TILTS:
            too_few.append((name, count))
    if not too_few:
        return True

    iter_dirs = sorted(
        os.path.join(config.warp_tiltseries_dir, entry)
        for entry in os.listdir(config.warp_tiltseries_dir)
        if _ITER_DIR_RE.match(entry)
        and os.path.isdir(os.path.join(config.warp_tiltseries_dir, entry)))

    lines = [
        "MissAlignment needs at least %d used tilts per tilt series, but "
        "%d tilt series have fewer:" % (_MIN_TILTS, len(too_few)),
        "",
    ]
    lines += ["    %-40s %d used tilt(s)" % (name, count) for name, count in too_few]
    lines += [
        "",
        "On these, miss-alignment's reconstruction workers crash and training "
        "then waits for them forever, so nothing was started.",
        "",
        "To leave them out: take their mdocs out of mdocs.preprocessing (%s) "
        "and re-run 'tomo_denoising run_pipe' WITHOUT --force. Every step "
        "resumes; align_tiltseries lists the files an earlier run made for "
        "them, for you to remove." % config.mdoc_dir("preprocessing"),
    ]
    if iter_dirs:
        lines += [
            "",
            "Also remove MissAlignment's training state from an earlier "
            "attempt - it was set up with these tilt series included, and "
            "resuming from it could bring them back:",
            "",
            "    rm -r " + " ".join(shlex.quote(d) for d in iter_dirs),
        ]
    message = "\n".join(lines)
    print("\nERROR: " + message + "\n", file=sys.stderr)
    log.fail("tilt count", "%d tilt series with fewer than %d used tilts: %s"
             % (len(too_few), _MIN_TILTS, ", ".join(n for n, _ in too_few)))
    log.quiet(message)
    return False


# --------------------------------------------------------- pre-training backup


#: In warp_tiltseries_prealign/: which alignment (warptools.alignment_id)
#: the backups next to it were taken from.
_BACKUP_ID_FILE = ".alignment_id"


def _backup_id_path(config):
    return os.path.join(config.warp_tiltseries_prealign_dir, _BACKUP_ID_FILE)


def _read_backup_id(config):
    try:
        with open(_backup_id_path(config), encoding="utf-8") as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def _backups_are_current(config):
    """Whether the backups belong to the alignment the project has NOW.

    False once align_tiltseries has written a new alignment since they were
    taken (its id changed). A backup folder without an id file - from before
    this existed - is taken to be current, as the old code assumed.
    """
    stored = _read_backup_id(config)
    return stored is None or stored == alignment_id(config)


def backup_alignment(config, series_names, log, force=False, dry_run=False):
    """Keep ``warp_tiltseries_prealign/<tomo>.xml`` the pre-MissAlignment
    state of the CURRENT alignment, and on *force* put it back.

    No guessing from the XMLs' content (WarpTools itself rewrites them, e.g.
    in ts_ctf). Instead:

      - No backup yet -> take it. This step always runs before step 2 and
        training, and backups are never removed by the pipeline, so an XML
        without a backup has never been through MissAlignment.
      - Backups of an EARLIER alignment (align_tiltseries re-aligned since -
        see :func:`_backups_are_current`) -> retaken from the current XMLs,
        which are clean: re-alignment restored them before ts_aretomo3
        rewrote them, and stopped until the old ``iterN/`` training state
        was removed (:func:`reset_for_realignment`).
      - Backup present, *force* -> the XML is restored from it, so a forced
        run retrains from the ts_aretomo3 alignment instead of refining (and
        compounding the errors of) the previous result.
      - Backup present, no *force* -> both left alone: a resume, and the
        partially refined XML is exactly what training continues on.

    Existing backups are never overwritten except as stale. The one case
    this cannot catch is a series that went through MissAlignment while its
    backup was missing (removed by hand) - its refined XML is then backed up
    as if clean.
    """
    backup_dir = ensure_dir(config.warp_tiltseries_prealign_dir)
    current = alignment_id(config)
    stale = not _backups_are_current(config)
    if stale:
        log.note("the backups in %s are of an earlier alignment - retaking "
                 "them from the current one" % backup_dir)

    backed_up, restored, kept = 0, 0, 0
    for name in series_names:
        source = os.path.join(config.warp_tiltseries_dir, name + ".xml")
        target = os.path.join(backup_dir, name + ".xml")
        if not os.path.isfile(source):
            continue
        if stale or not os.path.isfile(target):
            if not dry_run:
                shutil.copy2(source, target)
            backed_up += 1
        elif force:
            if not dry_run:
                shutil.copy2(target, source)
            restored += 1
        else:
            kept += 1

    if not dry_run and current is not None:
        with open(_backup_id_path(config), "w", encoding="utf-8") as handle:
            handle.write(current + "\n")

    if backed_up:
        info("backed up %d alignment XML(s) to %s" % (backed_up, backup_dir))
        log.note("backed up %d alignment XML(s) to %s" % (backed_up, backup_dir))
    if restored:
        info("restored %d alignment XML(s) from %s (--force: retraining from "
             "the pre-MissAlignment state)" % (restored, backup_dir))
        log.note("restored %d pre-MissAlignment XML(s) from %s"
                 % (restored, backup_dir))
    if kept:
        log.note("%d XML(s) already backed up - resuming on them, backup in "
                 "%s kept" % (kept, backup_dir))


def reset_for_realignment(config, log, dry_run=False):
    """Undo MissAlignment's traces before ``align_tiltseries`` runs
    ts_aretomo3 again - without deleting anything.

    1. Every XML with a backup OF THE CURRENT ALIGNMENT is put back to it, so
       ts_aretomo3 starts from the pre-MissAlignment XML - WarpTools
       updates an existing XML rather than writing it from nothing, and
       nothing guarantees it would reset MissAlignment's own fields.
       Backups of an even earlier alignment are left out: restoring them
       would roll the project back further than intended. The backups
       themselves can stay - the new alignment gets a new id, which makes
       :func:`backup_alignment` retake them on the next MissAlignment run.
    2. MissAlignment's ``iterN/`` training state, however, was trained
       against the alignment being replaced, and :func:`_resume_iteration`
       would resume (or skip) training on top of the new one. This pipeline
       never deletes files itself, so this STOPS and lists them for the user
       to remove - before the tilt axis search, so stopping costs nothing.
    """
    backup_dir = config.warp_tiltseries_prealign_dir
    restored = 0
    if os.path.isdir(backup_dir) and _backups_are_current(config):
        for entry in sorted(os.listdir(backup_dir)):
            if not entry.endswith(".xml"):
                continue
            xml = os.path.join(config.warp_tiltseries_dir, entry)
            if os.path.isfile(xml):
                if not dry_run:
                    shutil.copy2(os.path.join(backup_dir, entry), xml)
                restored += 1
    if restored:
        log.note("restored %d pre-MissAlignment XML(s) from %s before "
                 "re-alignment" % (restored, backup_dir))

    iter_dirs = []
    if os.path.isdir(config.warp_tiltseries_dir):
        iter_dirs = sorted(
            os.path.join(config.warp_tiltseries_dir, entry)
            for entry in os.listdir(config.warp_tiltseries_dir)
            if _ITER_DIR_RE.match(entry)
            and os.path.isdir(os.path.join(config.warp_tiltseries_dir, entry)))
    if not iter_dirs:
        return
    if dry_run:
        log.note("dry run: a real run would stop here - MissAlignment "
                 "training state from the previous alignment exists: %s"
                 % ", ".join(iter_dirs))
        return
    raise SystemExit(
        "ERROR: the tilt series are about to be aligned again, but "
        "MissAlignment's training state from the previous alignment is "
        "still there - a later run_missalignment would resume it on top of "
        "the new alignment.\n\n"
        "This pipeline never deletes files itself. Remove it, then re-run "
        "(nothing has been aligned yet):\n\n    rm -r %s"
        % " ".join(shlex.quote(d) for d in iter_dirs))


# --------------------------------------------------- training-volume patch


#: The <TiltSeries ...> opening tag, wherever it ends (real native ts_aretomo3
#: output writes it all on one line, but [^>] already spans newlines too).
_ROOT_TAG_RE = re.compile(r"<TiltSeries\b[^>]*>")


def _patch_dimensions_attr(text, attr_name, value):
    """Set ``attr_name="value"`` on the ``<TiltSeries>`` root element.

    Replaces the attribute if already present (even empty/zero) - but real,
    native ``ts_aretomo3`` output confirmed NOT to write
    ImageDimensionsAngstrom/VolumeDimensionsAngstrom at all (checked against
    real project XMLs - the attribute is simply absent, not present-and-
    zero), so the common case is inserting it fresh into the root tag.
    """
    pattern = re.compile(r'(\s%s=")[^"]*(")' % re.escape(attr_name))
    new_text, count = pattern.subn(lambda m: m.group(1) + value + m.group(2),
                                   text, count=1)
    if count:
        return new_text, True

    match = _ROOT_TAG_RE.search(text)
    if not match:
        return text, False
    tag = match.group(0)
    new_tag = tag[:-1] + ' %s="%s">' % (attr_name, value)
    return text[:match.start()] + new_tag + text[match.end():], True


def resolve_training_dimensions(config, series_list, log, dry_run=False):
    """(X, Y, Z) unbinned pixels to patch every XML's dimensions to.

    ``missalign.training_volume_dimensions`` if set explicitly (to
    deliberately tighten the volume MissAlignment trains against - see
    MissAlignment's own docs: "the tomogram shape should tightly fit your
    sample ... to avoid training on empty regions"); otherwise the same
    ``tomo_dimensions`` ``align_tiltseries`` already resolved for
    ``--tomo_dimensions``, so every XML ends up consistent with what the
    project was actually created with.
    """
    raw = str(config["missalign"]["training_volume_dimensions"] or "").strip()
    if not raw:
        return resolve_tomo_dimensions(config, series_list, log, dry_run=dry_run)
    try:
        x_str, y_str, z_str = raw.lower().split("x")
        return int(x_str), int(y_str), int(z_str)
    except ValueError:
        raise SystemExit(
            "ERROR: missalign.training_volume_dimensions must be 'XxYxZ', "
            "got %r" % raw
        )


def resolve_image_dimensions(config, series_list, log, dry_run=False):
    """(X, Y) unbinned pixels of the actual tilt images, for
    ``ImageDimensionsAngstrom``.

    Deliberately NOT taken from ``tomo_dimensions``/
    ``training_volume_dimensions``: those describe the VOLUME, and an
    explicit XY that differs from the camera frame (e.g. swapped for a ~90 deg
    tilt axis, or a tightened training volume) would otherwise mis-centre
    every projection by half the size difference - Warp's geometry adds
    ``ImageDimensions / 2`` after projecting - wrecking MissAlignment's
    training and the ts_reconstruct that uses its result.
    """
    found = frame_image_xy(config, series_list)
    if found is not None:
        x, y, path = found
        log.note("tilt image dimensions: %dx%d unbinned px (from %s)"
                 % (x, y, os.path.basename(path)))
        return x, y
    if dry_run:
        log.note("tilt image dimensions cannot be read yet (no frame-series "
                 "average exists) - using a 0x0 placeholder for this dry run.")
        return 0, 0
    raise SystemExit(
        "ERROR: no frame-series average found under %s to read the tilt "
        "image dimensions from (needed for MissAlignment's "
        "ImageDimensionsAngstrom). Run 'tomo_denoising motion_correct' first."
        % config.warp_frameseries_dir
    )


def patch_training_dimensions(config, series_names, apix, image_xy, volume_xyz,
                              log, dry_run=False):
    """Patch every tomogram's ``ImageDimensionsAngstrom`` to *image_xy* and
    ``VolumeDimensionsAngstrom`` to *volume_xyz* (unbinned pixels) x *apix*.

    The two are independent: *image_xy* is the physical tilt-image size
    (:func:`resolve_image_dimensions`), *volume_xyz* the volume
    (:func:`resolve_training_dimensions`).

    Always run, unconditionally, before MissAlignment trains - NOT a no-op:
    real, native ``ts_aretomo3`` output can leave these at ``[0.0, 0.0]``
    (confirmed against real WarpTools output, not just MissAlignment's own
    reference material assuming the worst case), which MissAlignment itself
    refuses to train on (``ValueError: ... has zero values in
    'ImageDimensionsAngstrom'``). Always overwritten with the same computed
    value on every run, regardless of what was there before - this is
    project-wide geometry, not something MissAlignment's own training
    refines, so re-patching to the same value is idempotent and safe even on
    a resumed run.
    """
    img_x, img_y = image_xy
    x, y, z = volume_xyz
    image_val = "%.6f, %.6f" % (img_x * apix, img_y * apix)
    volume_val = "%.6f, %.6f, %.6f" % (x * apix, y * apix, z * apix)

    patched = 0
    for name in series_names:
        path = os.path.join(config.warp_tiltseries_dir, name + ".xml")
        if not os.path.isfile(path) or dry_run:
            continue
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        text, hit1 = _patch_dimensions_attr(text, "ImageDimensionsAngstrom", image_val)
        text, hit2 = _patch_dimensions_attr(text, "VolumeDimensionsAngstrom", volume_val)
        if hit1 and hit2:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            patched += 1
        else:
            warn("%s: no ImageDimensionsAngstrom/VolumeDimensionsAngstrom "
                 "attribute found on the TiltSeries element - dimensions not "
                 "patched for this tomogram." % name)

    if patched:
        log.note("dimensions patched to image %dx%d / volume %dx%dx%d "
                 "(unbinned px) x %.4f A on %d XML(s)"
                 % (img_x, img_y, x, y, z, apix, patched))


# --------------------------------------------------------------- MissAlignment


def _partition_gpus(gpus, processes_per_gpu=3):
    """training_devices / reconstruction_devices, matching the pattern in
    every runMiss_align_*.sh example: roughly half the GPUs train, the rest
    reconstruct (each repeated *processes_per_gpu* times). A single GPU shares
    both roles.

    *processes_per_gpu* comes from the memory profile, not from the examples:
    on a 24 GB card three reconstruction processes per GPU is free throughput,
    on a 12 GB one it is three CUDA contexts plus three sets of reconstruction
    buffers competing with the trainer on the same card.
    """
    gpus = list(gpus) or [0]
    if len(gpus) == 1:
        return list(gpus), list(gpus) * processes_per_gpu
    n_train = max(1, len(gpus) // 2)
    training = gpus[:n_train]
    remaining = gpus[n_train:] or gpus[:1]
    reconstruction = []
    for gpu in remaining:
        reconstruction.extend([gpu] * processes_per_gpu)
    return training, reconstruction


def _as_gpu_list(value):
    """Accept either a YAML list ([2, 3]) or a string ("2,3,3")."""
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    return [int(part) for part in str(value).replace(",", " ").split()]


def resolve_devices(config, processes_per_gpu, log=None):
    """``(training_devices, reconstruction_devices)`` as miss-alignment sees them.

    miss-alignment is launched with ``CUDA_VISIBLE_DEVICES`` set to the
    configured ``gpus``, which renumbers them 0..N-1 inside the process, so
    both lists have to be LOGICAL ids - a config of ``gpus: [2, 3]`` means
    ``--training-devices 0 --reconstruction-devices 1``. Passing the real ids
    through (as this used to) asks torch for a device index that does not
    exist in its own view, which is a hard failure for any ``gpus`` list not
    starting at 0.

    ``missalign.training_devices``/``reconstruction_devices`` are written as
    REAL GPU ids, the same as everywhere else in the config, and translated
    here - repetitions preserved, since that is how miss-alignment asks for
    several reconstruction processes on one GPU.
    """
    gpus = config.gpus
    settings = config["missalign"]
    training_raw = settings["training_devices"]
    reconstruction_raw = settings["reconstruction_devices"]

    if training_raw and reconstruction_raw:
        training = to_logical_gpu_ids(gpus, _as_gpu_list(training_raw),
                                      "missalign.training_devices")
        reconstruction = to_logical_gpu_ids(gpus, _as_gpu_list(reconstruction_raw),
                                            "missalign.reconstruction_devices")
    else:
        training, reconstruction = _partition_gpus(
            gpu_ids_for("missalign", gpus), processes_per_gpu)

    if log is not None:
        log.note(gpu_id_note("missalign", gpus))
        real = {position: gpu for position, gpu in enumerate(gpus)}
        log.note("MissAlignment devices: training %s, reconstruction %s "
                 "(real GPUs %s / %s)"
                 % (training, reconstruction,
                    [real.get(g, g) for g in training],
                    [real.get(g, g) for g in reconstruction]))
    return training, reconstruction


def write_missalign_config(config, profile, dry_run=False):
    """Write ``warp_tiltseries/missalign_config.yaml``.

    A cheap, pure-Python file write with no GPU/compute cost - but still
    skipped under ``dry_run`` so a dry run never touches the project.

    *profile* is :func:`_memory_profile`'s result, which is what decides the
    two batch sizes.
    """
    config_path = os.path.join(config.warp_tiltseries_dir, "missalign_config.yaml")
    payload = {
        "general": {
            "training_directory": config.warp_tiltseries_dir,
            # Hardcoded, not configurable: CTF correction happens once, in
            # WarpTools' own reconstruction (ts_ctf/ts_reconstruct);
            # enabling it here too only doubles MissAlignment's own
            # processing time for no benefit to alignment.
            "apply_ctf": False,
            "iteration_settings": _ITERATION_SETTINGS,
            "seed": 45132,
        },
        "model_training": {
            "model_architecture": "default",
            "model_checkpoint": None,
            "loss_margin": 0.5,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "max_epochs_per_iteration": 30,
            "warmup_steps": 500,
            "multistep_lr_scheduler": {"milestones": [5, 15], "gamma": 0.5},
        },
        "data_loading": {
            "batch_size": profile["batch_size"],
            # NOT scaled with GPU memory - this is the model's input size,
            # not a throughput knob. See the memory-profile notes above.
            "patch_size": 96,
            "steps_per_epoch": 1000,
        },
        "shift_generation": {
            "trajectory_probability": 0.5,
            "trajectory_max_shift": 10.0,
            "jitter_probability": 0.5,
            "jitter_max_std": 2.0,
            "outlier_probability": 0.5,
            "outlier_max_shift": 20.0,
            "fracture_probability": 0.5,
            "fracture_max_shift": 20.0,
        },
        "tilt_series_alignment": {
            # MissAlignment requires this to match the training patch_size.
            "patch_size": 96,
            "patch_overlap": 0.1,
            # A SEPARATE pool from the training batch above, on the
            # reconstruction devices - pure throughput, no effect on the
            # result, so it is scaled on its own setting.
            "batch_size": profile["alignment_batch_size"],
        },
    }
    if not dry_run:
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        info("wrote %s" % config_path)
    return config_path


def _resume_iteration(config):
    """The value to pass as ``--start-at-iteration``, or 0 if never started.

    NOT ``highest_existing_iterN + 1``: miss-alignment's own resume logic
    (``sync_start_iteration_xmls``) requires ``iter<N>/`` to already exist
    for ``--start-at-iteration N`` - that directory IS the snapshot it
    resumes from, not the last one it finished. iter0/ is the baseline
    written before any training happens, so the highest ``iterN/`` actually
    on disk is itself the correct resume point.
    """
    highest = -1
    for entry in os.listdir(config.warp_tiltseries_dir):
        match = _ITER_DIR_RE.match(entry)
        if match:
            highest = max(highest, int(match.group(1)))
    return max(highest, 0)


def run_missalign_train(config, missalign_config_path, profile, log,
                        force=False, dry_run=False):
    gpus = config.gpus
    settings = config["missalign"]
    training_devices, reconstruction_devices = resolve_devices(
        config, profile["reconstruction_processes_per_gpu"], log=log)
    dataloaders = int(profile["dataloaders_per_trainer"])
    start_at = 0 if force else _resume_iteration(config)
    if start_at >= len(_ITERATION_SETTINGS):
        info("MissAlignment: all %d iterations already completed - skipping "
             "training. Use --force to retrain from scratch." % len(_ITERATION_SETTINGS))
        return True

    command = config.command(
        "missalign", "train",
        "--config-file", missalign_config_path,
        "--training-devices", ",".join(str(g) for g in training_devices),
        "--reconstruction-devices", ",".join(str(g) for g in reconstruction_devices),
        "--dataloaders-per-trainer", str(dataloaders),
        "--start-at-iteration", str(start_at),
        "--prepare-stacks", str(float(settings["prepare_stacks_gb"])))
    conda_env = config.conda_env("missalign")
    if conda_env:
        info("running miss-alignment inside conda environment %r" % conda_env)
    env = dict(gpu_env(gpus))
    env.update(resolve_nccl_env(config, log=log))
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        # Without this, every process that hits a torch.compile'd region
        # spins up its OWN TorchInductor compile-worker pool sized to the
        # node's full CPU count. With N training processes x
        # dataloaders_per_trainer dataloader workers each potentially
        # triggering a compile, that is many such pools launching at once,
        # oversubscribing the node badly enough that no single compile ever
        # finishes - which then means the dataloader depending on it never
        # produces a batch, and training looks permanently "stuck" at near-
        # zero GPU memory with 0% CPU everywhere (confirmed via py-spy: the
        # main thread blocks forever in dataloader._try_get_data). Capping
        # this keeps compilation serialized/small per process instead.
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
    logs = ensure_dir(config.missalignment_logs_dir)
    code = run_command(command, cwd=config.warp_tiltseries_dir, env=env,
              log_file=os.path.join(logs, "missalign_train.log"),
              check=False, dry_run=dry_run)

    if code != 0:
        log.fail("miss-alignment train",
                 "exit status %d - see %s" % (code, os.path.join(logs, "missalign_train.log")))
        return False
    log.task("miss-alignment train", True, "-")
    return True


# ------------------------------------------------------------------ entrypoint


def run(config, force=None, dry_run=False):
    """``tomo_denoising run_missalignment``"""
    force = config.overwrite if force is None else force
    with StepLog(config, STEP, force=force) as log:
        return _run(config, log, force=force, dry_run=dry_run)


def _run(config, log, force=False, dry_run=False):
    banner("MissAlignment-refined reconstruction")
    if not config["missalign"]["enabled"]:
        log.note("skipped: missalign.enabled is false")
        return 0

    # Config.require checks the right thing by itself: conda/mamba when the
    # program lives in its own environment (conda_env_missalign /
    # conda_env_warptools), the program itself otherwise.
    config.require("warptools", "WarpTools")
    config.require("missalign", "miss-alignment")

    if not dry_run and not os.path.isfile(config.warp_tiltseries_settings_path):
        log.fail("run_missalignment", "no %s - run 'tomo_denoising "
                 "align_tiltseries' first" % config.warp_tiltseries_settings_path)
        return 1

    series_names = aligned_series_names(config)
    if not series_names and not dry_run:
        log.fail("run_missalignment", "no aligned tilt series (*.xml) in %s "
                 "- run 'tomo_denoising align_tiltseries' first"
                 % config.warp_tiltseries_dir)
        return 1
    info("%d tilt series considered for MissAlignment" % len(series_names))

    # Before anything is backed up, patched or trained: the run would
    # otherwise hang, not fail (see _MIN_TILTS).
    if not dry_run and not (check_excluded_xmls(config, log)
                            and check_tilt_counts(config, series_names, log)):
        return 1

    ensure_dir(config.missalignment_dir)
    ensure_dir(config.missalignment_logs_dir)

    banner("MissAlignment 1/4  back up the pre-MissAlignment alignment")
    backup_alignment(config, series_names, log, force=force, dry_run=dry_run)

    banner("MissAlignment 2/4  patch TiltSeries XML dimensions")
    preproc = collect_tilt_series(config.mdoc_dir("preprocessing"), "preprocessing")
    apix = dataset_apix(config, preproc)
    image_xy = resolve_image_dimensions(config, preproc, log, dry_run=dry_run)
    volume_xyz = resolve_training_dimensions(config, preproc, log,
                                             dry_run=dry_run)
    patch_training_dimensions(config, series_names, apix, image_xy, volume_xyz,
                              log, dry_run=dry_run)

    banner("MissAlignment 3/4  miss-alignment train")
    profile = _memory_profile(config, log=log)
    missalign_config_path = write_missalign_config(config, profile,
                                                   dry_run=dry_run)
    ok = run_missalign_train(config, missalign_config_path, profile, log,
                             force=force, dry_run=dry_run)
    if not ok:
        return 1

    banner("MissAlignment 4/4  ts_reconstruct")
    recon_angpix = resolve_recon_angpix(config, log=log)
    ok = run_ts_reconstruct(
        config, config.warp_tiltseries_settings_path,
        config.missalignment_halfmaps_dir, config.missalignment_tiltsplit_halfmaps_dir,
        config.missalignment_logs_dir, series_names, recon_angpix,
        config.missalignment_reconstruction_dir,
        full_dir=config.missalignment_tomograms_dir, do_full=True,
        do_tiltsplit=config.tilt_split, log=log, force=force, dry_run=dry_run,
        warp_links_dir=config.warp_reconstruction_dir,
    )
    if not ok:
        return 1

    info("MissAlignment half maps: %s" % config.missalignment_halfmaps_dir)
    log.note("full tomograms: %s (also linked under WarpTools' own names "
             "in %s)" % (config.missalignment_tomograms_dir,
                         config.warp_reconstruction_dir))
    log.note("half maps: %s" % config.missalignment_halfmaps_dir)
    return 0 if not log.failed else 1
