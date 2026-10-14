"""Loading, validating and generating the tomo_denoising configuration."""

import os
import sys

import yaml

from . import __version__
from .utils import require_executable, warn, wrap_conda_run

try:  # py3.9+
    from importlib.resources import files as _resource_files
except ImportError:  # pragma: no cover - py3.8
    _resource_files = None

DEFAULT_CONFIG_NAME = "tomo_denoising_config.yaml"

#: Executables that may live in a conda/mamba environment of their own, run
#: via ``conda run -n <env> ...`` (see :meth:`Config.command`). The config key
#: for each is ``conda_env_<name>`` at the top level, next to ``paths``.
CONDA_WRAPPABLE = ("warptools", "missalign", "aretomo", "isonet", "resolve")

#: Fallback values merged under whatever the user supplies, so an incomplete
#: config never raises a KeyError deep inside a processing step.
DEFAULTS = {
    "paths": {
        "input_frames": "",
        "processing_directory": "",
    },
    "mdocs": {
        # Standalone directories, never joined to processing_directory.
        "preprocessing": "",
        "train": "",
        "predict": "",
    },
    "tiltseries_filter": {
        # exclude | stop | off - see tiltfilter.py and the template comments.
        "mode": "exclude",
        "min_tilts": 10,
        "min_tilts_fraction": 0.5,
        "min_span_fraction": 0.5,
        "max_gap_factor": 5,
    },
    "executables": {
        # The SAME AreTomo build/version is used everywhere - both for our own
        # external tilt-axis-search calls and passed to WarpTools'
        # ts_aretomo3 --exe.
        "aretomo": "AreTomo3",
        "header": "header",                # IMOD, pixel-size fallback only
        "isonet": "isonet.py",
        "resolve": "RESOLVE",
        "warptools": "WarpTools",
        "missalign": "miss-alignment",
        # Only used when one of the conda_env_* settings is set.
        "conda": "conda",
    },
    # Per-program conda/mamba environments - "" (the default) means the
    # matching "executables" entry is run directly. See CONDA_WRAPPABLE.
    "conda_env_missalign": "",
    "conda_env_warptools": "",
    "conda_env_aretomo": "",
    "conda_env_isonet": "",
    "conda_env_resolve": "",
    "overwrite": False,
    "gpus": [0, 1],
    "acquisition": {
        "pixel_size": "read",
        "frame_extension": "tiff",         # tif/tiff/eer/mrc
        "gain": "",
        "gain_flip_x": False,
        "gain_flip_y": False,
        "gain_transpose": False,
        "voltage": 300.0,
        "spherical_aberration": 2.7,
        "amplitude_contrast": 0.1,
    },
    "frameseries": {
        # log2 Fourier binning WarpTools applies when producing averages
        # (WarpTools' own --bin convention: 0 = none, 1 = bin by 2, ...).
        "bin": 0,
        # Dose PER TILT, e-/A^2 - WarpTools needs this explicitly for its own
        # dose-weighting; MotionCor3 never needed it, so this is new.
        "exposure": 3.0,
        # Only used when acquisition.frame_extension is "eer". Passed on
        # create_settings (not fs_motion_and_ctf, which rejects it) - EER
        # grouping is a property of how the raw data is read, so it's fixed
        # once when the .settings file is created.
        "eer_ngroups": "",
        "m_grid": "1x1x3",
        "c_grid": "2x2x1",
        "c_range_max": 7.0,
        "c_defocus_max": 8.0,
        "c_use_sum": True,
        "perdevice": 1,
        "extra_args": "",
    },
    "tomo_dimensions": {
        # "auto" -> taken from a motion-corrected average's own pixel shape,
        # scaled back up to unbinned. Or set explicitly, e.g. "4400x6000".
        "xy": "auto",
        "z": 1200,
    },
    "aretomo": {
        # Mostly for OUR OWN external, alignment-only tilt-axis-search calls
        # (tiltaxis.py) - the real per-tomogram alignment runs through native
        # WarpTools ts_aretomo3 instead (see the warptools section). "alignz"
        # is the exception: it feeds BOTH.
        "dark_tolerance": 0.5,
        "wbp": True,
        # Alignment thickness in UNBINNED pixels (same convention as
        # tomo_dimensions, but deliberately a SEPARATE value - the volume
        # worth aligning on is not the volume worth reconstructing). Fed to
        # both consumers, each in the units it wants: WarpTools
        # ts_aretomo3 --alignz in ANGSTROM, our own AreTomo3 -AlignZ in
        # pixels at the input stack's own pixel size (align_angpix).
        # "" (default) -> neither call gets the argument at all.
        "alignz": "",
        # "" (default) -> the first iteration is a free search. Set to a
        # degrees value to use it as the starting point for iteration 1
        # instead (refined within +/-3 deg, same as every later iteration);
        # iteration 2 onward still takes the previous iteration's result as
        # usual. With tilt_axis_iterations: 0, this value is instead used
        # directly AS the converged tilt axis, with no external search at
        # all - the opposite of tilt_axis_iterations: 0 with this left
        # empty, which searches each tomogram independently via ts_aretomo3.
        "initial_tilt_axis": "",
        "tilt_axis_iterations": 3,
        "tilt_axis_subset_size": 10,
        "tilt_axis_seed": 0,
        "extra_args": "",
    },
    "warptools": {
        "dont_invert": False,
        "min_intensity": 0.3,
        # Pixel size (A) for ts_stack / our own axis-search AreTomo calls /
        # native ts_aretomo3. A coarser value than the final reconstruction,
        # matching how alignment is normally done.
        "align_angpix": 10.0,
        # Final reconstruction pixel size (A), for ts_reconstruct.
        "recon_angpix": 9.0,
        "min_fov": 0,
        # Tilt-ANGLE offset handling on ts_import - the 0-tilt zero point /
        # specimen pre-tilt, NOT the tilt axis rotation. Both off by default,
        # so the mdoc's own tilt angles are used unchanged.
        "auto_zero": False,
        "tilt_offset": "",
        "auto_fix_defocus_hand": True,
        "ts_ctf_range_high": 7.0,
        "ts_ctf_defocus_max": 8.0,
        "perdevice": 1,
    },
    "quality_check": {
        "fast": True,
        "gpu_enabled": True,
        "cpu_threads": 16,
        "choose_training_set": True,
        "min_signal_ratio": 0.25,
        "outlier_factor": 1.5,
        # true  -> RESOLVE measures dedicated TILT-split half maps, an extra
        #          reconstruction made only for this (at the coarse
        #          tilt_split_target_pixel_size below).
        # false -> RESOLVE measures the FRAME-split half maps that are
        #          reconstructed anyway for IsoNet, and no tilt-split
        #          reconstruction is made at all.
        "enable_tilt_split": True,
        "tilt_split_target_pixel_size": 50.0,
    },
    "isonet": {
        "denoise_epochs": 50,
        "refine_epochs": 100,
        # "None" (the exact string IsoNet2's CLI expects - not "none", not
        # a null/empty value): the half maps IsoNet trains on are already
        # CTF-corrected by WarpTools (ts_defocus_hand -> ts_ctf ->
        # ts_reconstruct, run before IsoNet ever sees them). IsoNet2's other
        # modes ("network"/"wiener"/"phase_only") each assume RAW,
        # un-CTF-corrected weighted-back-projection input and apply their
        # own CTF-shaped correction on top - using one of them here would
        # double-correct already-corrected data.
        "ctf_mode": "None",
        "density_percentage": 50,
        "std_percentage": 80,
        "mw_weight": 200,
        "bfactor": 0,
        "fix_tilt_range": [],
        "denoise_network": "auto",
        "refine_network": "auto",
        "extra_args": "",
    },
    "missalign": {
        "enabled": True,
        # "XxYxZ" (unbinned px, same convention as tomo_dimensions) to
        # tighten the volume MissAlignment trains against so it does not
        # spend capacity on empty margins - see MissAlignment's own docs.
        # Leave empty (default) to train against the same dimensions as the
        # final reconstruction (tomo_dimensions).
        "training_volume_dimensions": "",
        # VRAM per GPU, in GB. >= 24 uses MissAlignment's own reference
        # settings unchanged; anything lower selects the low-memory profile
        # (see missalign._memory_profile), which is what makes the step run
        # at all on a 12 GB card.
        "gpu_memory_gb": 12,
        # null = taken from the gpu_memory_gb profile / config.gpus. Set any
        # of these explicitly to override it.
        "batch_size": None,
        "alignment_batch_size": None,
        "reconstruction_processes_per_gpu": None,
        "training_devices": None,
        "reconstruction_devices": None,
        "dataloaders_per_trainer": None,
        "prepare_stacks_gb": 10.0,
        # NCCL transport control for multi-GPU training - see
        # missalign.resolve_nccl_env. "" leaves NCCL's own defaults alone.
        "nccl_p2p": "PIX",
    },
}


#: Replaced by the running version when create_config writes the template.
_VERSION_PLACEHOLDER = "__TOMO_DENOISING_VERSION__"

#: Settings that have since moved and that :meth:`Config._migrate_legacy`
#: still honours - recognised, but reported as moved rather than unknown.
_LEGACY_KEYS = {
    "tilt_split": "quality_check.enable_tilt_split / "
                  "quality_check.tilt_split_target_pixel_size",
    "missalign.conda_env": "conda_env_missalign",
}


class ConfigError(Exception):
    """Raised when the configuration is missing or inconsistent."""


def _key_paths(data, prefix=""):
    """Dotted paths of every key in *data*, descending only into sections
    that are sections (dicts) in DEFAULTS too - so a list or free-form value
    is one setting, not a tree of them."""
    paths = []
    for key, value in data.items():
        path = prefix + str(key)
        paths.append(path)
        default = _lookup(DEFAULTS, path)
        if isinstance(value, dict) and isinstance(default, dict):
            paths.extend(_key_paths(value, path + "."))
    return paths


def _lookup(tree, path):
    node = tree
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _key_differences(raw):
    """(unknown, moved, missing) dotted key paths of the user's *raw* config
    relative to DEFAULTS (which mirrors the shipped template exactly)."""
    user = set(_key_paths(raw)) - {"config_version"}
    known = set(_key_paths(DEFAULTS))
    moved = sorted(k for k in user if k in _LEGACY_KEYS)
    # Children of a moved section are covered by reporting the section.
    unknown = sorted(k for k in user - known
                     if k not in _LEGACY_KEYS
                     and not any(k.startswith(m + ".") for m in moved))
    missing = sorted(k for k in known - user
                     if not any(k.startswith(m + ".") for m in known - user
                                if m != k))
    return unknown, moved, missing


def check_config_version(raw, source):
    """Refuse a config written for a different tomo_denoising version.

    A hard stop on ANY difference, by design: settings get added, renamed or
    change meaning between versions, and a config carried over from an older
    install would otherwise run with silently different behaviour. The error
    lists the concrete key differences to review. With matching versions,
    keys this version does not know (typos, most likely) are only warned
    about.
    """
    unknown, moved, missing = _key_differences(raw)
    details = []
    if unknown:
        details.append("  not recognised by %s (ignored if left in):\n%s"
                       % (__version__, "\n".join("    " + k for k in unknown)))
    if moved:
        details.append("  moved:\n%s" % "\n".join(
            "    %s -> %s" % (k, _LEGACY_KEYS[k]) for k in moved))
    if missing:
        details.append("  not in your file (the built-in default is used):\n%s"
                       % "\n".join("    " + k for k in missing))

    written_for = raw.get("config_version")
    written_for = "" if written_for is None else str(written_for).strip()
    if written_for != __version__:
        found = ("has no config_version (written before versioned configs)"
                 if not written_for else
                 "was written for tomo_denoising %s" % written_for)
        raise ConfigError(
            "%s %s, but this is tomo_denoising %s.\n"
            "Settings may have been added, renamed or changed meaning since. "
            "Write a fresh template with\n"
            "    tomo_denoising create_config -o new_config.yaml\n"
            "compare it with your file, carry over what changed, then set\n"
            "    config_version: \"%s\"\n"
            "in your file.%s"
            % (source, found, __version__, __version__,
               ("\n\nDifferences between your file and this version's "
                "template:\n" + "\n".join(details)) if details else
               "\n\nIts keys match this version's template exactly, but a "
               "setting may still have changed meaning - compare against the "
               "new template's comments.")
        )
    if unknown:
        warn("%s has setting(s) tomo_denoising %s does not know - they are "
             "ignored. Typo?\n%s" % (source, __version__,
                                     "\n".join("    " + k for k in unknown)))


def _template_text():
    """Return the commented default config shipped with the package."""
    if _resource_files is not None:
        res = _resource_files("tomo_denoising").joinpath(
            "templates/default_config.yaml"
        )
        return res.read_text(encoding="utf-8")
    here = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(here, "templates", "default_config.yaml"), encoding="utf-8"
    ) as handle:
        return handle.read()


def write_template(path, force=False):
    """Write the commented default configuration to *path*."""
    path = os.path.abspath(path)
    if os.path.exists(path) and not force:
        raise ConfigError(
            "%s already exists. Use --force to overwrite it." % path
        )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_template_text().replace(_VERSION_PLACEHOLDER, __version__))
    return path


def _merge(defaults, user):
    """Recursively overlay *user* onto *defaults* without mutating either."""
    merged = dict(defaults)
    for key, value in (user or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class Config(object):
    """Parsed configuration plus the derived directory layout."""

    def __init__(self, data, source):
        self.source = source
        check_config_version(data or {}, source)
        self.data = _merge(DEFAULTS, data)
        self._migrate_legacy(data or {})
        self._validate()

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path=None):
        path = os.path.abspath(path or DEFAULT_CONFIG_NAME)
        if not os.path.isfile(path):
            raise ConfigError(
                "Configuration file not found: %s\n"
                "Create one with:  tomo_denoising create_config" % path
            )
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError("%s does not contain a YAML mapping." % path)
        return cls(data, path)

    def _migrate_legacy(self, raw):
        """Honour settings that have since moved, so an older, hand-edited
        config keeps working instead of silently reverting to a default.

        *raw* is the user's own data, BEFORE the defaults were merged under
        it - the only way to tell "the user wrote this" from "this is the
        default value".
        """
        legacy_split = raw.get("tilt_split")
        quality = raw.get("quality_check") or {}
        if isinstance(legacy_split, dict):
            moved = []
            if "enabled" in legacy_split and "enable_tilt_split" not in quality:
                self.data["quality_check"]["enable_tilt_split"] = bool(
                    legacy_split["enabled"])
                moved.append("enabled -> quality_check.enable_tilt_split")
            if ("target_pixel_size" in legacy_split
                    and "tilt_split_target_pixel_size" not in quality):
                self.data["quality_check"]["tilt_split_target_pixel_size"] = float(
                    legacy_split["target_pixel_size"])
                moved.append("target_pixel_size -> "
                             "quality_check.tilt_split_target_pixel_size")
            if moved:
                warn("%s still has a 'tilt_split' section; it now lives in "
                     "'quality_check'. Using it for now (%s) - move it there "
                     "to silence this." % (self.source, "; ".join(moved)))
            self.data.pop("tilt_split", None)

        legacy_env = str((raw.get("missalign") or {}).get("conda_env") or "").strip()
        if legacy_env and not str(raw.get("conda_env_missalign") or "").strip():
            self.data["conda_env_missalign"] = legacy_env
            warn("%s still sets missalign.conda_env; it is now the top-level "
                 "'conda_env_missalign'. Using it for now - move it there to "
                 "silence this." % self.source)

    def _validate(self):
        frames = self.data["paths"].get("input_frames") or ""
        processing = self.data["paths"].get("processing_directory") or ""

        if not processing.strip() or processing.startswith("/path/to"):
            raise ConfigError(
                "paths.processing_directory is not set in %s" % self.source
            )
        if not frames.strip() or frames.startswith("/path/to"):
            raise ConfigError(
                "paths.input_frames is not set in %s" % self.source
            )
        if not os.path.isdir(frames):
            raise ConfigError(
                "paths.input_frames does not exist: %s" % frames
            )

        ext = str(self.data["acquisition"]["frame_extension"]).lower().lstrip(".")
        if ext not in ("tif", "tiff", "eer", "mrc"):
            raise ConfigError(
                "acquisition.frame_extension must be one of tif/tiff/eer/mrc, "
                "got %r" % ext
            )
        self.data["acquisition"]["frame_extension"] = ext

        mode = str(self.data["tiltseries_filter"]["mode"]).strip().lower()
        if mode not in ("exclude", "stop", "off"):
            raise ConfigError(
                "tiltseries_filter.mode must be one of exclude/stop/off, "
                "got %r" % self.data["tiltseries_filter"]["mode"]
            )

    # ------------------------------------------------------------- accessors
    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    @property
    def input_frames(self):
        return os.path.abspath(self.data["paths"]["input_frames"])

    @property
    def processing_dir(self):
        return os.path.abspath(self.data["paths"]["processing_directory"])

    @property
    def tilt_split(self):
        """Whether quality_check measures dedicated TILT-split half maps.

        False (``quality_check.enable_tilt_split: false``) means RESOLVE
        measures the FRAME-split half maps that are reconstructed anyway for
        IsoNet, and the extra tilt-split reconstruction is never made at all.
        """
        return bool(self.data["quality_check"]["enable_tilt_split"])

    @property
    def tilt_split_angpix(self):
        """Pixel size (A) the tilt-split half maps are reconstructed at.

        Deliberately much coarser than ``warptools.recon_angpix``: tilt-split
        half-tomograms differ from each other far more than frame-split ones
        do, so RESOLVE's estimate is only meaningful at a coarse pixel size.
        """
        return float(self.data["quality_check"]["tilt_split_target_pixel_size"])

    @property
    def gpus(self):
        gpus = self.data.get("gpus") or [0]
        if isinstance(gpus, (int, str)):
            gpus = [gpus]
        return [int(g) for g in gpus]

    def exe(self, name):
        return self.data["executables"][name]

    # ------------------------------------------------------- conda wrapping
    def conda_env(self, name):
        """The conda/mamba environment *name* should be run in, or "".

        Read from the top-level ``conda_env_<name>`` setting (see
        :data:`CONDA_WRAPPABLE`), which sits next to ``paths`` rather than
        inside each program's own section because it is the same question for
        every one of them.
        """
        if name not in CONDA_WRAPPABLE:
            return ""
        return str(self.data.get("conda_env_%s" % name) or "").strip()

    def command(self, name, *args):
        """The argv for running executable *name*, conda-wrapped if configured.

        Every call site in the package goes through this instead of
        ``exe(name)`` directly, so adding an environment to one program never
        needs the program's own module to know about it. List/tuple arguments
        are flattened, so a call can mix plain flags with pre-built argument
        lists.
        """
        argv = [self.exe(name)]
        for arg in args:
            if isinstance(arg, (list, tuple)):
                argv.extend(str(part) for part in arg)
            else:
                argv.append(str(arg))
        env = self.conda_env(name)
        if not env:
            return argv
        return wrap_conda_run(argv, self.exe("conda"), env)

    def require(self, name, purpose):
        """Warn if *name* cannot be run - checking the right thing.

        When *name* has a conda environment configured it lives inside that
        environment, not necessarily on the PATH tomo_denoising itself runs
        with, so looking for it here would just produce a false warning. What
        actually has to be reachable in that case is conda/mamba.
        """
        env = self.conda_env(name)
        if env:
            return require_executable(
                self.exe("conda"), "conda/mamba, to run %s in %r" % (purpose, env))
        return require_executable(self.exe(name), purpose)

    @property
    def gain_name(self):
        """Bare file name of the gain reference, for excluding it from frames."""
        return os.path.basename(
            str(self.data["acquisition"].get("gain") or "").strip()
        )

    def resolve_gain(self):
        """Absolute path of the gain reference, or "" when none is configured.

        A standalone path, like the mdoc directories: an absolute path is used
        as given, a relative one is resolved against the current working
        directory. It is never joined to input_frames. Raises if one is
        configured but missing, since WarpTools would otherwise fail deep
        inside fs_motion_and_ctf with a less obvious message.
        """
        raw = str(self.data["acquisition"].get("gain") or "").strip()
        if not raw:
            return ""

        path = os.path.abspath(raw)
        if os.path.isfile(path):
            return path

        raise SystemExit(
            "ERROR: gain reference %r not found at:\n  %s\n"
            "Relative paths are resolved against the directory you run "
            "tomo_denoising from. Use an absolute path, or leave "
            "acquisition.gain empty if the frames are already gain corrected."
            % (raw, path)
        )

    # ----------------------------------------------------------- directories
    def sub(self, *parts):
        """Absolute path inside the processing directory."""
        return os.path.join(self.processing_dir, *parts)

    @property
    def overwrite(self):
        return bool(self.data.get("overwrite", False))

    @property
    def uncorrected_tomograms_dir(self):
        """Links to the full, non-split tomograms reconstructed from the
        uncorrected (pre-MissAlignment) alignment - always produced by
        reconstruct_tomograms, independent of whether half maps are. Next to
        missalignment_tomograms_dir (tomos_MissAlignment/)."""
        return self.sub("warp_tiltseries", "tomos_uncorrected")

    @property
    def raw_halfmaps_dir(self):
        return self.sub("warp_tiltseries", "raw_halfMaps")

    @property
    def tiltsplit_halfmaps_dir(self):
        return self.sub("warp_tiltseries", "raw_halfMaps_tiltSplit")

    @property
    def logs_dir(self):
        return self.sub("logs")

    @property
    def resolve_dir(self):
        return self.sub("resolve")

    @property
    def state_dir(self):
        """Where completion markers for the resume logic live."""
        return self.sub("logs", ".state")

    @property
    def isonet_config_dir(self):
        return self.sub("isonet_config")

    @property
    def isonet_dir(self):
        return self.sub("isonet")

    # ------------------------------------------------------- WarpTools project
    @property
    def warp_frameseries_settings_path(self):
        return self.sub("warp_frameseries.settings")

    @property
    def warp_frameseries_dir(self):
        return self.sub("warp_frameseries")

    @property
    def tomostar_dir(self):
        return self.sub("tomostar")

    @property
    def warp_tiltseries_settings_path(self):
        return self.sub("warp_tiltseries.settings")

    @property
    def warp_tiltseries_dir(self):
        return self.sub("warp_tiltseries")

    @property
    def warp_tiltseries_prealign_dir(self):
        """Backup of each per-tomogram XML, taken before miss-alignment
        modifies the originals in place."""
        return self.sub("warp_tiltseries_prealign")

    @property
    def tiltstack_dir(self):
        """ts_stack output: <tomo>/<tomo>.st + .rawtlt, one dir per series."""
        return self.sub("warp_tiltseries", "tiltstack")

    @property
    def tiltaxis_search_dir(self):
        """Scratch home for our own external, alignment-only AreTomo runs
        (-OutDir).

        Never warp_tiltseries/tiltstack itself: with -VolZ 0 AreTomo writes
        an aligned stack here rather than a reconstruction, and that must
        never be mistaken by anything for real project state.
        """
        return self.sub("warp_tiltseries", "tiltaxis_search")

    @property
    def uncorrected_reconstruction_dir(self):
        """reconstruct_tomograms' own ts_reconstruct output (one
        --output_processing per call under it - see
        warptools.run_ts_reconstruct): the physical files
        uncorrected_tomograms_dir and raw_halfMaps*/ link to."""
        return self.sub("warp_tiltseries", "reconstruction_uncorrected")

    @property
    def warp_reconstruction_dir(self):
        """WarpTools' OWN reconstruction folder for this project
        (<processing folder>/reconstruction/), where its later commands -
        ts_template_match, for one - look for <name>_<angpix>Apx.mrc. Holds
        only links, to the MissAlignment-corrected full tomograms, under
        exactly that WarpTools name."""
        return self.sub("warp_tiltseries", "reconstruction")

    @property
    def mdocs_filtered_dir(self):
        """What ts_import reads whenever a tilt was dropped or a tilt series
        excluded: filtered copies of, and links to, mdocs.preprocessing's
        mdocs - see warptools._prepare_mdocs. mdocs.preprocessing itself is
        never written to."""
        return self.sub("mdocs_filtered")

    # -------------------------------------------------------- MissAlignment
    @property
    def missalignment_dir(self):
        return self.sub("missalignment")

    @property
    def missalignment_tomograms_dir(self):
        """Links to MissAlignment's full, non-split tomograms - the analogue
        of uncorrected_tomograms_dir, from the refined alignment instead of
        the native one. Under warp_tiltseries/, next to it; the physical
        files stay in missalignment_reconstruction_dir."""
        return self.sub("warp_tiltseries", "tomos_MissAlignment")

    @property
    def missalignment_halfmaps_dir(self):
        return self.sub("missalignment", "raw_halfMaps")

    @property
    def missalignment_tiltsplit_halfmaps_dir(self):
        return self.sub("missalignment", "raw_halfMaps_tiltSplit")

    @property
    def missalignment_reconstruction_dir(self):
        """MissAlignment's own ts_reconstruct output (one --output_processing
        per call under it - see warptools.run_ts_reconstruct) - deliberately
        SEPARATE from uncorrected_reconstruction_dir, which
        reconstruct_tomograms uses for the pre-MissAlignment reconstruction.
        Sharing one between the two used to mean MissAlignment's later
        reconstruction silently overwrote the physical files
        reconstruct_tomograms's own raw_halfMaps symlinks already pointed
        to."""
        return self.sub("missalignment", "reconstruction")

    @property
    def missalignment_logs_dir(self):
        return self.sub("missalignment", "logs")

    def mdoc_dir(self, which):
        """Absolute path of the mdoc directory for 'preprocessing'/'train'/'predict'.

        These are standalone locations - they are deliberately NOT joined to
        processing_directory, so the mdocs can live anywhere. An absolute path
        is used as given; a relative one is resolved against the current
        working directory.
        """
        raw = str(self.data["mdocs"].get(which) or "").strip()
        if not raw or raw.startswith("/path/to"):
            raise SystemExit(
                "ERROR: mdocs.%s is not set in %s\n"
                "Set it to the directory holding the .mdoc files for this "
                "step. It may be anywhere - it is not required to be inside "
                "the processing directory." % (which, self.source)
            )
        return os.path.abspath(raw)


def load_config(path=None):
    """Load a config, printing a friendly message and exiting on failure."""
    try:
        return Config.load(path)
    except ConfigError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
