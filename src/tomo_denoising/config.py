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
CONDA_WRAPPABLE = ("warptools", "missalign", "aretomo", "isonet", "resolve",
                   "cryocare", "imod")

#: Executables that share ONE conda environment setting with others: the
#: three CryoCARE scripts all live in the same environment, and IMOD's
#: ``header`` in IMOD's.
_CONDA_ENV_OF = {
    "cryocare_extract": "cryocare",
    "cryocare_train": "cryocare",
    "cryocare_predict": "cryocare",
    "header": "imod",
}

#: The denoisers run_pipe can use - exactly one per run (see "denoiser").
DENOISERS = ("isonet", "cryocare")

#: The programs that can do the final tilt-series alignment - exactly one
#: per run (see "aligner"): WarpTools ts_aretomo3, or ts_etomo_patches
#: (IMOD patch tracking).
ALIGNERS = ("aretomo", "imod")

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
        # IMOD's install FOLDER (IMOD_DIR), not a program: its bin/ is put
        # first on the PATH of ts_etomo_patches (which runs batchruntomo),
        # and its bin/header is the pixel-size fallback. "" -> whatever IMOD
        # is already on the PATH (or in conda_env_imod).
        "imod": "",
        "isonet": "isonet.py",
        "resolve": "RESOLVE",
        "warptools": "WarpTools",
        "missalign": "miss-alignment",
        # Only used with denoiser: "cryocare".
        "cryocare_extract": "cryoCARE_extract_train_data.py",
        "cryocare_train": "cryoCARE_train.py",
        "cryocare_predict": "cryoCARE_predict.py",
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
    "conda_env_cryocare": "",
    # IMOD is never run through "conda run" by ts_etomo_patches: the
    # environment's bin/ (and its IMOD_DIR, if it sets one) is put on the
    # PATH of the WarpTools call instead - see imod.imod_env.
    "conda_env_imod": "",
    "overwrite": False,
    "gpus": [0, 1],
    # VRAM per GPU, in GB. Shared by MissAlignment (>= 24: its own reference
    # settings, lower: the low-memory profile - see
    # missalign._memory_profile) and IsoNet (isonet.cube_size "auto": 128
    # at >= 24, 96 otherwise).
    "gpu_memory_gb": 12,
    # Which program does the final tilt-series alignment - one of ALIGNERS.
    "aligner": "aretomo",
    # Which denoiser run_pipe trains and applies - one of DENOISERS.
    "denoiser": "isonet",
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
        # log2 Fourier binning WarpTools applies when loading raw data
        # (WarpTools' own --bin convention: 0 = none, 1 = bin by 2, ...) -
        # passed to BOTH create_settings calls, frame series and tilt series.
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
    "imod": {
        # Only used with aligner: "imod". "" -> the starting axis comes from
        # the AreTomo tilt axis search (the aretomo section); a degrees
        # value is used instead, and AreTomo is not run at all.
        "initial_tilt_axis": "",
        # ts_etomo_patches --do_axis_search rounds on a random subset, each
        # starting from the previous result; 0 -> the starting axis is used
        # as it is for the final alignment.
        "tilt_axis_iterations": 1,
        "tilt_axis_subset_size": 10,
        "tilt_axis_seed": 0,
        # ts_etomo_patches --patch_size, Angstrom, or "auto": the field of
        # view's short side / 7, rounded to 100, at least 500 (see
        # warptools.imod_patch_size).
        "patch_size": "auto",
    },
    "autolevel": {
        # EXPERIMENTAL. WarpTools ts_autolevel after the final alignment:
        # levels the sample slab (LevelAngleX/Y in each XML) so it lies flat
        # in the tomogram. Can hurt IsoNet2's missing-wedge correction (the
        # wedge is tilted by the level angles). Changing it re-aligns.
        "enabled": False,
        # ts_autolevel --patch_size, Angstrom (WarpTools' default).
        "patch_size": 500,
    },
    "warptools": {
        "dont_invert": False,
        "min_intensity": 0.3,
        # Pixel size (A) for ts_stack / our own axis-search AreTomo calls /
        # native ts_aretomo3. A multiple of the raw pixel size, so the
        # binning is the same whatever the dataset's own pixel size.
        "align_angpix": "4*pixel_size",
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
        # "auto": 128 when gpu_memory_gb >= 24, else 96. Or a number.
        "cube_size": "auto",
        # "network" (default) or "None" (the exact string IsoNet2's CLI
        # expects - not "none", not a null/empty value). WarpTools'
        # ts_reconstruct only phase-flips the CTF; its amplitude
        # modulation is still in the half maps, which "network" models
        # using each tomogram's defocus (rlnDefocus, from the XML's
        # GridCTF) - see isonet.tomogram_defocus.
        "ctf_mode": "network",
        # --isCTFflipped for denoise/refine/predict, only passed when
        # ctf_mode is not "None". WarpTools' half maps are phase-flipped.
        "is_ctf_flipped": True,
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
        # The volume MissAlignment trains against, unbinned px - the same
        # layout as tomo_dimensions ("XxY" and Z). Set to tighten it so it
        # does not spend capacity on empty margins (see MissAlignment's own
        # docs). Each one left empty takes tomo_dimensions' own value.
        "training_volume_dimensions": {
            "xy": "",
            "z": "",
        },
        # null = taken from the gpu_memory_gb profile / config.gpus. Set any
        # of these explicitly to override it.
        "batch_size": None,
        "alignment_batch_size": None,
        "reconstruction_processes_per_gpu": None,
        "training_devices": None,
        "reconstruction_devices": None,
        "dataloaders_per_trainer": None,
        # miss-alignment --prepare-stacks: a PIXEL SIZE in Angstrom (the
        # tilt stacks MissAlignment builds for itself), not disk space.
        "stack_pixel_size": 10.0,
        # NCCL transport control for multi-GPU training - see
        # missalign.resolve_nccl_env. "" leaves NCCL's own defaults alone.
        "nccl_p2p": "PIX",
    },
    "cryocare": {
        "epochs": 100,
        "patch_shape": [72, 72, 72],
        "num_slices": 1800,
        "n_tiles": [8, 8, 8],
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
    "missalign.gpu_memory_gb": "gpu_memory_gb (top level, next to gpus)",
    "missalign.prepare_stacks_gb": "missalign.stack_pixel_size (it always "
                                   "was a pixel size in Angstrom, not GB)",
    "executables.header": "executables.imod (IMOD's install folder - "
                          "header is taken from its bin/)",
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

        # missalign.training_volume_dimensions used to be ONE "XxYxZ" string;
        # it is now {xy, z} like tomo_dimensions. A string still found here
        # (an earlier 1.1.0 config) is split into the two.
        legacy_dims = (raw.get("missalign") or {}).get("training_volume_dimensions")
        if not isinstance(legacy_dims, dict) and legacy_dims is not None:
            text = str(legacy_dims).strip()
            parts = text.lower().split("x")
            converted = {"xy": "", "z": ""}
            if len(parts) == 3:
                converted = {"xy": "%sx%s" % (parts[0], parts[1]), "z": parts[2]}
            elif text:
                raise ConfigError(
                    "missalign.training_volume_dimensions in %s is %r - it now "
                    "takes xy and z separately, like tomo_dimensions:\n"
                    "  training_volume_dimensions:\n    xy: \"XxY\"\n    z: Z"
                    % (self.source, text))
            self.data["missalign"]["training_volume_dimensions"] = converted
            if text:
                warn("%s sets missalign.training_volume_dimensions as one "
                     "string (%r); it now takes xy and z separately, like "
                     "tomo_dimensions. Using xy %r, z %r for now - update the "
                     "config to silence this."
                     % (self.source, text, converted["xy"], converted["z"]))

        raw_missalign = raw.get("missalign") or {}
        if "gpu_memory_gb" in raw_missalign and "gpu_memory_gb" not in raw:
            self.data["gpu_memory_gb"] = raw_missalign["gpu_memory_gb"]
            warn("%s still sets missalign.gpu_memory_gb; it is now the "
                 "top-level 'gpu_memory_gb' (shared with IsoNet). Using it for "
                 "now - move it there to silence this." % self.source)
        if ("prepare_stacks_gb" in raw_missalign
                and "stack_pixel_size" not in raw_missalign):
            self.data["missalign"]["stack_pixel_size"] = raw_missalign["prepare_stacks_gb"]
            warn("%s still sets missalign.prepare_stacks_gb; it is now "
                 "missalign.stack_pixel_size - it always was the pixel size "
                 "(Angstrom) of MissAlignment's own tilt stacks, not GB. "
                 "Using it for now - rename it to silence this." % self.source)
        self.data["missalign"].pop("gpu_memory_gb", None)
        self.data["missalign"].pop("prepare_stacks_gb", None)

        # executables.header is now derived from executables.imod. A header
        # set to anything but the bare default is still honoured.
        legacy_header = str((raw.get("executables") or {}).get("header") or "").strip()
        self._legacy_header = ""
        if legacy_header and legacy_header != "header":
            self._legacy_header = legacy_header
            warn("%s still sets executables.header; IMOD is now configured "
                 "as a whole with executables.imod (its install folder) or "
                 "conda_env_imod. Using %r for now - remove it to silence "
                 "this." % (self.source, legacy_header))
        self.data["executables"].pop("header", None)

        legacy_env = str((raw.get("missalign") or {}).get("conda_env") or "").strip()
        if legacy_env and not str(raw.get("conda_env_missalign") or "").strip():
            self.data["conda_env_missalign"] = legacy_env
            warn("%s still sets missalign.conda_env; it is now the top-level "
                 "'conda_env_missalign'. Using it for now - move it there to "
                 "silence this." % self.source)

    def _validate(self):
        processing = self.data["paths"].get("processing_directory") or ""

        if not processing.strip() or processing.startswith("/path/to"):
            raise ConfigError(
                "paths.processing_directory is not set in %s" % self.source
            )

        ext = str(self.data["acquisition"]["frame_extension"]).lower().lstrip(".")
        if ext not in ("tif", "tiff", "eer", "mrc"):
            raise ConfigError(
                "acquisition.frame_extension must be one of tif/tiff/eer/mrc, "
                "got %r" % ext
            )
        self.data["acquisition"]["frame_extension"] = ext

        self._check_distinct_folders()

        denoiser = str(self.data.get("denoiser") or "").strip().lower()
        if denoiser not in DENOISERS:
            raise ConfigError(
                "denoiser must be one of %s, got %r"
                % ("/".join(DENOISERS), self.data.get("denoiser")))
        self.data["denoiser"] = denoiser

        aligner = str(self.data.get("aligner") or "").strip().lower()
        if aligner not in ALIGNERS:
            raise ConfigError(
                "aligner must be one of %s, got %r"
                % ("/".join(ALIGNERS), self.data.get("aligner")))
        self.data["aligner"] = aligner

        patch = self.data["imod"].get("patch_size")
        if str(patch).strip().lower() == "auto":
            self.data["imod"]["patch_size"] = "auto"
        else:
            try:
                ok = float(patch) > 0
            except (TypeError, ValueError):
                ok = False
            if not ok:
                raise ConfigError(
                    "imod.patch_size must be \"auto\" or a positive number of "
                    "Angstrom, got %r" % patch)

        if not isinstance(self.data["autolevel"].get("enabled"), bool):
            raise ConfigError(
                "autolevel.enabled must be true or false, got %r"
                % self.data["autolevel"].get("enabled"))

        mode =str(self.data["tiltseries_filter"]["mode"]).strip().lower()
        if mode not in ("exclude", "stop", "off"):
            raise ConfigError(
                "tiltseries_filter.mode must be one of exclude/stop/off, "
                "got %r" % self.data["tiltseries_filter"]["mode"]
            )

    def _check_distinct_folders(self):
        """Refuse folder settings that would overwrite or mix data:

        - processing_directory == input_frames: WarpTools treats every
          *.mrc in the frames folder as a movie, and results would be
          written next to the raw data that must only ever be read;
        - mdocs.train == mdocs.preprocessing: every tilt series would be
          trained on, and the automatic training-set choice (which fills
          mdocs.train with links) would write into the user's own mdocs.

        Only settings that are actually filled in are compared.
        """
        def folder(raw):
            raw = str(raw or "").strip()
            if not raw or raw.startswith("/path/to"):
                return None
            return os.path.normcase(os.path.realpath(os.path.abspath(raw)))

        processing = folder(self.data["paths"].get("processing_directory"))
        frames = folder(self.data["paths"].get("input_frames"))
        if processing and frames and processing == frames:
            raise ConfigError(
                "paths.processing_directory and paths.input_frames are the "
                "same folder (%s) in %s. The frames folder is only ever read; "
                "all results must go to a separate processing directory."
                % (self.data["paths"]["processing_directory"], self.source))

        train = folder(self.data["mdocs"].get("train"))
        preprocessing = folder(self.data["mdocs"].get("preprocessing"))
        if train and preprocessing and train == preprocessing:
            raise ConfigError(
                "mdocs.train and mdocs.preprocessing are the same folder (%s) "
                "in %s. mdocs.train must be its own folder: a subset of the "
                "tilt series to train on, chosen by you or filled in "
                "automatically by the quality check."
                % (self.data["mdocs"]["train"], self.source))

    # ------------------------------------------------------------- accessors
    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    @property
    def input_frames(self):
        return os.path.abspath(self.data["paths"]["input_frames"])

    def require_input_frames(self):
        """Stop unless paths.input_frames is set and exists.

        Checked only where the raw frames are actually read - motion
        correction - not on every command: later steps never touch them (the
        one exception, IMOD's pixel-size fallback, reports a missing frame
        itself), so they keep working where the frames are not reachable.
        """
        frames = str(self.data["paths"].get("input_frames") or "").strip()
        if not frames or frames.startswith("/path/to"):
            raise SystemExit("ERROR: paths.input_frames is not set in %s"
                             % self.source)
        if not os.path.isdir(frames):
            raise SystemExit("ERROR: paths.input_frames does not exist: %s"
                             % frames)

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
        if name == "header":
            return self._header_exe()
        return self.data["executables"][name]

    @property
    def imod_dir(self):
        """IMOD's install folder (``executables.imod``), or "" for whatever
        IMOD the PATH / conda_env_imod provides."""
        return str(self.data["executables"].get("imod") or "").strip()

    def _header_exe(self):
        """IMOD's ``header``: an old-style executables.header if one is set
        (see _migrate_legacy), else ``<imod>/bin/header``, else ``header``
        on the PATH (or inside conda_env_imod - see :meth:`command`)."""
        legacy = getattr(self, "_legacy_header", "")
        if legacy:
            return legacy
        if self.imod_dir:
            return os.path.join(self.imod_dir, "bin", "header")
        return "header"

    # ------------------------------------------------------- conda wrapping
    def conda_env(self, name):
        """The conda/mamba environment *name* should be run in, or "".

        Read from the top-level ``conda_env_<name>`` setting (see
        :data:`CONDA_WRAPPABLE`), which sits next to ``paths`` rather than
        inside each program's own section because it is the same question for
        every one of them.
        """
        name = _CONDA_ENV_OF.get(name, name)
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
    def inverted_halfmaps_dir(self):
        """Contrast-inverted copies of the MissAlignment half maps, made for
        IsoNet2 (which expects dark features; WarpTools reconstructs them
        bright) - see isonet._linked_half_maps."""
        return self.sub("warp_tiltseries", "inverted_half-maps_IsoNet2")

    @property
    def inverted_uncorrected_halfmaps_dir(self):
        """The same for the uncorrected half maps (raw_halfMaps), when
        IsoNet2 runs on those."""
        return self.sub("warp_tiltseries", "inverted_half-maps_uncorrected_IsoNet2")

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

    @property
    def cryocare_dir(self):
        return self.sub("cryocare")

    @property
    def removed_mdocs_dir(self):
        """Where error messages suggest the USER moves the mdocs of broken
        tilt series - never written by the pipeline itself."""
        return self.sub("removed_mdocs")

    @property
    def denoiser(self):
        return self.data["denoiser"]

    @property
    def aligner(self):
        return self.data["aligner"]

    @property
    def autolevel(self):
        """Whether ts_autolevel levels the samples after the alignment."""
        return self.data["autolevel"]["enabled"]

    @property
    def gpu_memory_gb(self):
        return float(self.data.get("gpu_memory_gb") or 12)

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
    def imod_axis_search_dir(self):
        """Scratch home for the IMOD tilt axis search rounds (aligner:
        "imod"): one ``round_<n>/`` per round, each a ts_etomo_patches
        ``--output_processing`` - so warp_tiltseries/<tomo>.xml itself is
        only ever written by the final alignment."""
        return self.sub("warp_tiltseries", "tiltaxis_search_imod")

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
