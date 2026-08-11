"""Loading, validating and generating the tomo_denoising configuration."""

import os
import sys

import yaml

try:  # py3.9+
    from importlib.resources import files as _resource_files
except ImportError:  # pragma: no cover - py3.8
    _resource_files = None

DEFAULT_CONFIG_NAME = "tomo_denoising_config.yaml"

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
    "executables": {
        "motioncor": "MotionCor3",
        "aretomo": "AreTomo2",
        "newstack": "newstack",
        "header": "header",
        "ctffind": "ctffind",
        "cryocare_extract": "cryoCARE_extract_train_data.py",
        "cryocare_train": "cryoCARE_train.py",
        "cryocare_predict": "cryoCARE_predict.py",
        "isonet": "isonet.py",
        "resolve": "RESOLVE",
    },
    "overwrite": False,
    "gpus": [0, 1],
    "acquisition": {
        "pixel_size": "read",
        "frame_extension": "tiff",
        "gain": "",
        "voltage": 300.0,
        "spherical_aberration": 2.7,
        "amplitude_contrast": 0.1,
    },
    "motioncor": {
        "flip_gain": 2,
        "binning": 2,
        "iterations": 7,
        "tolerance": 0.5,
        "eer_sampling": 1,
        "fm_int_file": "",
        "extra_args": "",
    },
    "aretomo": {
        "binning": 4,
        "vol_z": 1200,
        "local_align": "-Patch 4 5",
        "tilt_axis_correction": True,
        "dark_tolerance": 0.5,
        "wbp": True,
        "tilt_axis_iterations": 3,
        "tilt_axis_subset_size": 10,
        "tilt_axis_seed": 0,
        "extra_args": "",
    },
    "tilt_split": {
        "enabled": True,
        "target_pixel_size": 50.0,
    },
    "quality_check": {
        "fast": True,
        "gpu_enabled": True,
        "cpu_threads": 16,
        "choose_training_set": True,
        "min_signal_ratio": 0.25,
        "outlier_factor": 1.5,
    },
    "cryocare": {
        "model_name": "denoise_model",
        "patch_shape": [72, 72, 72],
        "num_slices": 1800,
        "split": 0.9,
        "tilt_axis": "Y",
        "n_normalization_samples": 500,
        "epochs": 100,
        "steps_per_epoch": 200,
        "batch_size": 16,
        "unet_kern_size": 3,
        "unet_n_depth": 3,
        "unet_n_first": 32,
        "learning_rate": 0.0004,
        "n_tiles": [8, 8, 8],
        "overwrite": True,
    },
    "isonet": {
        "denoise_epochs": 50,
        "refine_epochs": 100,
        "ctf_mode": "network",
        "density_percentage": 50,
        "std_percentage": 80,
        "mw_weight": 200,
        "bfactor": 0,
        "tilt_range": [],
        "denoise_network": "auto",
        "refine_network": "auto",
        "extra_args": "",
    },
    "ctffind": {
        "box_size": 512,
        "min_resolution": 50.0,
        "max_resolution": 5.0,
        "min_defocus": 5000.0,
        "max_defocus": 70000.0,
        "defocus_step": 500.0,
    },
}


class ConfigError(Exception):
    """Raised when the configuration is missing or inconsistent."""


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
        handle.write(_template_text())
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
        self.data = _merge(DEFAULTS, data)
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
        if ext not in ("tif", "tiff", "mrc", "eer"):
            raise ConfigError(
                "acquisition.frame_extension must be one of tif/tiff/mrc/eer, "
                "got %r" % ext
            )
        self.data["acquisition"]["frame_extension"] = ext

        # Tilt splitting is additive now: it reuses the alignment solved on the
        # full frame-split run, so local patch alignment stays enabled.

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
        return bool(self.data["tilt_split"]["enabled"])

    @property
    def gpus(self):
        gpus = self.data.get("gpus") or [0]
        if isinstance(gpus, (int, str)):
            gpus = [gpus]
        return [int(g) for g in gpus]

    def exe(self, name):
        return self.data["executables"][name]

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
        configured but missing, since MotionCor3 would otherwise fail once per
        tilt with a less obvious message.
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
    def raw_halfmaps_dir(self):
        return self.sub("raw_halfMaps")

    @property
    def tiltsplit_halfmaps_dir(self):
        return self.sub("raw_halfMaps_tiltSplit")

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
    def cryocare_config_dir(self):
        return self.sub("cryocare_config")

    @property
    def cryocare_dir(self):
        return self.sub("cryocare")

    @property
    def isonet_config_dir(self):
        return self.sub("isonet_config")

    @property
    def isonet_dir(self):
        return self.sub("isonet")

    def rec_dir(self, tomo_name):
        return self.sub("rec_" + tomo_name)

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
