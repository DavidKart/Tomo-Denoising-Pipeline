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
        # The SAME AreTomo build/version is used everywhere - both for our own
        # external tilt-axis-search calls and passed to WarpTools'
        # ts_aretomo3 --exe.
        "aretomo": "AreTomo3",
        "header": "header",                # IMOD, pixel-size fallback only
        "isonet": "isonet.py",
        "resolve": "RESOLVE",
        "warptools": "WarpTools",
        "missalign": "miss-alignment",
        # Only used when missalign.conda_env is set - see that section.
        "conda": "conda",
    },
    "overwrite": False,
    "gpus": [0, 1],
    "acquisition": {
        "pixel_size": "read",
        "frame_extension": "tiff",         # tif/tiff/eer
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
    "tilt_split": {
        "enabled": True,
        "target_pixel_size": 40.0,
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
        # Name (or path) of the conda/mamba environment miss-alignment is
        # installed in, if it lives in a separate one from the rest of the
        # pipeline. When set, "miss-alignment train" is run via
        # `conda run -n <env> --no-capture-output ...` (executables.conda),
        # so that environment does not need to be activated beforehand.
        # Leave empty (default) if executables.missalign is already directly
        # runnable (e.g. an absolute path into that environment's own bin/,
        # or miss-alignment is installed in the same environment
        # tomo_denoising itself runs in).
        "conda_env": "",
        "gpu_memory_gb": 12,
        # null = auto-compute from gpu_memory_gb / config.gpus. Set any of
        # these explicitly to override the auto-partitioning.
        "batch_size": None,
        "training_devices": None,
        "reconstruction_devices": None,
        "dataloaders_per_trainer": None,
        "prepare_stacks_gb": 10.0,
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
        if ext not in ("tif", "tiff", "eer"):
            raise ConfigError(
                "acquisition.frame_extension must be one of tif/tiff/eer, "
                "got %r" % ext
            )
        self.data["acquisition"]["frame_extension"] = ext

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
    def tomograms_dir(self):
        """Full, non-split reconstructions - always produced by
        reconstruct_tomograms, independent of whether half maps are. Lives
        under warp_tiltseries/ (not directly under processing_directory) so
        the layout is the analogue of missalignment_tomograms_dir - both
        stages produce the same three named directories
        (tomograms/raw_halfMaps/raw_halfMaps_tiltSplit), one uncorrected,
        one MissAlignment-corrected."""
        return self.sub("warp_tiltseries", "tomograms")

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
    def reconstruction_dir(self):
        return self.sub("warp_tiltseries", "reconstruction")

    @property
    def mdocs_filtered_dir(self):
        """Copies of mdocs.preprocessing with tilts that have no WarpTools
        frame-series data removed - see warptools._sanitize_mdocs. Only
        created (and only used for ts_import) when at least one mdoc
        actually needed filtering; mdocs.preprocessing itself is never
        written to."""
        return self.sub("mdocs_filtered")

    # -------------------------------------------------------- MissAlignment
    @property
    def missalignment_dir(self):
        return self.sub("missalignment")

    @property
    def missalignment_tomograms_dir(self):
        """MissAlignment's own full, non-split reconstruction - the
        analogue of tomograms_dir, from the refined alignment instead of the
        native one."""
        return self.sub("missalignment", "tomograms")

    @property
    def missalignment_halfmaps_dir(self):
        return self.sub("missalignment", "raw_halfMaps")

    @property
    def missalignment_tiltsplit_halfmaps_dir(self):
        return self.sub("missalignment", "raw_halfMaps_tiltSplit")

    @property
    def missalignment_reconstruction_dir(self):
        """MissAlignment's own ts_reconstruct scratch location
        (--output_processing) - deliberately SEPARATE from
        reconstruction_dir (warp_tiltseries/reconstruction), which
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
