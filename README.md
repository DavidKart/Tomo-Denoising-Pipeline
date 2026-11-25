# Tomo-Denoising-Pipeline

One-click cryo-ET processing: raw movie frames → denoised tomograms, built on
WarpTools, AreTomo, an optional MissAlignment refinement step, and IsoNet
(or, alternatively, CryoCARE).

## Installation

```bash
conda create -n tomo_denoising python=3.11
conda activate tomo_denoising

git clone https://github.com/DavidKart/Tomo-Denoising-Pipeline.git
cd Tomo-Denoising-Pipeline
pip install .
```

This puts `tomo_denoising` on your `PATH`.

### External software

Install these separately and make sure the right one is reachable when you
call the corresponding step (`executables` section of the config). Anything
that lives in a conda environment of its own - WarpTools and miss-alignment
commonly do - can be named there instead of activated: see the
`conda_env_warptools` / `conda_env_missalign` settings.

IMOD is **optional**. It is needed for `aligner: "imod"` (WarpTools'
`ts_etomo_patches` runs IMOD's `batchruntomo`), and otherwise only for
reading the pixel size off a raw frame when `acquisition.pixel_size` is
`read` *and* the mdocs carry no `PixelSpacing`. Point `executables.imod` at
IMOD's install folder (or set `conda_env_imod`), or have it on the `PATH`.
Without IMOD, keep `aligner: "aretomo"` and set `acquisition.pixel_size`
explicitly.

| Program | Used by | Source |
| --- | --- | --- |
| [WarpTools](https://warpem.github.io/warp/) | `motion_correct`, `align_tiltseries`, `reconstruct_tomograms`, `run_missalignment` | warpem |
| [AreTomo3](https://github.com/czimaginginstitute/AreTomo3) | `align_tiltseries` (not needed with `aligner: "imod"` and `imod.initial_tilt_axis` set) | czimaginginstitute |
| [IMOD](https://bio3d.colorado.edu/imod/) | *optional* - `align_tiltseries` with `aligner: "imod"`, pixel-size fallback | bio3d.colorado.edu |
| [RESOLVE](https://github.com/DavidKart/RESOLVE) | `quality_check` | DavidKart |
| [IsoNet](https://github.com/IsoNet-cryoET/IsoNet2) | `*_isonet` (default denoiser) | IsoNet-cryoET |
| [CryoCARE](https://github.com/juglab/cryoCARE_pip) | `*_cryocare` - *optional*, only with `denoiser: "cryocare"` | juglab |
| [miss-alignment](https://github.com/warpem/miss-alignment) | `run_missalignment` | warpem |

## Quick start

```bash
tomo_denoising create_config      # writes a fully commented config - edit it
tomo_denoising run_pipe           # runs everything, in order
```

Every setting is documented inline in the generated `tomo_denoising_config.yaml`.
A config is tied to the version that wrote it: after updating tomo_denoising,
regenerate it with `create_config` and carry your settings over (the pipeline
refuses to run with an outdated one).
For what each step actually does and in what order, see
**[INSTRUCTIONS.md](INSTRUCTIONS.md)**.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
