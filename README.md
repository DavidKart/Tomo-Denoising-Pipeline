# Tomo-Denoising-Pipeline

One-click cryo-ET processing: raw movie frames → denoised tomograms, built on
WarpTools, AreTomo, an optional MissAlignment refinement step, and IsoNet.

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

IMOD is the one **optional** entry. Its `header` is used for exactly one
thing: reading the pixel size off a raw frame when `acquisition.pixel_size` is
`read` *and* the mdocs carry no `PixelSpacing`. Without IMOD, set
`acquisition.pixel_size` explicitly and nothing else changes.

| Program | Used by | Source |
| --- | --- | --- |
| [WarpTools](https://warpem.github.io/warp/) | `motion_correct`, `align_tiltseries`, `reconstruct_tomograms`, `run_missalignment` | warpem |
| [AreTomo3](https://github.com/czimaginginstitute/AreTomo3) | `align_tiltseries` | czimaginginstitute |
| [IMOD](https://bio3d.colorado.edu/imod/) (`header`) | *optional* - pixel-size fallback only | bio3d.colorado.edu |
| [RESOLVE](https://github.com/DavidKart/RESOLVE) | `quality_check` | DavidKart |
| [IsoNet](https://github.com/IsoNet-cryoET/IsoNet2) | `*_isonet` | IsoNet-cryoET |
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
