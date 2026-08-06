# Tomo-Denoising-Pipeline

One-click cryo-ET processing: from raw movie frames to denoised tomograms.

`tomo_denoising` chains **MotionCor3**, **IMOD newstack**, **AreTomo2**,
**cryoCARE** and **IsoNet** into a single configuration-driven workflow. Every
step is a subcommand, so you can run the whole thing or just the part you need.

---

## Installation

```bash
git clone https://github.com/DavidKart/Tomo-Denoising-Pipeline.git
cd Tomo-Denoising-Pipeline
pip install .
```

This puts `tomo_denoising` on your `PATH`.

### External software

The pipeline drives these programs; install them separately and make sure the
right one is reachable when you call the corresponding step.

| Program | Used by | Source |
| --- | --- | --- |
| [IMOD](https://bio3d.colorado.edu/imod/) (`newstack`, `header`) | `create_halfMaps` | bio3d.colorado.edu |
| [MotionCor3](https://github.com/czimaginginstitute/MotionCor3) | `create_halfMaps` | czimaginginstitute |
| [AreTomo2](https://github.com/czimaginginstitute/AreTomo2) | `create_halfMaps` | czimaginginstitute |
| [CTFFIND4](https://grigoriefflab.umassmed.edu/ctffind4) | `*_isonet` | Grigorieff lab |
| [cryoCARE](https://github.com/juglab/cryoCARE_pip) | `*_cryoCARE` | juglab |
| [IsoNet](https://github.com/IsoNet-cryoET/IsoNet) | `*_isonet` | IsoNet-cryoET |

> **Conda environments.** cryoCARE and IsoNet normally live in their own conda
> environments. Either activate the right environment before calling that step,
> or put the absolute path to the executable inside that environment into the
> `executables` section of the config. `tomo_denoising` itself only needs
> PyYAML, numpy and mrcfile, so it is happy in any of them.

---

## Workflow

```
tomo_denoising create_config       # 1. write a config, then edit it
tomo_denoising create_halfMaps     # 2. frames -> tomograms + even/odd half maps

tomo_denoising train_cryoCARE      # 3a. cryoCARE
tomo_denoising predict_cryoCARE

tomo_denoising train_isonet        # 3b. IsoNet
tomo_denoising predict_isonet
```

Every step reads `./tomo_denoising_config.yaml`. Point at a different one with
`-c/--config`. Add `-n/--dry-run` to any step to print the exact command lines
it would run without running them — useful for checking your parameters before
committing a GPU node.

### 1. `create_config`

```bash
tomo_denoising create_config
```

Writes a fully commented `tomo_denoising_config.yaml`. At minimum set:

```yaml
paths:
  input_frames:         "/data/session42/frames"     # read only, never written to
  processing_directory: "/scratch/session42"         # everything is written here
```

`input_frames` and `processing_directory` are deliberately separate — the raw
data stays untouched.

Then point at your `.mdoc` directories:

```yaml
mdocs:
  preprocessing: "/data/session42/mdocs"        # all series to reconstruct
  train:         "/data/session42/mdocsTrain"   # the 3-5 used for training
  predict:       "/data/session42/mdocs"        # all series to denoise
```

These are **independent locations** — they are not interpreted relative to
`processing_directory` and may live anywhere, including next to your raw data.
An absolute path is used as given; a relative one resolves against the
directory you run `tomo_denoising` from.

Every `*.mdoc` in each directory is used. The tomogram name is the mdoc file
name with `.mdoc` (and a trailing `.mrc`/`.st`) stripped, so `TS_01.mrc.mdoc`
becomes tomogram `TS_01`, reconstructed into `rec_TS_01/`.

### 2. `create_halfMaps`

MotionCor3 → newstack → AreTomo2, for each tilt series in `mdocs.preprocessing`.

Each tilt series gets one self-contained directory, and every half map lands in
one shared directory:

```
<processing_directory>/
  rec_TS_01/
    TS_01.rawtlt              tilt angles, sorted, failed tilts excluded
    motioncor/                *_averaged.mrc (+ _EVN/_ODD when splitting frames)
    stacks/                   full and half tilt series stacks, .aln
    aretomo_full/             TS_01_rec_tomo.mrc  (the full-dose tomogram)
    aretomo_even/             even_TS_01_rec_tomo.mrc + _projXY/_projXZ
    aretomo_odd/              odd_TS_01_rec_tomo.mrc  + _projXY/_projXZ
    logs/                     one log per program call
  rec_TS_02/
    ...
  raw_halfMaps/               symlinks - exactly two entries per tomogram
    even_TS_01_rec_tomo.mrc
    odd_TS_01_rec_tomo.mrc
    ...
```

AreTomo2 derives its side products (`_projXY.mrc`, `_projXZ.mrc`) from the
`-OutMrc` path rather than the working directory, so the half maps are
reconstructed inside `rec_<tomo>/` and only the volume itself is symlinked into
`raw_halfMaps/`. That keeps `raw_halfMaps/` to exactly the two half maps per
tomogram, with nothing moved or deleted. Downstream steps open them through the
link, so cryoCARE and IsoNet see them as ordinary files.

Already-finished tilt series are skipped; pass `--force` to redo them.

**Two ways to make half maps**, selected by `tilt_split.enabled`:

- `false` (default) — MotionCor3 splits the **frames** of every tilt into an
  even and an odd sum (`-SplitSum 1`). Both half maps contain all tilts.
- `true` — MotionCor3 averages all frames and the aligned **tilt series** is
  split instead: even-numbered tilts into one half map, odd-numbered into the
  other. Both the stack and the AreTomo `.aln` are split accordingly. Local
  patch alignment is switched off automatically, since patches cannot be
  aligned on half stacks.

### 3a. cryoCARE

```bash
tomo_denoising train_cryoCARE      # uses mdocs.train
tomo_denoising predict_cryoCARE    # uses mdocs.predict
```

The three cryoCARE `.json` files are **generated** from your YAML config into
`<processing_directory>/cryocare_config/` right before they are used, so they
always match the current settings and the current set of half maps. Results:

```
cryocare_config/    train_data_config.json, train_config.json, predict_config.json
cryocare/           train_data/, model/, logs/
denoised_cryoCARE/  the denoised tomograms
```

If prediction runs out of GPU memory, raise `cryocare.n_tiles` (e.g. to
`[16, 16, 16]`).

### 3b. IsoNet

```bash
tomo_denoising train_isonet        # uses mdocs.train
tomo_denoising predict_isonet      # uses mdocs.predict
```

`train_isonet` runs the whole preparation and training chain:

| # | Step | Notes |
| --- | --- | --- |
| 0 | CTFFIND4 defocus | on the first collected tilt of each series |
| 1 | `prepare_star` | plus per-tomogram defocus and tilt range |
| 2 | `denoise` | `isonet.denoise_epochs`, default 50 |
| 3 | `predict` | applies the denoiser, needed for the masks |
| 4 | `make_mask` | from the denoised volumes |
| 5 | `refine` | `isonet.refine_epochs`, default 100 |

`predict_isonet` is step 6: apply the refined network to the prediction set.

**Defocus.** IsoNet needs a defocus per tomogram. CTFFIND4 is run on the
motion-corrected average of the **first collected tilt** of each series — the
full `*_averaged.mrc`, never one of the `_EVN`/`_ODD` half images, so the
estimate uses that tilt's full dose. The value used is the mean of columns 2
and 3 (defocus 1 and defocus 2), in Angstrom. The output is cached as
`isonet_config/<tomo>_PS.txt` and reused on later runs; force a re-run with
`--force-ctf`.

These are handed to `prepare_star` as a **list, one value per tomogram**, in
the order the linked half maps are sorted (by tomogram name). The order is
printed before the call so you can check it:

```
[tomo_denoising] tomograms, in the order prepare_star receives them:
[tomo_denoising]     TS_01     defocus   58309.9 A   tilt  -60.0 to   60.0
[tomo_denoising]     TS_02     defocus   41000.5 A   tilt  -60.0 to   60.0
```

A tomogram whose CTFFIND4 run failed is **excluded** from the run rather than
skipped in place — a gap would shift every later value of the list onto the
wrong tomogram.

The star file `prepare_star` writes is used exactly as produced; nothing
rewrites it afterwards.

**Tilt range.** Read per tomogram as the minimum and maximum of
`rec_<tomo>/<tomo>.rawtlt`. `prepare_star` accepts only one range for the whole
set, so if the tomograms disagree by more than 2°, the **widest** range is used
and the outliers are named:

```
WARNING: tilt ranges differ across 3 tomograms:
WARNING:     TS_01     -60.0 to   60.0
WARNING:     TS_07     -40.0 to   42.0   <- 20.0/18.0 deg narrower
WARNING: Using the widest range: -60.0 to 60.0 deg.
WARNING: TS_07 will be under-corrected ...
```

That direction is deliberate. Declaring a range **wider** than a tomogram
really has makes IsoNet assume the outer shells hold data, so it leaves them
alone and that tomogram is merely under-corrected. Declaring a **narrower**
range makes IsoNet treat measured shells as missing and synthesise over them,
destroying real information. The first is recoverable, the second is not.

If the spread matters for your data, process the outliers in a separate run, or
pin the range explicitly with `isonet.tilt_range: [-60, 60]`.

**Constant optics.** `voltage`, `spherical_aberration` and
`amplitude_contrast` apply to the whole dataset and come from the
`acquisition` section (300 kV, 2.7 mm, 0.1 by default). They are used for the
CTFFIND4 estimate. If your IsoNet build accepts them on `prepare_star` too,
add them via `isonet.extra_args`.

Layout:

```
isonet_config/    <tomo>_PS.txt (CTFFIND4 results, cached)
                  prepare_train/tomograms.star, prepare_predict/tomograms.star
isonet/           train/{even,odd}, predict/{even,odd}  (links to raw_halfMaps)
                  denoise/, isonet_maps/, mask/, logs/
```

Half maps are made visible to IsoNet by **linking**, never by copying — the
originals in `raw_halfMaps/` are the only copy on disk.

---

## Notes

- **Nothing is moved or deleted.** All directories are created up front and
  every program is given an explicit output path; where a program insists on
  writing next to its input, it is run with its working directory set to where
  that output belongs. There is no `rm` and no `mv` anywhere in the pipeline.
- **Pixel size.** `acquisition.pixel_size: read` takes `PixelSpacing` from the
  mdoc and falls back to IMOD `header` on the first raw frame. Set a number if
  both are wrong. Total binning of the final tomogram is
  `motioncor.binning * aretomo.binning`.
- **`VolZ`** is exposed as `aretomo.vol_z` (default 1200, in binned pixels).
  Raise it for thick lamellae.
- **CTF correction** of the tilt series is intentionally not part of this
  pipeline. cryoCARE and IsoNet both operate on uncorrected reconstructions;
  IsoNet handles the CTF itself via `isonet.ctf_mode`.
- **Training subsets.** For large datasets 3-5 representative tomograms in
  `mdocsTrain/` are normally plenty.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
