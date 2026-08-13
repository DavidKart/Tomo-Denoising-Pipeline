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
| [RESOLVE](https://github.com/DavidKart/RESOLVE) | `quality_check` | DavidKart |
| [cryoCARE](https://github.com/juglab/cryoCARE_pip) | `*_cryoCARE` | juglab |
| [IsoNet](https://github.com/IsoNet-cryoET/IsoNet2) | `*_isonet` | IsoNet-cryoET |

---

## Workflow

```
tomo_denoising create_config       # 1. write a config, then edit it
tomo_denoising create_halfMaps     # 2. frames -> tomograms + half maps
tomo_denoising quality_check       # 3. RESOLVE, training set, results figure

tomo_denoising train_cryoCARE      # 4a. cryoCARE
tomo_denoising predict_cryoCARE

tomo_denoising train_isonet        # 4b. IsoNet
tomo_denoising predict_isonet
```

Or all of it at once:

```bash
tomo_denoising run_pipe --denoiser cryocare
tomo_denoising run_pipe --denoiser isonet
```

Every step reads `./tomo_denoising_config.yaml`. Point at a different one with
`-c/--config`. Add `-n/--dry-run` to any step to print the exact command lines
it would run without running them — useful for checking your parameters before
committing a GPU node.

### Resuming and logs

With `overwrite: false` (the default) every step checks what is already
finished and only redoes what is missing or was interrupted: tilt series that
already have half maps are skipped, a completed training is not repeated, and a
converged tilt axis is reused. `-f/--force` on any step overrides this once;
`overwrite: true` disables it permanently.

Every step writes a report to `<processing_directory>/logs/<step>.log`.

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
  train:         "/data/session42/mdocsTrain"   # the 3-5 used for training, selection can be automated.
  predict:       "/data/session42/mdocs"        # all series to denoise
```

### 2. `create_halfMaps`

MotionCor3 → newstack → AreTomo2.

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
    aretomo_tiltSplit_even/   even_TS_01_rec_tomo.mrc  (coarse, for RESOLVE)
    aretomo_tiltSplit_odd/    odd_TS_01_rec_tomo.mrc
  rec_TS_02/
    ...
  raw_halfMaps/               symlinks - exactly two entries per tomogram
    even_TS_01_rec_tomo.mrc
    odd_TS_01_rec_tomo.mrc
  raw_halfMaps_tiltSplit/     symlinks - the same, tilt-split (for RESOLVE)
  logs/                       one report per pipeline step
  tilt_axis_refinement.txt    what the tilt axis converged to, and how
```

Already-finished tilt series are skipped; pass `--force` to redo them.

#### Tilt axis refinement

The tilt axis belongs to the microscope, not to one tilt series, so it is
solved once for the dataset and then held fixed.
`aretomo.tilt_axis_iterations` alignment rounds each take a
different random subset of `tilt_axis_subset_size` tilt series and
run AreTomo2. The angles are combined, the result is passed to the next round as,
and after the last round **all** tilt series are reconstructed 
once with `-TiltAxis <converged> -1` (fixed, no further refinement). 
Set `tilt_axis_iterations: 0` to align every series independently instead.

The angles are combined with the **geometric median of their unit vectors**
(the mean for three or fewer), similar to what WarpTools does.

Each round and the final value are recorded in
`<processing_directory>/tilt_axis_refinement.txt`.

#### Frame splitting and tilt splitting

**Frame splitting always happens.** MotionCor3 splits the frames of every tilt
into an even and an odd sum (`-SplitSum 1`), and those reconstructions become
`raw_halfMaps/` — the ones cryoCARE and IsoNet train on.

**Tilt splitting is done in addition** when `tilt_split.enabled` is true (the
default), and reuses everything the frame-split run already produced. 
The result goes to `raw_halfMaps_tiltSplit/` and is what `quality_check` measures.

Because those only feed the resolution estimate, they are reconstructed much
more coarsely (Resolution measured on tilt-split tomograms has many advantages but requires 
higher pixel size, tilt-split half-tomos are much more different than frame-split half-tomos.
The resulting resolution estimate is on a different scale from the actual 
tomogram resolution or resolution measured on frame-split tomograms, 
but more meaningful and more artifact-free!): `tilt_split.target_pixel_size` 
(default 50 Å) sets the pixel size aimed for and the binning is derived from it. 
Binning is an integer so the target is approached, not hit, and the value reached is reported.

### 3. `quality_check`

```bash
tomo_denoising quality_check
```

Runs [RESOLVE](https://github.com/DavidKart/RESOLVE) in batch mode over
`raw_halfMaps_tiltSplit/`. It needs two genuinely independent reconstructions
of the same volume, which is what the tilt split provides. Everything RESOLVE
writes lands in `<processing_directory>/resolve/`, including `summary.tsv` with
one row per tomogram: `id`, `median_resolution`, `signal_ratio`.

**Choosing a training set.** If `mdocs.train` does not exist or holds no
`.mdoc` files, it is populated by symlinking the best tomograms' mdocs from
`mdocs.preprocessing`. An existing selection is never touched.

| tomograms measured | chosen |
| --- | --- |
| < 10 | 3 |
| ≥ 10 | 4 |
| ≥ 20 | 5 |

Anything with `signal_ratio` below `min_signal_ratio` (default 0.35) is never
eligible, however good its resolution. 

**The results figure.** `results_<denoiser>.pdf` in the base processing
directory: a central Z, Y and X slice of every **denoised** tomogram.

### 4a. cryoCARE

```bash
tomo_denoising train_cryoCARE      # uses mdocs.train
tomo_denoising predict_cryoCARE    # uses mdocs.predict
```

The three cryoCARE `.json` files are **generated** from your YAML config into
`<processing_directory>/cryocare_config/` right before they are used, so they
always match the current settings and the current set of half maps. 

### 4b. IsoNet

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
and 3 (defocus 1 and defocus 2), in Angstrom. 

These are handed to `prepare_star` as a **list, one value per tomogram**, in
the order the linked half maps are sorted (by tomogram name). The order is
printed before the call so you can check it.

**Tilt range.** Read per tomogram as the minimum and maximum of
`rec_<tomo>/<tomo>.rawtlt`. `tomograms.star` is by default corrected
row by row: each tomogram's `rlnTiltMin`/`rlnTiltMax` is rewritten from the
`TILT` column of its own AreTomo2 `.aln` (in `rec_<tomo>/aretomo_full/`),
which reflects dark tilts AreTomo2 dropped and, if `aretomo.tilt_axis_correction` 
is on, the corrected tilt angles too. Set `isonet.fix_tilt_range: [-60, 60]` to skip 
this and pin the **same** fixed range on every row instead; this is **not recommended** 
unless you know it applies identically to every tomogram in the set.

**Constant optics.** `voltage`, `spherical_aberration` and
`amplitude_contrast` apply to the whole dataset and come from the
`acquisition` section (300 kV, 2.7 mm, 0.1 by default). They are used for the
CTFFIND4 estimate.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
