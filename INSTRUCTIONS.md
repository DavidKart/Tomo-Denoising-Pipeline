# Instructions

See [README.md](README.md) for installation. Every step reads
`./tomo_denoising_config.yaml` (override with `-c/--config`); every setting is
documented inline in the config `create_config` writes, so this page only
covers what each step does and in what order.

Add `-n/--dry-run` to any step to print the exact commands it would run
without running them - useful before committing a GPU node.

## Resuming

With `overwrite: false` (the default), every step checks what's already
finished and only redoes what's missing or was interrupted - half maps that
already exist are skipped, a completed training isn't repeated, a converged
tilt axis is reused. `-f/--force` overrides this once; `overwrite: true`
disables resuming permanently. Every step writes a report to
`<processing_directory>/logs/<step>.log`.

Forcing a re-run also re-points the `raw_halfMaps*` symlinks at the new
reconstruction. That matters because their names are fixed
(`even_<tomo>_rec_tomo.mrc`) while the files they point at are named by
WarpTools after the pixel size - so changing `warptools.recon_angpix` writes
new files, and a stale link would quietly keep every downstream step on the
old ones. Likewise, an automatically chosen training set is re-picked from the
fresh RESOLVE ranking; mdocs you placed in `mdocs.train` yourself are never
touched, forced or not.

## Steps, in order

1. **`create_config`** — writes `tomo_denoising_config.yaml`. At minimum set
   `paths.input_frames`, `paths.processing_directory`, and the three
   `mdocs.*` directories (`preprocessing`, `train`, `predict`).
   The file records the version it was written for (`config_version`), and
   every step refuses to run with a config from a different version. After a
   software update, write a fresh template (`create_config -o new.yaml`),
   carry your settings over, and set `config_version` to the new version. The
   error lists which keys differ.

2. **`motion_correct`** — WarpTools `create_settings` + `fs_motion_and_ctf`:
   motion correction and CTF estimation on the raw frames.
   → `warp_frameseries/`

3. **`align_tiltseries`** — imports the mdocs (`ts_import`), builds a stack
   per tilt series (`ts_stack`), solves one dataset-wide tilt axis via
   external AreTomo iterations (`aretomo.tilt_axis_iterations`), then aligns
   every tilt series natively with that axis fixed (`ts_aretomo3`).
   → `warp_tiltseries/<tomo>.xml`, `tilt_axis_refinement.txt`

   Before anything is imported, each mdoc is checked for signs of a flawed
   acquisition: too few tilts, a narrow angular range, or a gap
   (`tiltseries_filter` in the config). By default, flagged series are left
   out of `<processing_directory>/mdocs_filtered/`, the directory `ts_import`
   actually reads. Your own mdoc directory is never touched. Tilt series
   AreTomo3 fails to align are excluded the same way, so the run continues
   with the rest. All exclusions and their reasons are listed in
   `excluded_tiltseries.json`, in the step report and at the end of
   `run_pipe`. The pipeline never deletes files: if files from an earlier run
   are in the way, it stops and lists them for you to remove.

4. **`reconstruct_tomograms`** — defocus handedness check
   (`ts_defocus_hand`), CTF refinement (`ts_ctf`), then reconstructs the
   full tomogram (always), the frame-split half maps, and the tilt-split half
   maps only if `quality_check.enable_tilt_split` is set (RESOLVE is the only
   thing that uses those). As part of `run_pipe`, the half maps are
   skipped by default when MissAlignment is about to run right after (step 5
   reconstructs its own instead) - the full tomogram is reconstructed either
   way. Run on its own, they are always reconstructed.
   → `warp_tiltseries/tomograms/`, `warp_tiltseries/raw_halfMaps/`,
   `warp_tiltseries/raw_halfMaps_tiltSplit/`

5. **`run_missalignment`** *(runs by default as part of `run_pipe`; skip with
   `--skip-missalignment` or `missalign.enabled: false`)* — refines each
   tilt series' alignment in place with MissAlignment's local deformation
   model, then reconstructs the full tomogram and both half-map kinds from
   the refined alignment - the same three directories as step 4, just under
   `missalignment/`. Once populated, its half maps are preferred
   automatically over `warp_tiltseries/raw_halfMaps*` everywhere downstream.
   A failure here stops `run_pipe` rather than falling back to the
   uncorrected alignment silently.
   → `missalignment/tomograms/`, `missalignment/raw_halfMaps*`

6. **`quality_check`** — runs RESOLVE over a set of even/odd half maps
   (whichever alignment is preferred - see step 5) to estimate resolution and
   signal ratio per tomogram, auto-picks a training set into `mdocs.train` if
   it's empty, and (re)builds the results figure once denoised tomograms
   exist. Run on its own with no half maps yet, it reconstructs them first.
   → `resolve/summary.tsv`, `results_isonet.pdf`

   `quality_check.enable_tilt_split` decides *which* half maps:

   - `true` — a dedicated tilt-split reconstruction at
     `tilt_split_target_pixel_size` (50 Å), made for this measurement and
     nothing else. Conservative number, costs an extra reconstruction pass.
     If it doesn't exist yet, this step reconstructs it.
   - `false` — the frame-split half maps already reconstructed for IsoNet. No
     extra reconstruction at all, and `raw_halfMaps_tiltSplit/` is never
     created. More optimistic number, same ranking.

   The results figure is captioned accordingly ("Tilt-split resolution" /
   "Frame-split resolution"), and flipping the setting re-measures instead of
   reusing the previous `summary.tsv`.

7. **`train_isonet`** / **`predict_isonet`** — IsoNet training (on
   `mdocs.train`) and prediction (on `mdocs.predict`): denoise, mask,
   missing-wedge refine, then apply the refined network to the prediction
   set.
   → `isonet/corrected_tomos_final/` (the final deliverable)

## Run everything at once

Once step 1 (`create_config`) has been run and the config edited, steps 2-7
can be run in one go instead of one at a time:

```bash
tomo_denoising run_pipe
```
