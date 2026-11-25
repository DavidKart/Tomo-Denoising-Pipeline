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

After stopping a run yourself (Ctrl-C, `kill`), check `nvidia-smi`:
WarpTools' `WarpWorker` processes can outlive it and keep their GPUs busy.
Kill any left over by their PID before starting again.

## Steps, in order

1. **`create_config`** — writes `tomo_denoising_config.yaml`. At minimum set
   `paths.input_frames`, `paths.processing_directory`, and the three
   `mdocs.*` directories (`preprocessing`, `train`, `predict`).
   The pipeline refuses to start if the processing directory is the frames
   directory, or if `mdocs.train` is the same folder as
   `mdocs.preprocessing`.
   The file records the version it was written for (`config_version`), and
   every step refuses to run with a config from a different version. After a
   software update, write a fresh template (`create_config -o new.yaml`),
   carry your settings over, and set `config_version` to the new version. The
   error lists which keys differ.

2. **`motion_correct`** — WarpTools `create_settings` + `fs_motion_and_ctf`:
   motion correction and CTF estimation on the raw frames. Counts as finished
   only once it has completed successfully (`logs/.state/motion_correct.complete`)
   - after a crash, a restart runs it again instead of skipping it.
   → `warp_frameseries/`

3. **`align_tiltseries`** — imports the mdocs (`ts_import`), builds a stack
   per tilt series (`ts_stack`), solves one dataset-wide tilt axis via
   external AreTomo iterations (`aretomo.tilt_axis_iterations`), then aligns
   every tilt series natively with that axis fixed (`ts_aretomo3`).
   → `warp_tiltseries/<tomo>.xml`, `tilt_axis_refinement.txt`

   **Alternative: IMOD patch tracking** *(experimental)* — with
   `aligner: "imod"`, the final alignment is WarpTools' `ts_etomo_patches`
   instead of `ts_aretomo3`. IMOD only refines a starting axis, and the
   mdoc's is not trusted, so the starting axis is the AreTomo search's result
   as above — or `imod.initial_tilt_axis`, and then AreTomo is not run at all.
   `imod.tilt_axis_iterations` rounds (default 1) of
   `ts_etomo_patches --do_axis_search` on a random subset refine it (each in
   its own `warp_tiltseries/tiltaxis_search_imod/round_<n>/`, never touching
   the project's XMLs), stopping early once a round moves it by less than
   0.1°. Every tilt series is then aligned with that axis fixed. Both axes
   are in `tilt_axis_refinement.txt`. Changing `aligner` re-aligns on the
   next run. IMOD is found through `executables.imod` (its install folder),
   `conda_env_imod`, or the `PATH`.

   **Optional: leveling** *(experimental, off by default)* — with
   `autolevel.enabled: true`, WarpTools' `ts_autolevel` runs after the final
   alignment and stores how each sample slab is tilted (`LevelAngleX/Y` in
   its XML); every later reconstruction is rotated so the slab lies flat,
   which lets a thinner volume hold it. It can hurt IsoNet2's missing-wedge
   correction: `LevelAngleY` is added to the tilt range IsoNet2 gets (and
   called out in the log), but `LevelAngleX` tilts the wedge in a way IsoNet2
   cannot model. A `ts_autolevel` failure stops the run. Changing the
   setting re-aligns on the next run.

   Before anything is imported, each mdoc is checked for signs of a flawed
   acquisition: too few tilts, a narrow angular range, or a gap
   (`tiltseries_filter` in the config). By default, flagged series are left
   out of `<processing_directory>/mdocs_filtered/`, the directory `ts_import`
   actually reads. Your own mdoc directory is never touched. Tilt series
   AreTomo3 fails to align are excluded the same way, so the run continues
   with the rest. All exclusions and their reasons are listed in
   `excluded_tiltseries.json`, in the step report and at the end of
   `run_pipe` (IMOD failures the same way). The pipeline never deletes files: if files from an earlier run
   are in the way, it stops and lists them for you to remove - together with
   the commands that move the series' mdocs out of every mdoc folder into
   `removed_mdocs/`, if you want to drop the series for good.

   After `ts_aretomo3` / `ts_etomo_patches` (and again at the start of
   `reconstruct_tomograms`),
   every XML's tilt angles are checked: non-finite values, values beyond
   ±90° or all-identical angles stop the run with the affected series named,
   instead of a misleading crash in `ts_defocus_hand`.

4. **`reconstruct_tomograms`** — defocus handedness check
   (`ts_defocus_hand`), CTF refinement (`ts_ctf`), then reconstructs the
   full tomogram (always), the frame-split half maps, and the tilt-split half
   maps only if `quality_check.enable_tilt_split` is set (RESOLVE is the only
   thing that uses those). As part of `run_pipe`, the half maps are
   skipped by default when MissAlignment is about to run right after (step 5
   reconstructs its own instead) - the full tomogram is reconstructed either
   way. Run on its own, they are always reconstructed.
   → `warp_tiltseries/tomos_uncorrected/`, `warp_tiltseries/raw_halfMaps/`,
   `warp_tiltseries/raw_halfMaps_tiltSplit/` (links; the files themselves are
   in `warp_tiltseries/reconstruction_uncorrected/`)

5. **`run_missalignment`** *(runs by default as part of `run_pipe`; skip with
   `--skip-missalignment` or `missalign.enabled: false`)* — refines each
   tilt series' alignment in place with MissAlignment's local deformation
   model, then reconstructs the full tomogram and both half-map kinds from
   the refined alignment, into `missalignment/`. Once populated, its half
   maps are preferred automatically over `warp_tiltseries/raw_halfMaps*`
   everywhere downstream. A failure here stops `run_pipe` rather than
   falling back to the uncorrected alignment silently.
   → `warp_tiltseries/tomos_MissAlignment/`, `missalignment/raw_halfMaps*`
   (links; the files themselves are in `missalignment/reconstruction/`). The
   full tomograms are also linked under WarpTools' own names
   (`<tomo>_<angpix>Apx.mrc`) in `warp_tiltseries/reconstruction/`, where
   WarpTools' later commands such as `ts_template_match` look for them.

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
   set. IsoNet2 expects dark features, while WarpTools reconstructs them
   bright, so IsoNet works on contrast-inverted copies of the half maps
   (`warp_tiltseries/inverted_half-maps_IsoNet2/`, or
   `…_uncorrected_IsoNet2/` without MissAlignment), made the first time
   IsoNet needs them. The final tomograms therefore have dark features.
   Each tomogram's tilt range is taken from its WarpTools XML and converted
   to IsoNet2's sign convention (the XML's angles have the opposite sign:
   XML −52/46 → IsoNet −46/52), and its defocus (used only with
   `isonet.ctf_mode: "network"`) from the XML's per-tilt CTF fit.
   → `isonet/corrected_tomos_final/` (the final deliverable)

   **Alternative: `train_cryocare`** / **`predict_cryocare`** — with
   `denoiser: "cryocare"`, `run_pipe` uses CryoCARE instead of IsoNet (never
   both). It gets the same contrast-inverted half maps and the same
   `mdocs.train` / `mdocs.predict` sets, and denoises only (no missing-wedge
   correction).
   → `cryocare/denoised/` (the final deliverable), `results_cryocare.pdf`

## Run everything at once

Once step 1 (`create_config`) has been run and the config edited, steps 2-7
can be run in one go instead of one at a time:

```bash
tomo_denoising run_pipe
```
