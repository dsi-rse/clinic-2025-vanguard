# UChicago enhancement-curve input

Real-data input for `../movie_curves.svg`, the talk's Duke-vs-UChicago
enhancement-curve comparison on the "movie" slide. Generated -- never
hand-edited. Regenerating overwrites `uchicago_curve.json`.

Numbers only, on purpose: UChicago DCE-MRI is not deidentified, so this
directory (and the figure built from it) must never contain an MRI frame,
mask, or other image -- see `../../symposium_poster/figures/architecture_data/README.md`
for the same policy applied to the poster's architecture figure.

## Regenerate

```
sbatch gnn/slurm/submit_uchicago_curve_export.slurm
  (CASE=uchicago_zhen_8b8f05c8b8d1eea9751542d0 DATASET=uchicago_zhen)
```

Then rebuild the comparison figure:

```
cd presentations/symposium_talk
micromamba run -n vanguard python figures/make_movie_curves_figure.py
```

Source: `analysis/uchicago_curve_export.py`, git commit
`93324ec5f82de0c5784bdb063c22081054caa410`.

## Inputs

- Exam `uchicago_zhen_8b8f05c8b8d1eea9751542d0` from the published
  `uchicago_ultrafast_pretreatment_cohort_v1` cohort root.
- Centerlines: `/gpfs/data/karczmar-lab/vanguard/uchicago_ultrafast_pretreatment_cohort_v1/centerlines`
- DCE images: `/gpfs/data/karczmar-lab/vanguard/uchicago_ultrafast_pretreatment_cohort_v1/images/uchicago_zhen`
- Frames: 24, physical acquisition seconds from the exam's own
  `ufast_times_seconds.npy` sidecar.

## What is here

- `uchicago_curve.json` -- one vessel node's real baseline-referenced DCE
  enhancement curve (`gnn.kinetics.baseline_relative_curve`, the same
  transform the forecasting pretraining task uses, via
  `gnn.pretrain.node_series.voxel_node_series`), plus the physical
  acquisition seconds and provenance. `baseline_frame_count=5`,
  `relative_enhancement=true`, per the exam's own `run_summary.json`
  kinetic-feature policy -- not a hardcoded assumption.

## Node selection

The skeleton voxel nearest the centroid of all skeleton voxels for this
exam: a deterministic, non-cherry-picked choice of "a typical vessel
location," not the node with the most dramatic-looking curve.

## Things a reader should know

- The Duke curve it's plotted against (`presentations/symposium_poster/figures/architecture_data/curves.tex`,
  case `DUKE_107`) has no Vanguard-produced `ufast_times_seconds.npy`
  sidecar, but its real per-phase acquisition seconds are published in the
  public MAMA-MIA `clinical_and_imaging_info.xlsx` (`acquisition_times`
  column, row `patient_id=DUKE_107`: `[0, 228, 349, 470, 591]`), hardcoded
  with that citation in `make_movie_curves_figure.py`. Both curves plot
  against this shared real-seconds axis, not frame index.
- `_resolve_case` (used by the poster's Duke export script) walks the
  centerline tree with `Path.rglob`, which does not descend into symlinked
  directories on this Python version. The published UChicago cohort exposes
  each exam's centerline directory as a symlink into a `_build/` tree, so
  `uchicago_curve_export.py` joins the study directory directly from
  `--dataset`/`--case-id` instead of searching for it.
