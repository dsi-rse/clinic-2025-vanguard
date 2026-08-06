# Five-minute symposium talk

This Quarto reveal.js deck presents the vessel-GNN project to a nontechnical audience.
The storyline is based on `LAB_NOTEBOOK.md`, `docs/design/PLAN_weight_transfer.md`,
and `presentations/symposium_poster/`.

Build the HTML deck from this directory. The cluster's `/apps/default/bin/quarto`
launcher currently points to a missing YAML resource; the existing local Quarto
1.4.557 installation works:

```bash
/ess/home/home1/t-9svena/quarto-1.4.557/bin/quarto render slides.qmd
```

Speaker notes are embedded in `slides.qmd` and available in reveal.js presenter view.
The talk is written for approximately five minutes, including the title slide.

The central reported comparisons, matching the final `symposium_poster`, are:

- Paired pretraining ΔAUC for the voxel GNN: +0.149 [+0.046, +0.255].
- Paired pretraining ΔAUC for the graph-free (simple-summary) model: +0.034 [−0.049, +0.120] (95% CI crosses zero).
- Pretrained GNN vs. pretrained graph-free: +0.079 [+0.010, +0.155].
- Acquisition-timing-only model (frame count, duration, cadence; no imaging): AUC 0.587, vs. the pretrained GNN's 0.583 (Pearson r = 0.50 between their predictions).

The forecast-pretraining comparison is a completed result, not planned work.

## Architecture figure (`figures/arch5.svg`)

This figure is drawn outside the repo and checked in only as an SVG export, with
its text flattened to outlines. Its four input-column labels are therefore real
`<text>` elements laid over the export: "Ultrafast MRI / frames" and "MRI frame
order / and scan timing". Their font size (21.3px Arial-metric) and baselines are
measured from the outlines they replace, so they land in the same spot; headless
Chrome resolves the family to Liberation Sans, which is metric-compatible.
Re-exporting the figure from its source drawing drops these labels, so the source
needs the same wording.

The poster's copy of the same figure (`../symposium_poster/figures/arch4.pdf`,
also a source-less export) still carries the older "UFast MRI volumes" and "DCE
phase ordering and metadata" wording.
