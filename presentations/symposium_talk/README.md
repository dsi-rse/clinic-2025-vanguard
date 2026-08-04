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
