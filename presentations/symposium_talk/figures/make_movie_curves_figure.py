"""Render the Duke-vs-UChicago enhancement-curve comparison for the talk's "movie" slide.

Both curves are real, baseline-referenced DCE enhancement at one vessel node,
plotted against real acquisition seconds since injection -- a shared, honest
time axis, not just frame index. The point: UChicago's ultrafast protocol
samples the same kind of contrast wash-in far more densely, in far less real
time, than Duke's standard protocol.

Duke curve values: presentations/symposium_poster/figures/architecture_data/curves.tex
(case DUKE_107, node A, git commit 2ce2b25bd4a7d321dc9dbd58c22b4039e06008e4).
Duke phase timing: /gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx,
row patient_id=DUKE_107, column acquisition_times = [0, 228, 349, 470, 591]
(seconds; the public MAMA-MIA clinical spreadsheet, not a Vanguard-produced
sidecar -- there is no ufast_times_seconds.npy for Duke, unlike UChicago).
UChicago values: figures/uchicago_curve_data/uchicago_curve.json (case
uchicago_zhen_8b8f05c8b8d1eea9751542d0, produced by
analysis/uchicago_curve_export.py). Both curves go through the same
transform, gnn.kinetics.baseline_relative_curve, via gnn.pretrain.node_series.

Run from presentations/symposium_talk with:
    micromamba run -n vanguard python figures/make_movie_curves_figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MAROON = "#800000"
GRAY = "#737b88"
INK = "#252525"

HERE = Path(__file__).parent
OUT_SVG = HERE / "movie_curves.svg"
UCHICAGO_JSON = HERE / "uchicago_curve_data" / "uchicago_curve.json"

# DUKE_107, node A, from architecture_data/curves.tex (frame index -> enhancement).
DUKE_CURVE = [0.0000, 0.6206, 1.0000, 0.2914, 0.6115]
# DUKE_107's real per-phase acquisition seconds, from the public MAMA-MIA
# clinical_and_imaging_info.xlsx (see module docstring).
DUKE_TIMES_SECONDS = [0, 228, 349, 470, 591]


def main() -> None:
    """Render the Duke-vs-UChicago curve comparison to movie_curves.svg."""
    uchicago = json.loads(UCHICAGO_JSON.read_text())
    uchicago_curve = uchicago["enhancement"]
    uchicago_times = uchicago["times_seconds"]

    fig, ax = plt.subplots(figsize=(8.4, 2.7))

    ax.plot(
        uchicago_times,
        uchicago_curve,
        color=MAROON,
        lw=2.4,
        marker="o",
        ms=5,
        markerfacecolor=MAROON,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label=f"UChicago ultrafast ({len(uchicago_curve)} frames, {uchicago_times[-1]:.0f}s)",
        zorder=3,
    )
    ax.plot(
        DUKE_TIMES_SECONDS,
        DUKE_CURVE,
        color=GRAY,
        lw=2.4,
        marker="s",
        ms=6,
        markerfacecolor=GRAY,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label=f"Duke, public reference ({len(DUKE_CURVE)} phases, {DUKE_TIMES_SECONDS[-1]}s)",
        zorder=3,
    )

    ax.set_xlabel("Seconds since injection", color=INK, fontsize=11)
    ax.set_ylabel("Relative enhancement", color=INK, fontsize=11)
    ax.tick_params(colors=INK, labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)
    ax.set_xlim(-10, DUKE_TIMES_SECONDS[-1] + 10)
    ax.legend(frameon=False, fontsize=10, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT_SVG, transparent=True)
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
