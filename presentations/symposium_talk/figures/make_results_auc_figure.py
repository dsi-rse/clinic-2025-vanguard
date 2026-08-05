"""Render the five-row forest plot for the talk's graph-model results slide.

Extends the poster's four-arm comparison (``presentations/symposium_poster/
figures/make_results_figure.py``, ``transfer_results.csv``) with a fifth row:
the protocol-only audit that predicts pCR from acquisition timing alone
(frame count, scan duration, median frame-to-frame interval), no vessel
imaging at all. The poster keeps this audit as prose in a separate block;
the talk puts it on the same plot as the four matched arms so the
batch-effect concern lands next to the result it complicates. This script
only reads the poster's CSV -- it does not modify the poster's script,
data, or PDF.

Protocol-only numbers: ``/gpfs/data/karczmar-lab/workspaces/spencervenancio/
experiments/uchicago_temporal_transfer/finetune_179_seed0_v1/metrics.json``,
key ``protocol_only_audit`` (same patient-grouped OOF folds and bootstrap CI
procedure as the four matched arms). GNN-pretrained vs. protocol-only
prediction correlation, Pearson r = 0.50, is from the same experiment's
README.md.

Run from presentations/symposium_talk with:
    micromamba run -n vanguard python figures/make_results_auc_figure.py
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

DATA_CSV = (
    Path(__file__).parent.parent.parent
    / "symposium_poster"
    / "figures"
    / "transfer_results.csv"
)
OUT_SVG = Path(__file__).parent / "results_auc.svg"

MAROON = "#800000"
GREY = "#6B7280"
GOLD = "#D6A800"
INK = "#282828"
LIGHT = "#D1D5DB"

# protocol_only_audit from finetune_179_seed0_v1/metrics.json (see module docstring).
PROTOCOL_ONLY = {
    "label": "Timing only, no imaging",
    "estimate": 0.587141577060932,
    "ci_low": 0.48271014304598236,
    "ci_high": 0.6856864821134905,
}


def _interval(
    ax: plt.Axes,
    y: float,
    estimate: float,
    lo: float,
    hi: float,
    color: str,
    marker: str = "o",
    ls: str = "-",
) -> None:
    """Draw one horizontal estimate and confidence interval."""
    ax.plot([lo, hi], [y, y], color=color, lw=3.2, solid_capstyle="round", ls=ls)
    ax.plot(
        [estimate],
        [y],
        marker=marker,
        ms=10,
        color=color,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=3,
    )


def _estimate_label(ax: plt.Axes, y: float, text: str) -> None:
    """Print one row's point estimate just outside the axes' right edge."""
    ax.annotate(
        text,
        xy=(1.0, y),
        xycoords=blended_transform_factory(ax.transAxes, ax.transData),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=12,
        color=INK,
        annotation_clip=False,
    )


def main() -> None:
    """Render the four matched arms plus the protocol-only audit to results_auc.svg."""
    table = pd.read_csv(DATA_CSV)
    auc = table.loc[table["kind"] == "auc"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.2, 3.05))

    for y, row in enumerate(auc.itertuples(index=False)):
        color = MAROON if row.architecture == "message passing" else GREY
        marker = "o" if row.initialization == "forecast pretrained" else "s"
        _interval(ax, y, row.estimate, row.ci_low, row.ci_high, color, marker)
        _estimate_label(ax, y, f"{row.estimate:.3f}")

    protocol_y = len(auc) + 0.55
    ax.axhline(len(auc) - 0.2, color=LIGHT, lw=1.0, zorder=0)
    _interval(
        ax,
        protocol_y,
        PROTOCOL_ONLY["estimate"],
        PROTOCOL_ONLY["ci_low"],
        PROTOCOL_ONLY["ci_high"],
        GOLD,
        marker="^",
        ls="--",
    )
    _estimate_label(ax, protocol_y, f"{PROTOCOL_ONLY['estimate']:.3f}")

    ax.axvline(0.5, color=INK, ls="--", lw=1.4, alpha=0.55, zorder=0)
    ax.set_yticks([*range(len(auc)), protocol_y])
    ax.set_yticklabels([*auc["label"], PROTOCOL_ONLY["label"]], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0.30, 0.72)
    ax.set_xlabel("Patient-level pooled OOF AUC (95% CI)", fontsize=12)
    ax.annotate(
        "chance",
        xy=(0.5, 1.0),
        xycoords=blended_transform_factory(ax.transData, ax.transAxes),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        color=INK,
    )

    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", color=LIGHT, lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK)

    fig.subplots_adjust(left=0.34, right=0.97, top=0.92, bottom=0.14)
    fig.savefig(OUT_SVG, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
