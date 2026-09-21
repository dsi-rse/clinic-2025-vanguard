r"""Judge a pCR model against the site structure of a multi-dataset cohort.

A pooled cross-validated AUROC on a cohort assembled from several datasets
can come from two very different places: the imaging features, or the fact
that datasets differ in pCR prevalence and the model has learned to tell the
datasets apart. This script separates the two, per the 2026-09-01 findings
that the n=230 vessel-skeleton model lost to a no-imaging baseline and did not
transfer across sites.

For each arm (an ``oof_predictions.csv`` from ``tabular.gnn_feature_baseline``)
it reports:

* pooled OOF AUROC with a label-stratified, patient-level bootstrap CI;
* the **dataset-prevalence baseline** (every exam scored with its dataset's
  in-sample pCR rate, no imaging) and the paired bootstrap of model - baseline;
* per-dataset AUROC (datasets with one class are listed, not scored) and the
  size-weighted within-dataset AUROC, which removes between-dataset structure;
* **leave-one-dataset-out** (LODO): refit the elastic net on the arm's feature
  table holding out each dataset, pooled and per-held-out AUROC;
* a sensitivity repeat on non-provisional labels when the labels file flags them.

With two arms it also reports the paired bootstrap of arm B - arm A on the
cases they share, which is the "does the complement add signal" contrast.

Inputs: one or two ``--arm NAME=DIR`` (DIR holds ``oof_predictions.csv`` and
``tabular_baseline_features.csv``), a ``--labels`` CSV with ``case_id``,
``dataset``, ``pcr``, optional ``patient_group_key`` (bootstrap unit; case_id
when absent) and optional ``pcr_label_is_provisional``.

Outputs (``--out-dir``): ``site_effect_summary.json``, ``per_dataset_auroc.csv``,
``lodo_auroc.csv``, ``per_dataset_auroc.png``.

Usage::

    python analysis/site_effect_evaluation.py \\
        --arm hr_only=<dir> --arm final=<dir> \\
        --labels <staging>/labels.csv --out-dir <cohort>/results/site_effect
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tabular.gnn_feature_baseline import _build_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_BOOT = 5000
BOOT_SEED = 0
LODO_SEEDS = (42, 142, 242)
NON_FEATURE_COLUMNS = {"case_id", "dataset", "fold", "pcr"}
PAIRED_ARM_COUNT = 2


def _stratified_unit_bootstrap(
    y: np.ndarray, units: np.ndarray, n_boot: int, seed: int
) -> list[np.ndarray]:
    """Row-index resamples that draw positive and negative *units* separately.

    A unit is a patient (or a case when no patient key exists); every row of a
    drawn unit comes along, so exams of one patient are never split. Positives
    and negatives are drawn separately so every resample keeps both classes.
    """
    rng = np.random.default_rng(seed)
    unit_label = pd.DataFrame({"unit": units, "y": y}).groupby("unit")["y"].max()
    pos_units = unit_label.index[unit_label == 1].to_numpy()
    neg_units = unit_label.index[unit_label == 0].to_numpy()
    rows_by_unit: dict[Any, np.ndarray] = {
        unit: np.flatnonzero(units == unit) for unit in unit_label.index
    }
    draws = []
    for _ in range(n_boot):
        chosen = np.concatenate(
            [
                rng.choice(pos_units, len(pos_units), replace=True),
                rng.choice(neg_units, len(neg_units), replace=True),
            ]
        )
        draws.append(np.concatenate([rows_by_unit[u] for u in chosen]))
    return draws


def _auc_ci(
    y: np.ndarray, prob: np.ndarray, draws: list[np.ndarray]
) -> tuple[float, float, float]:
    point = float(roc_auc_score(y, prob))
    boots = np.array([roc_auc_score(y[d], prob[d]) for d in draws])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _paired_delta(
    y: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray, draws: list[np.ndarray]
) -> dict[str, float]:
    """Bootstrap of AUROC(b) - AUROC(a) on the same resamples."""
    deltas = np.array(
        [roc_auc_score(y[d], prob_b[d]) - roc_auc_score(y[d], prob_a[d]) for d in draws]
    )
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": float(roc_auc_score(y, prob_b) - roc_auc_score(y, prob_a)),
        "ci95": [float(lo), float(hi)],
        "p_b_better": float(np.mean(deltas > 0)),
    }


def prevalence_baseline(frame: pd.DataFrame) -> np.ndarray:
    """Score every exam with its dataset's in-sample pCR rate (no imaging)."""
    return frame.groupby("dataset")["pcr"].transform("mean").to_numpy(dtype=float)


def per_dataset_auroc(frame: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    """AUROC per dataset; single-class datasets are listed with ``auroc`` None."""
    rows = []
    for dataset, group in frame.assign(_prob=prob).groupby("dataset"):
        n_pos = int(group["pcr"].sum())
        row = {
            "dataset": dataset,
            "n": len(group),
            "n_pos": n_pos,
            "auroc": None,
        }
        if 0 < n_pos < len(group):
            row["auroc"] = float(roc_auc_score(group["pcr"], group["_prob"]))
        rows.append(row)
    return pd.DataFrame(rows)


def size_weighted_within(per_dataset: pd.DataFrame) -> dict[str, Any]:
    """Size-weighted mean of the per-dataset AUROCs (between-dataset structure removed)."""
    scored = per_dataset.dropna(subset=["auroc"])
    if scored.empty:
        return {"auroc": None, "n": 0}
    return {
        "auroc": float(np.average(scored["auroc"], weights=scored["n"])),
        "n": int(scored["n"].sum()),
        "n_datasets": int(len(scored)),
    }


def lodo(
    features: pd.DataFrame, seeds: tuple[int, ...], model_name: str
) -> tuple[pd.DataFrame, dict]:
    """Refit ``model_name`` (a ``tabular.gnn_feature_baseline`` model) holding out each dataset."""
    feature_cols = [
        c
        for c in features.columns
        if c not in NON_FEATURE_COLUMNS and features[c].std(skipna=True) > 0
    ]
    x = features[feature_cols].to_numpy(dtype=float)
    y = features["pcr"].to_numpy(dtype=int)
    datasets = features["dataset"].to_numpy()
    held_out_prob = np.full(len(features), np.nan)
    rows = []
    for dataset in sorted(set(datasets)):
        test = datasets == dataset
        per_seed = []
        for seed in seeds:
            model = _build_model(model_name, seed)
            model.fit(x[~test], y[~test])
            per_seed.append(model.predict_proba(x[test])[:, 1])
        prob = np.mean(per_seed, axis=0)
        held_out_prob[test] = prob
        n_pos = int(y[test].sum())
        rows.append(
            {
                "held_out": dataset,
                "n": int(test.sum()),
                "n_pos": n_pos,
                "auroc": float(roc_auc_score(y[test], prob))
                if 0 < n_pos < test.sum()
                else None,
            }
        )
    table = pd.DataFrame(rows)
    scored_datasets = set(table.dropna(subset=["auroc"])["held_out"])
    scorable = np.isin(datasets, list(scored_datasets))
    summary = {
        "model": model_name,
        "n_features": len(feature_cols),
        "pooled_auroc_two_class_datasets": float(
            roc_auc_score(y[scorable], held_out_prob[scorable])
        ),
        "n_pooled": int(scorable.sum()),
        "size_weighted_auroc": size_weighted_within(
            table.rename(columns={"held_out": "dataset"})
        ),
    }
    return table, summary


def evaluate_arm(
    name: str,
    arm_dir: Path,
    labels: pd.DataFrame,
    model_column: str,
    *,
    subset_label: str,
    case_filter: pd.Series | None,
    lodo_model: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate one arm on one subset: pooled OOF, baseline, per-dataset, LODO."""
    oof = pd.read_csv(arm_dir / "oof_predictions.csv")
    frame = oof[["case_id", model_column]].merge(labels, on="case_id", how="inner")
    if len(frame) != len(oof):
        raise ValueError(
            f"{name}: {len(oof) - len(frame)} OOF cases missing from the labels file"
        )
    if case_filter is not None:
        frame = frame[frame["case_id"].isin(case_filter)].reset_index(drop=True)
    y = frame["pcr"].to_numpy(dtype=int)
    prob = frame[model_column].to_numpy(dtype=float)
    units = frame["unit"].to_numpy()
    draws = _stratified_unit_bootstrap(y, units, N_BOOT, BOOT_SEED)

    model_auc = _auc_ci(y, prob, draws)
    baseline = prevalence_baseline(frame)
    baseline_auc = _auc_ci(y, baseline, draws)
    per_ds = per_dataset_auroc(frame, prob)
    per_ds.insert(0, "arm", name)
    per_ds.insert(1, "subset", subset_label)

    features = pd.read_csv(arm_dir / "tabular_baseline_features.csv")
    features = features[features["case_id"].isin(frame["case_id"])]
    features = features.drop(columns=["dataset", "pcr"], errors="ignore").merge(
        frame[["case_id", "dataset", "pcr"]], on="case_id"
    )
    lodo_table, lodo_summary = lodo(features, LODO_SEEDS, lodo_model)
    lodo_table.insert(0, "arm", name)
    lodo_table.insert(1, "subset", subset_label)

    summary = {
        "arm": name,
        "subset": subset_label,
        "model_column": model_column,
        "n": int(len(frame)),
        "n_pos": int(y.sum()),
        "n_units": int(len(set(units))),
        "pooled_oof": {"auroc": model_auc[0], "ci95": list(model_auc[1:])},
        "prevalence_baseline": {
            "auroc": baseline_auc[0],
            "ci95": list(baseline_auc[1:]),
        },
        "model_minus_baseline": _paired_delta(y, baseline, prob, draws),
        "within_dataset_size_weighted": size_weighted_within(per_ds),
        "lodo": lodo_summary,
    }
    return summary, per_ds, lodo_table


def _plot_per_dataset(per_ds: pd.DataFrame, out_path: Path) -> None:
    """Dot plot of per-dataset AUROC per arm; single colour, chance line."""
    full = per_ds[per_ds["subset"] == "all"].dropna(subset=["auroc"])
    arms = list(dict.fromkeys(full["arm"]))
    datasets = list(dict.fromkeys(full["dataset"]))
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(datasets) * max(len(arms), 1) + 1.5))
    offsets = np.linspace(-0.2, 0.2, len(arms)) if len(arms) > 1 else [0.0]
    for arm, off in zip(arms, offsets, strict=True):
        sub = full[full["arm"] == arm].set_index("dataset").reindex(datasets)
        ypos = np.arange(len(datasets)) + off
        ax.plot(sub["auroc"], ypos, "o", color="#444444", label=arm)
        for yp, (_, row) in zip(ypos, sub.iterrows(), strict=True):
            ax.annotate(
                f"n={int(row['n'])}",
                (row["auroc"], yp),
                xytext=(6, -3),
                textcoords="offset points",
                fontsize=8,
                color="#444444",
            )
    ax.axvline(0.5, color="#999999", linewidth=1, linestyle="--")
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_xlabel("within-dataset AUROC (pooled OOF)")
    if len(arms) > 1:
        ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Evaluate each arm, the paired arm contrast, and the provisional-label sensitivity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-column", default="oof_prob_elastic_net")
    parser.add_argument(
        "--lodo-model",
        default="elastic_net",
        choices=("elastic_net", "logistic_regression", "xgboost"),
        help="model refit for leave-one-dataset-out; match --model-column",
    )
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    labels["unit"] = (
        labels["patient_group_key"]
        if "patient_group_key" in labels
        else labels["case_id"]
    )
    label_cols = ["case_id", "dataset", "pcr", "unit"]
    if "pcr_label_is_provisional" in labels:
        label_cols.append("pcr_label_is_provisional")
    labels = labels[label_cols]

    arms = [spec.split("=", 1) for spec in args.arm]
    subsets: list[tuple[str, pd.Series | None]] = [("all", None)]
    if "pcr_label_is_provisional" in labels:
        firm = labels.loc[~labels["pcr_label_is_provisional"].astype(bool), "case_id"]
        subsets.append(("non_provisional", firm))

    summaries, per_ds_tables, lodo_tables = [], [], []
    for subset_label, case_filter in subsets:
        for name, arm_dir in arms:
            summary, per_ds, lodo_table = evaluate_arm(
                name,
                Path(arm_dir),
                labels,
                args.model_column,
                subset_label=subset_label,
                case_filter=case_filter,
                lodo_model=args.lodo_model,
            )
            summaries.append(summary)
            per_ds_tables.append(per_ds)
            lodo_tables.append(lodo_table)
            print(
                f"[{subset_label}] {name}: n={summary['n']} "
                f"AUROC={summary['pooled_oof']['auroc']:.4f} "
                f"{summary['pooled_oof']['ci95']} | prevalence baseline "
                f"{summary['prevalence_baseline']['auroc']:.4f} | LODO pooled "
                f"{summary['lodo']['pooled_auroc_two_class_datasets']:.4f}",
                flush=True,
            )

    contrasts = []
    if len(arms) == PAIRED_ARM_COUNT:
        (name_a, dir_a), (name_b, dir_b) = arms
        oof_a = pd.read_csv(Path(dir_a) / "oof_predictions.csv")
        oof_b = pd.read_csv(Path(dir_b) / "oof_predictions.csv")
        shared = (
            oof_a[["case_id", args.model_column]]
            .merge(
                oof_b[["case_id", args.model_column]],
                on="case_id",
                suffixes=("_a", "_b"),
            )
            .merge(labels, on="case_id")
        )
        y = shared["pcr"].to_numpy(dtype=int)
        draws = _stratified_unit_bootstrap(
            y, shared["unit"].to_numpy(), N_BOOT, BOOT_SEED
        )
        delta = _paired_delta(
            y,
            shared[f"{args.model_column}_a"].to_numpy(),
            shared[f"{args.model_column}_b"].to_numpy(),
            draws,
        )
        contrasts.append(
            {"a": name_a, "b": name_b, "n_shared": int(len(shared)), **delta}
        )
        print(f"paired {name_b} - {name_a}: {delta}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_ds_all = pd.concat(per_ds_tables, ignore_index=True)
    per_ds_all.to_csv(args.out_dir / "per_dataset_auroc.csv", index=False)
    pd.concat(lodo_tables, ignore_index=True).to_csv(
        args.out_dir / "lodo_auroc.csv", index=False
    )
    (args.out_dir / "site_effect_summary.json").write_text(
        json.dumps(
            {
                "labels": str(args.labels),
                "model_column": args.model_column,
                "bootstrap": {
                    "n_boot": N_BOOT,
                    "seed": BOOT_SEED,
                    "unit": "patient_group_key or case_id",
                    "stratified_by_label": True,
                },
                "lodo_seeds": list(LODO_SEEDS),
                "lodo_model": args.lodo_model,
                "arms": summaries,
                "paired_contrasts": contrasts,
            },
            indent=2,
        )
    )
    _plot_per_dataset(per_ds_all, args.out_dir / "per_dataset_auroc.png")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
