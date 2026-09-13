"""Tabular baseline: logistic regression / XGBoost on hand-summarized GNN inputs.

Per the code review's outcome-signature framework: if this tabular baseline
also stays near chance, the six inputs (peak_time, peak_enhancement,
time_to_enhancement, washin_slope, auc_positive, radius) likely lack
generalizable pCR signal, independent of architecture; if it clearly beats
the GNN, pooling/message-passing is the more likely bottleneck.

Reuses arm 1 (mixed baseline)'s existing cache
(gnn_cache_voxel_mixed_baseline) and frozen folds -- same 1332-case cohort,
same 6 features, same 5 predefined folds -- so results are directly
comparable to that arm's GNN AUC (0.509 mean, 3 seeds x 5 folds, see
experiments/harmonized_single_breast_v1/README.md). Per case, summarizes
each of the 6 node-level features to (mean, std, q10, q50, q90), adds the
TTE no-arrival fraction (fraction of nodes at TTE_NO_ARRIVAL_SENTINEL) and
the already-recorded node/edge counts -- 33 tabular columns total. The
time_to_enhancement summaries are computed over observed-arrival nodes only
(the no-arrival sentinel is excluded), so they measure arrival timing rather
than a blend of timing and missingness prevalence -- the latter is carried
separately by tte_no_arrival_fraction. No new graph build: loads the existing
cache read-only.

Writes the feature table and per-fold results to --out-dir (default
experiments/tabular_baseline_v1/, see that directory's README.md for the
full writeup), matching the results-directory convention used elsewhere in
this repo. Lives in tabular/, not analysis/: this fits real predictive
models (logistic regression, XGBoost), which is what tabular/models.py and
tabular/train.py are for -- analysis/ is for scripts that summarize or
diagnose already-computed results, not ones that fit their own models.

Usage::

    python -m tabular.gnn_feature_baseline --config configs/gnn_voxel_mixed_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from xgboost import XGBClassifier

from gnn.data_loader import TTE_NO_ARRIVAL_SENTINEL, VanguardCenterlineDataset
from load_cohort import load_config

_SUMMARY_QUANTILES = (0.10, 0.50, 0.90)
_SEEDS = (42, 142, 242)
_N_BOOTSTRAP = 2000
_CI_PCT = (2.5, 97.5)


def _build_model(model_name: str, seed: int) -> Pipeline:
    """The imputer -> scaler -> classifier pipeline for one model/seed.

    SimpleImputer (mean) is fit on each fold's training split only (see
    ``oof_predictions`` / ``run_cv``), so the NaN ``summarize_graph`` emits for
    an all-no-arrival case's TTE timing summaries never draws on validation
    data. It is a near-no-op otherwise (only the handful of all-no-arrival cases
    carry any NaN).
    """
    if model_name == "logistic_regression":
        return make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed),
        )
    if model_name == "xgboost":
        return make_pipeline(
            SimpleImputer(strategy="mean"),
            XGBClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=seed,
                eval_metric="logloss",
            ),
        )
    if model_name == "elastic_net":
        # penalty/solver/l1_ratio/C match this repo's elastic-net convention in
        # tabular/models.py's build_model_pipeline (C=1.0, l1_ratio=0.5,
        # max_iter=3000, class_weight="balanced" for the pCR class imbalance).
        return make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=1.0,
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    raise ValueError(f"Unknown model_name: {model_name!r}")


MODEL_NAMES = ("logistic_regression", "xgboost", "elastic_net")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gnn_voxel_mixed_baseline.yaml")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("experiments/tabular_baseline_v1")
    )
    return parser.parse_args()


def load_dataset_from_config(
    config_path: Path,
) -> tuple[VanguardCenterlineDataset, tuple[str, ...]]:
    """Load an existing cache read-only, exactly as gnn/train.py does for training."""
    config = load_config(config_path)
    mp, dp = config.model_params, config.data_paths
    node_features = tuple(mp.gnn_node_features)
    dataset = VanguardCenterlineDataset(
        root=dp.gnn_centerline_root,
        labels_path=dp.gnn_labels_path,
        dce_root=dp.gnn_dce_root,
        cases=list(dp.gnn_cases) if dp.gnn_cases else None,
        id_column=dp.gnn_id_column,
        label_column=dp.gnn_label_column,
        cache_dir=dp.gnn_cache_dir,
        node_mode=str(mp.gnn_node_mode),
        node_features=node_features,
        # Must match the cache manifest's floor: a floored cache (UChicago 0.05)
        # refuses an unfloored (0.0) request and vice versa, so this cannot be
        # left at the default when reading a floored cache.
        kinetic_baseline_floor_frac=float(mp.gnn_kinetic_baseline_floor_frac),
    )
    return dataset, node_features


def _summary_stats(values: np.ndarray, name: str) -> dict[str, float]:
    """mean/std/quantiles for one feature column.

    An empty ``values`` (e.g. the ``time_to_enhancement`` column after every
    no-arrival node is masked out -- a case where no node ever enhanced) has no
    defined timing summary, so every stat is ``NaN``. That ``NaN`` is handled by
    a per-fold mean-imputer in ``run_cv`` (fit on the training split only, never
    the whole cohort); the missingness itself is already carried explicitly by
    ``tte_no_arrival_fraction``.
    """
    if values.size == 0:
        stats = {f"{name}_mean": float("nan"), f"{name}_std": float("nan")}
        for q in _SUMMARY_QUANTILES:
            stats[f"{name}_q{int(q * 100)}"] = float("nan")
        return stats
    stats = {
        f"{name}_mean": float(np.mean(values)),
        f"{name}_std": float(np.std(values)),
    }
    for q in _SUMMARY_QUANTILES:
        stats[f"{name}_q{int(q * 100)}"] = float(np.quantile(values, q))
    return stats


def summarize_graph(data: Data, node_features: tuple[str, ...]) -> dict[str, float]:
    """Collapse one graph's per-node feature matrix into tabular summary stats.

    ``time_to_enhancement`` is summarized over **observed-arrival nodes only**
    (nodes not at ``TTE_NO_ARRIVAL_SENTINEL``): including the sentinel in the
    mean/quantiles would make those features a mixture of actual arrival timing
    and the prevalence of missing arrivals, even though the missingness is
    already represented explicitly by ``tte_no_arrival_fraction``. Every other
    feature is summarized over all nodes.
    """
    x = data.x.numpy()
    tte_idx = (
        node_features.index("time_to_enhancement")
        if "time_to_enhancement" in node_features
        else None
    )
    row: dict[str, float] = {}
    for i, name in enumerate(node_features):
        column = x[:, i]
        if tte_idx is not None and i == tte_idx:
            column = column[column != TTE_NO_ARRIVAL_SENTINEL]
        row.update(_summary_stats(column, name))
    if tte_idx is not None:
        tte_column = x[:, tte_idx]
        row["tte_no_arrival_fraction"] = float(
            np.mean(tte_column == TTE_NO_ARRIVAL_SENTINEL)
        )
    row["num_nodes"] = float(data.num_nodes)
    row["num_edges"] = int(data.edge_index.shape[1])
    row["case_id"] = str(data.case_id)
    row["pcr"] = int(data.y.item())
    row["dataset"] = data.dataset
    return row


def build_feature_table(
    dataset: VanguardCenterlineDataset, node_features: tuple[str, ...]
) -> pd.DataFrame:
    """Summarize every graph in the dataset into one tabular row each."""
    rows = [summarize_graph(dataset[i], node_features) for i in range(len(dataset))]
    return pd.DataFrame(rows)


def add_folds(
    feature_table: pd.DataFrame, labels_path: Path, id_column: str = "case_id"
) -> pd.DataFrame:
    """Join in each case's frozen fold assignment."""
    folds = pd.read_csv(labels_path)[[id_column, "fold"]]
    if id_column != "case_id":
        folds = folds.rename(columns={id_column: "case_id"})
    merged = feature_table.merge(folds, on="case_id", how="left")
    if merged["fold"].isna().any():
        missing = merged.loc[merged["fold"].isna(), "case_id"].tolist()
        raise ValueError(
            f"{len(missing)} case(s) have no fold assignment: {missing[:10]}..."
        )
    return merged


def run_cv(
    feature_table: pd.DataFrame, feature_cols: list[str], *, model_name: str, seed: int
) -> list[dict[str, object]]:
    """5-fold CV using the frozen fold column, one held-out AUC per fold."""
    rows = []
    for fold in sorted(feature_table["fold"].unique()):
        train = feature_table[feature_table["fold"] != fold]
        test = feature_table[feature_table["fold"] == fold]
        x_train, y_train = train[feature_cols].to_numpy(), train["pcr"].to_numpy()
        x_test, y_test = test[feature_cols].to_numpy(), test["pcr"].to_numpy()
        model = _build_model(model_name, seed)
        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "fold": int(fold),
                "auc": auc,
                "n_test": len(test),
            }
        )
    return rows


def oof_predictions(
    feature_table: pd.DataFrame, feature_cols: list[str], *, model_name: str, seed: int
) -> np.ndarray:
    """Out-of-fold predicted probabilities, aligned to ``feature_table`` row order.

    Each case is scored only by the model trained on the folds that exclude it,
    so the stacked vector holds exactly one held-out prediction per case. Feeding
    it to a single ``roc_auc_score`` over all cases gives the **pooled OOF AUC**
    -- the plan's metric convention and the estimand the GNN's ~0.61 is measured
    on -- as opposed to averaging the noisier per-fold AUCs ``run_cv`` reports.
    """
    oof = np.full(len(feature_table), np.nan)
    fold = feature_table["fold"].to_numpy()
    y = feature_table["pcr"].to_numpy()
    x = feature_table[feature_cols].to_numpy()
    for f in sorted(np.unique(fold)):
        train_mask, test_mask = fold != f, fold == f
        model = _build_model(model_name, seed)
        model.fit(x[train_mask], y[train_mask])
        oof[test_mask] = model.predict_proba(x[test_mask])[:, 1]
    return oof


def bootstrap_auc_ci(
    y: np.ndarray, prob: np.ndarray, *, n_boot: int = _N_BOOTSTRAP, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for a pooled AUC, resampling cases with replacement.

    Quantifies sampling uncertainty of the pooled-OOF AUC at this fixed N -- the
    Phase 0.3 requirement that tier/model deltas be read against a CI rather than
    a bare point estimate.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        aucs[b] = roc_auc_score(y[idx], prob[idx])
    lo, hi = np.percentile(aucs, _CI_PCT)
    return float(lo), float(hi)


def main() -> None:
    """Build the tabular feature table once, then run both models across seeds."""
    args = _parse_args()
    config = load_config(args.config)
    dataset, node_features = load_dataset_from_config(args.config)
    print(f"Loaded {len(dataset)} graphs from {config.data_paths.gnn_cache_dir}")

    feature_table = build_feature_table(dataset, node_features)
    feature_table = add_folds(
        feature_table,
        Path(config.data_paths.gnn_labels_path),
        id_column=config.data_paths.gnn_id_column,
    )
    feature_cols = [
        c
        for c in feature_table.columns
        if c not in ("case_id", "pcr", "dataset", "fold")
    ]
    print(f"{len(feature_table)} cases, {len(feature_cols)} tabular feature columns")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(args.out_dir / "tabular_baseline_features.csv", index=False)

    all_rows: list[dict[str, object]] = []
    for model_name in MODEL_NAMES:
        for seed in _SEEDS:
            all_rows.extend(
                run_cv(feature_table, feature_cols, model_name=model_name, seed=seed)
            )

    results = pd.DataFrame(all_rows)
    print()
    print("=== Per-model per-fold AUC (mean +/- std across seed x fold) ===")
    print(results.groupby("model")["auc"].agg(["mean", "std", "count"]))
    print()
    print(f"=== Per-fold AUC, one model at a time (seed={_SEEDS[0]}) ===")
    print(
        results[results["seed"] == _SEEDS[0]].pivot_table(
            index="fold", columns="model", values="auc"
        )
    )

    # Pooled OOF AUC is the plan's headline metric (comparable to the GNN's
    # ~0.61): score every case once from the fold that held it out, average those
    # OOF probabilities across seeds to damp init noise, then take a single AUC
    # over all cases plus a bootstrap CI on it (Phase 0.3).
    y = feature_table["pcr"].to_numpy()
    pooled: list[dict[str, object]] = []
    oof_columns: dict[str, np.ndarray] = {}
    print()
    print("=== Pooled OOF AUC over all cases (headline metric) ===")
    for model_name in MODEL_NAMES:
        per_seed = np.stack(
            [
                oof_predictions(
                    feature_table, feature_cols, model_name=model_name, seed=s
                )
                for s in _SEEDS
            ]
        )
        seed_pooled = [float(roc_auc_score(y, per_seed[i])) for i in range(len(_SEEDS))]
        mean_oof = per_seed.mean(axis=0)
        oof_columns[model_name] = mean_oof
        auc = float(roc_auc_score(y, mean_oof))
        ci_lo, ci_hi = bootstrap_auc_ci(y, mean_oof)
        pooled.append(
            {
                "model": model_name,
                "pooled_oof_auc_seed_mean": auc,
                "pooled_oof_auc_ci95": [ci_lo, ci_hi],
                "pooled_oof_auc_per_seed": seed_pooled,
                "n_cases": int(len(y)),
                "n_features": len(feature_cols),
            }
        )
        print(
            f"  {model_name:20s} AUC={auc:.3f}  "
            f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]  "
            f"per-seed={[round(v, 3) for v in seed_pooled]}"
        )

    out_path = args.out_dir / "tabular_baseline_results.json"
    out_path.write_text(
        json.dumps({"per_fold": all_rows, "pooled_oof": pooled}, indent=2)
    )
    print(f"\nWrote {out_path}")

    # Per-case seed-mean OOF probabilities, one column per model. Kept
    # separately from tabular_baseline_results.json (which only has the
    # pooled scalar AUC) so a downstream script can pair cases across two
    # arms (e.g. HR-only vs complemented) for a paired bootstrap delta.
    oof_table = feature_table[["case_id", "dataset", "fold", "pcr"]].copy()
    for model_name, values in oof_columns.items():
        oof_table[f"oof_prob_{model_name}"] = values
    oof_path = args.out_dir / "oof_predictions.csv"
    oof_table.to_csv(oof_path, index=False)
    print(f"Wrote {oof_path}")


if __name__ == "__main__":
    main()
