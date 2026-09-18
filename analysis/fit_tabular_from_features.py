"""Fit the tabular pCR models from a feature CSV (no graph cache needed).

``tabular.gnn_feature_baseline`` builds its feature table from the graph cache
and then fits logistic regression, XGBoost and the elastic net with pooled OOF
predictions. This runner takes an already-built (or transformed, e.g. ComBat-
harmonised) ``tabular_baseline_features.csv`` plus the labels file and
produces the same outputs -- ``oof_predictions.csv`` and
``tabular_baseline_results.json`` -- with the same models, seeds, folds and
bootstrap, so ``analysis/site_effect_evaluation.py`` can compare arms and
feature variants like for like.

Usage::

    python analysis/fit_tabular_from_features.py --features <dir>/tabular_baseline_features.csv \\
        --labels <staging>/labels.csv --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tabular.gnn_feature_baseline import (
    _SEEDS,
    MODEL_NAMES,
    bootstrap_auc_ci,
    oof_predictions,
)

NON_FEATURE_COLUMNS = {"case_id", "dataset", "fold", "pcr"}


def main() -> None:
    """Fit every model across seeds and write pooled-OOF outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features).drop(
        columns=["dataset", "pcr", "fold"], errors="ignore"
    )
    labels = pd.read_csv(args.labels)[["case_id", "dataset", "pcr", "fold"]]
    table = features.merge(labels, on="case_id", how="inner", validate="one_to_one")
    if len(table) != len(features):
        raise ValueError(f"{len(features) - len(table)} feature rows have no label row")
    feature_cols = [c for c in features.columns if c not in NON_FEATURE_COLUMNS]
    y = table["pcr"].to_numpy(dtype=int)

    pooled = []
    oof_table = table[["case_id", "dataset", "fold", "pcr"]].copy()
    for model_name in MODEL_NAMES:
        per_seed = np.stack(
            [
                oof_predictions(table, feature_cols, model_name=model_name, seed=s)
                for s in _SEEDS
            ]
        )
        mean_oof = per_seed.mean(axis=0)
        oof_table[f"oof_prob_{model_name}"] = mean_oof
        auc = float(roc_auc_score(y, mean_oof))
        lo, hi = bootstrap_auc_ci(y, mean_oof)
        pooled.append(
            {
                "model": model_name,
                "pooled_oof_auc_seed_mean": auc,
                "pooled_oof_auc_ci95": [lo, hi],
                "pooled_oof_auc_per_seed": [
                    float(roc_auc_score(y, p)) for p in per_seed
                ],
                "n_cases": int(len(y)),
                "n_features": len(feature_cols),
            }
        )
        print(f"  {model_name:20s} AUC={auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    (args.out_dir / "tabular_baseline_results.json").write_text(
        json.dumps(
            {"per_fold": [], "pooled_oof": pooled, "features": str(args.features)},
            indent=2,
        )
    )
    oof_table.to_csv(args.out_dir / "oof_predictions.csv", index=False)
    if not (args.out_dir / "tabular_baseline_features.csv").exists():
        pd.read_csv(args.features).to_csv(
            args.out_dir / "tabular_baseline_features.csv", index=False
        )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
