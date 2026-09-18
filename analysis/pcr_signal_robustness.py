"""Stress-test a pooled-OOF pCR AUROC before it is believed.

Given one arm's feature table (``tabular.gnn_feature_baseline`` output), the
labels file (patient-grouped folds), and the EDA's per-target permutation
importances, this refits the model under four perturbations that a real signal
survives and a confound does not:

1. **Label permutation** -- shuffle pCR within dataset (so per-dataset
   prevalence is preserved), refit with the same folds, and record the pooled
   OOF AUROC ``n_perm`` times. The empirical p-value is the fraction of
   permutations reaching the observed AUROC.
2. **Scanner-feature ablation** -- drop the top-k features that carry scanner
   model / software version (from ``batch_importance.csv``) and refit. A signal
   that lives only in the acquisition-carrying features disappears.
3. **Within-stratum refits** -- restrict train *and* test to one
   scanner x field x software stratum (the largest ones) so no era/scanner
   structure is available to the model at all.
4. **Hyperparameter sensitivity** -- refit across a small XGBoost grid to show
   the headline is not a lucky default.

Every fit uses the labels file's folds (patient-grouped) and averages three
seeds, exactly as ``tabular.gnn_feature_baseline`` does.

Outputs (``--out-dir``): ``robustness_summary.json``, ``permutation_aucs.csv``,
``ablation.csv``, ``within_stratum.csv``, ``hyperparameters.csv``.

Usage::

    python analysis/pcr_signal_robustness.py --features <arm>/tabular_baseline_features.csv \\
        --labels <staging>/labels.csv --joined <eda>/joined_table.csv \\
        --importance <eda>/batch_importance.csv --model xgboost --out-dir <results>/robustness_final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from tabular.gnn_feature_baseline import _build_model

NON_FEATURE_COLUMNS = {"case_id", "dataset", "fold", "pcr"}
SEEDS = (42, 142, 242)
SCANNER_TARGETS = ("hr_manufacturer_model_name", "hr_software_major")
MIN_STRATUM = 30
XGB_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr}
    for n in (50, 100, 300)
    for d in (2, 3, 4)
    for lr in (0.1, 0.3)
]


def _xgb(params: dict, seed: int):  # noqa: ANN202
    return make_pipeline(
        SimpleImputer(strategy="mean"),
        XGBClassifier(random_state=seed, eval_metric="logloss", **params),
    )


def pooled_oof_auc(
    x: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    build,  # noqa: ANN001
) -> float:
    """Seed-mean pooled OOF AUROC with the frozen fold column."""
    oof = np.zeros((len(SEEDS), len(y)))
    for s, seed in enumerate(SEEDS):
        for f in np.unique(fold):
            tr, te = fold != f, fold == f
            model = build(seed)
            model.fit(x[tr], y[tr])
            oof[s, te] = model.predict_proba(x[te])[:, 1]
    return float(roc_auc_score(y, oof.mean(axis=0)))


def main() -> None:
    """Run the four perturbations and write their tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--joined", type=Path, required=True, help="EDA joined_table.csv"
    )
    parser.add_argument("--importance", type=Path, required=True)
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--ablate-k", type=int, nargs="+", default=[3, 6, 10])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features).drop(
        columns=["dataset", "pcr", "fold"], errors="ignore"
    )
    labels = pd.read_csv(args.labels)[
        ["case_id", "dataset", "pcr", "fold", "patient_group_key"]
    ]
    table = features.merge(labels, on="case_id", validate="one_to_one")
    joined = pd.read_csv(args.joined, low_memory=False)
    joined["stratum"] = (
        joined["hr_manufacturer_model_name"].astype(str)
        + "_"
        + joined["hr_magnetic_field_strength"].astype(str)
        + "T_"
        + joined["hr_software_versions"].map(
            lambda v: json.loads(v)[0] if isinstance(v, str) else "na"
        )
    )
    table = table.merge(
        joined[["exam_id", "stratum"]].rename(columns={"exam_id": "case_id"}),
        on="case_id",
        how="left",
    )
    feats = [
        c
        for c in features.columns
        if c not in NON_FEATURE_COLUMNS and table[c].std() > 0
    ]
    x = table[feats].to_numpy(dtype=float)
    y = table["pcr"].to_numpy(dtype=int)
    fold = table["fold"].to_numpy()
    build = lambda seed: _build_model(args.model, seed)  # noqa: E731
    observed = pooled_oof_auc(x, y, fold, build)
    print(
        f"observed pooled OOF AUROC ({args.model}, {len(feats)} features, n={len(y)}): {observed:.4f}"
    )

    # 1. label permutation within dataset
    rng = np.random.default_rng(0)
    perm_aucs = []
    for i in range(args.n_perm):
        y_perm = y.copy()
        for _, idx in table.groupby("dataset").indices.items():
            y_perm[idx] = rng.permutation(y[idx])
        perm_aucs.append(pooled_oof_auc(x, y_perm, fold, build))
        if (i + 1) % 20 == 0:
            print(
                f"  permutation {i + 1}/{args.n_perm}: running max {max(perm_aucs):.3f}",
                flush=True,
            )
    perm_aucs = np.array(perm_aucs)
    p_value = float((np.sum(perm_aucs >= observed) + 1) / (len(perm_aucs) + 1))
    pd.DataFrame({"perm_auc": perm_aucs}).to_csv(
        args.out_dir / "permutation_aucs.csv", index=False
    )
    print(
        f"permutation null: mean {perm_aucs.mean():.3f} sd {perm_aucs.std():.3f} max {perm_aucs.max():.3f}; p = {p_value:.4f}"
    )

    # 2. scanner-feature ablation
    importance = pd.read_csv(args.importance)
    carriers = (
        importance[
            importance["target"].isin(SCANNER_TARGETS) & (importance["model"] == "rf")
        ]
        .groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
    )
    ablation = []
    for k in args.ablate_k:
        dropped = [f for f in carriers.index[:k] if f in feats]
        keep = [i for i, f in enumerate(feats) if f not in dropped]
        auc = pooled_oof_auc(x[:, keep], y, fold, build)
        ablation.append(
            {
                "dropped_k": k,
                "dropped": json.dumps(dropped),
                "n_features": len(keep),
                "auroc": auc,
            }
        )
        print(
            f"ablate top-{k} scanner carriers -> AUROC {auc:.4f}  (dropped {dropped})"
        )
    pd.DataFrame(ablation).to_csv(args.out_dir / "ablation.csv", index=False)

    # 3. within-stratum refits
    within = []
    for stratum, group in table.groupby("stratum"):
        if len(group) < MIN_STRATUM or group["pcr"].nunique() < 2:
            continue
        idx = group.index.to_numpy()
        f_local = fold[idx]
        if any(
            y[idx][f_local == f].sum() in (0, (f_local == f).sum())
            for f in np.unique(f_local)
        ):
            continue
        auc = pooled_oof_auc(x[idx], y[idx], f_local, build)
        within.append(
            {
                "stratum": stratum,
                "n": len(idx),
                "n_pos": int(y[idx].sum()),
                "auroc": auc,
            }
        )
        print(f"within {stratum}: n={len(idx)} pos={int(y[idx].sum())} AUROC {auc:.4f}")
    pd.DataFrame(within).to_csv(args.out_dir / "within_stratum.csv", index=False)

    # 4. hyperparameter sensitivity (xgboost only)
    hyper = []
    if args.model == "xgboost":
        for params in XGB_GRID:
            auc = pooled_oof_auc(x, y, fold, lambda seed, p=params: _xgb(p, seed))
            hyper.append({**params, "auroc": auc})
        hyper_df = pd.DataFrame(hyper)
        hyper_df.to_csv(args.out_dir / "hyperparameters.csv", index=False)
        print(
            f"hyperparameter grid AUROC: min {hyper_df.auroc.min():.3f} "
            f"median {hyper_df.auroc.median():.3f} max {hyper_df.auroc.max():.3f}"
        )

    (args.out_dir / "robustness_summary.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "n": int(len(y)),
                "n_features": len(feats),
                "observed_auroc": observed,
                "permutation": {
                    "n_perm": int(len(perm_aucs)),
                    "null_mean": float(perm_aucs.mean()),
                    "null_sd": float(perm_aucs.std()),
                    "null_max": float(perm_aucs.max()),
                    "p_value": p_value,
                    "shuffled_within": "dataset",
                },
                "ablation": ablation,
                "within_stratum": within,
                "hyperparameters": hyper,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
