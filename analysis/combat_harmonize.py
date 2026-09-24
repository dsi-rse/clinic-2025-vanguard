r"""ComBat harmonisation of a tabular feature table across acquisition batches.

Parametric empirical-Bayes ComBat (Johnson, Li & Rabinovic 2007, as used by
neuroCombat) applied to one arm's ``tabular_baseline_features.csv``. The batch
is an acquisition stratum -- by default scanner model x field strength x
software version from the EDA's joined table -- because the EDA showed those
are what the skeleton features encode. No biological covariate is preserved:
adding pCR as a ComBat covariate would write the label into the features and
inflate every downstream cross-validated number, so this is purely
unsupervised location/scale alignment. It is fit on the whole cohort
(transductive, label-free); a fold-wise variant is a follow-up if the result
matters.

Batches smaller than ``--min-batch`` are pooled into ``<scanner>_other`` so
every batch has enough exams to estimate its shift; the mapping is recorded.

Output: a feature CSV with the same columns/rows as the input (harmonised
values), plus ``combat_provenance.json`` and ``combat_batch_shift.csv`` (the
estimated per-batch additive and multiplicative effects per feature).

Usage::

    python analysis/combat_harmonize.py --features <arm>/tabular_baseline_features.csv \\
        --joined <eda>/joined_table.csv --out-dir <results>/combat_final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NON_FEATURE_COLUMNS = {"case_id", "dataset", "fold", "pcr"}
MIN_BATCH_DEFAULT = 10
EB_TOLERANCE = 1e-4


def stratum_labels(
    joined: pd.DataFrame, min_batch: int
) -> tuple[pd.Series, dict[str, str]]:
    """Scanner x field x software per exam, with small batches pooled by scanner."""
    software = joined["hr_software_versions"].map(
        lambda v: json.loads(v)[0] if isinstance(v, str) else "na"
    )
    scanner = (
        joined["hr_manufacturer_model_name"].astype(str)
        + "_"
        + joined["hr_magnetic_field_strength"].astype(str)
        + "T"
    )
    stratum = scanner + "_" + software
    counts = stratum.value_counts()
    mapping = {
        s: (s if counts[s] >= min_batch else f"{scanner[stratum == s].iloc[0]}_other")
        for s in counts.index
    }
    pooled = stratum.map(mapping)
    pooled_counts = pooled.value_counts()
    # A pooled "other" bucket can itself still be tiny; fold it into the scanner's largest batch.
    for name, n in pooled_counts.items():
        if n < min_batch:
            scanner_prefix = name.rsplit("_", 1)[0]
            candidates = pooled_counts[
                pooled_counts.index.str.startswith(scanner_prefix)
                & (pooled_counts.index != name)
            ]
            target = candidates.idxmax() if len(candidates) else pooled_counts.idxmax()
            mapping = {k: (target if v == name else v) for k, v in mapping.items()}
            pooled = stratum.map(mapping)
            pooled_counts = pooled.value_counts()
    return pd.Series(pooled.to_numpy(), index=joined["exam_id"].to_numpy()), mapping


def _aprior(gamma_hat: np.ndarray) -> float:
    m, s2 = gamma_hat.mean(), gamma_hat.var(ddof=1)
    return (2 * s2 + m**2) / s2


def _bprior(gamma_hat: np.ndarray) -> float:
    m, s2 = gamma_hat.mean(), gamma_hat.var(ddof=1)
    return (m * s2 + m**3) / s2


def _postmean(
    g_hat: np.ndarray, g_bar: float, n: int, d_star: np.ndarray, t2: float
) -> np.ndarray:
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2: np.ndarray, n: int, a: float, b: float) -> np.ndarray:
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1)


def _iterate_eb(
    s_data: np.ndarray,
    g_hat: np.ndarray,
    d_hat: np.ndarray,
    g_bar: float,
    t2: float,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = s_data.shape[1]
    g_old, d_old = g_hat.copy(), d_hat.copy()
    for _ in range(500):
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((s_data - g_new[:, None]) ** 2).sum(axis=1)
        d_new = _postvar(sum2, n, a, b)
        change = max(
            np.abs(g_new - g_old).max() / np.abs(g_old).max(),
            np.abs(d_new - d_old).max() / d_old.max(),
        )
        g_old, d_old = g_new, d_new
        if change < EB_TOLERANCE:
            break
    return g_old, d_old


def combat(x: np.ndarray, batch: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Parametric EB ComBat; ``x`` is (n_samples, n_features). Returns (adjusted, shifts)."""
    x = x.T.astype(float)  # features x samples, ComBat's native orientation
    batches = sorted(set(batch))
    n_batch = len(batches)
    batch_idx = [np.flatnonzero(batch == b) for b in batches]
    n_per = np.array([len(i) for i in batch_idx])
    n_total = x.shape[1]

    design = np.zeros((n_total, n_batch))
    for j, idx in enumerate(batch_idx):
        design[idx, j] = 1.0
    beta_hat = np.linalg.lstsq(design, x.T, rcond=None)[0]  # n_batch x features
    grand_mean = (n_per / n_total) @ beta_hat
    var_pooled = ((x - (design @ beta_hat).T) ** 2).mean(axis=1)
    var_pooled = np.where(var_pooled <= 0, 1e-12, var_pooled)
    stand_mean = np.outer(grand_mean, np.ones(n_total))
    s_data = (x - stand_mean) / np.sqrt(var_pooled)[:, None]

    gamma_hat = np.linalg.lstsq(design, s_data.T, rcond=None)[0]  # n_batch x features
    delta_hat = np.array([s_data[:, idx].var(axis=1, ddof=1) for idx in batch_idx])
    delta_hat = np.where(delta_hat <= 0, 1e-12, delta_hat)
    gamma_star = np.empty_like(gamma_hat)
    delta_star = np.empty_like(delta_hat)
    for j, idx in enumerate(batch_idx):
        g_bar, t2 = gamma_hat[j].mean(), gamma_hat[j].var(ddof=1)
        a, b = _aprior(delta_hat[j]), _bprior(delta_hat[j])
        gamma_star[j], delta_star[j] = _iterate_eb(
            s_data[:, idx], gamma_hat[j], delta_hat[j], g_bar, t2, a, b
        )

    adjusted = s_data.copy()
    for j, idx in enumerate(batch_idx):
        adjusted[:, idx] = (s_data[:, idx] - gamma_star[j][:, None]) / np.sqrt(
            delta_star[j]
        )[:, None]
    adjusted = adjusted * np.sqrt(var_pooled)[:, None] + stand_mean
    shifts = pd.DataFrame(
        {
            "batch": np.repeat(batches, x.shape[0]),
            "feature_index": np.tile(np.arange(x.shape[0]), n_batch),
            "gamma_star": gamma_star.ravel(),
            "delta_star": delta_star.ravel(),
            "n": np.repeat(n_per, x.shape[0]),
        }
    )
    return adjusted.T, shifts


def main() -> None:
    """Harmonise the feature table and write it next to its provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--joined", type=Path, required=True, help="EDA joined_table.csv"
    )
    parser.add_argument("--min-batch", type=int, default=MIN_BATCH_DEFAULT)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features)
    joined = pd.read_csv(args.joined, low_memory=False)
    stratum, mapping = stratum_labels(joined, args.min_batch)
    missing = set(features["case_id"]) - set(stratum.index)
    if missing:
        raise ValueError(
            f"{len(missing)} feature rows have no acquisition stratum: {sorted(missing)[:5]}"
        )
    batch = stratum.loc[features["case_id"]].to_numpy()
    feats = [
        c
        for c in features.columns
        if c not in NON_FEATURE_COLUMNS and features[c].std(skipna=True) > 0
    ]
    x = features[feats].to_numpy(dtype=float)
    col_means = np.nanmean(x, axis=0)
    nan_mask = np.isnan(x)
    x = np.where(nan_mask, col_means, x)
    adjusted, shifts = combat(x, batch)
    adjusted[nan_mask] = np.nan
    out = features.copy()
    out[feats] = adjusted
    out.to_csv(args.out_dir / "tabular_baseline_features.csv", index=False)
    shifts["feature"] = [feats[i] for i in shifts["feature_index"]]
    shifts.drop(columns="feature_index").to_csv(
        args.out_dir / "combat_batch_shift.csv", index=False
    )
    counts = pd.Series(batch).value_counts()
    (args.out_dir / "combat_provenance.json").write_text(
        json.dumps(
            {
                "source_features": str(args.features),
                "batch_definition": "hr scanner model x field strength x software version (pooled below min_batch)",
                "min_batch": args.min_batch,
                "batches": counts.to_dict(),
                "stratum_to_batch": mapping,
                "covariates_preserved": [],
                "fit": "whole-cohort, label-free, parametric empirical Bayes",
                "n_features": len(feats),
                "n_exams": int(len(out)),
            },
            indent=2,
        )
    )
    print(f"ComBat over {len(counts)} batches: {counts.to_dict()}")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
