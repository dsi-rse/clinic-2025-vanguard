r"""EDA: how much of the vessel-skeleton feature space is acquisition, not biology?

Joins one arm's tabular feature table (``tabular.gnn_feature_baseline``) with
three per-exam metadata tables and asks, from several directions, whether the
features track the scanner/protocol/site rather than pCR:

1. **Census** -- per dataset: vendor x model x software x field strength,
   study-date range, spacing / slice thickness / UFAST cadence. The "what
   differs between sites" view.
2. **Feature vs numeric metadata** -- Spearman |rho| for every feature against
   spacing, slice thickness, FOV, matrix, TR/TE/FA, UFAST cadence, study year.
3. **Feature vs categorical metadata** -- one-way ANOVA eta^2 (fraction of a
   feature's variance lying *between* groups) for dataset, source, model,
   software, protocol, plus pCR for scale.
4. **PCA** of the standardised features; PC1/PC2 scatter coloured by dataset,
   scanner model, and study year.
5. **Batch detectability** -- patient-grouped, stratified 5-fold CV classifiers
   (logistic regression and random forest, from ``scripts/batch_effect``)
   predicting dataset, scanner model, software major version, study-year
   bucket, and pCR from the features; macro one-vs-rest AUROC plus the
   permutation importance of each feature for each target. A target that is
   predicted far above chance while pCR is not is the signature of a batch
   effect; the importances say which features carry it.
6. **Size features vs field of view** -- ``num_nodes``/``num_edges`` against
   HR FOV and voxel volume, since those were the top pCR features at n=230.

Dates: DICOM ``StudyDate`` can be a de-identification placeholder (e.g.
Jan 1); the census reports how many dates fall on Jan 1 so the year analyses
are read with that in mind.

Inputs::

    --features            tabular_baseline_features.csv (one arm)
    --labels              labels.csv (case_id, dataset, sources, patient_group_key, pcr, ...)
    --dicom-metadata      dicom_acquisition_metadata_per_exam.csv
    --preprocessing-summary preprocessing_summary.csv
    --manifest-metadata   exam_metadata_from_manifest.csv

Outputs (``--out-dir``): ``census_*.csv``, ``feature_vs_numeric_spearman.csv``
(+ ``.png``), ``feature_vs_categorical_eta2.csv`` (+ ``.png``), ``pca_scores.csv``,
``pca_by_*.png``, ``batch_detectability.csv``, ``batch_importance.csv``,
``batch_detectability.png``, ``size_vs_fov.png``, ``joined_table.csv``.

Usage::

    python analysis/feature_metadata_eda.py --features ... --labels ... \\
        --dicom-metadata ... --preprocessing-summary ... --manifest-metadata ... \\
        --out-dir <cohort>/results/eda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.batch_effect.batch_detectability_ispy2 import macro_auc, make_model
from scripts.batch_effect.pca_ispy2_batch_effect import eta_squared_and_p

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NON_FEATURE_COLUMNS = {"case_id", "dataset", "fold", "pcr"}
NUMERIC_METADATA = [
    "hr_pixel_spacing_row_mm",
    "hr_slice_thickness_mm",
    "hr_rows",
    "hr_fov_row_mm",
    "hr_voxel_volume_mm3",
    "hr_repetition_time_ms",
    "hr_echo_time_ms",
    "hr_flip_angle_deg",
    "ufast_pixel_spacing_row_mm",
    "ufast_slice_thickness_mm",
    "ufast_median_dt_s",
    "ufast_phases",
    "study_year",
]
CATEGORICAL_METADATA = [
    "dataset",
    "sources",
    "hr_manufacturer_model_name",
    "hr_software_major",
    "hr_protocol_name",
    "ufast_protocol_name",
    "study_year_bucket",
    "pcr",
]
DETECT_TARGETS = [
    "dataset",
    "hr_manufacturer_model_name",
    "hr_software_major",
    "study_year_bucket",
    "pcr",
]
MIN_CLASS_SIZE = 5
MIN_CLASSES = 2
N_FOLDS = 5
SEED = 42
GREY = "#444444"
PALETTE = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#000000",
    "#DDAA33",
    "#004488",
]


def _first_json_value(series: pd.Series) -> pd.Series:
    def first(value: object) -> float:
        if isinstance(value, str) and value.startswith("["):
            return float(json.loads(value)[0])
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    return series.map(first)


def build_joined_table(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    dicom: pd.DataFrame,
    summary: pd.DataFrame,
    manifest_meta: pd.DataFrame | None,
) -> pd.DataFrame:
    """One row per exam: features + labels + acquisition + pipeline metadata."""
    feats = features.drop(columns=["dataset", "pcr", "fold"], errors="ignore")
    lab = (
        labels.rename(columns={"case_id": "exam_id"}) if "case_id" in labels else labels
    )
    table = (
        feats.rename(columns={"case_id": "exam_id"})
        .merge(lab, on="exam_id", how="inner", validate="one_to_one")
        .merge(
            dicom.drop(columns=["dataset"]),
            on="exam_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            summary.drop(columns=["dataset"]),
            on="exam_id",
            how="left",
            validate="one_to_one",
        )
    )
    if manifest_meta is not None:
        table = table.merge(
            manifest_meta.drop(columns=["dataset", "pcr"], errors="ignore"),
            on="exam_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "_manifest"),
        )
    if "patient_group_key" not in table:
        table["patient_group_key"] = table["exam_id"]
    if "pcr_label_is_provisional" not in table:
        table["pcr_label_is_provisional"] = np.nan
    if len(table) != len(features):
        raise ValueError(
            f"{len(features) - len(table)} feature rows have no label row; "
            "the features and labels files disagree"
        )
    for role in ("hr", "ufast"):
        table[f"{role}_pixel_spacing_row_mm"] = _first_json_value(
            table[f"{role}_pixel_spacing_mm"]
        )
        table[f"{role}_software_major"] = (
            table[f"{role}_software_versions"]
            .map(
                lambda v: json.loads(v)[0].split(".")[0] if isinstance(v, str) else None
            )
            .astype("string")
        )
    table["hr_fov_row_mm"] = table["hr_pixel_spacing_row_mm"] * table["hr_rows"]
    table["hr_voxel_volume_mm3"] = (
        table["hr_pixel_spacing_row_mm"] ** 2 * table["hr_slice_thickness_mm"]
    )
    dates = pd.to_datetime(
        table["hr_study_date"].astype("Int64").astype(str), errors="coerce"
    )
    table["study_date"] = dates
    table["study_year"] = dates.dt.year
    table["study_date_is_jan1"] = (dates.dt.month == 1) & (dates.dt.day == 1)
    table["study_year_bucket"] = pd.cut(
        table["study_year"],
        bins=[0, 2017, 2020, 2022, 9999],
        labels=["<=2017", "2018-2020", "2021-2022", ">=2023"],
    ).astype("string")
    return table


def feature_columns(table: pd.DataFrame, features: pd.DataFrame) -> list[str]:
    """Non-constant feature columns of the arm's feature table."""
    return [
        c
        for c in features.columns
        if c not in NON_FEATURE_COLUMNS and table[c].std(skipna=True) > 0
    ]


def census(table: pd.DataFrame, out_dir: Path) -> None:
    """Per-dataset acquisition summary tables."""
    scanner = (
        table.groupby(
            [
                "dataset",
                "hr_manufacturer",
                "hr_manufacturer_model_name",
                "hr_magnetic_field_strength",
                "hr_software_major",
            ],
            dropna=False,
        )
        .size()
        .rename("n_exams")
        .reset_index()
    )
    scanner.to_csv(out_dir / "census_scanner_by_dataset.csv", index=False)
    numeric = (
        table.groupby("dataset")
        .agg(
            n=("exam_id", "size"),
            pcr_rate=("pcr", "mean"),
            provisional_frac=("pcr_label_is_provisional", "mean"),
            year_min=("study_year", "min"),
            year_max=("study_year", "max"),
            jan1_dates=("study_date_is_jan1", "sum"),
            hr_spacing_median=("hr_pixel_spacing_row_mm", "median"),
            hr_slice_mm_median=("hr_slice_thickness_mm", "median"),
            hr_rows_median=("hr_rows", "median"),
            ufast_slice_mm_median=("ufast_slice_thickness_mm", "median"),
            ufast_dt_median=("ufast_median_dt_s", "median"),
            ufast_phases_median=("ufast_phases", "median"),
            hr_tr_median=("hr_repetition_time_ms", "median"),
            hr_fa_median=("hr_flip_angle_deg", "median"),
        )
        .reset_index()
    )
    numeric.to_csv(out_dir / "census_acquisition_by_dataset.csv", index=False)
    protocols = (
        table.groupby(
            ["dataset", "hr_protocol_name", "ufast_protocol_name"], dropna=False
        )
        .size()
        .rename("n_exams")
        .reset_index()
    )
    protocols.to_csv(out_dir / "census_protocol_by_dataset.csv", index=False)


def feature_vs_numeric(table: pd.DataFrame, feats: list[str], out_dir: Path) -> None:
    """Spearman correlation of every feature with each numeric acquisition parameter."""
    cols = [c for c in NUMERIC_METADATA if c in table and table[c].std(skipna=True) > 0]
    rho = pd.DataFrame(index=feats, columns=cols, dtype=float)
    pval = rho.copy()
    for f in feats:
        for m in cols:
            pair = table[[f, m]].dropna()
            if len(pair) < MIN_CLASS_SIZE * 2:
                continue
            r, p = stats.spearmanr(pair[f], pair[m])
            rho.loc[f, m], pval.loc[f, m] = r, p
    rho.to_csv(out_dir / "feature_vs_numeric_spearman.csv")
    pval.to_csv(out_dir / "feature_vs_numeric_spearman_p.csv")
    _heatmap(
        rho.abs(),
        out_dir / "feature_vs_numeric_spearman.png",
        "|Spearman rho| feature vs acquisition metadata",
        vmax=1.0,
    )


def feature_vs_categorical(
    table: pd.DataFrame, feats: list[str], out_dir: Path
) -> None:
    """ANOVA eta^2 of every feature across each categorical acquisition variable."""
    cols = [
        c
        for c in CATEGORICAL_METADATA
        if c in table and len(table[c].dropna().unique()) > 1
    ]
    eta = pd.DataFrame(index=feats, columns=cols, dtype=float)
    pval = eta.copy()
    for f in feats:
        for m in cols:
            r2, p = eta_squared_and_p(table[f].to_numpy(dtype=float), table[m])
            eta.loc[f, m], pval.loc[f, m] = r2, p
    eta.to_csv(out_dir / "feature_vs_categorical_eta2.csv")
    pval.to_csv(out_dir / "feature_vs_categorical_eta2_p.csv")
    _heatmap(
        eta,
        out_dir / "feature_vs_categorical_eta2.png",
        "eta^2: fraction of feature variance between groups",
        vmax=max(0.05, float(np.nanmax(eta.to_numpy()))),
    )


def _heatmap(frame: pd.DataFrame, out_path: Path, title: str, *, vmax: float) -> None:
    fig, ax = plt.subplots(
        figsize=(1.1 * len(frame.columns) + 3, 0.28 * len(frame) + 2)
    )
    image = ax.imshow(
        frame.to_numpy(dtype=float), aspect="auto", cmap="Greys", vmin=0, vmax=vmax
    )
    ax.set_xticks(range(len(frame.columns)))
    ax.set_xticklabels(frame.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels(frame.index, fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _standardised(table: pd.DataFrame, feats: list[str]) -> np.ndarray:
    x = table[feats].to_numpy(dtype=float)
    x = SimpleImputer(strategy="median").fit_transform(x)
    return StandardScaler().fit_transform(x)


def pca_views(table: pd.DataFrame, feats: list[str], out_dir: Path) -> dict:
    """PCA of the standardised features with PC1/PC2 scatters per colouring."""
    x = _standardised(table, feats)
    pca = PCA(n_components=min(10, len(feats)), random_state=SEED).fit(x)
    scores = pca.transform(x)
    score_table = table[["exam_id", "dataset", "pcr"]].copy()
    score_table["pc1"], score_table["pc2"] = scores[:, 0], scores[:, 1]
    score_table.to_csv(out_dir / "pca_scores.csv", index=False)
    evr = pca.explained_variance_ratio_
    eta = {}
    for colour_by in (
        "dataset",
        "hr_manufacturer_model_name",
        "study_year_bucket",
        "pcr",
    ):
        if (
            colour_by not in table
            or len(table[colour_by].dropna().unique()) < MIN_CLASSES
        ):
            continue
        groups = table[colour_by].astype("string").fillna("missing")
        eta[colour_by] = {
            f"pc{k + 1}_eta2": float(eta_squared_and_p(scores[:, k], groups)[0])
            for k in range(2)
        }
        fig, ax = plt.subplots(figsize=(6, 5))
        for colour, (name, idx) in zip(
            PALETTE, groups.groupby(groups).groups.items(), strict=False
        ):
            ax.scatter(
                scores[idx, 0],
                scores[idx, 1],
                s=14,
                color=colour,
                alpha=0.8,
                label=f"{name} (n={len(idx)})",
                edgecolors="none",
            )
        ax.set_xlabel(f"PC1 ({evr[0]:.0%})")
        ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
        ax.set_title(f"feature PCA coloured by {colour_by}", fontsize=10)
        ax.legend(frameon=False, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / f"pca_by_{colour_by}.png", dpi=150)
        plt.close(fig)
    loadings = pd.DataFrame(
        pca.components_[:3].T, index=feats, columns=["pc1", "pc2", "pc3"]
    )
    loadings.to_csv(out_dir / "pca_loadings.csv")
    return {"explained_variance_ratio": evr.tolist(), "eta2_by_colouring": eta}


def batch_detectability(
    table: pd.DataFrame, feats: list[str], out_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Can the features predict the acquisition variable? Patient-grouped CV."""
    x = table[feats].to_numpy(dtype=float)
    groups = table["patient_group_key"].to_numpy()
    rows, importance_rows = [], []
    for target in DETECT_TARGETS:
        if target not in table:
            continue
        labels = table[target].astype("string").fillna("missing")
        counts = labels.value_counts()
        keep = labels.isin(counts.index[counts >= MIN_CLASS_SIZE]).to_numpy()
        if len(labels[keep].unique()) < MIN_CLASSES:
            continue
        classes = sorted(labels[keep].unique())
        y = labels[keep].map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=int)
        xk, gk = x[keep], groups[keep]
        splitter = StratifiedGroupKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=SEED
        )
        folds = list(splitter.split(xk, y, gk))
        for model_name in ("logreg", "rf"):
            proba = np.zeros((len(y), len(classes)))
            fold_aucs = []
            importance = np.zeros(len(feats))
            for train_idx, test_idx in folds:
                model = make_model(model_name, SEED)
                model.fit(xk[train_idx], y[train_idx])
                fold_proba = model.predict_proba(xk[test_idx])
                proba[test_idx] = fold_proba
                if len(set(y[test_idx])) == len(classes):
                    fold_aucs.append(macro_auc(y[test_idx], fold_proba, len(classes)))
                perm = permutation_importance(
                    model,
                    xk[test_idx],
                    y[test_idx],
                    n_repeats=5,
                    random_state=SEED,
                    scoring="balanced_accuracy",
                )
                importance += perm.importances_mean / len(folds)
            pooled = macro_auc(y, proba, len(classes))
            chance = 0.5
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "n": int(len(y)),
                    "n_classes": len(classes),
                    "classes": json.dumps(classes),
                    "pooled_macro_auroc": pooled,
                    "fold_auroc_mean": float(np.mean(fold_aucs))
                    if fold_aucs
                    else np.nan,
                    "fold_auroc_std": float(np.std(fold_aucs)) if fold_aucs else np.nan,
                    "chance": chance,
                }
            )
            for f, imp in zip(feats, importance, strict=True):
                importance_rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "feature": f,
                        "importance": imp,
                    }
                )
    result = pd.DataFrame(rows)
    importance_table = pd.DataFrame(importance_rows)
    result.to_csv(out_dir / "batch_detectability.csv", index=False)
    importance_table.to_csv(out_dir / "batch_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 0.5 * len(result) + 1.5))
    labels_txt = [
        f"{r.target} / {r.model} (K={r.n_classes}, n={r.n})"
        for r in result.itertuples()
    ]
    ypos = np.arange(len(result))
    ax.barh(ypos, result["pooled_macro_auroc"], color=GREY)
    ax.axvline(0.5, color="#999999", linestyle="--", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels_txt, fontsize=8)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("pooled OOF macro AUROC (patient-grouped 5-fold)")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "batch_detectability.png", dpi=150)
    plt.close(fig)
    return result, importance_table


def size_vs_fov(table: pd.DataFrame, out_dir: Path) -> None:
    """Skeleton-size features against field of view, voxel volume and cadence."""
    pairs = [
        ("hr_fov_row_mm", "num_nodes"),
        ("hr_voxel_volume_mm3", "num_nodes"),
        ("hr_fov_row_mm", "num_edges"),
        ("ufast_median_dt_s", "num_nodes"),
    ]
    pairs = [(a, b) for a, b in pairs if a in table and b in table]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.6))
    for ax, (a, b) in zip(np.atleast_1d(axes), pairs, strict=True):
        pair = table[[a, b]].dropna()
        ax.scatter(pair[a], pair[b], s=10, color=GREY, alpha=0.7, edgecolors="none")
        r, p = stats.spearmanr(pair[a], pair[b])
        ax.set_title(f"rho={r:.2f}, p={p:.1e}, n={len(pair)}", fontsize=9)
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "size_vs_fov.png", dpi=150)
    plt.close(fig)


def date_drift(table: pd.DataFrame, feats: list[str], out_dir: Path) -> None:
    """Feature medians by dataset and study year (placeholder dates excluded)."""
    keep = table[~table["study_date_is_jan1"].fillna(True)]
    drift = keep.groupby(["dataset", "study_year"])[feats].median()
    drift.insert(0, "n", keep.groupby(["dataset", "study_year"]).size())
    drift.to_csv(out_dir / "feature_medians_by_dataset_year.csv")


def main() -> None:
    """Join the tables and write every EDA output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dicom-metadata", type=Path, required=True)
    parser.add_argument("--preprocessing-summary", type=Path, required=True)
    parser.add_argument("--manifest-metadata", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    table = build_joined_table(
        features,
        pd.read_csv(args.labels),
        pd.read_csv(args.dicom_metadata),
        pd.read_csv(args.preprocessing_summary),
        pd.read_csv(args.manifest_metadata) if args.manifest_metadata else None,
    )
    feats = feature_columns(table, features)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "joined_table.csv", index=False)
    print(f"{len(table)} exams, {len(feats)} non-constant features")

    census(table, args.out_dir)
    feature_vs_numeric(table, feats, args.out_dir)
    feature_vs_categorical(table, feats, args.out_dir)
    pca_summary = pca_views(table, feats, args.out_dir)
    detect, _ = batch_detectability(table, feats, args.out_dir)
    size_vs_fov(table, args.out_dir)
    date_drift(table, feats, args.out_dir)

    (args.out_dir / "eda_summary.json").write_text(
        json.dumps(
            {
                "n_exams": int(len(table)),
                "n_features": len(feats),
                "jan1_study_dates": int(
                    table["study_date_is_jan1"].fillna(False).sum()
                ),
                "pca": pca_summary,
                "batch_detectability": detect.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    print(
        detect[["target", "model", "n", "n_classes", "pooled_macro_auroc"]].to_string(
            index=False
        )
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
