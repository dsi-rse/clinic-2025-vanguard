"""Build the paired HR/UFAST case manifest from Anna's canonical UChicago DCE manifest.

The canonical interface is one file, ``current_manifest.csv`` under
``/gpfs/data/karczmar-lab/hfdp/derived_datasets/hfdp/uchicago_dce_cache/``:
one row per exam carrying every source acquisition (``series_json``), its
original DICOM locations (``original_dicom_locations_json``), Anna's typed
preprocessing outcome per acquisition (``preprocessing_json``, which is where
``baseline_frame_count`` lives), and the pCR / treatment-time metadata. This
script reads that file and nothing else, so cohort definition has exactly one
upstream source.

Selection (Anna's rule, applied verbatim)::

    treatment_time == "pretreatment"
    and pcr in {0, 1}
    and ultrafast_series_count > 0
    and high_resolution_series_count > 0

Acquisition choice. Some exams carry more than one HR or more than one UFAST
acquisition. For each role the candidates are the acquisitions whose Anna-side
``preprocessing_status`` is ``complete`` and whose phases all live in one
SeriesInstanceUID (``preprocessing/dicom.py::load_dicom_series`` loads one UID
per series); UFAST candidates must also have ``baseline_status == available``.
Among candidates the one with the most ``native_frame_count`` wins; ties go
to the lowest ``acquisition_order_index``. Every candidate, the winner, and
the reason are written to ``case_selection_provenance.json`` so the choice is
auditable. An exam with no candidate for either role is excluded with a
reason, never patched.

Folds. The manifest carries no fold assignment (``split_role`` is ``train``
for every exam), so this script mints one: 5-fold, grouped by
``patient_group_key`` (no patient in two folds), stratified on
``pcr x dataset``, ``random_state=42``. Both arms of any downstream comparison
read the same ``labels.csv``, so they are paired by construction.

Outputs (all in ``--output-dir``):
    case_manifest.csv               the six columns ``preprocessing/cases.py`` requires
    labels.csv                      case_id, dataset, sources, patient_group_key, pcr,
                                    pcr_label_is_provisional, pcr_label_authority, fold
    excluded.csv                    exam_id, dataset, reason, detail
    case_selection_provenance.json  per-exam candidates and choice
    exam_metadata_from_manifest.csv per-exam manifest-derived fields for the EDA
    manifest_snapshot.csv(.sha256)  the consumed manifest, byte-for-byte, plus hash

Usage::

    python scripts/build_uchicago_dce_case_manifest.py \
        --manifest /gpfs/.../uchicago_dce_cache/current_manifest.csv \
        --output-dir /gpfs/.../uchicago_dce_g16/staging
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CASE_COLUMNS = [
    "exam_id",
    "dataset",
    "study_instance_uid",
    "hr_series_instance_uid",
    "ufast_series_instance_uid",
    "ufast_baseline_frame_count",
]
LABEL_COLUMNS = [
    "case_id",
    "dataset",
    "sources",
    "patient_group_key",
    "pcr",
    "pcr_label_is_provisional",
    "pcr_label_authority",
    "fold",
]
EXCLUSION_COLUMNS = ["exam_id", "dataset", "reason", "detail"]
ROLES = ("high_resolution", "ultrafast")
N_FOLDS = 5
FOLD_SEED = 42


def select_exams(manifest: pd.DataFrame) -> pd.DataFrame:
    """Apply Anna's selection rule exactly as documented in the module docstring."""
    mask = (
        manifest["treatment_time"].eq("pretreatment")
        & manifest["pcr"].isin([0, 1])
        & manifest["ultrafast_series_count"].gt(0)
        & manifest["high_resolution_series_count"].gt(0)
    )
    selected = manifest.loc[mask].copy()
    if selected["exam_key"].duplicated().any():
        raise ValueError("manifest has duplicate exam_key rows after selection")
    return selected


def dataset_name(dataset_labels_json: str) -> str:
    """One safe path component naming the exam's dataset label set, e.g. ``a+b``."""
    labels = json.loads(dataset_labels_json)
    if not labels:
        raise ValueError("dataset_labels_json is empty")
    name = "+".join(labels)
    if Path(name).name != name or "/" in name:
        raise ValueError(f"dataset name is not a safe path component: {name!r}")
    return name


def _candidates(acquisitions: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    """Per-acquisition eligibility for one role, with the reason it is or is not eligible."""
    rows = []
    for acq in acquisitions:
        if acq["acquisition_role"] != role:
            continue
        phase_uids = list(dict.fromkeys(acq["phase_series_instance_uids"]))
        reasons = []
        if acq["preprocessing_status"] != "complete":
            reasons.append(f"preprocessing_status={acq['preprocessing_status']}")
        if len(phase_uids) != 1:
            reasons.append(f"phases_span_{len(phase_uids)}_series_uids")
        if role == "ultrafast" and acq.get("baseline_status") != "available":
            reasons.append(f"baseline_status={acq.get('baseline_status')}")
        rows.append(
            {
                "acquisition_order_index": int(acq["acquisition_order_index"]),
                "series_instance_uid": phase_uids[0] if len(phase_uids) == 1 else None,
                "phase_series_instance_uids": phase_uids,
                "native_frame_count": int(acq["native_frame_count"]),
                "baseline_frame_count": acq.get("baseline_frame_count"),
                "eligible": not reasons,
                "ineligible_reasons": reasons,
            }
        )
    return rows


def choose_acquisition(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    """Most native frames wins; ties go to the lowest acquisition order."""
    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        return None, "no_eligible_acquisition"
    eligible.sort(
        key=lambda c: (-c["native_frame_count"], c["acquisition_order_index"])
    )
    winner = eligible[0]
    if len(eligible) == 1:
        reason = "only_eligible_acquisition"
    elif eligible[1]["native_frame_count"] < winner["native_frame_count"]:
        reason = "most_native_frames"
    else:
        reason = "tie_on_native_frames_lowest_acquisition_order"
    return winner, reason


def _location_for(
    locations_payload: dict[str, Any], series_uid: str, study_uid: str
) -> dict[str, Any]:
    """The resolved DICOM location entry for one series, checked against the study."""
    matches = [
        s for s in locations_payload["series"] if s["series_instance_uid"] == series_uid
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one location entry for series {series_uid}, got {len(matches)}"
        )
    entry = matches[0]
    if entry["status"] != "resolved" or not entry["locations"]:
        raise ValueError(f"series {series_uid} location is not resolved")
    for loc in entry["locations"]:
        if loc["study_instance_uid"] != study_uid:
            raise ValueError(
                f"series {series_uid} location study {loc['study_instance_uid']} "
                f"!= exam study {study_uid}"
            )
    return entry


def build_cases(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Resolve every selected exam to one HR + one UFAST series, or an exclusion."""
    case_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for row in selected.itertuples(index=False):
        exam_id = str(row.exam_key)
        dataset = dataset_name(row.dataset_labels_json)
        study_uid = str(row.study_instance_uid)
        if not isinstance(row.preprocessing_json, str) or not row.preprocessing_json:
            excluded_rows.append(
                {
                    "exam_id": exam_id,
                    "dataset": dataset,
                    "reason": "no_preprocessing_json",
                    "detail": f"preprocessing_status={row.preprocessing_status}",
                }
            )
            continue
        acquisitions = json.loads(row.preprocessing_json)
        locations = json.loads(row.original_dicom_locations_json)
        exam_prov: dict[str, Any] = {"dataset": dataset, "roles": {}}
        chosen: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        for role in ROLES:
            candidates = _candidates(acquisitions, role)
            winner, reason = choose_acquisition(candidates)
            exam_prov["roles"][role] = {
                "candidates": candidates,
                "chosen_acquisition_order_index": (
                    winner["acquisition_order_index"] if winner else None
                ),
                "choice_reason": reason,
            }
            if winner is None:
                failures.append(
                    f"{role}: "
                    + "; ".join(
                        f"acq{c['acquisition_order_index']}="
                        + ",".join(c["ineligible_reasons"])
                        for c in candidates
                    )
                )
            else:
                chosen[role] = winner
        provenance[exam_id] = exam_prov
        if failures:
            excluded_rows.append(
                {
                    "exam_id": exam_id,
                    "dataset": dataset,
                    "reason": "no_eligible_acquisition",
                    "detail": " | ".join(failures),
                }
            )
            continue
        for role, winner in chosen.items():
            entry = _location_for(locations, winner["series_instance_uid"], study_uid)
            exam_prov["roles"][role]["location_kinds"] = [
                loc["kind"] for loc in entry["locations"]
            ]
        case_rows.append(
            {
                "exam_id": exam_id,
                "dataset": dataset,
                "study_instance_uid": study_uid,
                "hr_series_instance_uid": chosen["high_resolution"][
                    "series_instance_uid"
                ],
                "ufast_series_instance_uid": chosen["ultrafast"]["series_instance_uid"],
                "ufast_baseline_frame_count": int(
                    chosen["ultrafast"]["baseline_frame_count"]
                ),
            }
        )
    cases = pd.DataFrame(case_rows, columns=CASE_COLUMNS)
    excluded = pd.DataFrame(excluded_rows, columns=EXCLUSION_COLUMNS)
    if cases["exam_id"].duplicated().any():
        raise ValueError("duplicate exam_id in case manifest")
    return cases, excluded, provenance


def assign_folds(labels: pd.DataFrame) -> pd.Series:
    """Patient-grouped, pcr x dataset-stratified fold ids aligned to ``labels`` rows."""
    strata = labels["pcr"].astype(str) + "|" + labels["dataset"]
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED
    )
    fold = np.full(len(labels), -1, dtype=int)
    for k, (_, test_idx) in enumerate(
        splitter.split(labels, strata, groups=labels["patient_group_key"])
    ):
        fold[test_idx] = k
    if (fold < 0).any():
        raise RuntimeError("some cases received no fold")
    patient_folds = pd.DataFrame(
        {"patient": labels["patient_group_key"].to_numpy(), "fold": fold}
    ).drop_duplicates()
    if patient_folds["patient"].duplicated().any():
        raise RuntimeError("a patient was assigned to more than one fold")
    return pd.Series(fold, index=labels.index)


def build_labels(selected: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """Labels, grouping key, provisional flag and folds for the runnable cases."""
    keep = selected.set_index("exam_key").loc[cases["exam_id"]]
    labels = pd.DataFrame(
        {
            "case_id": cases["exam_id"].to_numpy(),
            "dataset": cases["dataset"].to_numpy(),
            "sources": [
                "+".join(json.loads(s)) for s in keep["source_names_json"].to_numpy()
            ],
            "patient_group_key": keep["patient_group_key"].to_numpy(),
            "pcr": keep["pcr"].astype(int).to_numpy(),
            "pcr_label_is_provisional": keep["pcr_label_is_provisional"]
            .astype(bool)
            .to_numpy(),
            "pcr_label_authority": keep["pcr_label_authority"].to_numpy(),
        }
    )
    labels["fold"] = assign_folds(labels).to_numpy()
    return labels[LABEL_COLUMNS]


def _median_dt(times: list[float]) -> float | None:
    diffs = np.diff(np.asarray(times, dtype=float))
    return float(statistics.median(diffs)) if len(diffs) else None


def build_exam_metadata(
    selected: pd.DataFrame, cases: pd.DataFrame, provenance: dict[str, Any]
) -> pd.DataFrame:
    """Manifest-derived per-exam fields that the metadata EDA joins on."""
    keep = selected.set_index("exam_key").loc[cases["exam_id"]]
    rows = []
    for exam_id, row in keep.iterrows():
        acqs = {
            int(a["acquisition_order_index"]): a
            for a in json.loads(row.preprocessing_json)
        }
        out: dict[str, Any] = {
            "exam_id": exam_id,
            "dataset": provenance[exam_id]["dataset"],
            "sources": "+".join(json.loads(row.source_names_json)),
            "patient_group_key": row.patient_group_key,
            "manifest_generation": row.manifest_generation,
            "pcr": int(row.pcr),
            "pcr_label_is_provisional": bool(row.pcr_label_is_provisional),
            "pcr_label_scope": row.pcr_label_scope,
            "exam_context_is_provisional": row.exam_context_is_provisional,
            "reviewed_first_pass_category": row.reviewed_first_pass_category,
            "automatic_first_pass_observed": row.automatic_first_pass_observed,
        }
        for role in ROLES:
            short = "hr" if role == "high_resolution" else "ufast"
            idx = provenance[exam_id]["roles"][role]["chosen_acquisition_order_index"]
            acq = acqs[idx]
            out[f"{short}_native_frame_count"] = acq["native_frame_count"]
            out[f"{short}_median_dt_seconds"] = _median_dt(acq["native_times_seconds"])
            out[f"{short}_location_kinds"] = "+".join(
                provenance[exam_id]["roles"][role]["location_kinds"]
            )
            out[f"{short}_native_clock_origin_date"] = acq.get(
                "native_clock_origin_date"
            )
            out[f"{short}_motion_status"] = acq.get("motion_status")
            out[f"{short}_aif_status"] = acq.get("aif_status")
            out[f"{short}_n_candidate_acquisitions"] = len(
                provenance[exam_id]["roles"][role]["candidates"]
            )
        out["ufast_baseline_frame_count"] = acqs[
            provenance[exam_id]["roles"]["ultrafast"]["chosen_acquisition_order_index"]
        ]["baseline_frame_count"]
        rows.append(out)
    return pd.DataFrame(rows)


def snapshot_manifest(manifest_path: Path, output_dir: Path) -> str:
    """Copy the consumed manifest next to the outputs and record its SHA-256."""
    target = output_dir / "manifest_snapshot.csv"
    shutil.copyfile(manifest_path, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (output_dir / "manifest_snapshot.sha256").write_text(f"{digest}  {manifest_path}\n")
    return digest


def main() -> None:
    """Build the case manifest, labels, exclusions and provenance from the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    digest = snapshot_manifest(args.manifest, args.output_dir)
    manifest = pd.read_csv(args.manifest)
    selected = select_exams(manifest)
    cases, excluded, provenance = build_cases(selected)
    labels = build_labels(selected, cases)
    metadata = build_exam_metadata(selected, cases, provenance)

    cases.to_csv(args.output_dir / "case_manifest.csv", index=False)
    labels.to_csv(args.output_dir / "labels.csv", index=False)
    excluded.to_csv(args.output_dir / "excluded.csv", index=False)
    metadata.to_csv(args.output_dir / "exam_metadata_from_manifest.csv", index=False)
    (args.output_dir / "case_selection_provenance.json").write_text(
        json.dumps(
            {
                "manifest_path": str(args.manifest),
                "manifest_sha256": digest,
                "manifest_generation": sorted(
                    int(g) for g in selected["manifest_generation"].unique()
                ),
                "selection_rule": (
                    "treatment_time == pretreatment and pcr in {0,1} and "
                    "ultrafast_series_count > 0 and high_resolution_series_count > 0"
                ),
                "acquisition_choice_rule": (
                    "eligible = preprocessing_status complete, one SeriesInstanceUID "
                    "across phases, (ultrafast) baseline_status available; winner = "
                    "most native_frame_count, tie -> lowest acquisition_order_index"
                ),
                "fold_policy": {
                    "n_splits": N_FOLDS,
                    "random_state": FOLD_SEED,
                    "group": "patient_group_key",
                    "stratify": "pcr x dataset",
                },
                "exams": provenance,
            },
            indent=2,
        )
    )

    print(f"manifest sha256 {digest}")
    print(
        f"selected {len(selected)} exams; runnable {len(cases)}; excluded {len(excluded)}"
    )
    print(f"pcr: {labels['pcr'].value_counts().to_dict()}")
    print(f"provisional: {labels['pcr_label_is_provisional'].value_counts().to_dict()}")
    print(labels.groupby("dataset")["pcr"].agg(["count", "sum"]).to_string())
    print(labels.groupby("fold")["pcr"].agg(["count", "sum"]).to_string())
    if len(excluded):
        print(excluded.to_string(index=False))


if __name__ == "__main__":
    main()
