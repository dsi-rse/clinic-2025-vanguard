"""Tally one paired-preprocessing output root against its case manifest.

For every case in the manifest this reports how far it got through
``preprocessing/run_cohort.py`` (prepare -> infer -> tc4d -> map -> qc), the
policy it was built under, the HR/UFAST alignment status the mapping step
recorded, the skeleton voxel count, the vessel tiling used, and the HR volume
shape. Nothing is inferred from directory names alone: each stage is judged by
the same artifacts ``run_cohort.py::_stage_complete`` checks.

Outputs (in ``--out-dir``):
    preprocessing_summary.csv   one row per case-manifest exam
    preprocessing_summary.json  per-dataset counts by furthest stage and alignment status

Usage::

    python scripts/summarize_preprocessing_run.py \
        --case-manifest <staging>/case_manifest.csv \
        --output-root <cohort>/preprocessing_out \
        --out-dir <cohort>/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.cases import read_case_manifest
from segmentation.batch_segmentation import (
    VESSEL_MIN_XY_DIVISIONS,
    VESSEL_MIN_Z_DIVISIONS,
)


def _furthest_stage(output_root: Path, exam_id: str, dataset: str) -> str:
    work = output_root / "work" / exam_id
    centerline = output_root / "centerlines" / dataset / exam_id
    provenance_path = work / "preprocessing_provenance.json"
    if not provenance_path.is_file():
        return "none"
    provenance = json.loads(provenance_path.read_text())
    reached = "none"
    if provenance.get("status") == "prepared":
        reached = "prepare"
    else:
        return reached
    if "inference" in provenance and any(
        (work / "hr_vessel_predictions").glob("*.npz")
    ):
        reached = "infer"
    else:
        return reached
    if "tc4d" in provenance and (work / "hr_tc4d" / "exam_skeleton_zyx.npy").is_file():
        reached = "tc4d"
    else:
        return reached
    if (centerline / f"{exam_id}_skeleton_4d_exam_mask.npy").is_file():
        reached = "map"
    else:
        return reached
    if (centerline / "mapping_qc.png").is_file():
        reached = "qc"
    return reached


def summarize_case(output_root: Path, exam_id: str, dataset: str) -> dict[str, Any]:
    """One summary row for an exam, tolerant of any stage being absent."""
    work = output_root / "work" / exam_id
    centerline = output_root / "centerlines" / dataset / exam_id
    row: dict[str, Any] = {
        "exam_id": exam_id,
        "dataset": dataset,
        "furthest_stage": _furthest_stage(output_root, exam_id, dataset),
        "policy": None,
        "hr_shape_zyx": None,
        "hr_spacing_xyz_mm": None,
        "hr_phases": None,
        "ufast_phases": None,
        "ufast_median_dt_s": None,
        "vessel_x_y_divisions": None,
        "vessel_z_division": None,
        "alignment_qc_status": None,
        "alignment_components": None,
        "tc4d_skeleton_voxels": None,
        "mapped_skeleton_voxels": None,
    }
    provenance_path = work / "preprocessing_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        row["policy"] = (provenance.get("policy") or {}).get("name")
        hr = provenance.get("hr_source") or {}
        ufast = provenance.get("ufast_source") or {}
        geometry = hr.get("geometry") or {}
        row["hr_shape_zyx"] = json.dumps(geometry.get("shape_zyx"))
        row["hr_spacing_xyz_mm"] = json.dumps(geometry.get("spacing_xyz_mm"))
        row["hr_phases"] = len(hr.get("times_seconds") or [])
        ufast_times = ufast.get("times_seconds") or []
        row["ufast_phases"] = len(ufast_times)
        if len(ufast_times) > 1:
            row["ufast_median_dt_s"] = float(np.median(np.diff(ufast_times)))
        tiling = (provenance.get("inference") or {}).get("vessel_tiling") or {}
        row["vessel_x_y_divisions"] = tiling.get("x_y_divisions")
        row["vessel_z_division"] = tiling.get("z_division")
        row["tc4d_skeleton_voxels"] = (provenance.get("tc4d") or {}).get(
            "skeleton_voxels"
        )
    summary_path = centerline / "run_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        row["alignment_qc_status"] = summary.get("alignment_qc_status")
        row["alignment_components"] = json.dumps(summary.get("alignment_qc_components"))
    mask_path = centerline / f"{exam_id}_skeleton_4d_exam_mask.npy"
    if mask_path.is_file():
        row["mapped_skeleton_voxels"] = int(np.load(mask_path, mmap_mode="r").sum())
    return row


def main() -> None:
    """Write the per-exam table and the per-dataset counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    records = read_case_manifest(args.case_manifest)
    table = pd.DataFrame(
        [summarize_case(args.output_root, r.exam_id, r.dataset) for r in records]
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "preprocessing_summary.csv", index=False)

    counts = {
        "n_cases": len(table),
        "by_stage": table["furthest_stage"].value_counts().to_dict(),
        "by_dataset_stage": {
            dataset: group["furthest_stage"].value_counts().to_dict()
            for dataset, group in table.groupby("dataset")
        },
        "alignment_qc_status": table["alignment_qc_status"]
        .fillna("not_mapped")
        .value_counts()
        .to_dict(),
        "by_dataset_alignment": {
            dataset: group["alignment_qc_status"]
            .fillna("not_mapped")
            .value_counts()
            .to_dict()
            for dataset, group in table.groupby("dataset")
        },
        "policies": table["policy"].fillna("none").value_counts().to_dict(),
        "non_default_tiling": int(
            (
                table["vessel_x_y_divisions"].gt(VESSEL_MIN_XY_DIVISIONS)
                | table["vessel_z_division"].gt(VESSEL_MIN_Z_DIVISIONS)
            ).sum()
        ),
    }
    (args.out_dir / "preprocessing_summary.json").write_text(
        json.dumps(counts, indent=2)
    )
    print(json.dumps(counts, indent=2))
    incomplete = table[table["furthest_stage"] != "qc"]
    if len(incomplete):
        print(f"\n{len(incomplete)} exams short of qc:")
        print(
            incomplete[["exam_id", "dataset", "furthest_stage"]].to_string(index=False)
        )


if __name__ == "__main__":
    main()
