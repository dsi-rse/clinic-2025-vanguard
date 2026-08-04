"""Export one real UChicago vessel-node enhancement curve, numbers only.

For the symposium talk: a "movie" slide contrasts a public Duke enhancement
curve (already exported by ``poster_architecture_data.py``) with a UChicago
one. UChicago DCE-MRI is not deidentified and must not be printed on
audience-facing material (see ``presentations/symposium_poster/figures/
architecture_data/README.md``), so this script emits only the numeric curve
-- physical acquisition seconds and baseline-referenced enhancement at one
voxel -- never an MRI frame, mask overlay, or other image.

The node is the skeleton voxel closest to the centroid of all skeleton
voxels: a deterministic, non-cherry-picked choice of "a typical vessel
location" for this exam.

Reuses the same loaders and the same enhancement transform the real
training pipeline uses, so the curve is not a one-off computation:
``gnn.data_loader._load_study_metadata`` for the per-exam baseline contract,
``gnn.raw_dce`` for the raw series, and ``gnn.pretrain.node_series.
voxel_node_series`` (``gnn.kinetics.baseline_relative_curve`` under the hood).

Loads one exam's raw 4D DCE series -- run via Slurm, not the head node.

The published cohort exposes each exam's centerline directory as a symlink
into a ``_build/`` tree (see the cohort README), and ``pathlib.Path.rglob``
does not descend into symlinked directories on this Python version -- so the
study directory is joined directly from ``--centerline-root``/``--dataset``/
``--case-id`` rather than searched for, unlike the Duke script's ``rglob``-based
``_resolve_case``.

Usage:
    python -m analysis.uchicago_curve_export --case-id <exam_id> --dataset <dataset> \
        --centerline-root /gpfs/data/karczmar-lab/vanguard/uchicago_ultrafast_pretreatment_cohort_v1/centerlines \
        --dce-root /gpfs/data/karczmar-lab/vanguard/uchicago_ultrafast_pretreatment_cohort_v1/images/<dataset> \
        --out-dir presentations/symposium_talk/figures/uchicago_curve_data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gnn.data_loader import (
    _CENTERLINE_SUFFIX,
    _SUPPORT_PATTERN,
    _git_commit,
    _load_study_metadata,
)
from gnn.pretrain.node_series import voxel_node_series
from gnn.raw_dce import discover_raw_dce_paths, load_raw_dce_series, load_raw_dce_times


def _centroid_skeleton_voxel(skeleton: np.ndarray) -> tuple[int, int, int]:
    """The skeleton voxel nearest the centroid of all skeleton voxels, as (x, y, z)."""
    zyx = np.argwhere(skeleton)
    if zyx.shape[0] == 0:
        raise ValueError("skeleton mask has no voxels")
    centroid = zyx.mean(axis=0)
    distances = np.linalg.norm(zyx.astype(np.float64) - centroid, axis=1)
    z, y, x = zyx[int(np.argmin(distances))]
    return int(x), int(y), int(z)


def main() -> None:
    """Export one exam's real enhancement curve (numbers only) to JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--centerline-root", type=Path, required=True)
    parser.add_argument("--dce-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    study_dir = args.centerline_root / args.dataset / args.case_id
    mask_path = study_dir / f"{args.case_id}{_CENTERLINE_SUFFIX}"
    support_path = study_dir / _SUPPORT_PATTERN.format(case_id=args.case_id)
    if not mask_path.is_file():
        raise FileNotFoundError(
            f"case={args.case_id}: skeleton mask not found: {mask_path}"
        )
    if not support_path.is_file():
        raise FileNotFoundError(
            f"case={args.case_id}: support mask not found: {support_path}"
        )
    skeleton = np.load(mask_path).astype(bool)
    support = np.load(support_path).astype(bool)

    timepoints, baseline_frame_count, relative_enhancement = _load_study_metadata(
        args.case_id, mask_path.parent
    )

    dce_paths = discover_raw_dce_paths(args.dce_root, args.case_id, timepoints)
    dce_4d = load_raw_dce_series(dce_paths, expected_shape_zyx=support.shape)
    times_seconds = load_raw_dce_times(args.dce_root, args.case_id, timepoints)
    if times_seconds is None:
        raise ValueError(f"case={args.case_id}: no physical time sidecar found")

    node = _centroid_skeleton_voxel(skeleton)
    enhancement = voxel_node_series(
        dce_4d,
        [node],
        baseline_frame_count=baseline_frame_count,
        relative_enhancement=relative_enhancement,
    )[0]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "uchicago_curve.json"
    out_path.write_text(
        json.dumps(
            {
                "case_id": args.case_id,
                "git_commit": _git_commit(),
                "n_frames": len(timepoints),
                "baseline_frame_count": baseline_frame_count,
                "relative_enhancement": relative_enhancement,
                "node_selection": "skeleton voxel nearest the centroid of all skeleton voxels",
                "times_seconds": [round(float(t), 3) for t in times_seconds],
                "enhancement": [round(float(v), 5) for v in enhancement],
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
