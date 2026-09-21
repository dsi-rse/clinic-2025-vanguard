r"""Build a per-arm centerline tree from one pipeline output root.

``preprocessing.pipeline`` writes three skeleton variants per exam under
``<output_root>/centerlines/<dataset>/<exam>/``:

* ``<exam>_skeleton_4d_exam_mask.npy``                      final (HR + UFAST merge + complement)
* ``<exam>_skeleton_4d_exam_mask_hr_only.npy``              HR-mapped TC4D only
* ``<exam>_skeleton_4d_exam_mask_hr_ufast_merge_only.npy``  HR + UFAST merge, no complement

``gnn.data_loader`` resolves one fixed filename (``config.py``
``centerline_file_pattern``), so comparing arms means giving each arm its own
tree in which that filename points at the wanted variant. This script builds
such a tree of symlinks. Every exam must carry the requested variant; an exam
that has a final skeleton but lacks the variant aborts the build, so a tree
can never silently mix arms.

Each exam directory in the tree gets: the arm's skeleton under the canonical
name, the matching support mask where the pipeline wrote one (final support
otherwise, recorded as such), ``run_summary.json``, and the provenance JSONs.
The tree root gets ``arm_tree_manifest.csv`` and ``arm_tree_provenance.json``.

Usage::

    python analysis/build_skeleton_arm_tree.py \\
        --centerline-root <cohort>/preprocessing_out/centerlines \\
        --arm hr_only --out-tree <cohort>/arm_trees/hr_only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARM_SUFFIX = {
    "final": "",
    "hr_only": "_hr_only",
    "hr_ufast_merge_only": "_hr_ufast_merge_only",
}
CANONICAL = "{exam}_skeleton_4d_exam_mask.npy"
CANONICAL_SUPPORT = "{exam}_skeleton_4d_exam_support_mask.npy"
SIDE_FILES = (
    "run_summary.json",
    "merge_provenance.json",
    "complement_provenance.json",
    "{exam}_morphometry.json",
)


def build_tree(centerline_root: Path, arm: str, out_tree: Path) -> pd.DataFrame:
    """Symlink one arm's variant under the canonical name for every exam."""
    suffix = ARM_SUFFIX[arm]
    rows = []
    for final_path in sorted(centerline_root.glob("*/*/*_skeleton_4d_exam_mask.npy")):
        exam_dir = final_path.parent
        exam = final_path.name[: -len("_skeleton_4d_exam_mask.npy")]
        dataset = exam_dir.parent.name
        variant = exam_dir / f"{exam}_skeleton_4d_exam_mask{suffix}.npy"
        if not variant.is_file():
            raise FileNotFoundError(
                f"{exam}: has a final skeleton but no {arm} variant ({variant.name}); "
                "refusing to build a mixed-arm tree"
            )
        support_variant = exam_dir / f"{exam}_skeleton_4d_exam_support_mask{suffix}.npy"
        support = (
            support_variant
            if support_variant.is_file()
            else exam_dir / CANONICAL_SUPPORT.format(exam=exam)
        )
        target = out_tree / dataset / exam
        target.mkdir(parents=True, exist_ok=True)
        (target / CANONICAL.format(exam=exam)).symlink_to(variant.resolve())
        if support.is_file():
            (target / CANONICAL_SUPPORT.format(exam=exam)).symlink_to(support.resolve())
        for name in SIDE_FILES:
            side = exam_dir / name.format(exam=exam)
            if side.is_file():
                (target / side.name).symlink_to(side.resolve())
        rows.append(
            {
                "exam_id": exam,
                "dataset": dataset,
                "arm": arm,
                "skeleton_source": str(variant),
                "support_source": str(support) if support.is_file() else None,
                "support_is_arm_specific": support_variant.is_file(),
                "skeleton_voxels": int(np.load(variant, mmap_mode="r").sum()),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no skeletons under {centerline_root}")
    return pd.DataFrame(rows)


def main() -> None:
    """Build the tree and write its manifest and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centerline-root", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARM_SUFFIX), required=True)
    parser.add_argument("--out-tree", type=Path, required=True)
    args = parser.parse_args()
    if args.out_tree.exists():
        raise FileExistsError(
            f"{args.out_tree} exists; build arm trees into a fresh path"
        )

    manifest = build_tree(args.centerline_root, args.arm, args.out_tree)
    manifest.to_csv(args.out_tree / "arm_tree_manifest.csv", index=False)
    (args.out_tree / "arm_tree_provenance.json").write_text(
        json.dumps(
            {
                "centerline_root": str(args.centerline_root.resolve()),
                "arm": args.arm,
                "variant_suffix": ARM_SUFFIX[args.arm],
                "n_exams": int(len(manifest)),
                "n_arm_specific_support": int(
                    manifest["support_is_arm_specific"].sum()
                ),
                "median_skeleton_voxels": float(manifest["skeleton_voxels"].median()),
            },
            indent=2,
        )
    )
    print(
        f"{args.arm}: {len(manifest)} exams -> {args.out_tree} "
        f"(median {manifest['skeleton_voxels'].median():.0f} voxels)"
    )


if __name__ == "__main__":
    main()
