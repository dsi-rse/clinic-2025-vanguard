"""Tests for the canonical-manifest DICOM stager and the per-arm skeleton tree builder."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from analysis.build_skeleton_arm_tree import build_tree
from preprocessing.stage_uchicago_dce_dicom import (
    _matching_instances,
    stage_exam,
)

STUDY = "1.2.3.4"
SERIES_HR = "1.2.3.4.1"
SERIES_UF = "1.2.3.4.2"
MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"
PHILIPS_SERIES_DATA = "1.3.46.670589.11.0.0.12.2"
N_PHASES = 3


def _instance(
    series_uid: str,
    *,
    sop_class: str = MR_IMAGE_STORAGE,
    temporal: int | None = 1,
    instance_number: int = 1,
) -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = STUDY
    ds.SeriesInstanceUID = series_uid
    ds.InstanceNumber = instance_number
    if temporal is not None:
        ds.TemporalPositionIdentifier = temporal
    ds.Rows = 2
    ds.Columns = 2
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.zeros((2, 2), dtype=np.uint16).tobytes()
    stream = BytesIO()
    pydicom.dcmwrite(stream, ds, write_like_original=False)
    return stream.getvalue()


def _write_container(root: Path) -> Path:
    """A directory holding two series plus a non-image and an unrelated series."""
    container = root / "DICOM"
    container.mkdir()
    for phase in range(1, N_PHASES + 1):
        (container / f"hr_{phase}.dcm").write_bytes(
            _instance(SERIES_HR, temporal=phase, instance_number=phase)
        )
        (container / f"uf_{phase}.dcm").write_bytes(
            _instance(SERIES_UF, temporal=phase, instance_number=phase)
        )
    (container / "hr_private.dcm").write_bytes(
        _instance(SERIES_HR, sop_class=PHILIPS_SERIES_DATA, temporal=None)
    )
    (container / "other.dcm").write_bytes(_instance("9.9.9", temporal=1))
    (container / "notes.txt").write_text("not dicom")
    return container


def test_matching_instances_filters_series_and_excludes_non_image(
    tmp_path: Path,
) -> None:
    """Only the wanted series' image instances are kept; the private object is logged."""
    container = _write_container(tmp_path)
    kept, excluded, scanned = _matching_instances(
        "directory", str(container), STUDY, SERIES_HR
    )
    assert scanned == 2 * N_PHASES + 3
    assert sorted(k["temporal_position_identifier"] for k in kept) == [1, 2, 3]
    assert [e["reason"] for e in excluded] == [
        "non_image:philips_private_mr_series_data_storage"
    ]


def test_matching_instances_fails_loud_without_temporal_position(
    tmp_path: Path,
) -> None:
    """An image instance with no TemporalPositionIdentifier is never guessed."""
    container = tmp_path / "DICOM"
    container.mkdir()
    (container / "a.dcm").write_bytes(_instance(SERIES_HR, temporal=None))
    with pytest.raises(ValueError, match="no TemporalPositionIdentifier"):
        _matching_instances("directory", str(container), STUDY, SERIES_HR)


def test_stage_exam_writes_archive_inventory_and_provenance(tmp_path: Path) -> None:
    """The staged package matches the v5 contract and records the location used."""
    container = _write_container(tmp_path)
    case_manifest = tmp_path / "case_manifest.csv"
    pd.DataFrame(
        [
            {
                "exam_id": "exam_a",
                "dataset": "ds",
                "study_instance_uid": STUDY,
                "hr_series_instance_uid": SERIES_HR,
                "ufast_series_instance_uid": SERIES_UF,
                "ufast_baseline_frame_count": 1,
            }
        ]
    ).to_csv(case_manifest, index=False)
    selection = tmp_path / "selection.csv"
    rows = []
    for role, uid in (("hr", SERIES_HR), ("ufast", SERIES_UF)):
        rows.append(
            {
                "exam_id": "exam_a",
                "dataset": "ds",
                "series_role": role,
                "study_instance_uid": STUDY,
                "series_instance_uid": uid,
                "location_rank": 0,
                "location_kind": "directory",
                "location_path": str(container),
            }
        )
    pd.DataFrame(rows).to_csv(selection, index=False)
    destination = tmp_path / "staged"
    stage_exam(case_manifest, selection, destination, 0)

    inventory = pd.read_parquet(
        destination / "inventory_shards" / "ds" / "exam_a.parquet"
    )
    assert len(inventory) == 2 * N_PHASES
    assert set(inventory["series_role"]) == {"hr", "ufast"}
    assert inventory["read_ok"].all()
    with zipfile.ZipFile(destination / "archives" / "ds" / "exam_a.zip") as archive:
        assert sorted(archive.namelist()) == sorted(inventory["archive_member"])
    provenance = json.loads(
        (destination / "provenance_shards" / "ds" / "exam_a.json").read_text()
    )
    hr = provenance["series"][SERIES_HR]
    assert hr["n_temporal_positions"] == N_PHASES
    assert hr["location_used"]["kind"] == "directory"
    assert len(hr["excluded_non_image_instances"]) == 1


def _skeleton_dir(root: Path, dataset: str, exam: str, *, with_hr_only: bool) -> None:
    exam_dir = root / dataset / exam
    exam_dir.mkdir(parents=True)
    np.save(
        exam_dir / f"{exam}_skeleton_4d_exam_mask.npy", np.ones((2, 2, 2), np.uint8)
    )
    np.save(
        exam_dir / f"{exam}_skeleton_4d_exam_support_mask.npy",
        np.ones((2, 2, 2), np.uint8),
    )
    if with_hr_only:
        np.save(
            exam_dir / f"{exam}_skeleton_4d_exam_mask_hr_only.npy",
            np.zeros((2, 2, 2), np.uint8),
        )
    (exam_dir / "run_summary.json").write_text("{}")


def test_build_tree_links_requested_variant(tmp_path: Path) -> None:
    """The canonical name in the tree resolves to the arm's variant file."""
    root = tmp_path / "centerlines"
    _skeleton_dir(root, "ds", "e1", with_hr_only=True)
    out = tmp_path / "hr_only"
    manifest = build_tree(root, "hr_only", out)
    link = out / "ds" / "e1" / "e1_skeleton_4d_exam_mask.npy"
    assert link.is_symlink()
    assert link.resolve().name == "e1_skeleton_4d_exam_mask_hr_only.npy"
    assert int(np.load(link).sum()) == 0
    assert (out / "ds" / "e1" / "run_summary.json").is_symlink()
    assert manifest.loc[0, "skeleton_voxels"] == 0
    assert not manifest.loc[0, "support_is_arm_specific"]


def test_build_tree_refuses_mixed_arms(tmp_path: Path) -> None:
    """An exam missing the variant aborts instead of silently falling back to final."""
    root = tmp_path / "centerlines"
    _skeleton_dir(root, "ds", "e1", with_hr_only=True)
    _skeleton_dir(root, "ds", "e2", with_hr_only=False)
    with pytest.raises(FileNotFoundError, match="mixed-arm"):
        build_tree(root, "hr_only", tmp_path / "hr_only")
