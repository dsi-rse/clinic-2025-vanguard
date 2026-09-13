"""Tests for the canonical-manifest case builder."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.build_uchicago_dce_case_manifest import (
    assign_folds,
    build_cases,
    choose_acquisition,
    dataset_name,
    select_exams,
)

EXPECTED_BASELINE = 3


def _acq(
    order: int,
    role: str,
    frames: int,
    *,
    uids: list[str] | None = None,
    status: str = "complete",
    baseline: int | None = 5,
) -> dict[str, object]:
    uid = uids or [f"{role}-uid-{order}"] * frames
    return {
        "acquisition_order_index": order,
        "acquisition_role": role,
        "native_frame_count": frames,
        "phase_series_instance_uids": uid,
        "preprocessing_status": status,
        "baseline_frame_count": baseline,
        "baseline_status": "available" if baseline is not None else "unavailable",
    }


def _exam(
    key: str,
    acquisitions: list[dict[str, object]],
    *,
    datasets: list[str] | None = None,
    pcr: float = 1.0,
    treatment_time: str = "pretreatment",
    patient: str = "p",
) -> dict[str, object]:
    uids = {
        a["phase_series_instance_uids"][0]
        for a in acquisitions
        if len(set(a["phase_series_instance_uids"])) == 1
    }
    locations = {
        "series": [
            {
                "series_instance_uid": uid,
                "status": "resolved",
                "locations": [
                    {
                        "kind": "directory",
                        "path": f"/src/{uid}",
                        "study_instance_uid": f"study-{key}",
                    }
                ],
            }
            for uid in sorted(uids)
        ]
    }
    return {
        "exam_key": key,
        "patient_group_key": patient,
        "study_instance_uid": f"study-{key}",
        "dataset_labels_json": json.dumps(datasets or ["retro"]),
        "source_names_json": json.dumps(["src"]),
        "treatment_time": treatment_time,
        "pcr": pcr,
        "pcr_label_is_provisional": False,
        "pcr_label_authority": "auth",
        "ultrafast_series_count": sum(
            a["acquisition_role"] == "ultrafast" for a in acquisitions
        ),
        "high_resolution_series_count": sum(
            a["acquisition_role"] == "high_resolution" for a in acquisitions
        ),
        "preprocessing_status": "complete",
        "preprocessing_json": json.dumps(acquisitions),
        "original_dicom_locations_json": json.dumps(locations),
    }


def test_select_exams_applies_annas_rule() -> None:
    """Only pretreatment, labelled exams with both UF and HR survive selection."""
    hr, uf = _acq(0, "high_resolution", 5), _acq(1, "ultrafast", 24)
    manifest = pd.DataFrame(
        [
            _exam("keep", [hr, uf]),
            _exam("on_treatment", [hr, uf], treatment_time="on_treatment"),
            _exam("no_label", [hr, uf], pcr=float("nan")),
            _exam("no_uf", [hr]),
        ]
    )
    assert select_exams(manifest)["exam_key"].tolist() == ["keep"]


def test_dataset_name_joins_labels_into_one_path_component() -> None:
    """Multi-label exams get a joined name; path separators are rejected."""
    assert dataset_name('["retro_caps_pcr_imaging","zhen_retro_nact"]') == (
        "retro_caps_pcr_imaging+zhen_retro_nact"
    )
    with pytest.raises(ValueError, match="safe path"):
        dataset_name('["a/b"]')


def test_choose_acquisition_prefers_most_frames_then_lowest_order() -> None:
    """The documented choice rule and its recorded reasons."""
    candidates = [
        {"acquisition_order_index": 2, "native_frame_count": 24, "eligible": True},
        {"acquisition_order_index": 1, "native_frame_count": 6, "eligible": True},
    ]
    winner, reason = choose_acquisition(candidates)
    assert winner is candidates[0]
    assert reason == "most_native_frames"

    tied = [
        {"acquisition_order_index": 3, "native_frame_count": 24, "eligible": True},
        {"acquisition_order_index": 1, "native_frame_count": 24, "eligible": True},
    ]
    winner, reason = choose_acquisition(tied)
    assert winner["acquisition_order_index"] == 1
    assert reason == "tie_on_native_frames_lowest_acquisition_order"
    assert choose_acquisition([]) == (None, "no_eligible_acquisition")


def test_build_cases_excludes_multi_uid_and_missing_baseline_without_guessing() -> None:
    """Exams the rule cannot resolve are excluded with a reason, never patched."""
    good = _exam(
        "good",
        [
            _acq(0, "high_resolution", 5),
            _acq(1, "ultrafast", 13, baseline=3),
            _acq(2, "ultrafast", 24, baseline=None),
        ],
    )
    multi_uid_hr = _exam(
        "multi",
        [
            _acq(0, "high_resolution", 3, uids=["a", "b", "c"]),
            _acq(1, "ultrafast", 24),
        ],
    )
    no_baseline = _exam(
        "nobase",
        [_acq(0, "high_resolution", 5), _acq(1, "ultrafast", 24, baseline=None)],
    )
    cases, excluded, provenance = build_cases(
        pd.DataFrame([good, multi_uid_hr, no_baseline])
    )

    assert cases["exam_id"].tolist() == ["good"]
    row = cases.iloc[0]
    assert row["ufast_series_instance_uid"] == "ultrafast-uid-1"
    assert row["ufast_baseline_frame_count"] == EXPECTED_BASELINE
    assert provenance["good"]["roles"]["ultrafast"]["choice_reason"] == (
        "only_eligible_acquisition"
    )
    assert set(excluded["exam_id"]) == {"multi", "nobase"}
    assert (
        "phases_span_3_series_uids"
        in excluded.set_index("exam_id").loc["multi", "detail"]
    )
    assert (
        "baseline_status=unavailable"
        in excluded.set_index("exam_id").loc["nobase", "detail"]
    )


def test_assign_folds_keeps_patients_together() -> None:
    """Every fold id is used and no patient is split across folds."""
    rows = []
    for i in range(40):
        patient = f"p{i // 2}"
        rows.append(
            {
                "case_id": f"e{i}",
                "dataset": "retro" if i % 3 else "sim",
                "patient_group_key": patient,
                "pcr": i % 2,
            }
        )
    labels = pd.DataFrame(rows)
    folds = assign_folds(labels)
    assert set(folds.unique()) == {0, 1, 2, 3, 4}
    per_patient = pd.DataFrame(
        {"p": labels["patient_group_key"], "f": folds}
    ).drop_duplicates()
    assert not per_patient["p"].duplicated().any()
