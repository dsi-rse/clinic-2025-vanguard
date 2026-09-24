r"""Extract per-series DICOM acquisition metadata for a staged cohort.

For every case in a case manifest, read one representative header per staged
series (the HR and the UFAST series named in the manifest) from the staging
archive and record the acquisition fields a batch-effect analysis needs:
vendor, model, field strength, sequence, TR/TE/flip angle, in-plane spacing,
slice thickness, matrix size, temporal positions, contrast agent, study date.
The instance is located through the finalized inventory, so this reads the
same bytes the preprocessing pipeline read. Absent tags are left empty and
counted in the run summary, never filled in.

Output (``--out``): one row per (exam_id, series_role) with the columns in
``HEADER_FIELDS`` plus ``exam_id, dataset, series_role, series_instance_uid,
n_instances, n_temporal_positions`` and a wide per-exam companion
``<out stem>_per_exam.csv`` with ``hr_``/``ufast_`` prefixed columns.

Usage::

    python scripts/extract_dicom_acquisition_metadata.py \\
        --case-manifest <staging>/case_manifest.csv \\
        --inventory <staging>/dicom_file_manifest.parquet \\
        --out <cohort>/metadata/dicom_acquisition_metadata.csv
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from preprocessing.cases import read_case_manifest

# (column name, DICOM keyword). Multi-valued tags are JSON-encoded lists.
HEADER_FIELDS: tuple[tuple[str, str], ...] = (
    ("study_date", "StudyDate"),
    ("series_date", "SeriesDate"),
    ("manufacturer", "Manufacturer"),
    ("manufacturer_model_name", "ManufacturerModelName"),
    ("magnetic_field_strength", "MagneticFieldStrength"),
    ("institution_name", "InstitutionName"),
    ("station_name", "StationName"),
    ("software_versions", "SoftwareVersions"),
    ("series_description", "SeriesDescription"),
    ("protocol_name", "ProtocolName"),
    ("sequence_name", "SequenceName"),
    ("scanning_sequence", "ScanningSequence"),
    ("sequence_variant", "SequenceVariant"),
    ("scan_options", "ScanOptions"),
    ("mr_acquisition_type", "MRAcquisitionType"),
    ("repetition_time_ms", "RepetitionTime"),
    ("echo_time_ms", "EchoTime"),
    ("flip_angle_deg", "FlipAngle"),
    ("inversion_time_ms", "InversionTime"),
    ("pixel_spacing_mm", "PixelSpacing"),
    ("slice_thickness_mm", "SliceThickness"),
    ("spacing_between_slices_mm", "SpacingBetweenSlices"),
    ("rows", "Rows"),
    ("columns", "Columns"),
    ("pixel_bandwidth", "PixelBandwidth"),
    ("number_of_averages", "NumberOfAverages"),
    ("number_of_temporal_positions", "NumberOfTemporalPositions"),
    ("temporal_resolution", "TemporalResolution"),
    ("contrast_bolus_agent", "ContrastBolusAgent"),
    ("contrast_bolus_volume", "ContrastBolusVolume"),
    ("patient_position", "PatientPosition"),
    ("receive_coil_name", "ReceiveCoilName"),
    ("transmit_coil_name", "TransmitCoilName"),
    ("image_type", "ImageType"),
    ("bits_stored", "BitsStored"),
    ("rescale_slope", "RescaleSlope"),
    ("rescale_intercept", "RescaleIntercept"),
)


def _tag_value(header: Any, keyword: str) -> Any:
    value = getattr(header, keyword, None)
    if value is None or value == "":
        return None
    if isinstance(value, list | tuple) or type(value).__name__ == "MultiValue":
        return json.dumps([_scalar(v) for v in value])
    return _scalar(value)


def _scalar(value: Any) -> Any:
    for cast in (int, float):
        if isinstance(value, cast):
            return value
    try:
        return float(value) if "." in str(value) else int(value)
    except (TypeError, ValueError):
        return str(value)


def _representative_member(rows: pd.DataFrame) -> str:
    """First instance of the first temporal position, by instance number."""
    ordered = rows.sort_values(
        ["temporal_position_identifier", "instance_number", "archive_member"]
    )
    return str(ordered.iloc[0]["archive_member"])


def read_series_header(archive_path: Path, member: str) -> Any:
    """Read one instance's header (no pixels) from a staged archive."""
    import pydicom

    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read(member)
    return pydicom.dcmread(io.BytesIO(payload), stop_before_pixels=True, force=True)


def extract(case_manifest: Path, inventory_path: Path) -> pd.DataFrame:
    """One metadata row per (exam, role) for every case in the manifest."""
    inventory = pd.read_parquet(inventory_path)
    rows = []
    for record in read_case_manifest(case_manifest):
        for role, series_uid in (
            ("hr", record.hr_series_instance_uid),
            ("ufast", record.ufast_series_instance_uid),
        ):
            series_rows = inventory[
                (inventory["exam_id"] == record.exam_id)
                & (inventory["series_instance_uid"] == series_uid)
            ]
            if series_rows.empty:
                raise ValueError(f"{record.exam_id}: {role} series not in inventory")
            archives = set(series_rows["archive_path"])
            if len(archives) != 1:
                raise ValueError(
                    f"{record.exam_id}: {role} spans {len(archives)} archives"
                )
            member = _representative_member(series_rows)
            header = read_series_header(Path(archives.pop()), member)
            row: dict[str, Any] = {
                "exam_id": record.exam_id,
                "dataset": record.dataset,
                "series_role": role,
                "series_instance_uid": series_uid,
                "n_instances": int(len(series_rows)),
                "n_temporal_positions": int(
                    series_rows["temporal_position_identifier"].nunique()
                ),
                "header_member": member,
            }
            for column, keyword in HEADER_FIELDS:
                row[column] = _tag_value(header, keyword)
            rows.append(row)
    return pd.DataFrame(rows)


def widen_per_exam(table: pd.DataFrame) -> pd.DataFrame:
    """One row per exam with ``hr_``/``ufast_`` prefixed metadata columns."""
    keys = ["exam_id", "dataset"]
    value_cols = [c for c in table.columns if c not in {*keys, "series_role"}]
    parts = []
    for role in ("hr", "ufast"):
        part = table[table["series_role"] == role].set_index(keys)[value_cols]
        parts.append(part.add_prefix(f"{role}_"))
    return pd.concat(parts, axis=1).reset_index()


def main() -> None:
    """Write the long per-series table, the wide per-exam table, and a missingness tally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    table = extract(args.case_manifest, args.inventory)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    wide = widen_per_exam(table)
    wide_path = args.out.with_name(f"{args.out.stem}_per_exam.csv")
    wide.to_csv(wide_path, index=False)

    missing = (
        table.groupby("series_role")[[c for c, _ in HEADER_FIELDS]]
        .apply(lambda g: g.isna().sum())
        .T
    )
    missing_path = args.out.with_name(f"{args.out.stem}_missing_tags.csv")
    missing.to_csv(missing_path)
    print(f"{len(table)} series rows, {len(wide)} exams -> {args.out}")
    print("tags missing per role (count of series):")
    print(missing[(missing > 0).any(axis=1)].to_string())


if __name__ == "__main__":
    main()
