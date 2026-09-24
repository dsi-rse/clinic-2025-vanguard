"""Stage raw DICOM for the canonical UChicago DCE cohort into the ZIP + inventory format.

The target contract is the per-exam ZIP archive + ``read_ok`` inventory that
``preprocessing/dicom.py::_read_inventory`` requires, identical to what
``stage_v5_raw_dicom.py`` writes, so ``preprocessing/run_cohort.py`` consumes
the result unchanged.

What differs from the v5 stager is where instances come from. Anna's canonical
manifest (``current_manifest.csv``) lists, per series, one or more *containers*
in ``original_dicom_locations_json`` -- a ``directory`` or a ``zip`` -- and a
container holds several series. There is no delivered per-file manifest, so
this script discovers instances itself: it reads every candidate file's header
(``stop_before_pixels=True``) and keeps the ones whose StudyInstanceUID and
SeriesInstanceUID match the case manifest. ``TemporalPositionIdentifier`` and
``InstanceNumber`` are taken from the header; an image instance without a
temporal position fails the exam (listed in ``stage_failures``) rather than
being guessed. Known non-image objects (Raw Data Storage and Philips private
series-data/spectrum/examcard objects) are excluded and logged.

Commands:
    prepare-selection  Write ``staged_dicom_selection.csv``: one row per
                        (exam, role, location) from the manifest, so array
                        tasks never load the ~100 MB manifest.
    stage --index N    Stage one exam (by sorted case-manifest position).
    finalize            Merge all per-exam shards into one
                        ``dicom_file_manifest.parquet`` and write checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

# Non-image objects that Philips scanners bundle into a dynamic series' own
# SeriesInstanceUID. None carries PixelData or a temporal position, so they are
# not phases; they are excluded and logged rather than forced into one.
NON_IMAGE_SOP_CLASS_UIDS = {
    "1.2.840.10008.5.1.4.1.1.66": "raw_data_storage",
    "1.3.46.670589.11.0.0.12.1": "philips_private_mr_spectrum_storage",
    "1.3.46.670589.11.0.0.12.2": "philips_private_mr_series_data_storage",
    "1.3.46.670589.11.0.0.12.4": "philips_private_mr_examcard_storage",
}
ROLE_BY_MANIFEST_ROLE = {"high_resolution": "hr", "ultrafast": "ufast"}

SELECTION_COLUMNS = [
    "exam_id",
    "dataset",
    "series_role",
    "study_instance_uid",
    "series_instance_uid",
    "location_rank",
    "location_kind",
    "location_path",
]
SHARED_INVENTORY_COLUMNS = [
    "exam_id",
    "dataset",
    "series_role",
    "study_instance_uid",
    "series_instance_uid",
    "archive_path",
    "archive_member",
    "read_ok",
    "temporal_position_identifier",
    "instance_number",
    "sop_instance_uid",
    "file_size_bytes",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_cases(case_manifest_path: Path) -> pd.DataFrame:
    cases = pd.read_csv(case_manifest_path, dtype=str)
    return cases.sort_values("exam_id").reset_index(drop=True)


def prepare_selection(
    case_manifest_path: Path, manifest_path: Path, selection_path: Path
) -> None:
    """Extract each case's HR/UFAST DICOM locations from the canonical manifest."""
    cases = _load_cases(case_manifest_path)
    manifest = pd.read_csv(
        manifest_path,
        usecols=["exam_key", "study_instance_uid", "original_dicom_locations_json"],
        dtype=str,
    ).set_index("exam_key")
    rows: list[dict[str, object]] = []
    for case in cases.itertuples(index=False):
        if case.exam_id not in manifest.index:
            raise KeyError(f"{case.exam_id} is not in {manifest_path}")
        exam = manifest.loc[case.exam_id]
        if exam["study_instance_uid"] != case.study_instance_uid:
            raise ValueError(f"{case.exam_id}: study UID differs from the manifest")
        wanted = {
            case.hr_series_instance_uid: "hr",
            case.ufast_series_instance_uid: "ufast",
        }
        payload = json.loads(exam["original_dicom_locations_json"])
        found: set[str] = set()
        for series in payload["series"]:
            uid = series["series_instance_uid"]
            if uid not in wanted:
                continue
            if series["status"] != "resolved" or not series["locations"]:
                raise ValueError(f"{case.exam_id}: series {uid} is not resolved")
            if ROLE_BY_MANIFEST_ROLE.get(series["role"]) != wanted[uid]:
                raise ValueError(
                    f"{case.exam_id}: series {uid} role {series['role']} "
                    f"does not match case manifest role {wanted[uid]}"
                )
            found.add(uid)
            for rank, location in enumerate(series["locations"]):
                if location["study_instance_uid"] != case.study_instance_uid:
                    raise ValueError(f"{case.exam_id}: location study UID mismatch")
                rows.append(
                    {
                        "exam_id": case.exam_id,
                        "dataset": case.dataset,
                        "series_role": wanted[uid],
                        "study_instance_uid": case.study_instance_uid,
                        "series_instance_uid": uid,
                        "location_rank": rank,
                        "location_kind": location["kind"],
                        "location_path": location["path"],
                    }
                )
        if found != set(wanted):
            raise ValueError(
                f"{case.exam_id}: manifest lacks locations for "
                f"{sorted(set(wanted) - found)}"
            )
    selection = pd.DataFrame(rows, columns=SELECTION_COLUMNS)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    temp = selection_path.with_suffix(".csv.tmp")
    selection.to_csv(temp, index=False)
    temp.chmod(0o640)
    temp.replace(selection_path)
    print(
        f"[complete] {len(selection)} location rows for "
        f"{selection['exam_id'].nunique()} exams -> {selection_path}",
        flush=True,
    )


def location_variants(kind: str, path: str) -> list[tuple[str, str]]:
    """The listed location plus its zipped/unzipped alternates, existing ones only.

    Anna compresses source directories in place over time, so a location
    listed as a directory when a job is submitted may be a ``dicoms.zip``
    (or ``<dir>.zip``) by the time it runs, and vice versa. Every candidate
    that exists is returned, listed location first.
    """
    base = Path(path)
    candidates = [(kind, base)]
    if kind == "directory":
        candidates += [("zip", base / "dicoms.zip"), ("zip", base.with_suffix(".zip"))]
    elif kind == "zip":
        candidates += [
            (
                "directory",
                base.parent if base.name == "dicoms.zip" else base.with_suffix(""),
            )
        ]
    return [
        (k, str(c))
        for k, c in candidates
        if (c.is_dir() if k == "directory" else c.is_file())
    ]


def _live_locations(
    manifest_path: Path, exam_id: str, series_uid: str
) -> list[dict[str, str]]:
    """Locations for one series from the live manifest, read for one exam only."""
    manifest = pd.read_csv(
        manifest_path,
        usecols=["exam_key", "original_dicom_locations_json"],
        dtype=str,
    )
    row = manifest.loc[manifest["exam_key"] == exam_id]
    if len(row) != 1:
        raise KeyError(
            f"{exam_id}: expected one row in {manifest_path}, got {len(row)}"
        )
    payload = json.loads(row.iloc[0]["original_dicom_locations_json"])
    for series in payload["series"]:
        if series["series_instance_uid"] == series_uid:
            return list(series["locations"])
    raise KeyError(f"{exam_id}: series {series_uid} not in live manifest")


def _iter_container(kind: str, path: str) -> Iterator[tuple[str, bytes]]:
    """Yield (member name, bytes) for every regular file in a directory or zip."""
    if kind == "directory":
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"directory location missing: {root}")
        for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
            yield str(file_path.relative_to(root)), file_path.read_bytes()
    elif kind == "zip":
        archive_path = Path(path)
        if not archive_path.is_file():
            raise FileNotFoundError(f"zip location missing: {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            for info in sorted(archive.infolist(), key=lambda i: i.filename):
                if info.is_dir():
                    continue
                yield info.filename, archive.read(info)
    else:
        raise ValueError(f"unsupported location kind {kind!r}")


def _matching_instances(
    kind: str, path: str, study_uid: str, series_uid: str
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    """Header-scan one container; return (kept, excluded_non_image, n_scanned)."""
    import pydicom
    from pydicom.errors import InvalidDicomError

    kept: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    scanned = 0
    for member, payload in _iter_container(kind, path):
        scanned += 1
        try:
            header = pydicom.dcmread(
                io.BytesIO(payload), stop_before_pixels=True, force=True
            )
        except InvalidDicomError:
            continue
        if (
            str(getattr(header, "SeriesInstanceUID", "")) != series_uid
            or str(getattr(header, "StudyInstanceUID", "")) != study_uid
        ):
            continue
        sop_uid = str(getattr(header, "SOPInstanceUID", ""))
        if not sop_uid:
            raise ValueError(
                f"{member} in {path}: matching series but no SOPInstanceUID"
            )
        sop_class = str(getattr(header, "SOPClassUID", ""))
        if sop_class in NON_IMAGE_SOP_CLASS_UIDS:
            excluded.append(
                {
                    "sop_instance_uid": sop_uid,
                    "sop_class_uid": sop_class,
                    "member": member,
                    "reason": f"non_image:{NON_IMAGE_SOP_CLASS_UIDS[sop_class]}",
                }
            )
            continue
        temporal = getattr(header, "TemporalPositionIdentifier", None)
        if temporal in (None, ""):
            raise ValueError(
                f"{member} in {path}: image instance {sop_uid} has no "
                f"TemporalPositionIdentifier (SOPClassUID="
                f"{getattr(header, 'SOPClassUID', 'MISSING')}) -- needs review"
            )
        instance_number = getattr(header, "InstanceNumber", None)
        kept.append(
            {
                "sop_instance_uid": sop_uid,
                "temporal_position_identifier": int(temporal),
                "instance_number": int(instance_number)
                if instance_number not in (None, "")
                else -1,
                "member": member,
                "payload": payload,
            }
        )
    return kept, excluded, scanned


def stage_exam(
    case_manifest_path: Path,
    selection_path: Path,
    destination: Path,
    index: int,
    live_manifest: Path | None = None,
) -> None:
    """Stage one exam selected by its stable sorted array index.

    Locations come from the prepared selection; if none of them (or their
    zipped/unzipped variants) exists any more and ``live_manifest`` is given,
    the exam's current locations are re-read from it and tried as well.
    """
    cases = _load_cases(case_manifest_path)
    if index < 0 or index >= len(cases):
        raise IndexError(f"array index {index} outside [0, {len(cases) - 1}]")
    case = cases.iloc[index]
    exam_id = case["exam_id"]
    dataset = case["dataset"]

    archive = destination / "archives" / dataset / f"{exam_id}.zip"
    shard = destination / "inventory_shards" / dataset / f"{exam_id}.parquet"
    provenance = destination / "provenance_shards" / dataset / f"{exam_id}.json"
    for parent in (archive.parent, shard.parent, provenance.parent):
        parent.mkdir(parents=True, exist_ok=True)
    existing = [path.exists() for path in (archive, shard, provenance)]
    if any(existing) and not all(existing):
        raise RuntimeError(
            f"{exam_id}: refusing to overwrite an incomplete staged DICOM "
            "package; remove the partial outputs before retrying"
        )
    if all(existing):
        print(f"[skip] complete outputs already exist for {exam_id}", flush=True)
        return

    selection = pd.read_csv(selection_path, dtype=str)
    selection = selection.loc[selection["exam_id"] == exam_id]
    expected = {
        case["hr_series_instance_uid"]: "hr",
        case["ufast_series_instance_uid"]: "ufast",
    }
    if set(selection["series_instance_uid"]) != set(expected):
        raise ValueError(f"{exam_id}: selection rows do not match the case manifest")

    temp_archive = archive.with_suffix(f".zip.tmp.{os.getpid()}")
    temp_shard = shard.with_suffix(f".parquet.tmp.{os.getpid()}")
    temp_provenance = provenance.with_suffix(f".json.tmp.{os.getpid()}")
    for temporary in (temp_archive, temp_shard, temp_provenance):
        temporary.unlink(missing_ok=True)

    output_rows: list[dict[str, object]] = []
    series_provenance: dict[str, dict[str, object]] = {}
    payload_digest = hashlib.sha256()
    with zipfile.ZipFile(
        temp_archive, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as output_zip:
        for series_uid, role in sorted(expected.items(), key=lambda kv: kv[1]):
            locations = selection.loc[
                selection["series_instance_uid"] == series_uid
            ].sort_values("location_rank")
            kept: list[dict[str, object]] = []
            excluded: list[dict[str, object]] = []
            used: dict[str, object] | None = None
            skipped: list[dict[str, object]] = []
            listed = [
                (
                    loc.location_kind,
                    loc.location_path,
                    "selection",
                    int(loc.location_rank),
                )
                for loc in locations.itertuples(index=False)
            ]
            if live_manifest is not None:
                listed += [
                    (loc["kind"], loc["path"], "live_manifest", rank)
                    for rank, loc in enumerate(
                        _live_locations(live_manifest, exam_id, series_uid)
                    )
                ]
            tried: set[str] = set()
            for kind, path, source, rank in listed:
                variants = location_variants(kind, path)
                if not variants:
                    skipped.append(
                        {
                            "kind": kind,
                            "path": path,
                            "source": source,
                            "reason": "missing",
                        }
                    )
                for variant_kind, variant_path in variants:
                    if variant_path in tried:
                        continue
                    tried.add(variant_path)
                    kept, excluded, scanned = _matching_instances(
                        variant_kind,
                        variant_path,
                        case["study_instance_uid"],
                        series_uid,
                    )
                    if kept:
                        used = {
                            "kind": variant_kind,
                            "path": variant_path,
                            "listed_kind": kind,
                            "listed_path": path,
                            "source": source,
                            "rank": rank,
                            "files_scanned": scanned,
                        }
                        break
                    skipped.append(
                        {
                            "kind": variant_kind,
                            "path": variant_path,
                            "source": source,
                            "files_scanned": scanned,
                            "reason": "no_matching_image_instances",
                        }
                    )
                if used is not None:
                    break
            if used is None:
                raise ValueError(
                    f"{exam_id}: no location yielded instances for {role} "
                    f"series {series_uid}: {skipped}"
                )
            sop_uids = [k["sop_instance_uid"] for k in kept]
            if len(set(sop_uids)) != len(sop_uids):
                raise ValueError(
                    f"{exam_id}: duplicate SOPInstanceUIDs inside {used['path']} "
                    f"for series {series_uid}"
                )
            kept.sort(
                key=lambda k: (
                    k["temporal_position_identifier"],
                    k["instance_number"],
                    k["sop_instance_uid"],
                )
            )
            for counter, instance in enumerate(kept):
                member = f"series/{series_uid}/{counter:06d}.dcm"
                payload = instance["payload"]
                output_zip.writestr(member, payload)
                payload_digest.update(series_uid.encode())
                payload_digest.update(member.encode())
                payload_digest.update(payload)
                output_rows.append(
                    {
                        "exam_id": exam_id,
                        "dataset": dataset,
                        "series_role": role,
                        "study_instance_uid": case["study_instance_uid"],
                        "series_instance_uid": series_uid,
                        "archive_path": str(archive.resolve()),
                        "archive_member": member,
                        "read_ok": True,
                        "temporal_position_identifier": instance[
                            "temporal_position_identifier"
                        ],
                        "instance_number": instance["instance_number"],
                        "sop_instance_uid": instance["sop_instance_uid"],
                        "file_size_bytes": len(payload),
                    }
                )
            temporal_positions = sorted(
                {k["temporal_position_identifier"] for k in kept}
            )
            series_provenance[series_uid] = {
                "series_role": role,
                "location_used": used,
                "locations_skipped": skipped,
                "n_files": len(kept),
                "n_temporal_positions": len(temporal_positions),
                "temporal_position_range": [
                    temporal_positions[0],
                    temporal_positions[-1],
                ],
                "excluded_non_image_instances": excluded,
            }
            print(
                f"[series] {exam_id} {role}: {len(kept)} files, "
                f"{len(temporal_positions)} temporal positions, "
                f"{len(excluded)} non-image excluded, from {used['kind']} "
                f"({used['files_scanned']} files scanned)",
                flush=True,
            )

    output = pd.DataFrame(output_rows, columns=SHARED_INVENTORY_COLUMNS)
    output.to_parquet(temp_shard, index=False)
    metadata = {
        "exam_id": exam_id,
        "dataset": dataset,
        "study_instance_uid": case["study_instance_uid"],
        "series_instance_uids": sorted(expected),
        "n_files": len(output),
        "payload_sha256": payload_digest.hexdigest(),
        "archive_sha256": _sha256(temp_archive),
        "archive_bytes": temp_archive.stat().st_size,
        "source": "uchicago_dce_cache/current_manifest.csv original_dicom_locations_json",
        "series": series_provenance,
    }
    temp_provenance.write_text(json.dumps(metadata, indent=2) + "\n")
    for temporary in (temp_archive, temp_shard, temp_provenance):
        temporary.chmod(0o640)
    temp_archive.replace(archive)
    temp_shard.replace(shard)
    temp_provenance.replace(provenance)
    print(
        f"[complete] {exam_id}: {len(output)} files, "
        f"{archive.stat().st_size / 1024**3:.2f} GiB",
        flush=True,
    )


def finalize(
    case_manifest_path: Path, destination: Path, inventory_path: Path | None
) -> None:
    """Merge completed per-exam shards into one PAIRED_INVENTORY parquet.

    ``inventory_path`` defaults to ``<destination>/dicom_file_manifest.parquet``.
    Validation tiers pass their own path so every case manifest gets its own
    inventory file: ``run_cohort.py`` pins the inventory path and hash in each
    exam's provenance, so a shared inventory must never be rewritten in place.
    """
    cases = _load_cases(case_manifest_path)
    shard_paths = [
        destination / "inventory_shards" / row.dataset / f"{row.exam_id}.parquet"
        for row in cases.itertuples(index=False)
    ]
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} inventory shards: {missing[:5]}"
        )
    inventory = pd.concat(
        [pd.read_parquet(path) for path in shard_paths], ignore_index=True
    )
    if inventory["exam_id"].nunique() != len(cases):
        raise ValueError("merged inventory does not cover every case-manifest exam")

    combined_path = inventory_path or destination / "dicom_file_manifest.parquet"
    if combined_path.exists():
        raise FileExistsError(
            f"{combined_path} already exists; write a new inventory path instead "
            "of rewriting one that prepare runs may already be pinned to"
        )
    temp_combined = combined_path.with_suffix(".parquet.tmp")
    inventory.to_parquet(temp_combined, index=False)
    temp_combined.chmod(0o640)
    temp_combined.replace(combined_path)

    checksum_path = combined_path.with_suffix(".SHA256SUMS")
    with checksum_path.open("w") as stream:
        for row in cases.itertuples(index=False):
            metadata = json.loads(
                (
                    destination
                    / "provenance_shards"
                    / row.dataset
                    / f"{row.exam_id}.json"
                ).read_text()
            )
            stream.write(
                f"{metadata['archive_sha256']}  archives/{row.dataset}/{row.exam_id}.zip\n"
            )
    checksum_path.chmod(0o640)
    print(
        f"[complete] merged {len(inventory)} files across "
        f"{inventory['exam_id'].nunique()} exams -> {combined_path}",
        flush=True,
    )


def main() -> None:
    """Parse the sub-command and dispatch to prepare-selection, stage, or finalize."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-selection", "stage", "finalize"))
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "canonical current_manifest.csv; required for prepare-selection, and "
            "for stage it is the live fallback when listed locations have moved"
        ),
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument(
        "--inventory-path",
        type=Path,
        help="finalize output (default <destination>/dicom_file_manifest.parquet)",
    )
    args = parser.parse_args()

    if args.command == "prepare-selection":
        if args.manifest is None:
            parser.error("--manifest is required for prepare-selection")
        prepare_selection(args.case_manifest, args.manifest, args.selection)
    elif args.command == "stage":
        index = args.index
        if index is None:
            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        stage_exam(
            args.case_manifest,
            args.selection,
            args.destination,
            index,
            live_manifest=args.manifest,
        )
    else:
        finalize(args.case_manifest, args.destination, args.inventory_path)


if __name__ == "__main__":
    main()
