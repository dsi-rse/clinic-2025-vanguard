#!/usr/bin/env bash
#SBATCH --job-name=finalize-uchicago-dce-dicom
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/finalize-uchicago-dce-dicom-%j.out
#SBATCH --error=logs/slurm/finalize-uchicago-dce-dicom-%j.err

set -euo pipefail
umask 0007

: "${REPO_ROOT:?Set REPO_ROOT to the Vanguard checkout}"
: "${CASE_MANIFEST:?Set CASE_MANIFEST to the uchicago_dce case manifest}"
: "${SELECTION:?Set SELECTION to the prepared DICOM location selection}"
: "${DESTINATION_ROOT:?Set DESTINATION_ROOT to the staging directory}"
: "${INVENTORY_PATH:?Set INVENTORY_PATH to the new inventory parquet to write}"

VANGUARD_PYTHON="${VANGUARD_PYTHON:-${CONDA_PREFIX:?activate the vanguard env or set VANGUARD_PYTHON}/bin/python}"

cd "${REPO_ROOT}"
echo "[job] id=${SLURM_JOB_ID} host=$(hostname)"
echo "[python] ${VANGUARD_PYTHON}"
"${VANGUARD_PYTHON}" preprocessing/stage_uchicago_dce_dicom.py finalize \
  --case-manifest "${CASE_MANIFEST}" \
  --selection "${SELECTION}" \
  --destination "${DESTINATION_ROOT}" \
  --inventory-path "${INVENTORY_PATH}"
