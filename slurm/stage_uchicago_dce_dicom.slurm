#!/usr/bin/env bash
#SBATCH --job-name=stage-uchicago-dce-dicom
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/stage-uchicago-dce-dicom-%A_%a.out
#SBATCH --error=logs/slurm/stage-uchicago-dce-dicom-%A_%a.err

set -euo pipefail
umask 0007

: "${REPO_ROOT:?Set REPO_ROOT to the Vanguard checkout}"
: "${CASE_MANIFEST:?Set CASE_MANIFEST to the uchicago_dce case manifest}"
: "${SELECTION:?Set SELECTION to the prepared DICOM location selection (prepare-selection output)}"
: "${DESTINATION_ROOT:?Set DESTINATION_ROOT to the staging directory}"
: "${LIVE_MANIFEST:?Set LIVE_MANIFEST to the canonical current_manifest.csv (fallback for moved sources)}"

VANGUARD_PYTHON="${VANGUARD_PYTHON:-${CONDA_PREFIX:?activate the vanguard env or set VANGUARD_PYTHON}/bin/python}"

cd "${REPO_ROOT}"
echo "[job] id=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
echo "[python] ${VANGUARD_PYTHON}"
"${VANGUARD_PYTHON}" preprocessing/stage_uchicago_dce_dicom.py stage \
  --case-manifest "${CASE_MANIFEST}" \
  --selection "${SELECTION}" \
  --destination "${DESTINATION_ROOT}" \
  --manifest "${LIVE_MANIFEST}" \
  --index "${SLURM_ARRAY_TASK_ID}"
