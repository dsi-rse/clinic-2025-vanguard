#!/bin/bash
# Per-series DICOM acquisition metadata for the uchicago_dce_g16 cohort, read from
# the staged archives on scratch through the pinned inventory.
#SBATCH --job-name=g16-dicom-meta
#SBATCH --output=logs/slurm/g16-dicom-meta-%j.out
#SBATCH --error=logs/slurm/g16-dicom-meta-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=tier1q
set -euo pipefail
PY="${VANGUARD_PYTHON:-/home/t-9sbose/micromamba/envs/vanguard/bin/python3}"
REPO="${REPO_ROOT:-/ess/home/home1/t-9sbose/vanguard}"
G=/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16
SC=/ess/scratch/scratch1/t-9sbose/uchicago_dce_g16
cd "${REPO}"
echo "host=$(hostname) commit=$(git rev-parse HEAD)"
"${PY}" scripts/extract_dicom_acquisition_metadata.py \
  --case-manifest "${G}/staging/case_manifest.csv" \
  --inventory "${SC}/staging/staged_dicom/dicom_file_manifest.parquet" \
  --out "${G}/metadata/dicom_acquisition_metadata.csv"
