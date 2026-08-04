#!/usr/bin/env bash
#SBATCH --job-name=uchicago-curve-export
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Real-data input for the symposium talk's "movie" slide: one UChicago vessel
# node's real baseline-referenced DCE enhancement curve (numbers only -- no
# MRI frame, mask, or other image is written; see analysis/uchicago_curve_export.py).
#
# Loads a raw 4D DCE series (13-24 frames), so it runs on Slurm, not the head node.
#
# Usage:
#   sbatch gnn/slurm/submit_uchicago_curve_export.slurm
#   CASE=<exam_id> DATASET=<dataset_subdir> sbatch gnn/slurm/submit_uchicago_curve_export.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

set +u
source "${HOME}/.bashrc" || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi
set -u

export PYTHONUNBUFFERED=1

COHORT_ROOT="/gpfs/data/karczmar-lab/vanguard/uchicago_ultrafast_pretreatment_cohort_v1"
DATASET="${DATASET:-uchicago_zhen}"
CASE="${CASE:-uchicago_zhen_8b8f05c8b8d1eea9751542d0}"

python -m analysis.uchicago_curve_export \
  --case-id "${CASE}" \
  --dataset "${DATASET}" \
  --centerline-root "${COHORT_ROOT}/centerlines" \
  --dce-root "${COHORT_ROOT}/images/${DATASET}" \
  --out-dir presentations/symposium_talk/figures/uchicago_curve_data
