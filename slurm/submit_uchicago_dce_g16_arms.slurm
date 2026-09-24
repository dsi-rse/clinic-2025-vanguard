#!/bin/bash
# Tabular pCR baseline (logistic / xgboost / elastic net, pooled OOF) for the
# uchicago_dce_g16 cohort, one arm per array task, all arms paired by construction
# (same labels.csv folds, per-arm symlink trees over one pipeline output root).
#
# Array index -> arm: 0 hr_only, 1 hr_ufast_merge_only, 2 final.
# Required env: OUT_ROOT (results dir). Optional: CONFIG_DIR (default configs/),
# CONFIG_SUFFIX (default "" -> configs/tabular_<arm>_uchicago_dce_g16.yaml),
# used by validation tiers that point the same configs at a tier's trees.
#SBATCH --job-name=g16-arms
#SBATCH --output=logs/slurm/g16-arms-%A_%a.out
#SBATCH --error=logs/slurm/g16-arms-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=tier1q
set -euo pipefail
: "${OUT_ROOT:?Set OUT_ROOT to the results directory for this run}"
PY="${VANGUARD_PYTHON:-/home/t-9sbose/micromamba/envs/vanguard/bin/python3}"
REPO="${REPO_ROOT:-/ess/home/home1/t-9sbose/vanguard}"
CONFIG_DIR="${CONFIG_DIR:-${REPO}/configs}"
CONFIG_SUFFIX="${CONFIG_SUFFIX:-}"

ARMS=(hr_only hr_ufast_merge_only final)
ARM="${ARMS[${SLURM_ARRAY_TASK_ID}]}"
CONFIG="${CONFIG_DIR}/tabular_${ARM}_uchicago_dce_g16${CONFIG_SUFFIX}.yaml"

cd "${REPO}"
echo "host=$(hostname) arm=${ARM} config=${CONFIG} commit=$(git rev-parse HEAD)"
"${PY}" -m tabular.gnn_feature_baseline \
  --config "${CONFIG}" \
  --out-dir "${OUT_ROOT}/tabular_${ARM}"
echo "done ${ARM}"
