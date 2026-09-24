#!/bin/bash
# Robustness checks (permutation / ablation / within-stratum / hyperparameters) for the
# uchicago_dce_g16 tabular arms. Env: OUT_ROOT (run results dir), REPO_ROOT. Array 0=final, 1=hr_only.
#SBATCH --job-name=g16-robust
#SBATCH --output=logs/slurm/g16-robust-%A_%a.out
#SBATCH --error=logs/slurm/g16-robust-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=tier1q
set -euo pipefail
ARMS=(final hr_only)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
cd ${REPO_ROOT:-/ess/home/home1/t-9sbose/vanguard}
/home/t-9sbose/micromamba/envs/vanguard/bin/python3 analysis/pcr_signal_robustness.py --features ${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/tabular_$ARM/tabular_baseline_features.csv --labels /gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/staging/labels.csv --joined ${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/eda_final/joined_table.csv --importance ${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/eda_final/batch_importance.csv --model xgboost --n-perm 200 --out-dir ${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/robustness_xgboost_$ARM
