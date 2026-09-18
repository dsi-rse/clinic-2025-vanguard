#!/bin/bash
# Site-effect evaluation on the XGBoost OOF column (LODO refits XGBoost). Env: OUT_ROOT, REPO_ROOT.
#SBATCH --job-name=g16-eval-xgb
#SBATCH --output=logs/slurm/g16-eval-xgb-%j.out
#SBATCH --error=logs/slurm/g16-eval-xgb-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=tier1q
set -euo pipefail
PY=/home/t-9sbose/micromamba/envs/vanguard/bin/python3
cd ${REPO_ROOT:-/ess/home/home1/t-9sbose/vanguard}
OUT=${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}
$PY analysis/site_effect_evaluation.py --model-column oof_prob_xgboost --lodo-model xgboost --arm hr_only=$OUT/tabular_hr_only --arm final=$OUT/tabular_final --labels /gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/staging/labels.csv --out-dir $OUT/site_effect_xgboost_hr_only_vs_final
$PY analysis/site_effect_evaluation.py --model-column oof_prob_xgboost --lodo-model xgboost --arm hr_only=$OUT/tabular_hr_only --arm hr_ufast_merge_only=$OUT/tabular_hr_ufast_merge_only --labels /gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/staging/labels.csv --out-dir $OUT/site_effect_xgboost_hr_only_vs_merge
