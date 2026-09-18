#!/bin/bash
# ComBat harmonisation pipeline for uchicago_dce_g16: raw refit control -> ComBat ->
# refit -> paired raw-vs-ComBat site-effect evaluation -> post-ComBat EDA. Env: OUT_ROOT, REPO_ROOT.
#SBATCH --job-name=g16-combat
#SBATCH --output=logs/slurm/g16-combat-%j.out
#SBATCH --error=logs/slurm/g16-combat-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=tier1q
set -euo pipefail
PY=/home/t-9sbose/micromamba/envs/vanguard/bin/python3
cd ${REPO_ROOT:-/ess/home/home1/t-9sbose/vanguard}
for ARM in final hr_only; do
  echo "== $ARM: raw refit control"
  $PY analysis/fit_tabular_from_features.py --features ${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/tabular_$ARM/tabular_baseline_features.csv --labels /gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/staging/labels.csv --out-dir ${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/refit_raw_$ARM
  echo "== $ARM: combat"
  $PY analysis/combat_harmonize.py --features ${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/tabular_$ARM/tabular_baseline_features.csv --joined ${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/eda_final/joined_table.csv --out-dir ${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/combat_$ARM
  $PY analysis/fit_tabular_from_features.py --features ${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/combat_$ARM/tabular_baseline_features.csv --labels /gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/staging/labels.csv --out-dir ${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/combat_$ARM
  for M in xgboost elastic_net; do
    $PY analysis/site_effect_evaluation.py --model-column oof_prob_$M --lodo-model $M --arm raw=${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/refit_raw_$ARM --arm combat=${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/combat_$ARM --labels /gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/staging/labels.csv --out-dir ${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/site_effect_combat_${M}_$ARM
  done
done
$PY analysis/feature_metadata_eda.py --features ${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/combat_final/tabular_baseline_features.csv --labels /gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/staging/labels.csv --dicom-metadata /gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/metadata/dicom_acquisition_metadata_per_exam.csv --preprocessing-summary ${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/preprocessing_summary/preprocessing_summary.csv --manifest-metadata /gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/staging/exam_metadata_from_manifest.csv --out-dir ${OUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/results/full_20260916}/eda_combat_final
echo done
