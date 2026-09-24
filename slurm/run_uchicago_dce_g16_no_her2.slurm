#!/bin/bash
# Rerun the three tabular arms and the site-effect evaluation with her2_naclike
# dropped (Anna's 2026-09-23 request). her2_naclike is 16 exams, 0 pCR+ -- a
# single-class site that contributes no within-site discrimination and inflates
# the dataset-prevalence baseline, so its removal tests both the model and the
# baseline it is judged against. Folds are left as assigned so the remaining 331
# cases keep the same partition as the full-cohort run.
#
# Env: OUT_ROOT (full-cohort results dir), REPO_ROOT, COHORT_ROOT.
#SBATCH --job-name=g16-nohers2
#SBATCH --output=logs/slurm/g16-noher2-%j.out
#SBATCH --error=logs/slurm/g16-noher2-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=tier1q
set -euo pipefail
PY="${VANGUARD_PYTHON:-/home/t-9sbose/micromamba/envs/vanguard/bin/python3}"
REPO="${REPO_ROOT:-/ess/home/home1/t-9sbose/vanguard}"
G="${COHORT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16}"
OUT="${OUT_ROOT:-${G}/results/full_20260916}"
EXCL="--exclude-dataset her2_naclike"
cd "${REPO}"
echo "host=$(hostname) commit=$(git rev-parse HEAD)"

for ARM in hr_only hr_ufast_merge_only final; do
  echo "== ${ARM} (no her2_naclike)"
  "${PY}" analysis/fit_tabular_from_features.py \
    --features "${OUT}/tabular_${ARM}/tabular_baseline_features.csv" \
    --labels "${G}/staging/labels.csv" ${EXCL} \
    --out-dir "${OUT}/noher2_${ARM}"
done

for M in xgboost elastic_net logistic_regression; do
  for B in final hr_ufast_merge_only; do
    "${PY}" analysis/site_effect_evaluation.py \
      --model-column "oof_prob_${M}" --lodo-model "${M}" ${EXCL} \
      --arm hr_only="${OUT}/noher2_hr_only" --arm "${B}=${OUT}/noher2_${B}" \
      --labels "${G}/staging/labels.csv" \
      --out-dir "${OUT}/noher2_site_effect_${M}_hr_only_vs_${B}"
  done
done
echo done
