#!/bin/bash
# Site-effect evaluation + feature-vs-metadata EDA for one uchicago_dce_g16 run.
# Expects OUT_ROOT/tabular_<arm>/ from slurm/submit_uchicago_dce_g16_arms.slurm and
# OUT_ROOT/preprocessing_summary/ from the downstream launcher.
#SBATCH --job-name=g16-eval
#SBATCH --output=logs/slurm/g16-eval-%j.out
#SBATCH --error=logs/slurm/g16-eval-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=tier1q
set -euo pipefail
: "${OUT_ROOT:?Set OUT_ROOT to the results directory of this run}"
PY="${VANGUARD_PYTHON:-/home/t-9sbose/micromamba/envs/vanguard/bin/python3}"
REPO="${REPO_ROOT:-/ess/home/home1/t-9sbose/vanguard}"
G="${COHORT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16}"
cd "${REPO}"
echo "host=$(hostname) commit=$(git rev-parse HEAD)"

"${PY}" analysis/site_effect_evaluation.py \
  --arm hr_only="${OUT_ROOT}/tabular_hr_only" \
  --arm final="${OUT_ROOT}/tabular_final" \
  --labels "${G}/staging/labels.csv" \
  --out-dir "${OUT_ROOT}/site_effect_hr_only_vs_final"
"${PY}" analysis/site_effect_evaluation.py \
  --arm hr_only="${OUT_ROOT}/tabular_hr_only" \
  --arm hr_ufast_merge_only="${OUT_ROOT}/tabular_hr_ufast_merge_only" \
  --labels "${G}/staging/labels.csv" \
  --out-dir "${OUT_ROOT}/site_effect_hr_only_vs_merge"

for ARM in hr_only final; do
  "${PY}" analysis/feature_metadata_eda.py \
    --features "${OUT_ROOT}/tabular_${ARM}/tabular_baseline_features.csv" \
    --labels "${G}/staging/labels.csv" \
    --dicom-metadata "${G}/metadata/dicom_acquisition_metadata_per_exam.csv" \
    --preprocessing-summary "${OUT_ROOT}/preprocessing_summary/preprocessing_summary.csv" \
    --manifest-metadata "${G}/staging/exam_metadata_from_manifest.csv" \
    --out-dir "${OUT_ROOT}/eda_${ARM}"
done
echo "done"
