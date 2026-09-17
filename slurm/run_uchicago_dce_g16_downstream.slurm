#!/bin/bash
# Downstream stage for the uchicago_dce_g16 cohort, run once preprocessing finished:
#   1. tally the preprocessing output root (scripts/summarize_preprocessing_run.py)
#   2. build one symlink tree per skeleton arm (analysis/build_skeleton_arm_tree.py)
#   3. submit the three tabular arms (slurm/submit_uchicago_dce_g16_arms.slurm)
#   4. submit evaluation + EDA with afterok on the arms (slurm/run_uchicago_dce_g16_eval.slurm)
# Submit with --dependency=afterany:<postprocess array job> so it runs when the chain ends;
# step 1 reports any exam short of qc and step 2 only links exams that reached the arm.
#SBATCH --job-name=g16-downstream
#SBATCH --output=logs/slurm/g16-downstream-%j.out
#SBATCH --error=logs/slurm/g16-downstream-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=tier1q
set -euo pipefail
PY="${VANGUARD_PYTHON:-/home/t-9sbose/micromamba/envs/vanguard/bin/python3}"
REPO="${REPO_ROOT:-/ess/home/home1/t-9sbose/vanguard}"
G="${COHORT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16}"
SC="${SCRATCH_ROOT:-/ess/scratch/scratch1/t-9sbose/uchicago_dce_g16}"
RUN="${RUN_NAME:-$(date +%Y%m%d)}"
OUT="${G}/results/${RUN}"
cd "${REPO}"
mkdir -p "${OUT}" logs/slurm
echo "host=$(hostname) commit=$(git rev-parse HEAD) run=${RUN}"

"${PY}" scripts/summarize_preprocessing_run.py \
  --case-manifest "${G}/staging/case_manifest.csv" \
  --output-root "${SC}/preprocessing_out" \
  --out-dir "${OUT}/preprocessing_summary"

for ARM in hr_only hr_ufast_merge_only final; do
  TREE="${SC}/arm_trees/${RUN}/${ARM}"
  if [ -e "${TREE}" ]; then echo "arm tree exists: ${TREE}"; else
    "${PY}" analysis/build_skeleton_arm_tree.py \
      --centerline-root "${SC}/preprocessing_out/centerlines" --arm "${ARM}" --out-tree "${TREE}"
  fi
done

# Per-run configs: the committed configs name the arm trees under scratch/arm_trees/<RUN>.
CONF="${OUT}/configs"; mkdir -p "${CONF}"
for ARM in hr_only hr_ufast_merge_only final; do
  sed -e "s#uchicago_dce_g16/arm_trees/${ARM}#uchicago_dce_g16/arm_trees/${RUN}/${ARM}#" \
      -e "s#/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/arm_trees#${SC}/arm_trees#" \
      -e "s#/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/preprocessing_out#${SC}/preprocessing_out#" \
      -e "s#uchicago_dce_g16/gnn_cache/${ARM}#uchicago_dce_g16/gnn_cache/${RUN}/${ARM}#" \
      -e "s#/gpfs/data/karczmar-lab/workspaces/saritbose/uchicago_dce_g16/gnn_cache#${SC}/gnn_cache#" \
      "configs/tabular_${ARM}_uchicago_dce_g16.yaml" > "${CONF}/tabular_${ARM}_uchicago_dce_g16.yaml"
done

ARMS_JOB=$(sbatch --parsable --array=0-2 \
  --export=ALL,OUT_ROOT="${OUT}",CONFIG_DIR="${CONF}",REPO_ROOT="${REPO}",VANGUARD_PYTHON="${PY}" \
  slurm/submit_uchicago_dce_g16_arms.slurm)
EVAL_JOB=$(sbatch --parsable --dependency=afterok:"${ARMS_JOB}" \
  --export=ALL,OUT_ROOT="${OUT}",COHORT_ROOT="${G}",REPO_ROOT="${REPO}",VANGUARD_PYTHON="${PY}" \
  slurm/run_uchicago_dce_g16_eval.slurm)
echo "arms_job=${ARMS_JOB} eval_job=${EVAL_JOB}" | tee "${OUT}/downstream_job_ids.txt"
