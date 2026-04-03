#!/bin/bash
#SBATCH -A def-twwebb_gpu
#SBATCH -p gpubase_bygpu_b4
#SBATCH --gres=gpu:h100:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -J fv_tail_noQ1_3d
#SBATCH -o /home/sunsik/my_fv_project/logs/%x_%j.out

set -euo pipefail

cd /home/sunsik/my_fv_project
mkdir -p logs

source /home/sunsik/.venvs/pt442/bin/activate
module load cuda/12.9
export LD_LIBRARY_PATH="$EBROOTCUDA/lib64:${LD_LIBRARY_PATH:-}"
export BNB_CUDA_VERSION=122
unset CUDA_HOME
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[slurm] job_id=${SLURM_JOB_ID:-unknown}"
scontrol show job "${SLURM_JOB_ID:-0}" | egrep 'ReqTRES|AllocTRES|NodeList|StartTime|TimeLimit' || true

# Tail-first list (Q1 excluded). Override at submit time with Q_LIST_OVERRIDE if needed.
# Example:
#   Q_LIST_OVERRIDE="Q9,Q8,Q7,Q6,Q5,Q4,Q3" sbatch scripts/slurm/run_q_tail_from_end_condition_pipeline_3d.sbatch
Q_LIST="${Q_LIST_OVERRIDE:-Q9,Q8,Q7,Q6,Q5,Q4,Q3}"
SCRATCH_OUT_ROOT="${SCRATCH_OUT_ROOT_OVERRIDE:-/scratch/${USER}/my_fv_project/results_fv/relation_condition_qwise}"
HOME_MIRROR_ROOT="${HOME_MIRROR_ROOT_OVERRIDE:-/home/sunsik/my_fv_project/results_fv/relation_condition_qwise}"
SYNC_MODE="${SYNC_MODE:-per_q}"
HOME_ARTIFACT_PROFILE="${HOME_ARTIFACT_PROFILE:-core}"
echo "[slurm] q_list=${Q_LIST}"
echo "[slurm] scratch_out_root=${SCRATCH_OUT_ROOT}"
echo "[slurm] home_mirror_root=${HOME_MIRROR_ROOT}"
echo "[slurm] sync_mode=${SYNC_MODE} profile=${HOME_ARTIFACT_PROFILE}"

python scripts/run_condition_qwise_pipeline.py \
  --model /scratch/sunsik/models/Llama-3.1-70B \
  --model_spec llama3 \
  --relation_a_csv datasets/relation/relationA_ex.csv \
  --relation_b_csv datasets/relation/relationB_ex.csv \
  --conditions AAA,BBB,BABA \
  --q_list "${Q_LIST}" \
  --n_trials_per_q 25 \
  --n_demos 9 \
  --topk 20 \
  --score_key mean_delta_p \
  --enable_union_ref 0 \
  --run_injection 0 \
  --device cuda \
  --dtype bf16 \
  --quant 4bit \
  --resume 1 \
  --stop_on_error 0 \
  --seed 0 \
  --stepd_layers 16-55 \
  --out_root "${SCRATCH_OUT_ROOT}" \
  --sync_home_root "${HOME_MIRROR_ROOT}" \
  --sync_mode "${SYNC_MODE}" \
  --home_artifact_profile "${HOME_ARTIFACT_PROFILE}" \
  --sync_on_failure 1
