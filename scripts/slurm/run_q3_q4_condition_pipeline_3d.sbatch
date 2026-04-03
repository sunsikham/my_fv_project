#!/bin/bash
#SBATCH -A def-twwebb_gpu
#SBATCH -p gpubase_bygpu_b4
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH -J fv_q34_retry
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

RELATION_NAME="${RELATION_NAME_OVERRIDE:-relA_relationA_ex__relB_relationB_ex__h51e123e95a}"
Q_LIST="${Q_LIST_OVERRIDE:-Q3,Q4}"
CONDITIONS="${CONDITIONS_OVERRIDE:-AAA,BBB,BABA}"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-/home/sunsik/my_fv_project/results_fv/relation_condition_qwise}"
SYNC_MODE="${SYNC_MODE:-none}"
HOME_ARTIFACT_PROFILE="${HOME_ARTIFACT_PROFILE:-core}"

echo "[slurm] relation_name=${RELATION_NAME}"
echo "[slurm] q_list=${Q_LIST}"
echo "[slurm] conditions=${CONDITIONS}"
echo "[slurm] out_root=${OUT_ROOT}"
echo "[slurm] sync_mode=${SYNC_MODE} profile=${HOME_ARTIFACT_PROFILE}"

python scripts/run_condition_qwise_pipeline.py \
  --model /scratch/sunsik/models/Llama-3.1-70B \
  --model_spec llama3 \
  --relation_a_csv datasets/relation/relationA_ex.csv \
  --relation_b_csv datasets/relation/relationB_ex.csv \
  --relation_name "${RELATION_NAME}" \
  --conditions "${CONDITIONS}" \
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
  --out_root "${OUT_ROOT}" \
  --sync_mode "${SYNC_MODE}" \
  --home_artifact_profile "${HOME_ARTIFACT_PROFILE}"
