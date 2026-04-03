#!/bin/bash
#SBATCH -A def-twwebb_gpu
#SBATCH -p gpubase_bygpu_b4
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH -J ptuni_refresh_qsubset
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
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true

export MODEL="/scratch/${USER}/models/Llama-3.1-70B"
export MODEL_SPEC="llama3"
export DEVICE="cuda"
export DTYPE="bf16"
export QUANT="4bit"
export QID="Q5,Q6,Q7,Q8,Q10,Q11,Q16,Q18"
export N_TRIALS="50"
export N_BOOT="10000"
export SHOT_LIST="0,1,3,5,7,9"
export SAVE_EDGE_TOPK="1"
export EDGE_TOPK_K="10"
export MODEL_TAG="llama31_70b_unified_pt_refresh"
export RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-nojob}"
export OUT_ROOT="/scratch/${USER}/my_fv_project/pt_analysis"

echo "[slurm] model=${MODEL}"
echo "[slurm] qid=${QID}"
echo "[slurm] n_trials=${N_TRIALS} n_boot=${N_BOOT} shot_list=${SHOT_LIST}"
echo "[slurm] out_root=${OUT_ROOT} model_tag=${MODEL_TAG} run_id=${RUN_ID}"

bash /home/sunsik/my_fv_project/scripts/run_pt_unified_drift_control_llama70b.sh
