#!/bin/bash
#SBATCH -A def-twwebb_gpu
#SBATCH -p gpubase_bygpu_b4
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -J fv_q1
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

python scripts/run_condition_qwise_pipeline.py \
  --model /scratch/sunsik/models/Llama-3.1-70B \
  --model_spec llama3 \
  --relation_a_csv datasets/relation/relationA_ex.csv \
  --relation_b_csv datasets/relation/relationB_ex.csv \
  --conditions AAA,BBB,BABA \
  --q_list Q1 \
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
  --seed 0 \
  --stepd_layers 16-55
