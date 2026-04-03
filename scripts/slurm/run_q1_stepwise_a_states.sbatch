#!/bin/bash
#SBATCH -A def-twwebb_gpu
#SBATCH -p gpubase_bygpu_b4
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH -J fv_q1_stepwise_a
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

PYTHON_BIN="/home/sunsik/.venvs/pt442/bin/python"
MODEL_PATH="/scratch/sunsik/models/Llama-3.1-70B"
Q_DIR="/home/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1"

echo "[stepwise] host=$(hostname)"
echo "[stepwise] q_dir=${Q_DIR}"
echo "[stepwise] refs=AAA_ref,union_ref"
echo "[stepwise] conditions=AAA,BABA,DADA"
echo "[stepwise] prompt_mode=corrupted"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true

"$PYTHON_BIN" scripts/extract_stepwise_a_states.py \
  --q_dir "${Q_DIR}" \
  --model "${MODEL_PATH}" \
  --model_spec llama3 \
  --device cuda \
  --dtype bf16 \
  --quant 4bit \
  --device_map auto \
  --refs AAA_ref,union_ref \
  --conditions AAA,BABA,DADA \
  --prompt_mode corrupted
