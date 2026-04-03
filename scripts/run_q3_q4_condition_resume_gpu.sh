#!/usr/bin/env bash
set -euo pipefail

cd /home/sunsik/my_fv_project

source /home/sunsik/.venvs/pt442/bin/activate
module load cuda/12.9
export LD_LIBRARY_PATH="$EBROOTCUDA/lib64:${LD_LIBRARY_PATH:-}"
export BNB_CUDA_VERSION=122
unset CUDA_HOME
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PYTHON_BIN="/home/sunsik/.venvs/pt442/bin/python"
RELATION_NAME="relA_relationA_ex__relB_relationB_ex__h51e123e95a"
OUT_ROOT="/home/sunsik/my_fv_project/results_fv/relation_condition_qwise"

echo "[resume] host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true
echo "[resume] relation_name=${RELATION_NAME}"
echo "[resume] q_list=Q3,Q4"
echo "[resume] conditions=AAA,BBB,BABA"
echo "[resume] out_root=${OUT_ROOT}"

"$PYTHON_BIN" scripts/run_condition_qwise_pipeline.py \
  --model /scratch/sunsik/models/Llama-3.1-70B \
  --model_spec llama3 \
  --relation_a_csv datasets/relation/relationA_ex.csv \
  --relation_b_csv datasets/relation/relationB_ex.csv \
  --relation_name "${RELATION_NAME}" \
  --conditions AAA,BBB,BABA \
  --q_list Q3,Q4 \
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
  --sync_mode none \
  --home_artifact_profile core
