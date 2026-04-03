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
MODEL="/scratch/sunsik/models/Llama-3.1-70B"
Q_DIR="/home/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__h51e123e95a/Q3"
TRIAL_JSON="$Q_DIR/_trials/condition_BABA.json"
RUN_BASE="$Q_DIR/_stepd/run_BABA"

echo "[resume] host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true
echo "[resume] q_dir=$Q_DIR"
echo "[resume] condition=BABA"
echo "[resume] run_base=$RUN_BASE"

"$PYTHON_BIN" scripts/run_stepD_aie_head_sweep.py \
  --model "$MODEL" \
  --model_spec llama3 \
  --device cuda \
  --dtype bf16 \
  --quant 4bit \
  --layers 16-55 \
  --heads all \
  --n_trials 25 \
  --n_icl_examples 9 \
  --score_key mean_delta_p \
  --seed 0 \
  --compute_prob_scores 1 \
  --fixed_trials_path "$TRIAL_JSON" \
  --fixed_out_dir "$RUN_BASE"

export Q_DIR RUN_BASE
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

from scripts.run_condition_qwise_pipeline import copy_stepd_outputs

q_dir = Path(os.environ["Q_DIR"])
run_base = Path(os.environ["RUN_BASE"])
copy_stepd_outputs(q_dir=q_dir, condition="BABA", stepd_run_base=run_base)
print(f"[resume] copied BABA outputs to {q_dir / '_stepd'}")
PY

echo "[resume] done"
