#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PY:-}" ]]; then
  if [[ -x "/home/${USER}/.venvs/pt442/bin/python" ]]; then
    PY="/home/${USER}/.venvs/pt442/bin/python"
  elif [[ -x "/home/sunsik/.venvs/pt442/bin/python" ]]; then
    PY="/home/sunsik/.venvs/pt442/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    PY=""
  fi
fi

MODEL="${MODEL:-/scratch/${USER}/models/Llama-3.1-70B}"
MODEL_SPEC="${MODEL_SPEC:-llama3}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
QUANT="${QUANT:-4bit}"
BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-122}"

REL_A="${REL_A:-datasets/relation/relationA_ex.csv}"
REL_B="${REL_B:-datasets/relation/relationB_ex.csv}"
REL_D="${REL_D:-datasets/relation/relationD_ex.csv}"
ICL_B="${ICL_B:-datasets/relation/icl_B_data.csv}"

SHOT_LIST="${SHOT_LIST:-1,3,5,7,9}"
N_TRIALS="${N_TRIALS:-50}"
SEED="${SEED:-0}"
QID="${QID:-}"
SAVE_EDGE_TOPK="${SAVE_EDGE_TOPK:-1}"
EDGE_TOPK_K="${EDGE_TOPK_K:-10}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

OUT_ROOT="${OUT_ROOT:-/scratch/${USER}/my_fv_project/pt_analysis}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_TAG="${MODEL_TAG:-llama31_70b_context_drift_probe}"
RUN_DIR="${RUN_DIR:-${OUT_ROOT}/${MODEL_TAG}_${RUN_ID}}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/run.log}"

SCORES_CSV="${RUN_DIR}/pt_context_drift_probe_shot_sweep.csv"
EDGE_TOPK_JSONL="${RUN_DIR}/pt_context_drift_probe_edge_topk.jsonl"
EDGE_TOPK_CHANGE_CSV="${RUN_DIR}/pt_context_drift_probe_edge_topk_change_summary.csv"

mkdir -p "$RUN_DIR"

die() {
  echo "[PTCTXP][ERROR] $*" | tee -a "$LOG_PATH"
  exit 1
}

[[ -x "$PY" ]] || die "python executable not found: $PY"
[[ -e "$MODEL" ]] || die "MODEL path not found: $MODEL"
[[ -f "$REL_A" ]] || die "missing relation file: $REL_A"
[[ -f "$REL_B" ]] || die "missing relation file: $REL_B"
[[ -f "$REL_D" ]] || die "missing relation file: $REL_D"
[[ -f "$ICL_B" ]] || die "missing relation file: $ICL_B"

{
  echo "[PTCTXP] ROOT_DIR=$ROOT_DIR"
  echo "[PTCTXP] PY=$PY"
  echo "[PTCTXP] MODEL=$MODEL"
  echo "[PTCTXP] MODEL_SPEC=$MODEL_SPEC DEVICE=$DEVICE DTYPE=$DTYPE QUANT=$QUANT"
  echo "[PTCTXP] SHOT_LIST=$SHOT_LIST N_TRIALS=$N_TRIALS SEED=$SEED QID=${QID:-ALL}"
  echo "[PTCTXP] SAVE_EDGE_TOPK=$SAVE_EDGE_TOPK EDGE_TOPK_K=$EDGE_TOPK_K"
  echo "[PTCTXP] RUN_DIR=$RUN_DIR"
  echo "[PTCTXP] SCORES_CSV=$SCORES_CSV"
  if [[ "$SAVE_EDGE_TOPK" == "1" ]]; then
    echo "[PTCTXP] EDGE_TOPK_JSONL=$EDGE_TOPK_JSONL"
    echo "[PTCTXP] EDGE_TOPK_CHANGE_CSV=$EDGE_TOPK_CHANGE_CSV"
  fi
} | tee "$LOG_PATH"

if [[ "$QUANT" == "4bit" || "$QUANT" == "8bit" ]]; then
  export BNB_CUDA_VERSION
fi

score_cmd=(
  env PYTHONPATH=. PYTHONUNBUFFERED="$PYTHONUNBUFFERED" "$PY" -u scripts/score_cross_relation_context_drift_logit.py
  --model "$MODEL"
  --model_spec "$MODEL_SPEC"
  --device "$DEVICE"
  --quant "$QUANT"
  --relationA_ex_path "$REL_A"
  --relationB_ex_path "$REL_B"
  --relationD_ex_path "$REL_D"
  --icl_B_path "$ICL_B"
  --icl_D_path "$ICL_B"
  --regime_mode candidate_probe
  --shot_list "$SHOT_LIST"
  --n_trials "$N_TRIALS"
  --seed "$SEED"
  --out_csv "$SCORES_CSV"
)
if [[ -n "$DTYPE" ]]; then
  score_cmd+=(--dtype "$DTYPE")
fi
if [[ -n "$QID" ]]; then
  score_cmd+=(--qid "$QID")
fi
if [[ "$SAVE_EDGE_TOPK" == "1" ]]; then
  score_cmd+=(
    --save_edge_topk 1
    --edge_topk_k "$EDGE_TOPK_K"
    --edge_topk_jsonl "$EDGE_TOPK_JSONL"
    --edge_topk_change_csv "$EDGE_TOPK_CHANGE_CSV"
  )
fi

"${score_cmd[@]}" 2>&1 | tee -a "$LOG_PATH"

echo "[PTCTXP] done" | tee -a "$LOG_PATH"
echo "[PTCTXP] outputs:" | tee -a "$LOG_PATH"
echo "  - $SCORES_CSV" | tee -a "$LOG_PATH"
if [[ "$SAVE_EDGE_TOPK" == "1" ]]; then
  echo "  - $EDGE_TOPK_JSONL" | tee -a "$LOG_PATH"
  echo "  - $EDGE_TOPK_CHANGE_CSV" | tee -a "$LOG_PATH"
fi
echo "  - $LOG_PATH" | tee -a "$LOG_PATH"
