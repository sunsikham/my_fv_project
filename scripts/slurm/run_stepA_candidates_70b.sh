#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-python}"
MODEL_PATH="${MODEL_PATH:-/scratch/${USER}/models/Llama-3.1-70B}"
MODEL_SPEC="${MODEL_SPEC:-llama3}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
QUANT="${QUANT:-4bit}"
DEVICE_MAP="${DEVICE_MAP:-}"

SAMPLED_TRIALS_PATH="${SAMPLED_TRIALS_PATH:-results_fv/relation_qwise/relationB_ex/_analysis/stepd_clean_acc_allq_9shot_t50_20260211_044034/sampled_trials_stepd_style.json}"
OUT_DIR="${OUT_DIR:-results_fv/relation_qwise/relationB_ex/_analysis/stepA_candidates_70b_$(date +%Y%m%d_%H%M%S)}"

N_CANDIDATES="${N_CANDIDATES:-10}"
MAX_ATTEMPT_ROUNDS="${MAX_ATTEMPT_ROUNDS:-6}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
DO_SAMPLE="${DO_SAMPLE:-1}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
INCLUDE_GOLD_IN_PROMPT="${INCLUDE_GOLD_IN_PROMPT:-0}"
REQUIRE_COMPLETE_10="${REQUIRE_COMPLETE_10:-0}"
PROMPT_TEMPLATE_PATH="${PROMPT_TEMPLATE_PATH:-}"

HF_HOME="${HF_HOME:-/scratch/${USER}/hf}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

LOG_PATH="${LOG_PATH:-${OUT_DIR}/stepA_run.log}"
mkdir -p "$(dirname "$LOG_PATH")"
mkdir -p "$OUT_DIR"

die() {
  echo "[StepA70B][ERROR] $*" | tee -a "$LOG_PATH"
  exit 1
}

[[ -x "$(command -v "$PY")" ]] || die "python executable not found: $PY"
[[ -e "$MODEL_PATH" ]] || die "MODEL_PATH not found: $MODEL_PATH"
[[ -f "$SAMPLED_TRIALS_PATH" ]] || die "SAMPLED_TRIALS_PATH not found: $SAMPLED_TRIALS_PATH"
if [[ -n "$PROMPT_TEMPLATE_PATH" && ! -f "$PROMPT_TEMPLATE_PATH" ]]; then
  die "PROMPT_TEMPLATE_PATH not found: $PROMPT_TEMPLATE_PATH"
fi

{
  echo "[StepA70B] ROOT_DIR=$ROOT_DIR"
  echo "[StepA70B] PY=$PY"
  echo "[StepA70B] MODEL_PATH=$MODEL_PATH MODEL_SPEC=$MODEL_SPEC"
  echo "[StepA70B] DEVICE=$DEVICE DTYPE=$DTYPE QUANT=$QUANT DEVICE_MAP=${DEVICE_MAP:-<none>}"
  echo "[StepA70B] SAMPLED_TRIALS_PATH=$SAMPLED_TRIALS_PATH"
  echo "[StepA70B] OUT_DIR=$OUT_DIR"
  echo "[StepA70B] N_CANDIDATES=$N_CANDIDATES MAX_ATTEMPT_ROUNDS=$MAX_ATTEMPT_ROUNDS"
  echo "[StepA70B] SEED=$SEED BATCH_SIZE=$BATCH_SIZE MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
  echo "[StepA70B] DO_SAMPLE=$DO_SAMPLE TEMPERATURE=$TEMPERATURE TOP_P=$TOP_P"
  echo "[StepA70B] INCLUDE_GOLD_IN_PROMPT=$INCLUDE_GOLD_IN_PROMPT REQUIRE_COMPLETE_10=$REQUIRE_COMPLETE_10"
  echo "[StepA70B] HF_HOME=$HF_HOME"
  echo "[StepA70B] LOG_PATH=$LOG_PATH"
} | tee "$LOG_PATH"

cmd=(
  env PYTHONPATH=. \
    HF_HOME="$HF_HOME" \
    HF_HUB_CACHE="$HF_HUB_CACHE" \
    TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    "$PY" scripts/run_stepA_candidate_generation.py
    --sampled_trials_path "$SAMPLED_TRIALS_PATH"
    --model_path "$MODEL_PATH"
    --model_spec "$MODEL_SPEC"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --quant "$QUANT"
    --n_candidates "$N_CANDIDATES"
    --max_attempt_rounds "$MAX_ATTEMPT_ROUNDS"
    --seed "$SEED"
    --batch_size "$BATCH_SIZE"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --do_sample "$DO_SAMPLE"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --out_dir "$OUT_DIR"
)

if [[ -n "$DEVICE_MAP" ]]; then
  cmd+=(--device_map "$DEVICE_MAP")
fi
if [[ "$INCLUDE_GOLD_IN_PROMPT" == "1" ]]; then
  cmd+=(--include_gold_in_prompt)
fi
if [[ "$REQUIRE_COMPLETE_10" == "1" ]]; then
  cmd+=(--require_complete_10)
fi
if [[ -n "$PROMPT_TEMPLATE_PATH" ]]; then
  cmd+=(--prompt_template_path "$PROMPT_TEMPLATE_PATH")
fi

"${cmd[@]}" 2>&1 | tee -a "$LOG_PATH"

echo "[StepA70B] done" | tee -a "$LOG_PATH"
echo "[StepA70B] outputs in: $OUT_DIR" | tee -a "$LOG_PATH"
