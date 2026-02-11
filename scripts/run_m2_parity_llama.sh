#!/usr/bin/env bash
set -euo pipefail

# Canonical M2 runner for Llama antonym parity gate.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-/mnt/ebs/venv/bin/python}"
MODEL="${MODEL:-/mnt/ebs/.hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b}"
FT="${FT:-datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json}"
FT_ID="${FT_ID:-fixed_trials_antonym_t10_s10_seed0_llama31_8b}"

DATASET_NAME="${DATASET_NAME:-antonym}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
N_TOP_HEADS="${N_TOP_HEADS:-10}"
TOKEN_CLASS_IDX="${TOKEN_CLASS_IDX:--1}"
EDIT_LAYER="${EDIT_LAYER:-9}"
MAX_TRIALS="${MAX_TRIALS:-5}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
RESULTS_FV_ROOT="${RESULTS_FV_ROOT:-results_fv}"

mkdir -p logs
LOG_PATH="${LOG_PATH:-logs/m2_parity_llama_$(date +%Y%m%d_%H%M%S).log}"
RUN_DIR="$RESULTS_ROOT/$DATASET_NAME/$FT_ID"
OUT_DIR="$RESULTS_FV_ROOT/$DATASET_NAME/$FT_ID"
M1_MEAN="$RUN_DIR/${DATASET_NAME}_mean_head_activations_FIXED.pt"
M1_IE="$RUN_DIR/${DATASET_NAME}_indirect_effect.pt"
M1_DUMMY="$RUN_DIR/${DATASET_NAME}_dummy_labels.json"

die() {
  echo "[M2][ERROR] $*" | tee -a "$LOG_PATH"
  exit 1
}

echo "[M2] ROOT_DIR=$ROOT_DIR" | tee "$LOG_PATH"
echo "[M2] PY=$PY" | tee -a "$LOG_PATH"
echo "[M2] MODEL=$MODEL" | tee -a "$LOG_PATH"
echo "[M2] FT=$FT" | tee -a "$LOG_PATH"
echo "[M2] FT_ID=$FT_ID" | tee -a "$LOG_PATH"
echo "[M2] RESULTS_ROOT=$RESULTS_ROOT" | tee -a "$LOG_PATH"
echo "[M2] RESULTS_FV_ROOT=$RESULTS_FV_ROOT" | tee -a "$LOG_PATH"
echo "[M2] RUN_DIR=$RUN_DIR" | tee -a "$LOG_PATH"
echo "[M2] OUT_DIR=$OUT_DIR" | tee -a "$LOG_PATH"
echo "[M2] LOG_PATH=$LOG_PATH" | tee -a "$LOG_PATH"
echo "[M2] MAX_TRIALS=$MAX_TRIALS" | tee -a "$LOG_PATH"

# Precheck for faster failure diagnosis.
[[ -x "$PY" ]] || die "python executable not found: $PY"
[[ -e "$MODEL" ]] || die "model path not found: $MODEL"
[[ -f "$FT" ]] || die "fixed_trials json not found: $FT"
[[ -f "$M1_MEAN" ]] || die "missing M1 artifact: $M1_MEAN"
[[ -f "$M1_IE" ]] || die "missing M1 artifact: $M1_IE"
[[ -f "$M1_DUMMY" ]] || die "missing M1 artifact: $M1_DUMMY"

env PYTHONPATH=. "$PY" scripts/run_parity_suite.py \
  --dataset_name "$DATASET_NAME" \
  --fixed_trials_path "$FT" \
  --fixed_trials_id "$FT_ID" \
  --model_name "$MODEL" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --n_top_heads "$N_TOP_HEADS" \
  --token_class_idx "$TOKEN_CLASS_IDX" \
  --edit_layer "$EDIT_LAYER" \
  --max_trials "$MAX_TRIALS" \
  --results_root "$RESULTS_ROOT" \
  --results_fv_root "$RESULTS_FV_ROOT" \
  2>&1 | tee -a "$LOG_PATH"

echo "[M2] done" | tee -a "$LOG_PATH"
echo "[M2] log: $LOG_PATH"
echo "[M2] report: $OUT_DIR/parity_suite_report.json"
echo "[M2] stages: $OUT_DIR/parity_suite_stages.csv"
