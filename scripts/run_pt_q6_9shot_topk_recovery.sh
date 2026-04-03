#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MODEL="${MODEL:-/scratch/${USER}/models/Llama-3.1-70B}"
export MODEL_SPEC="${MODEL_SPEC:-llama3}"
export DEVICE="${DEVICE:-cuda}"
export DTYPE="${DTYPE:-bf16}"
export QUANT="${QUANT:-4bit}"
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-122}"

export QID="${QID:-Q6}"
export SHOT_LIST="${SHOT_LIST:-1,3,5,7,9}"
export N_TRIALS="${N_TRIALS:-50}"
export N_BOOT="${N_BOOT:-10000}"
export SAVE_EDGE_TOPK="${SAVE_EDGE_TOPK:-1}"
export EDGE_TOPK_K="${EDGE_TOPK_K:-10}"
export EDGE_TOPK_EDGES="${EDGE_TOPK_EDGES:-AB,AC,AD,BC,BD}"

export MODEL_TAG="${MODEL_TAG:-llama31_70b_pt_q6_9shot_recovery}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export OUT_ROOT="${OUT_ROOT:-/scratch/${USER}/my_fv_project/pt_analysis}"

echo "[Q6-RECOVERY] model=${MODEL}"
echo "[Q6-RECOVERY] qid=${QID} shot_list=${SHOT_LIST} n_trials=${N_TRIALS} n_boot=${N_BOOT}"
echo "[Q6-RECOVERY] save_edge_topk=${SAVE_EDGE_TOPK} edge_topk_edges=${EDGE_TOPK_EDGES}"
echo "[Q6-RECOVERY] out_root=${OUT_ROOT} model_tag=${MODEL_TAG} run_id=${RUN_ID}"

bash "${ROOT_DIR}/scripts/run_pt_llama70b.sh"
