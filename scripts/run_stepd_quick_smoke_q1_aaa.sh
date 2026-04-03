#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/sunsik/.venvs/pt442/bin/python}"
MODEL_PATH="${MODEL_PATH:-/scratch/sunsik/models/Llama-3.1-70B}"
MODEL_SPEC="${MODEL_SPEC:-llama3}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
QUANT="${QUANT:-4bit}"

LAYER="${LAYER:-20}"
HEADS="${HEADS:-0,1,2,3,4,5,6,7}"
N_TRIALS="${N_TRIALS:-3}"
N_DEMOS="${N_DEMOS:-9}"
SEED="${SEED:-0}"

TRIAL_SRC="${TRIAL_SRC:-/home/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_trials/condition_AAA.json}"
TRIAL_TMP="${TRIAL_TMP:-/tmp/condition_AAA_Q1_t${N_TRIALS}.json}"
OUT_DIR="${OUT_DIR:-/tmp/stepd_quick_q1_aaa_l${LAYER}_h8_t${N_TRIALS}}"

if [[ ! -f "$TRIAL_SRC" ]]; then
  echo "Missing TRIAL_SRC: $TRIAL_SRC" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if type module >/dev/null 2>&1; then
  module load cuda/12.9 || true
fi
export LD_LIBRARY_PATH="${EBROOTCUDA:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-122}"
unset CUDA_HOME || true

TRIAL_SRC="$TRIAL_SRC" TRIAL_TMP="$TRIAL_TMP" N_TRIALS="$N_TRIALS" "$PYTHON_BIN" - <<'PY'
import json
import os

src = os.environ["TRIAL_SRC"]
dst = os.environ["TRIAL_TMP"]
n_trials = int(os.environ["N_TRIALS"])

with open(src, "r", encoding="utf-8") as f:
    payload = json.load(f)

payload["trials"] = payload.get("trials", [])[:n_trials]
payload.setdefault("meta", {})["n_trials"] = len(payload["trials"])

with open(dst, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=True, indent=2)

print(dst)
PY

echo "[run] layer=$LAYER heads=$HEADS n_trials=$N_TRIALS out=$OUT_DIR"
"$PYTHON_BIN" scripts/run_stepD_aie_head_sweep.py \
  --model "$MODEL_PATH" \
  --model_spec "$MODEL_SPEC" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --quant "$QUANT" \
  --layers "$LAYER" \
  --heads "$HEADS" \
  --n_trials "$N_TRIALS" \
  --n_mean_trials "$N_TRIALS" \
  --n_icl_examples "$N_DEMOS" \
  --score_key mean_delta_p \
  --fixed_trials_path "$TRIAL_TMP" \
  --fixed_out_dir "$OUT_DIR" \
  --baseline_cache_scope layer_trial \
  --cache_tokenized_inputs 1 \
  --persistent_layer_hook 1 \
  --perf_log 1 \
  --seed "$SEED"

echo "[done] outputs:"
echo "  - $OUT_DIR/artifacts/aie_scores.csv"
echo "  - $OUT_DIR/artifacts/perf_stats.json"
echo "  - $OUT_DIR/logs/stepD_aie.log"
