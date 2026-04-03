#!/usr/bin/env bash
set -euo pipefail

# PT pipeline runner for local Llama-70B checkpoints on GPU jobs.
# Usage example:
#   sbatch --gpus=1 --cpus-per-task=8 --mem=128G --time=24:00:00 scripts/run_pt_llama70b.sh
# Override example:
#   MODEL=/scratch/$USER/hf/llama-70b N_TRIALS=50 N_BOOT=10000 scripts/run_pt_llama70b.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PY:-}" ]]; then
  if [[ -x "/home/${USER}/.venvs/pt442/bin/python" ]]; then
    PY="/home/${USER}/.venvs/pt442/bin/python"
  elif [[ -x "/home/sunsik/.venvs/pt442/bin/python" ]]; then
    PY="/home/sunsik/.venvs/pt442/bin/python"
  elif [[ -x "/home/sunsik/.venvs/cedkv/bin/python" ]]; then
    PY="/home/sunsik/.venvs/cedkv/bin/python"
  elif [[ -x "/mnt/ebs/venv/bin/python" ]]; then
    PY="/mnt/ebs/venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
  else
    PY=""
  fi
fi
MODEL="${MODEL:-/scratch/${USER}/hf/models--meta-llama--Llama-3.1-70B-Instruct}"
MODEL_SPEC="${MODEL_SPEC:-llama3}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
QUANT="${QUANT:-4bit}"
BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-122}"

REL_A="${REL_A:-datasets/relation/relationA_ex.csv}"
REL_B="${REL_B:-datasets/relation/relationB_ex.csv}"
ICL_B="${ICL_B:-datasets/relation/icl_B_data.csv}"
ICL_C="${ICL_C:-datasets/relation/icl_C_data.csv}"
ICL_D="${ICL_D:-datasets/relation/icl_D_data.csv}"

SHOT_LIST="${SHOT_LIST:-1,3,5,7,10}"
N_TRIALS="${N_TRIALS:-50}"
N_BOOT="${N_BOOT:-10000}"
STAT="${STAT:-mean}"
SEED="${SEED:-0}"
QID="${QID:-}"
SAVE_EDGE_TOPK="${SAVE_EDGE_TOPK:-0}"
EDGE_TOPK_K="${EDGE_TOPK_K:-10}"
EDGE_TOPK_EDGES="${EDGE_TOPK_EDGES:-AB,AC,AD,BC,BD}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

OUT_ROOT="${OUT_ROOT:-/scratch/${USER}/my_fv_project/pt_analysis}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_TAG="${MODEL_TAG:-llama31_70b}"
RUN_DIR="${OUT_ROOT}/${MODEL_TAG}_${RUN_ID}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/run.log}"
RUN_META_JSON="${RUN_DIR}/run_meta.json"
RUN_STATUS_JSON="${RUN_DIR}/run_status.json"

SCORES_CSV="${RUN_DIR}/pt_5edge_shot_sweep.csv"
BOOTSTRAP_CSV="${RUN_DIR}/pt_bootstrap_summary.csv"
PLOT_PNG="${RUN_DIR}/pt_bootstrap_summary.png"
EDGE_TOPK_JSONL="${RUN_DIR}/pt_edge_topk.jsonl"
EDGE_TOPK_CHANGE_CSV="${RUN_DIR}/pt_edge_topk_change_summary.csv"

mkdir -p "$RUN_DIR"

die() {
  echo "[PT70B][ERROR] $*" | tee -a "$LOG_PATH"
  exit 1
}

[[ -x "$PY" ]] || die "python executable not found: $PY"
[[ -e "$MODEL" ]] || die "MODEL path not found: $MODEL"
[[ -f "$REL_A" ]] || die "missing relation file: $REL_A"
[[ -f "$REL_B" ]] || die "missing relation file: $REL_B"
[[ -f "$ICL_B" ]] || die "missing relation file: $ICL_B"
[[ -f "$ICL_C" ]] || die "missing relation file: $ICL_C"
[[ -f "$ICL_D" ]] || die "missing relation file: $ICL_D"

write_run_meta() {
  "$PY" - "$RUN_META_JSON" "$RUN_DIR" "$OUT_ROOT" "$RUN_ID" "$MODEL_TAG" "$MODEL" "$MODEL_SPEC" "$DEVICE" "${DTYPE:-}" "$QUANT" "$SHOT_LIST" "$N_TRIALS" "$N_BOOT" "$STAT" "$SEED" "${QID:-}" "$SAVE_EDGE_TOPK" "$EDGE_TOPK_K" "$EDGE_TOPK_EDGES" "$SCORES_CSV" "$BOOTSTRAP_CSV" "$PLOT_PNG" "$EDGE_TOPK_JSONL" "$EDGE_TOPK_CHANGE_CSV" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

(
    out_path,
    run_dir,
    out_root,
    run_id,
    model_tag,
    model,
    model_spec,
    device,
    dtype,
    quant,
    shot_list,
    n_trials,
    n_boot,
    stat,
    seed,
    qid,
    save_edge_topk,
    edge_topk_k,
    edge_topk_edges,
    scores_csv,
    bootstrap_csv,
    plot_png,
    edge_topk_jsonl,
    edge_topk_change_csv,
) = sys.argv[1:]

payload = {
    "created_at": datetime.utcnow().isoformat() + "Z",
    "run_family": "pt_baseline",
    "run_id": run_id,
    "model_tag": model_tag,
    "model": model,
    "model_spec": model_spec,
    "device": device,
    "dtype": dtype or None,
    "quant": quant,
    "shot_list": shot_list,
    "n_trials": int(n_trials),
    "n_boot": int(n_boot),
    "stat": stat,
    "seed": int(seed),
    "qid": qid or None,
    "save_edge_topk": int(save_edge_topk),
    "edge_topk_k": int(edge_topk_k),
    "edge_topk_edges": edge_topk_edges,
    "paths": {
        "run_dir": run_dir,
        "out_root": out_root,
        "scores_csv": scores_csv,
        "bootstrap_csv": bootstrap_csv,
        "plot_png": plot_png,
        "edge_topk_jsonl": edge_topk_jsonl,
        "edge_topk_change_csv": edge_topk_change_csv,
        "log_path": str(Path(run_dir) / "run.log"),
    },
    "canonical_root": run_dir,
    "sync_root": None,
    "sync_mode": "none",
    "artifact_profile": "full",
}
Path(out_path).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
PY
}

write_run_status() {
  "$PY" - "$RUN_STATUS_JSON" "$RUN_META_JSON" "$1" "$2" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

out_path, meta_path, status, exit_code = sys.argv[1:]
payload = {
    "updated_at": datetime.utcnow().isoformat() + "Z",
    "status": status,
    "exit_code": int(exit_code),
}
meta_file = Path(meta_path)
if meta_file.exists():
    payload["run_meta_path"] = str(meta_file)
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    for key in ("canonical_root", "sync_root", "sync_mode", "artifact_profile", "run_family", "run_id"):
        if key in meta:
            payload[key] = meta[key]
Path(out_path).write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
PY
}

write_run_meta
trap 'rc=$?; if [[ -x "${PY:-}" ]]; then if [[ $rc -eq 0 ]]; then write_run_status completed "$rc"; else write_run_status failed "$rc"; fi; fi' EXIT

{
  echo "[PT70B] ROOT_DIR=$ROOT_DIR"
  echo "[PT70B] PY=$PY"
  echo "[PT70B] MODEL=$MODEL"
  echo "[PT70B] MODEL_SPEC=$MODEL_SPEC DEVICE=$DEVICE DTYPE=$DTYPE QUANT=$QUANT"
  if [[ "$QUANT" == "4bit" || "$QUANT" == "8bit" ]]; then
    echo "[PT70B] BNB_CUDA_VERSION=$BNB_CUDA_VERSION"
  fi
  echo "[PT70B] SHOT_LIST=$SHOT_LIST N_TRIALS=$N_TRIALS N_BOOT=$N_BOOT STAT=$STAT SEED=$SEED QID=${QID:-ALL}"
  echo "[PT70B] SAVE_EDGE_TOPK=$SAVE_EDGE_TOPK EDGE_TOPK_K=$EDGE_TOPK_K EDGE_TOPK_EDGES=$EDGE_TOPK_EDGES"
  echo "[PT70B] PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
  echo "[PT70B] RUN_DIR=$RUN_DIR"
  echo "[PT70B] SCORES_CSV=$SCORES_CSV"
  echo "[PT70B] BOOTSTRAP_CSV=$BOOTSTRAP_CSV"
  echo "[PT70B] PLOT_PNG=$PLOT_PNG"
  if [[ "$SAVE_EDGE_TOPK" == "1" ]]; then
    echo "[PT70B] EDGE_TOPK_JSONL=$EDGE_TOPK_JSONL"
    echo "[PT70B] EDGE_TOPK_CHANGE_CSV=$EDGE_TOPK_CHANGE_CSV"
  fi
} | tee "$LOG_PATH"

if [[ "$QUANT" == "4bit" || "$QUANT" == "8bit" ]]; then
  export BNB_CUDA_VERSION

  if [[ "${LD_LIBRARY_PATH:-}" == *"cudacore/12.2"* ]]; then
    echo "[PT70B][WARN] LD_LIBRARY_PATH contains CUDA 12.2 paths." | tee -a "$LOG_PATH"
    echo "[PT70B][WARN] torch in this env is CUDA 12.9; mixed CUDA libs can cause nvJitLink symbol errors." | tee -a "$LOG_PATH"
  fi

  # Fail fast if the requested bitsandbytes CUDA binary does not exist in this env.
  bnb_lib="$("$PY" - <<'PY'
import os, sys
try:
    import bitsandbytes as bnb
except Exception:
    # Defer import/runtime failures to downstream script where full traceback is useful.
    print("")
    sys.exit(0)
pkg = os.path.dirname(bnb.__file__)
ver = os.environ.get("BNB_CUDA_VERSION", "").strip()
if not ver:
    print("")
else:
    print(os.path.join(pkg, f"libbitsandbytes_cuda{ver}.so"))
PY
)"
  if [[ -n "${bnb_lib}" && ! -f "${bnb_lib}" ]]; then
    die "bitsandbytes binary not found for BNB_CUDA_VERSION=${BNB_CUDA_VERSION}: ${bnb_lib}"
  fi
fi

score_cmd=(
  env PYTHONPATH=. PYTHONUNBUFFERED="$PYTHONUNBUFFERED" "$PY" -u scripts/score_cross_relation_target_logit.py
  --model "$MODEL"
  --model_spec "$MODEL_SPEC"
  --device "$DEVICE"
  --quant "$QUANT"
  --relationA_ex_path "$REL_A"
  --relationB_ex_path "$REL_B"
  --icl_B_path "$ICL_B"
  --icl_C_path "$ICL_C"
  --icl_D_path "$ICL_D"
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
    --edge_topk_edges "$EDGE_TOPK_EDGES"
    --edge_topk_jsonl "$EDGE_TOPK_JSONL"
    --edge_topk_change_csv "$EDGE_TOPK_CHANGE_CSV"
  )
fi

"${score_cmd[@]}" 2>&1 | tee -a "$LOG_PATH"

bootstrap_cmd=(
  env PYTHONPATH=. PYTHONUNBUFFERED="$PYTHONUNBUFFERED" "$PY" -u scripts/compute_product_test_bootstrap.py
  --in_csv "$SCORES_CSV"
  --out_csv "$BOOTSTRAP_CSV"
  --n_boot "$N_BOOT"
  --seed "$SEED"
  --shot_list "$SHOT_LIST"
  --stat "$STAT"
)
if [[ -n "$QID" ]]; then
  bootstrap_cmd+=(--qid "$QID")
fi

"${bootstrap_cmd[@]}" 2>&1 | tee -a "$LOG_PATH"

plot_cmd=(
  env PYTHONPATH=. PYTHONUNBUFFERED="$PYTHONUNBUFFERED" "$PY" -u scripts/plot_product_test_summary.py
  --in_csv "$BOOTSTRAP_CSV"
  --out_png "$PLOT_PNG"
  --show_p stars
  --title "PT summary (${MODEL_TAG})"
)
if [[ -n "$QID" ]]; then
  plot_cmd+=(--qid "$QID")
fi

"${plot_cmd[@]}" 2>&1 | tee -a "$LOG_PATH"

echo "[PT70B] done" | tee -a "$LOG_PATH"
echo "[PT70B] outputs:" | tee -a "$LOG_PATH"
echo "  - $RUN_META_JSON" | tee -a "$LOG_PATH"
echo "  - $RUN_STATUS_JSON" | tee -a "$LOG_PATH"
echo "  - $SCORES_CSV" | tee -a "$LOG_PATH"
echo "  - $BOOTSTRAP_CSV" | tee -a "$LOG_PATH"
echo "  - $PLOT_PNG" | tee -a "$LOG_PATH"
if [[ "$SAVE_EDGE_TOPK" == "1" ]]; then
  echo "  - $EDGE_TOPK_JSONL" | tee -a "$LOG_PATH"
  echo "  - $EDGE_TOPK_CHANGE_CSV" | tee -a "$LOG_PATH"
fi
echo "  - $LOG_PATH" | tee -a "$LOG_PATH"
