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

REL_B="${REL_B:-datasets/relation/relationB_ex.csv}"
REL_D="${REL_D:-datasets/relation/relationD_ex.csv}"
ICL_B="${ICL_B:-datasets/relation/icl_B_data.csv}"
ICL_D="${ICL_D:-datasets/relation/icl_D_data.csv}"

SHOT_LIST="${SHOT_LIST:-1,3,5,7,9}"
N_TRIALS="${N_TRIALS:-50}"
SEED="${SEED:-0}"
QID="${QID:-Q1}"
EDGE_TOPK_K="${EDGE_TOPK_K:-10}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

OUT_ROOT="${OUT_ROOT:-/scratch/${USER}/my_fv_project/pt_analysis}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_TAG="${MODEL_TAG:-llama31_70b_bd_shuffle_compare}"
RUN_DIR="${RUN_DIR:-${OUT_ROOT}/${MODEL_TAG}_${RUN_ID}}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/run.log}"
RUN_META_JSON="${RUN_DIR}/run_meta.json"
RUN_STATUS_JSON="${RUN_DIR}/run_status.json"
PROGRESS_JSON="${RUN_DIR}/progress_status.json"
TRIAL_PLAN_JSON="${RUN_DIR}/trial_plan.json"

SCORES_CSV="${RUN_DIR}/bd_shuffle_shot_sweep.csv"
EDGE_TOPK_JSONL="${RUN_DIR}/bd_shuffle_edge_topk.jsonl"
REGIME_METRICS_CSV="${RUN_DIR}/bd_shuffle_regime_metrics.csv"
CASE_DELTAS_CSV="${RUN_DIR}/bd_shuffle_case_deltas.csv"
SIDE_AGG_CSV="${RUN_DIR}/bd_shuffle_side_aggregate.csv"
EDGE_TOPK_AGG_CSV="${RUN_DIR}/bd_shuffle_edge_topk_trial_agg.csv"
SUMMARY_MD="${RUN_DIR}/bd_shuffle_summary.md"

mkdir -p "$RUN_DIR"

die() {
  echo "[BDSHUF][ERROR] $*" | tee -a "$LOG_PATH"
  exit 1
}

[[ -x "$PY" ]] || die "python executable not found: $PY"
[[ -e "$MODEL" ]] || die "MODEL path not found: $MODEL"
[[ -f "$REL_B" ]] || die "missing relation file: $REL_B"
[[ -f "$REL_D" ]] || die "missing relation file: $REL_D"
[[ -f "$ICL_B" ]] || die "missing relation file: $ICL_B"
[[ -f "$ICL_D" ]] || die "missing relation file: $ICL_D"

write_run_meta() {
  "$PY" - "$RUN_META_JSON" "$RUN_DIR" "$OUT_ROOT" "$RUN_ID" "$MODEL_TAG" "$MODEL" "$MODEL_SPEC" "$DEVICE" "${DTYPE:-}" "$QUANT" "$SHOT_LIST" "$N_TRIALS" "$SEED" "${QID:-}" "$EDGE_TOPK_K" "$SCORES_CSV" "$EDGE_TOPK_JSONL" "$REGIME_METRICS_CSV" "$CASE_DELTAS_CSV" "$SIDE_AGG_CSV" "$EDGE_TOPK_AGG_CSV" "$SUMMARY_MD" "$PROGRESS_JSON" "$TRIAL_PLAN_JSON" <<'PY'
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
    seed,
    qid,
    edge_topk_k,
    scores_csv,
    edge_topk_jsonl,
    regime_metrics_csv,
    case_deltas_csv,
    side_agg_csv,
    edge_topk_agg_csv,
    summary_md,
    progress_json,
    trial_plan_json,
) = sys.argv[1:]

payload = {
    "created_at": datetime.utcnow().isoformat() + "Z",
    "run_family": "pt_bd_shuffle_compare",
    "run_id": run_id,
    "model_tag": model_tag,
    "model": model,
    "model_spec": model_spec,
    "device": device,
    "dtype": dtype or None,
    "quant": quant,
    "shot_list": shot_list,
    "n_trials": int(n_trials),
    "seed": int(seed),
    "qid": qid or None,
    "edge_topk_k": int(edge_topk_k),
    "paths": {
        "run_dir": run_dir,
        "out_root": out_root,
        "scores_csv": scores_csv,
        "edge_topk_jsonl": edge_topk_jsonl,
        "regime_metrics_csv": regime_metrics_csv,
        "case_deltas_csv": case_deltas_csv,
        "side_agg_csv": side_agg_csv,
        "edge_topk_agg_csv": edge_topk_agg_csv,
        "summary_md": summary_md,
        "progress_json": progress_json,
        "trial_plan_json": trial_plan_json,
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
  echo "[BDSHUF] ROOT_DIR=$ROOT_DIR"
  echo "[BDSHUF] PY=$PY"
  echo "[BDSHUF] MODEL=$MODEL"
  echo "[BDSHUF] MODEL_SPEC=$MODEL_SPEC DEVICE=$DEVICE DTYPE=$DTYPE QUANT=$QUANT"
  echo "[BDSHUF] SHOT_LIST=$SHOT_LIST N_TRIALS=$N_TRIALS SEED=$SEED QID=$QID EDGE_TOPK_K=$EDGE_TOPK_K"
  echo "[BDSHUF] PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
  echo "[BDSHUF] RUN_DIR=$RUN_DIR"
  echo "[BDSHUF] SCORES_CSV=$SCORES_CSV"
  echo "[BDSHUF] EDGE_TOPK_JSONL=$EDGE_TOPK_JSONL"
  echo "[BDSHUF] REGIME_METRICS_CSV=$REGIME_METRICS_CSV"
  echo "[BDSHUF] CASE_DELTAS_CSV=$CASE_DELTAS_CSV"
  echo "[BDSHUF] SIDE_AGG_CSV=$SIDE_AGG_CSV"
  echo "[BDSHUF] EDGE_TOPK_AGG_CSV=$EDGE_TOPK_AGG_CSV"
  echo "[BDSHUF] SUMMARY_MD=$SUMMARY_MD"
  echo "[BDSHUF] PROGRESS_JSON=$PROGRESS_JSON"
} | tee "$LOG_PATH"

if [[ "$QUANT" == "4bit" || "$QUANT" == "8bit" ]]; then
  export BNB_CUDA_VERSION
fi

score_cmd=(
  env PYTHONPATH=. PYTHONUNBUFFERED="$PYTHONUNBUFFERED" "$PY" -u scripts/score_bd_shuffle_behavior.py
  --model "$MODEL"
  --model_spec "$MODEL_SPEC"
  --device "$DEVICE"
  --quant "$QUANT"
  --relationB_ex_path "$REL_B"
  --relationD_ex_path "$REL_D"
  --icl_B_path "$ICL_B"
  --icl_D_path "$ICL_D"
  --shot_list "$SHOT_LIST"
  --n_trials "$N_TRIALS"
  --seed "$SEED"
  --qid "$QID"
  --out_csv "$SCORES_CSV"
  --edge_topk_jsonl "$EDGE_TOPK_JSONL"
  --progress_json "$PROGRESS_JSON"
  --trial_plan_json "$TRIAL_PLAN_JSON"
  --edge_topk_k "$EDGE_TOPK_K"
)
if [[ -n "$DTYPE" ]]; then
  score_cmd+=(--dtype "$DTYPE")
fi

"${score_cmd[@]}" 2>&1 | tee -a "$LOG_PATH"

summary_cmd=(
  env PYTHONPATH=. PYTHONUNBUFFERED="$PYTHONUNBUFFERED" "$PY" -u scripts/build_bd_shuffle_behavior_summary.py
  --in_csv "$SCORES_CSV"
  --edge_topk_jsonl "$EDGE_TOPK_JSONL"
  --regime_metrics_csv "$REGIME_METRICS_CSV"
  --case_deltas_csv "$CASE_DELTAS_CSV"
  --side_aggregate_csv "$SIDE_AGG_CSV"
  --edge_topk_agg_csv "$EDGE_TOPK_AGG_CSV"
  --summary_md "$SUMMARY_MD"
)

"${summary_cmd[@]}" 2>&1 | tee -a "$LOG_PATH"

echo "[BDSHUF] done" | tee -a "$LOG_PATH"
