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
RUN_STAGE="${RUN_STAGE:-standard_full}"

REL_A="${REL_A:-datasets/relation/relationA_ex.csv}"
REL_B="${REL_B:-datasets/relation/relationB_ex.csv}"
REL_D="${REL_D:-datasets/relation/relationD_ex.csv}"
ICL_B="${ICL_B:-datasets/relation/icl_B_data.csv}"
ICL_D="${ICL_D:-datasets/relation/icl_D_data.csv}"

SHOT_LIST="${SHOT_LIST:-0,1,3,5,7,9}"
N_TRIALS="${N_TRIALS:-50}"
N_BOOT="${N_BOOT:-10000}"
SEED="${SEED:-0}"
QID="${QID:-}"
FAMILY_IDS="${FAMILY_IDS:-BASE_ABD,CTX_ABD,ZERO_CTRL,A_ONLY}"
SELECTED_TARGETS_JSON="${SELECTED_TARGETS_JSON:-}"
FORCED_SELECTED_TARGETS_JSON="${FORCED_SELECTED_TARGETS_JSON:-}"
SAVE_EDGE_TOPK="${SAVE_EDGE_TOPK:-1}"
EDGE_TOPK_K_DEFAULT="10"
if [[ "$RUN_STAGE" == "cache_build_only" ]]; then
  EDGE_TOPK_K_DEFAULT="20"
fi
EDGE_TOPK_K="${EDGE_TOPK_K:-$EDGE_TOPK_K_DEFAULT}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

OUT_ROOT="${OUT_ROOT:-/scratch/${USER}/my_fv_project/pt_analysis}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_TAG="${MODEL_TAG:-llama31_70b_unified_pt}"
RUN_DIR="${RUN_DIR:-${OUT_ROOT}/${MODEL_TAG}_${RUN_ID}}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/run.log}"
RUN_META_JSON="${RUN_DIR}/run_meta.json"
RUN_STATUS_JSON="${RUN_DIR}/run_status.json"

SCORES_CSV="${RUN_DIR}/pt_unified_shot_sweep.csv"
ELIGIBILITY_CSV="${RUN_DIR}/pt_unified_family_eligibility.csv"
BOOTSTRAP_CSV="${RUN_DIR}/pt_unified_bootstrap_summary.csv"
EDGE_TOPK_JSONL="${RUN_DIR}/pt_unified_edge_topk.jsonl"
EDGE_TOPK_CHANGE_CSV="${RUN_DIR}/pt_unified_edge_topk_change_summary.csv"
REPORT_DIR="${RUN_DIR}/human_report"

SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-}"
SOURCE_SWEEP_CSV="${SOURCE_SWEEP_CSV:-}"
SOURCE_ELIGIBILITY_CSV="${SOURCE_ELIGIBILITY_CSV:-}"
SOURCE_TOPK_JSONL="${SOURCE_TOPK_JSONL:-}"

mkdir -p "$RUN_DIR"

die() {
  echo "[PTUNI][ERROR] $*" | tee -a "$LOG_PATH"
  exit 1
}

[[ -x "$PY" ]] || die "python executable not found: $PY"
if [[ "$RUN_STAGE" == "standard_full" || "$RUN_STAGE" == "cache_build_only" ]]; then
  [[ -e "$MODEL" ]] || die "MODEL path not found: $MODEL"
  [[ -f "$REL_A" ]] || die "missing relation file: $REL_A"
  [[ -f "$REL_B" ]] || die "missing relation file: $REL_B"
  [[ -f "$REL_D" ]] || die "missing relation file: $REL_D"
  [[ -f "$ICL_B" ]] || die "missing relation file: $ICL_B"
  [[ -f "$ICL_D" ]] || die "missing relation file: $ICL_D"
fi
if [[ "$RUN_STAGE" == "offline_recompute" ]]; then
  [[ -n "$SOURCE_RUN_DIR" ]] || die "SOURCE_RUN_DIR is required for RUN_STAGE=offline_recompute"
  [[ -n "$SELECTED_TARGETS_JSON" ]] || die "SELECTED_TARGETS_JSON is required for RUN_STAGE=offline_recompute"
fi

write_run_meta() {
  "$PY" - "$RUN_META_JSON" "$RUN_DIR" "$OUT_ROOT" "$RUN_ID" "$MODEL_TAG" "$MODEL" "$MODEL_SPEC" "$DEVICE" "${DTYPE:-}" "$QUANT" "$SHOT_LIST" "$N_TRIALS" "$N_BOOT" "$SEED" "${QID:-}" "$FAMILY_IDS" "${SELECTED_TARGETS_JSON:-}" "${FORCED_SELECTED_TARGETS_JSON:-}" "$SAVE_EDGE_TOPK" "$EDGE_TOPK_K" "$SCORES_CSV" "$ELIGIBILITY_CSV" "$BOOTSTRAP_CSV" "$EDGE_TOPK_JSONL" "$EDGE_TOPK_CHANGE_CSV" "$REPORT_DIR" "$RUN_STAGE" "${SOURCE_RUN_DIR:-}" "${SOURCE_SWEEP_CSV:-}" "${SOURCE_ELIGIBILITY_CSV:-}" "${SOURCE_TOPK_JSONL:-}" <<'PY'
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
    seed,
    qid,
    family_ids,
    selected_targets_json,
    forced_selected_targets_json,
    save_edge_topk,
    edge_topk_k,
    scores_csv,
    eligibility_csv,
    bootstrap_csv,
    edge_topk_jsonl,
    edge_topk_change_csv,
    report_dir,
    run_stage,
    source_run_dir,
    source_sweep_csv,
    source_eligibility_csv,
    source_topk_jsonl,
) = sys.argv[1:]

payload = {
    "created_at": datetime.utcnow().isoformat() + "Z",
    "run_family": "pt_unified",
    "run_stage": run_stage,
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
    "seed": int(seed),
    "qid": qid or None,
    "family_ids": family_ids or None,
    "selected_targets_json": selected_targets_json or None,
    "forced_selected_targets_json": forced_selected_targets_json or None,
    "save_edge_topk": int(save_edge_topk),
    "edge_topk_k": int(edge_topk_k),
    "source_run_dir": source_run_dir or None,
    "source_sweep_csv": source_sweep_csv or None,
    "source_eligibility_csv": source_eligibility_csv or None,
    "source_topk_jsonl": source_topk_jsonl or None,
    "paths": {
        "run_dir": run_dir,
        "out_root": out_root,
        "scores_csv": scores_csv,
        "eligibility_csv": eligibility_csv,
        "bootstrap_csv": bootstrap_csv,
        "edge_topk_jsonl": edge_topk_jsonl,
        "edge_topk_change_csv": edge_topk_change_csv,
        "report_dir": report_dir,
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
  echo "[PTUNI] ROOT_DIR=$ROOT_DIR"
  echo "[PTUNI] PY=$PY"
  echo "[PTUNI] MODEL=$MODEL"
  echo "[PTUNI] MODEL_SPEC=$MODEL_SPEC DEVICE=$DEVICE DTYPE=$DTYPE QUANT=$QUANT"
  echo "[PTUNI] RUN_STAGE=$RUN_STAGE"
  echo "[PTUNI] SHOT_LIST=$SHOT_LIST N_TRIALS=$N_TRIALS N_BOOT=$N_BOOT SEED=$SEED QID=${QID:-ALL}"
  echo "[PTUNI] FAMILY_IDS=$FAMILY_IDS"
  echo "[PTUNI] SELECTED_TARGETS_JSON=${SELECTED_TARGETS_JSON:-NONE}"
  echo "[PTUNI] FORCED_SELECTED_TARGETS_JSON=${FORCED_SELECTED_TARGETS_JSON:-NONE}"
  echo "[PTUNI] SOURCE_RUN_DIR=${SOURCE_RUN_DIR:-NONE}"
  echo "[PTUNI] SOURCE_SWEEP_CSV=${SOURCE_SWEEP_CSV:-NONE}"
  echo "[PTUNI] SOURCE_ELIGIBILITY_CSV=${SOURCE_ELIGIBILITY_CSV:-NONE}"
  echo "[PTUNI] SOURCE_TOPK_JSONL=${SOURCE_TOPK_JSONL:-NONE}"
  echo "[PTUNI] SAVE_EDGE_TOPK=$SAVE_EDGE_TOPK EDGE_TOPK_K=$EDGE_TOPK_K"
  echo "[PTUNI] PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
  echo "[PTUNI] RUN_DIR=$RUN_DIR"
  echo "[PTUNI] SCORES_CSV=$SCORES_CSV"
  echo "[PTUNI] ELIGIBILITY_CSV=$ELIGIBILITY_CSV"
  echo "[PTUNI] BOOTSTRAP_CSV=$BOOTSTRAP_CSV"
  echo "[PTUNI] REPORT_DIR=$REPORT_DIR"
  if [[ "$SAVE_EDGE_TOPK" == "1" ]]; then
    echo "[PTUNI] EDGE_TOPK_JSONL=$EDGE_TOPK_JSONL"
    echo "[PTUNI] EDGE_TOPK_CHANGE_CSV=$EDGE_TOPK_CHANGE_CSV"
  fi
} | tee "$LOG_PATH"

if [[ "$QUANT" == "4bit" || "$QUANT" == "8bit" ]]; then
  export BNB_CUDA_VERSION
fi

run_score_stage() {
  local -a score_cmd=(
    env PYTHONPATH=. PYTHONUNBUFFERED="$PYTHONUNBUFFERED" "$PY" -u scripts/score_cross_relation_unified_drift_control.py
    --model "$MODEL"
    --model_spec "$MODEL_SPEC"
    --device "$DEVICE"
    --quant "$QUANT"
    --relationA_ex_path "$REL_A"
    --relationB_ex_path "$REL_B"
    --relationD_ex_path "$REL_D"
    --icl_B_path "$ICL_B"
    --icl_D_path "$ICL_D"
    --shot_list "$SHOT_LIST"
    --n_trials "$N_TRIALS"
    --seed "$SEED"
    --family_ids "$FAMILY_IDS"
    --out_csv "$SCORES_CSV"
    --eligibility_csv "$ELIGIBILITY_CSV"
  )
  if [[ -n "$DTYPE" ]]; then
    score_cmd+=(--dtype "$DTYPE")
  fi
  if [[ -n "$QID" ]]; then
    score_cmd+=(--qid "$QID")
  fi
  if [[ -n "$SELECTED_TARGETS_JSON" ]]; then
    score_cmd+=(--selected_targets_json "$SELECTED_TARGETS_JSON")
  fi
  if [[ -n "$FORCED_SELECTED_TARGETS_JSON" ]]; then
    score_cmd+=(--forced_selected_targets_json "$FORCED_SELECTED_TARGETS_JSON")
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
}

run_bootstrap_stage() {
  local -a bootstrap_cmd=(
    env PYTHONPATH=. PYTHONUNBUFFERED="$PYTHONUNBUFFERED" "$PY" -u scripts/compute_product_test_bootstrap_unified.py
    --in_csv "$SCORES_CSV"
    --out_csv "$BOOTSTRAP_CSV"
    --n_boot "$N_BOOT"
    --seed "$SEED"
    --shot_list "$SHOT_LIST"
  )
  if [[ -n "$QID" ]]; then
    bootstrap_cmd+=(--qid "$QID")
  fi
  "${bootstrap_cmd[@]}" 2>&1 | tee -a "$LOG_PATH"
}

run_report_stage() {
  local topk_arg="${1:-}"
  local eligibility_arg="${2:-}"
  local -a report_cmd=(
    env PYTHONPATH=. PYTHONUNBUFFERED="$PYTHONUNBUFFERED" "$PY" -u scripts/build_pt_unified_human_report.py
    --run_dir "$RUN_DIR"
    --out_dir "$REPORT_DIR"
  )
  if [[ -n "$topk_arg" ]]; then
    report_cmd+=(--topk_jsonl "$topk_arg")
  fi
  if [[ -n "$eligibility_arg" ]]; then
    report_cmd+=(--eligibility_csv "$eligibility_arg")
  fi
  "${report_cmd[@]}" 2>&1 | tee -a "$LOG_PATH"
}

run_recompute_stage() {
  local source_sweep="${SOURCE_SWEEP_CSV:-${SOURCE_RUN_DIR}/pt_unified_shot_sweep.csv}"
  local source_topk="${SOURCE_TOPK_JSONL:-${SOURCE_RUN_DIR}/pt_unified_edge_topk.jsonl}"
  local source_eligibility="${SOURCE_ELIGIBILITY_CSV:-${SOURCE_RUN_DIR}/pt_unified_family_eligibility.csv}"
  local -a recompute_cmd=(
    env PYTHONPATH=. PYTHONUNBUFFERED="$PYTHONUNBUFFERED" "$PY" -u scripts/recompute_pt_unified_from_edge_cache.py
    --source_run_dir "$SOURCE_RUN_DIR"
    --selected_targets_json "$SELECTED_TARGETS_JSON"
    --out_csv "$SCORES_CSV"
    --source_sweep_csv "$source_sweep"
    --topk_jsonl "$source_topk"
    --family_ids "$FAMILY_IDS"
    --shot_list "$SHOT_LIST"
  )
  [[ -f "$source_sweep" ]] || die "missing source sweep CSV: $source_sweep"
  [[ -f "$source_topk" ]] || die "missing source top-k JSONL: $source_topk"
  [[ -f "$source_eligibility" ]] || die "missing source eligibility CSV: $source_eligibility"
  if [[ -n "$QID" ]]; then
    recompute_cmd+=(--qid "$QID")
  fi
  cp "$source_eligibility" "$ELIGIBILITY_CSV"
  "${recompute_cmd[@]}" 2>&1 | tee -a "$LOG_PATH"
  run_bootstrap_stage
  run_report_stage "$source_topk" "$ELIGIBILITY_CSV"
}

case "$RUN_STAGE" in
  standard_full)
    run_score_stage
    run_bootstrap_stage
    run_report_stage
    ;;
  cache_build_only)
    run_score_stage
    ;;
  offline_recompute)
    run_recompute_stage
    ;;
  *)
    die "Unsupported RUN_STAGE: $RUN_STAGE"
    ;;
esac

echo "[PTUNI] done" | tee -a "$LOG_PATH"
echo "[PTUNI] outputs:" | tee -a "$LOG_PATH"
echo "  - $RUN_META_JSON" | tee -a "$LOG_PATH"
echo "  - $RUN_STATUS_JSON" | tee -a "$LOG_PATH"
echo "  - $SCORES_CSV" | tee -a "$LOG_PATH"
echo "  - $ELIGIBILITY_CSV" | tee -a "$LOG_PATH"
if [[ "$RUN_STAGE" == "standard_full" || "$RUN_STAGE" == "offline_recompute" ]]; then
  echo "  - $BOOTSTRAP_CSV" | tee -a "$LOG_PATH"
  echo "  - $REPORT_DIR/index.html" | tee -a "$LOG_PATH"
fi
if [[ "$RUN_STAGE" == "standard_full" || "$RUN_STAGE" == "cache_build_only" ]]; then
  if [[ "$SAVE_EDGE_TOPK" == "1" ]]; then
    echo "  - $EDGE_TOPK_JSONL" | tee -a "$LOG_PATH"
    echo "  - $EDGE_TOPK_CHANGE_CSV" | tee -a "$LOG_PATH"
  fi
fi
echo "  - $LOG_PATH" | tee -a "$LOG_PATH"
