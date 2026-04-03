#!/usr/bin/env bash
set -euo pipefail

# Re-run PT scoring for one q_id, merge into existing run CSV, and regenerate
# bootstrap summary + plot.
#
# Example:
#   bash scripts/rerun_merge_pt_qid.sh --qid Q17 \
#     --run-dir /home/sunsik/my_fv_project/pt_analysis/llama31_70b_20260224_031506

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${RUN_DIR:-$ROOT_DIR/pt_analysis/llama31_70b_20260224_031506}"
QID="${QID:-}"

MODEL="${MODEL:-/scratch/sunsik/models/Llama-3.1-70B}"
MODEL_SPEC="${MODEL_SPEC:-llama3}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
QUANT="${QUANT:-4bit}"

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

if [[ -z "${PY:-}" ]]; then
  if [[ -x "/home/sunsik/.venvs/pt442/bin/python" ]]; then
    PY="/home/sunsik/.venvs/pt442/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
  else
    echo "[rerun_pt_qid][ERROR] python executable not found" >&2
    exit 1
  fi
fi

usage() {
  cat <<EOF
Usage: $0 --qid QID [options]

Required:
  --qid QID           q_id to re-run (e.g., Q17)

Options:
  --run-dir PATH      PT run directory (default: $RUN_DIR)
  --model PATH        Model path (default: $MODEL)
  --model-spec STR    Model spec (default: $MODEL_SPEC)
  --device STR        Device (default: $DEVICE)
  --dtype STR         Dtype (default: $DTYPE)
  --quant STR         Quant mode (default: $QUANT)
  --shot-list STR     Shots (default: $SHOT_LIST)
  --n-trials INT      Trials per qid (default: $N_TRIALS)
  --n-boot INT        Bootstrap iterations (default: $N_BOOT)
  --stat STR          Bootstrap stat mean|median (default: $STAT)
  --seed INT          Seed (default: $SEED)
  -h, --help          Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qid)
      QID="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --model-spec)
      MODEL_SPEC="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --quant)
      QUANT="$2"
      shift 2
      ;;
    --shot-list)
      SHOT_LIST="$2"
      shift 2
      ;;
    --n-trials)
      N_TRIALS="$2"
      shift 2
      ;;
    --n-boot)
      N_BOOT="$2"
      shift 2
      ;;
    --stat)
      STAT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[rerun_pt_qid][ERROR] unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

[[ -n "$QID" ]] || { echo "[rerun_pt_qid][ERROR] --qid is required" >&2; exit 1; }

BASE_CSV="$RUN_DIR/pt_5edge_shot_sweep.csv"
BOOT_CSV="$RUN_DIR/pt_bootstrap_summary.csv"
PLOT_PNG="$RUN_DIR/pt_bootstrap_summary.png"
QID_CSV="$RUN_DIR/pt_${QID}_scores.csv"

[[ -d "$RUN_DIR" ]] || { echo "[rerun_pt_qid][ERROR] run dir not found: $RUN_DIR" >&2; exit 1; }
[[ -f "$BASE_CSV" ]] || { echo "[rerun_pt_qid][ERROR] missing base csv: $BASE_CSV" >&2; exit 1; }
[[ -f "$REL_A" ]] || { echo "[rerun_pt_qid][ERROR] missing relation file: $REL_A" >&2; exit 1; }
[[ -f "$REL_B" ]] || { echo "[rerun_pt_qid][ERROR] missing relation file: $REL_B" >&2; exit 1; }
[[ -f "$ICL_B" ]] || { echo "[rerun_pt_qid][ERROR] missing relation file: $ICL_B" >&2; exit 1; }
[[ -f "$ICL_C" ]] || { echo "[rerun_pt_qid][ERROR] missing relation file: $ICL_C" >&2; exit 1; }
[[ -f "$ICL_D" ]] || { echo "[rerun_pt_qid][ERROR] missing relation file: $ICL_D" >&2; exit 1; }

echo "[rerun_pt_qid] scoring $QID -> $QID_CSV"
PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PY" scripts/score_cross_relation_target_logit.py \
  --model "$MODEL" \
  --model_spec "$MODEL_SPEC" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --quant "$QUANT" \
  --relationA_ex_path "$REL_A" \
  --relationB_ex_path "$REL_B" \
  --icl_B_path "$ICL_B" \
  --icl_C_path "$ICL_C" \
  --icl_D_path "$ICL_D" \
  --shot_list "$SHOT_LIST" \
  --n_trials "$N_TRIALS" \
  --seed "$SEED" \
  --qid "$QID" \
  --out_csv "$QID_CSV"

backup="$RUN_DIR/pt_5edge_shot_sweep.before_${QID}.$(date +%Y%m%d_%H%M%S).csv"
cp "$BASE_CSV" "$backup"
echo "[rerun_pt_qid] backup created: $backup"

RUN_DIR="$RUN_DIR" QID="$QID" PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PY" - <<'PY'
import csv
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
qid = os.environ["QID"]
base = run_dir / "pt_5edge_shot_sweep.csv"
q_csv = run_dir / f"pt_{qid}_scores.csv"

with base.open("r", encoding="utf-8", newline="") as f:
    r = csv.DictReader(f)
    fields = r.fieldnames
    if not fields:
        raise SystemExit("base csv has no header")
    rows = [row for row in r if row.get("q_id") != qid]

with q_csv.open("r", encoding="utf-8", newline="") as f:
    r = csv.DictReader(f)
    if r.fieldnames != fields:
        raise SystemExit("header mismatch between base csv and qid csv")
    rows.extend(list(r))

rows.sort(
    key=lambda x: (
        x.get("q_id", ""),
        int(x.get("shot", 0)),
        int(x.get("trial_index", 0)),
        x.get("edge", ""),
    )
)

with base.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

qid_rows = sum(1 for row in rows if row.get("q_id") == qid)
print(f"[rerun_pt_qid] merged_total_rows={len(rows)}")
print(f"[rerun_pt_qid] {qid}_rows={qid_rows}")
PY

echo "[rerun_pt_qid] recomputing bootstrap -> $BOOT_CSV"
PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PY" scripts/compute_product_test_bootstrap.py \
  --in_csv "$BASE_CSV" \
  --out_csv "$BOOT_CSV" \
  --n_boot "$N_BOOT" \
  --seed "$SEED" \
  --shot_list "$SHOT_LIST" \
  --stat "$STAT"

echo "[rerun_pt_qid] redraw plot -> $PLOT_PNG"
PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PY" scripts/plot_product_test_summary.py \
  --in_csv "$BOOT_CSV" \
  --out_png "$PLOT_PNG" \
  --show_p stars \
  --title "PT summary (llama31_70b)"

echo "[rerun_pt_qid] done"
echo "  - $BASE_CSV"
echo "  - $BOOT_CSV"
echo "  - $PLOT_PNG"
