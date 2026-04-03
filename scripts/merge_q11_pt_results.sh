#!/usr/bin/env bash
set -euo pipefail

# Merge Q11-only PT score CSV into an existing PT run directory,
# then regenerate bootstrap summary + plot.
#
# Example:
#   bash scripts/merge_q11_pt_results.sh \
#     --run-dir /home/sunsik/my_fv_project/pt_analysis/llama31_70b_20260224_031506 \
#     --src-q11 /tmp/pt_q11_scores.csv

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${RUN_DIR:-$ROOT_DIR/pt_analysis/llama31_70b_20260224_031506}"
SRC_Q11="${SRC_Q11:-/tmp/pt_q11_scores.csv}"
SEED="${SEED:-0}"
SHOT_LIST="${SHOT_LIST:-1,3,5,7,10}"
N_BOOT="${N_BOOT:-10000}"
STAT="${STAT:-mean}"

if [[ -z "${PY:-}" ]]; then
  if [[ -x "/home/sunsik/.venvs/pt442/bin/python" ]]; then
    PY="/home/sunsik/.venvs/pt442/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
  else
    echo "[merge_q11][ERROR] python executable not found" >&2
    exit 1
  fi
fi

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --run-dir PATH     PT run directory (default: $RUN_DIR)
  --src-q11 PATH     Q11 score csv source (default: $SRC_Q11)
  --seed INT         Bootstrap seed (default: $SEED)
  --shot-list STR    Shot list (default: $SHOT_LIST)
  --n-boot INT       Bootstrap iterations (default: $N_BOOT)
  --stat STR         Bootstrap stat: mean|median (default: $STAT)
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --src-q11)
      SRC_Q11="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --shot-list)
      SHOT_LIST="$2"
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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[merge_q11][ERROR] unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

BASE_CSV="$RUN_DIR/pt_5edge_shot_sweep.csv"
BOOT_CSV="$RUN_DIR/pt_bootstrap_summary.csv"
PLOT_PNG="$RUN_DIR/pt_bootstrap_summary.png"
DST_Q11="$RUN_DIR/pt_q11_scores.csv"

[[ -d "$RUN_DIR" ]] || { echo "[merge_q11][ERROR] run dir not found: $RUN_DIR" >&2; exit 1; }
[[ -f "$BASE_CSV" ]] || { echo "[merge_q11][ERROR] missing base csv: $BASE_CSV" >&2; exit 1; }

if [[ -f "$SRC_Q11" ]]; then
  cp "$SRC_Q11" "$DST_Q11"
  echo "[merge_q11] copied q11 csv -> $DST_Q11"
fi
[[ -f "$DST_Q11" ]] || {
  echo "[merge_q11][ERROR] q11 csv not found: $SRC_Q11 (or $DST_Q11)" >&2
  exit 1
}

backup="$RUN_DIR/pt_5edge_shot_sweep.before_q11.$(date +%Y%m%d_%H%M%S).csv"
cp "$BASE_CSV" "$backup"
echo "[merge_q11] backup created: $backup"

RUN_DIR="$RUN_DIR" PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PY" - <<'PY'
import csv
from pathlib import Path

run_dir = Path(__import__("os").environ["RUN_DIR"])
base = run_dir / "pt_5edge_shot_sweep.csv"
q11 = run_dir / "pt_q11_scores.csv"

with base.open("r", encoding="utf-8", newline="") as f:
    r = csv.DictReader(f)
    base_fields = r.fieldnames
    if not base_fields:
        raise SystemExit("base csv has no header")
    rows = [row for row in r if row.get("q_id") != "Q11"]

with q11.open("r", encoding="utf-8", newline="") as f:
    r = csv.DictReader(f)
    q11_fields = r.fieldnames
    if not q11_fields:
        raise SystemExit("q11 csv has no header")
    if q11_fields != base_fields:
        raise SystemExit("header mismatch between base csv and q11 csv")
    q11_rows = list(r)

rows.extend(q11_rows)
rows.sort(
    key=lambda x: (
        x.get("q_id", ""),
        int(x.get("shot", 0)),
        int(x.get("trial_index", 0)),
        x.get("edge", ""),
    )
)

with base.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=base_fields)
    w.writeheader()
    w.writerows(rows)

q11_count = sum(1 for row in rows if row.get("q_id") == "Q11")
print(f"[merge_q11] merged_total_rows={len(rows)}")
print(f"[merge_q11] q11_rows={q11_count}")
PY

PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PY" scripts/compute_product_test_bootstrap.py \
  --in_csv "$BASE_CSV" \
  --out_csv "$BOOT_CSV" \
  --n_boot "$N_BOOT" \
  --seed "$SEED" \
  --shot_list "$SHOT_LIST" \
  --stat "$STAT"

PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PY" scripts/plot_product_test_summary.py \
  --in_csv "$BOOT_CSV" \
  --out_png "$PLOT_PNG" \
  --show_p stars \
  --title "PT summary (llama31_70b)"

echo "[merge_q11] done"
echo "  - $BASE_CSV"
echo "  - $BOOT_CSV"
echo "  - $PLOT_PNG"
