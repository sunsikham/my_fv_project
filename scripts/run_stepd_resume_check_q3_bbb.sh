#!/usr/bin/env bash
set -euo pipefail

cd /home/sunsik/my_fv_project

source /home/sunsik/.venvs/pt442/bin/activate
module load cuda/12.9
export LD_LIBRARY_PATH="$EBROOTCUDA/lib64:${LD_LIBRARY_PATH:-}"
export BNB_CUDA_VERSION=122
unset CUDA_HOME
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PYTHON_BIN="/home/sunsik/.venvs/pt442/bin/python"
MODEL="/scratch/sunsik/models/Llama-3.1-70B"

SRC_Q="/home/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__h51e123e95a/Q3"
SRC_RUN="$SRC_Q/_stepd/run_BBB"
TRIAL_JSON="$SRC_Q/_trials/condition_BBB.json"

TEST_BASE="/scratch/$USER/stepd_resume_checks"
TEST_NAME="q3_bbb_resume_check_$(date -u +%Y%m%d_%H%M%S)"
TEST_RUN="$TEST_BASE/$TEST_NAME/run_BBB"
TEST_ART="$TEST_RUN/artifacts"
ORIG_ART="$SRC_RUN/artifacts"

mkdir -p "$TEST_BASE"
rm -rf "$TEST_RUN"
mkdir -p "$TEST_RUN"

cp -a "$SRC_RUN/." "$TEST_RUN/"

rm -f "$TEST_ART/aie_scores.csv"
rm -f "$TEST_ART/aie_scores.json"
rm -f "$TEST_ART/aie_trials.csv"
rm -f "$TEST_ART/perf_stats.json"
rm -rf "$TEST_ART/_resume"

export TEST_ART
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

art = Path(os.environ["TEST_ART"])
trial_path = art / "trial_metrics.jsonl"

rows = []
with trial_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        rows.append(json.loads(line))

seen = []
seen_set = set()
kept = []

for row in rows:
    key = (int(row["layer"]), int(row["head"]))
    if key not in seen_set:
        seen_set.add(key)
        seen.append(key)
    if len(seen) <= 2500:
        kept.append(row)
    else:
        break

with trial_path.open("w", encoding="utf-8") as handle:
    for row in kept:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")

print("kept_completed_heads=", len(seen[:2500]))
print("first_missing_head=", seen[2500])
PY

"$PYTHON_BIN" scripts/run_stepD_aie_head_sweep.py \
  --model "$MODEL" \
  --model_spec llama3 \
  --device cuda \
  --dtype bf16 \
  --quant 4bit \
  --layers 16-55 \
  --heads all \
  --n_trials 25 \
  --n_icl_examples 9 \
  --score_key mean_delta_p \
  --seed 0 \
  --compute_prob_scores 1 \
  --fixed_trials_path "$TRIAL_JSON" \
  --fixed_out_dir "$TEST_RUN"

echo
echo "[compare] aie_scores.csv"
cmp -s "$ORIG_ART/aie_scores.csv" "$TEST_ART/aie_scores.csv" \
  && echo "MATCH" \
  || echo "DIFF"

echo "[compare] trial_metrics.jsonl"
cmp -s "$ORIG_ART/trial_metrics.jsonl" "$TEST_ART/trial_metrics.jsonl" \
  && echo "MATCH" \
  || echo "DIFF"

echo
sha256sum "$ORIG_ART/aie_scores.csv"
sha256sum "$TEST_ART/aie_scores.csv"
sha256sum "$ORIG_ART/trial_metrics.jsonl"
sha256sum "$TEST_ART/trial_metrics.jsonl"

echo
echo "test_run=$TEST_RUN"
