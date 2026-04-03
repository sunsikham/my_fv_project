#!/usr/bin/env bash
set -euo pipefail

cd /home/sunsik/my_fv_project

source /home/sunsik/.venvs/pt442/bin/activate

PY="/home/sunsik/.venvs/pt442/bin/python"

QID="Q1"
REF="AAA_ref"
BASIS_SCOPE="matched"
SLOT_NAME="A_query"
ALPHAS="0.0,0.5,1.0,1.5"
MODES="trial_exact,mean_vector"

STEPWISE_ROOT="/home/sunsik/my_fv_project/results_fv/relation_condition_qwise"
REWEIGHT_NPZ="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting/stepwise_reweighting_arrays_AAA_ref.npz"
INSIDE_OUTSIDE_NPZ="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_inside_outside/stepwise_inside_outside_endpoint_arrays.npz"

OUT_DIR="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_state_intervention"
VEC_NPZ="$OUT_DIR/inside_outside_state_intervention_vectors_Q1.npz"
RAW_CSV="$OUT_DIR/inside_outside_state_intervention_raw_rows_Q1.csv"
SUMMARY_CSV="$OUT_DIR/inside_outside_state_intervention_summary_Q1.csv"
STATES_NPZ="$OUT_DIR/inside_outside_state_intervention_synthetic_states_Q1.npz"
REPORT_MD="$OUT_DIR/inside_outside_state_intervention_report_Q1.md"

mkdir -p "$OUT_DIR"

echo "[io-state-intv] qid=$QID ref=$REF basis_scope=$BASIS_SCOPE slot=$SLOT_NAME"
echo "[io-state-intv] alphas=$ALPHAS modes=$MODES"
echo "[io-state-intv] out_dir=$OUT_DIR"

"$PY" scripts/run_q1_inside_outside_intervention.py \
  --qid "$QID" \
  --ref "$REF" \
  --basis_scope "$BASIS_SCOPE" \
  --slot_name "$SLOT_NAME" \
  --mode_list "$MODES" \
  --alpha_list "$ALPHAS" \
  --stepwise_root "$STEPWISE_ROOT" \
  --reweight_npz "$REWEIGHT_NPZ" \
  --inside_outside_npz "$INSIDE_OUTSIDE_NPZ" \
  --out_dir "$OUT_DIR" \
  --out_vectors_npz "$VEC_NPZ" \
  --out_csv "$RAW_CSV" \
  --out_summary_csv "$SUMMARY_CSV" \
  --out_states_npz "$STATES_NPZ" \
  --out_report_md "$REPORT_MD"
