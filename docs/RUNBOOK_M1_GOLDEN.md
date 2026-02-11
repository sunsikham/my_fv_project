# RUNBOOK: M1 Golden Artifacts (Llama-Only)

## Purpose
Create a reproducible `src` golden baseline for `antonym` using Llama, then gate it with parity checks.

## Scope
- Dataset: `antonym`
- Model family: `Llama-3.1-8B`
- Golden producer: `src/`
- Gate: parity scripts in `scripts/`

## Fixed Profile
- `dataset_json`: `datasets/antonym/raw/antonym.json`
- `model_name` (local snapshot): `/mnt/ebs/.hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- `seed`: `0`
- `device`: `cuda`
- `n_shots`: `10`
- `n_trials`: `10`

Use one fixed profile end-to-end. Do not mix model/tokenizer/profile during one run.

## Prerequisites
```bash
cd /mnt/ebs/my_fv_project
PY=/mnt/ebs/venv/bin/python
MODEL=/mnt/ebs/.hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
FT=datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json
```

## Step 0: Build Llama fixed_trials
```bash
env PYTHONPATH=. $PY src/make_fixed_trials.py \
  --dataset_json datasets/antonym/raw/antonym.json \
  --out_path "$FT" \
  --n_trials 10 \
  --n_shots 10 \
  --seed 0 \
  --model_name_for_tokenizer "$MODEL" \
  --model_prepend_bos true \
  --prepend_bos_token_used false
```

## Step 1: Validate fixed_trials format vs src expectations
```bash
env PYTHONPATH=. $PY scripts/verify_prompt_parity.py \
  --fixed_trials_path "$FT" \
  --max_trials 10 \
  --model_name_for_tokenizer "$MODEL"

env PYTHONPATH=. $PY scripts/verify_slot_parity_against_src.py \
  --fixed_trials_path "$FT" \
  --max_trials 10 \
  --mode corrupted \
  --tokenizer_name "$MODEL" \
  --assert_zero

env PYTHONPATH=. $PY src/make_fixed_trials.py \
  --verify true \
  --out_path "$FT" \
  --verify_n 5
```

Expected:
- `mismatch_count: 0` in prompt/slot checks.
- verify output shows `first_token_id == answer_ids_first`.

## Step 2: Generate M1 golden artifacts (Llama)
```bash
env PYTHONPATH=. $PY scripts/run_m1_golden_artifacts.py \
  --dataset_name antonym \
  --fixed_trials_path "$FT" \
  --model_name "$MODEL" \
  --device cuda \
  --seed 0 \
  --n_shots 10 \
  --n_trials 10 \
  --save_path_root results \
  --python_bin /mnt/ebs/venv/bin/python
```

Expected canonical output dir:
- `results/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b`

Required files:
- `results/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/antonym_mean_head_activations_FIXED.pt`
- `results/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/antonym_dummy_labels.json`
- `results/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/antonym_indirect_effect.pt`

## Step 3: Run parity suite (M0-M3 gate, Llama)
```bash
env PYTHONPATH=. $PY scripts/run_parity_suite.py \
  --dataset_name antonym \
  --fixed_trials_path "$FT" \
  --fixed_trials_id fixed_trials_antonym_t10_s10_seed0_llama31_8b \
  --model_name "$MODEL" \
  --device cuda \
  --seed 0 \
  --n_top_heads 10 \
  --token_class_idx -1 \
  --edit_layer 9 \
  --max_trials 5
```

Expected parity outputs:
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/parity_suite_report.json`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/parity_suite_stages.csv`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/parity_suite.log`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/injection_parity_trials.csv`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/injection_parity_report.json`

## Definition of Done
- Step 1 passes with `mismatch_count == 0`.
- Step 2 creates all 3 golden files in canonical directory.
- Step 3 parity suite overall PASS, and stage mismatch counts are zero.
- Re-running with same profile reproduces the same output paths and comparable metrics.
