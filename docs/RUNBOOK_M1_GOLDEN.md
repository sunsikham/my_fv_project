# RUNBOOK: M1 Golden Artifacts (Active-Root Version)

## Purpose

Create a reproducible `src` golden baseline for `antonym`, then gate it with parity checks.

## Active-Root Assumption

This runbook assumes the active environment described by `docs/brain/`:
- repo root: `/home/sunsik/my_fv_project`
- no `/mnt/ebs` dependency

## Scope

- dataset: `antonym`
- golden producer: `src/`
- gate: parity scripts in `scripts/`
- target output roots:
  - `results/antonym/<fixed_trials_id>/...`
  - `results_fv/antonym/<fixed_trials_id>/...`

## Fixed Profile Template

You must choose one model profile and keep it fixed for all steps.

Recommended shell variables:

```bash
ROOT=/home/sunsik/my_fv_project
PY=${PY:-/home/sunsik/.venvs/pt442/bin/python}
MODEL="${MODEL:-<set-llama-8b-model-path-or-hf-id>}"
FT_REL="datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json"
FT_ID="fixed_trials_antonym_t10_s10_seed0_llama31_8b"

cd "$ROOT"
```

Required profile values:
- `seed=0`
- `device=cuda`
- `n_shots=10`
- `n_trials=10`

## Preconditions

- `MODEL` must point to a valid local snapshot or Hugging Face model id accessible in your environment.
- GPU runtime must be available if you keep `device=cuda`.
- `PY` should be a working environment with project dependencies.

Quick preflight:

```bash
"$PY" -c "import torch; print('cuda_ok', torch.cuda.is_available(), 'count', torch.cuda.device_count())"
```

## Step 0: Build Fixed Trials

```bash
env PYTHONPATH=. "$PY" src/make_fixed_trials.py \
  --dataset_json datasets/antonym/raw/antonym.json \
  --out_path "$FT_REL" \
  --n_trials 10 \
  --n_shots 10 \
  --seed 0 \
  --model_name_for_tokenizer "$MODEL" \
  --model_prepend_bos true \
  --prepend_bos_token_used false
```

## Step 1: Validate Fixed-Trial Format

```bash
env PYTHONPATH=. "$PY" scripts/verify_prompt_parity.py \
  --fixed_trials_path "$FT_REL" \
  --max_trials 10 \
  --model_name_for_tokenizer "$MODEL"

env PYTHONPATH=. "$PY" scripts/verify_slot_parity_against_src.py \
  --fixed_trials_path "$FT_REL" \
  --max_trials 10 \
  --mode corrupted \
  --tokenizer_name "$MODEL" \
  --assert_zero

env PYTHONPATH=. "$PY" src/make_fixed_trials.py \
  --verify true \
  --out_path "$FT_REL" \
  --verify_n 5
```

Expected:
- `mismatch_count: 0` in prompt and slot checks
- fixed-trial verification confirms `first_token_id == answer_ids_first`

## Step 2: Generate Golden Artifacts

```bash
env PYTHONPATH=. "$PY" scripts/run_m1_golden_artifacts.py \
  --dataset_name antonym \
  --fixed_trials_path "$FT_REL" \
  --model_name "$MODEL" \
  --device cuda \
  --seed 0 \
  --n_shots 10 \
  --n_trials 10 \
  --save_path_root results \
  --python_bin "$PY"
```

Expected canonical output dir:
- `results/antonym/$FT_ID`

Required files:
- `results/antonym/$FT_ID/antonym_mean_head_activations_FIXED.pt`
- `results/antonym/$FT_ID/antonym_dummy_labels.json`
- `results/antonym/$FT_ID/antonym_indirect_effect.pt`

## Step 3: Run Parity Suite

```bash
env PYTHONPATH=. "$PY" scripts/run_parity_suite.py \
  --dataset_name antonym \
  --fixed_trials_path "$FT_REL" \
  --fixed_trials_id "$FT_ID" \
  --model_name "$MODEL" \
  --device cuda \
  --seed 0 \
  --n_top_heads 10 \
  --token_class_idx -1 \
  --edit_layer 9 \
  --max_trials 5
```

Expected outputs:
- `results_fv/antonym/$FT_ID/parity_suite_report.json`
- `results_fv/antonym/$FT_ID/parity_suite_stages.csv`
- `results_fv/antonym/$FT_ID/injection_parity_report.json`
- `results_fv/antonym/$FT_ID/injection_parity_trials.csv`

## Definition of Done

- Step 1 passes with `mismatch_count == 0`
- Step 2 creates all three golden files
- Step 3 overall parity status is `PASS`
- rerunning with the same fixed profile reproduces the same logical output locations

## Common Failure Modes

### tokenizer/model mismatch

Symptom:
- prompt or slot parity mismatches

Fix:
- ensure the same `MODEL` is used for fixed-trial tokenization and parity checks

### missing GPU runtime

Symptom:
- model load or parity run fails on CUDA

Fix:
- switch to a GPU node
- or rerun with CPU only if the specific profile permits it

### mixed profile usage

Symptom:
- outputs exist but comparisons drift or become incomparable

Fix:
- keep `MODEL`, `seed`, `n_shots`, `n_trials`, and device profile fixed end-to-end

