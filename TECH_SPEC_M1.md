# TECH_SPEC_M1.md

## Scope

### In (M1 only)
- Generate golden artifacts from `src/` using fixed trials:
- `mean head activations` + `dummy labels` + `indirect effect`
- Normalize outputs into the canonical `run_dir`.

### Out (explicitly excluded)
- `fv/` core implementation parity work (`M2+`)
- Injection parity (`M3+`)
- End-to-end parity suite (`M4+`)
- Model family expansion (`M5+`)

## Canonical Run Directory (SPEC-compliant)

- `dataset_name`: `antonym`
- `fixed_trials_id`: `fixed_trials_antonym_t10_s10_seed0` (fixed trials filename stem)
- Canonical `run_dir`:
- `results/antonym/fixed_trials_antonym_t10_s10_seed0/`

## Required Golden Files (final state in canonical run_dir)

- `antonym_mean_head_activations_FIXED.pt`
- `antonym_dummy_labels.json`
- `antonym_indirect_effect.pt`

## Clean vs Corrupted Regime (code-grounded)

- Mean head activations must use `prompt_data_clean`.
- Evidence:
- `src/compute_average_activations.py:83` calls `get_mean_head_activations_from_fixed_trials(..., mode="clean", ...)`.
- `src/utils/extract_utils.py:120` defines `get_mean_head_activations_from_fixed_trials`.
- `src/utils/extract_utils.py:140` maps `mode="clean"` to `data_key="prompt_data_clean"`.

- Indirect effect must use `prompt_data_corrupted`.
- Evidence:
- `src/compute_indirect_effect.py:109` defines `compute_indirect_effect(...)`.
- `src/compute_indirect_effect.py:133` reads first trial `prompt_data_corrupted` for setup.
- `src/compute_indirect_effect.py:184` uses each trial `prompt_data_corrupted` during the loop.

## Execution Commands (from actual `--help` flags)

All commands are from repo root and use `.venv` + `PYTHONPATH=.`.

```bash
export DATASET_NAME=antonym
export FIXED_TRIALS_PATH=datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json
export FIXED_TRIALS_ID=fixed_trials_antonym_t10_s10_seed0
export CANONICAL_RUN_DIR=results/${DATASET_NAME}/${FIXED_TRIALS_ID}
```

### 1) Mean head activations + dummy labels (`src/compute_average_activations.py`)

```bash
PYTHONPATH=. .venv/bin/python src/compute_average_activations.py \
  --dataset_name "${DATASET_NAME}" \
  --model_name gpt2 \
  --device cpu \
  --seed 42 \
  --n_shots 10 \
  --n_trials 5 \
  --fixed_trials_path "${FIXED_TRIALS_PATH}" \
  --save_path_root results
```

Expected producer path from script behavior:
- `results/antonym/antonym_mean_head_activations_FIXED.pt`
- `results/antonym/antonym_dummy_labels.json`

### 2) Indirect effect (`src/compute_indirect_effect.py`)

```bash
PYTHONPATH=. .venv/bin/python src/compute_indirect_effect.py \
  --dataset_name "${DATASET_NAME}" \
  --model_name gpt2 \
  --device cpu \
  --seed 42 \
  --n_shots 10 \
  --n_trials 5 \
  --fixed_trials_path "${FIXED_TRIALS_PATH}" \
  --save_path_root results \
  --last_token_only True
```

Expected producer path from script behavior:
- `results/antonym/antonym_indirect_effect.pt`

## Script Output Differences and Normalization Step

### Why normalization is needed

- Script default folders differ for fixed trials:
- `compute_average_activations.py` requires `--dataset_name` and writes under `<save_path_root>/<dataset_name>` (`src/compute_average_activations.py:23`, `src/compute_average_activations.py:43`).
- `compute_indirect_effect.py` can run with only `--fixed_trials_path`; then `dataset_name` defaults to fixed-trials stem (`src/compute_indirect_effect.py:255` to `src/compute_indirect_effect.py:256`) and writes under `<save_path_root>/<dataset_name>` (`src/compute_indirect_effect.py:267`), which can diverge from canonical naming intent.

### Normalization rule (copy principle)

After generation, copy artifacts into canonical run_dir:

```bash
mkdir -p "${CANONICAL_RUN_DIR}"
cp results/antonym/antonym_mean_head_activations_FIXED.pt "${CANONICAL_RUN_DIR}/antonym_mean_head_activations_FIXED.pt"
cp results/antonym/antonym_dummy_labels.json "${CANONICAL_RUN_DIR}/antonym_dummy_labels.json"
cp results/antonym/antonym_indirect_effect.pt "${CANONICAL_RUN_DIR}/antonym_indirect_effect.pt"
```

## DoD (Definition of Done)

- Canonical run_dir contains exactly the 3 required files.
- Tensor loads and shapes are recorded with reproducible commands/logs.
- Mean activations shape is confirmed as:
- `(n_layers, n_heads, n_slots, head_dim)`
- Indirect effect shape is confirmed as common case:
- `(n_trials, n_layers, n_heads)` when `--last_token_only True`.

Shape check command:

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
import torch
run_dir = "results/antonym/fixed_trials_antonym_t10_s10_seed0"
mean = torch.load(f"{run_dir}/antonym_mean_head_activations_FIXED.pt", map_location="cpu")
ie = torch.load(f"{run_dir}/antonym_indirect_effect.pt", map_location="cpu")
print("mean_head_activations shape:", tuple(mean.shape))
print("indirect_effect shape:", tuple(ie.shape))
PY
```

## Troubleshooting (5 common failures)

1. Output folder mismatch
- Symptom: expected files not found in canonical run_dir.
- Check: inspect producer locations first (`results/antonym/...` or `results/<fixed_trials_id>/...`) then apply normalization copy.

2. Tokenizer/model mismatch
- Symptom: unexpected shape or downstream parity drift.
- Check: ensure identical `--model_name` across both scripts and later parity runners (e.g., all `gpt2`).

3. Device/dtype inconsistency
- Symptom: unstable values across runs.
- Check: force `--device cpu`; run parity in `float32` contract environment.

4. Fixed trials JSON key missing
- Symptom: runtime `ValueError` for missing keys.
- Check: required keys include `trials`, `prompt_data_clean`, `prompt_data_corrupted`, `target_first_token_id`.

5. Cache/dependency issues
- Symptom: import/model loading failures.
- Check: use `.venv/bin/python`, verify dependencies in venv, and re-check model cache availability.

## Expected Files Tree

```text
results/
  antonym/
    fixed_trials_antonym_t10_s10_seed0/
      antonym_mean_head_activations_FIXED.pt
      antonym_dummy_labels.json
      antonym_indirect_effect.pt
```
