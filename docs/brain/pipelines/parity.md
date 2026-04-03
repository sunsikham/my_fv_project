# Parity Pipeline

## Purpose

The parity pipeline proves that `fv` reproduces the reference `src` behavior on fixed trials.

## Inputs

Main input:
- `datasets/fixed_trials/*.json`

Primary examples in this repo:
- `datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json`
- `datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json`

## Stage Flow

### Stage 1. Generate Golden Artifacts

Entry:
- `scripts/run_m1_golden_artifacts.py`

Underlying producers:
- `src/compute_average_activations.py`
- `src/compute_indirect_effect.py`

Outputs:
- `results/<dataset>/<fixed_trials_id>/<dataset>_mean_head_activations_FIXED.pt`
- `results/<dataset>/<fixed_trials_id>/<dataset>_dummy_labels.json`
- `results/<dataset>/<fixed_trials_id>/<dataset>_indirect_effect.pt`

### Stage 2. Prompt Parity

Entry:
- `scripts/verify_prompt_parity.py`

Checks:
- prompt string reconstruction
- tokenized `input_ids` parity vs fixed trials

### Stage 3. Slot Parity

Entry:
- `scripts/verify_slot_parity_against_src.py`

Checks:
- dummy labels
- token meta labels
- `idx_map`
- `idx_avg`

### Stage 4. FV Parity

Entry:
- `scripts/verify_fv_parity.py`

Checks:
- `top_heads`
- final `function_vector`

### Stage 5. Injection Parity

Entry:
- `scripts/verify_injection_parity.py`

Checks:
- clean logprob
- intervened logprob
- delta logprob

### Stage 6. Unified Suite

Entry:
- `scripts/run_parity_suite.py`

This orchestrates:
- prompt
- slot
- fv
- injection

## Main Outputs

Golden artifacts:
- `results/<dataset>/<fixed_trials_id>/...`

Verification artifacts:
- `results_fv/<dataset>/<fixed_trials_id>/parity_suite_report.json`
- `results_fv/<dataset>/<fixed_trials_id>/parity_suite_stages.csv`
- `results_fv/<dataset>/<fixed_trials_id>/injection_parity_report.json`
- `results_fv/<dataset>/<fixed_trials_id>/injection_parity_trials.csv`

## Interpretation

This pipeline is the semantic gate for the project.

It is not the whole project, but it is the base contract for any later FV runtime claim.

