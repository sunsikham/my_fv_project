# R0 Freeze Snapshot: relation runtime with antonym

- frozen_at_utc: `2026-02-10T16:47:38Z`
- profile: `relation_runtime_with_antonym_debug`

## Fixed Params
- model_path: `/mnt/ebs/.hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- dataset: `antonym`
- fixed_trials_file: `datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json`
- fixed_trials_id: `fixed_trials_antonym_t10_s10_seed0_llama31_8b`
- seed: `0`
- device: `cuda`
- max_trials: `5`
- edit_layer: `9`
- n_top_heads: `10`
- token_class_idx: `-1`

## Input/Code Integrity (sha256)
- `datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json`
  - `959d1657f497567722daa1f0514e7815cfcdb11c2c37f0aaae9bfe8172586350`
- `scripts/run_parity_suite.py`
  - `4e50a3a73873601a849b55a129e18eb259963930a9d2d25e4ce27ec113305b75`
- `scripts/run_step6_fv_injection_eval.py`
  - `4398db0b690d59e733093874eddc50b9bbefb8c08cc05d812c714b36916e136b`
- `src/utils/intervention_utils.py`
  - `1315dab1f6d9c46dc2f912ca0376684a9ba80e6caad89d9771247b8519dbb19d`
- `fv/intervene.py`
  - `636a96a8cbc9885819bc2b87423861aef93c0d327f7ca8f6616ba47599e245e6`

## R1 Baseline Check (current)
- parity report: `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/parity_suite_report.json`
- suite_status: `PASS`
- stage_status:
  - `prompt: PASS`
  - `slot: PASS`
  - `fv: PASS`
  - `injection: PASS`
- injection trial deltas (`delta_src_logprob`, n=5):
  - `[1.583984375, 3.625, 0.279296875, 0.3671875, 2.00390625]`
- mean_delta_src_logprob: `1.571875`

