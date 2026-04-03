# Condition Q-Wise Pipeline

## Purpose

Run condition-specific FV experiments and produce the main representation-analysis inputs.

Conditions commonly include:
- `AAA`
- `BBB`
- `BABA`
- extended runs may add `DDD` and `DADA`

## Main Entry Point

- `scripts/run_condition_qwise_pipeline.py`

## Locked Design Rules

The current stable reading of this pipeline is:
- strict per-q isolation
- deterministic resume gates using config/input/code fingerprints
- default PCA reference is `AAA_ref`
- optional secondary reference is `union_ref`
- `union_ref` head selection is based on rank aggregation, not raw-score averaging
- zero-shot injection, when enabled, must respect build/eval split boundaries

## Query-Predictive Contract

The important token-position contract is:
- StepD patch position
- stage3 vector capture position
- injection add position

must all refer to the same query-predictive token boundary.

## Core Q-Directory Layout

Each q directory is organized around stage folders:
- `_trials`
- `_stepd`
- `_top_heads`
- `_vectors`
- `_pca_common`
- `_fv`
- `_inject`
- `_status`
- `logs`

## Stage Flow

### Stage 0. Trial Generation

Input builder:
- `fv/condition_trials.py`

Outputs:
- `_trials/condition_*.json`

### Stage 1. StepD Per Condition

Producer:
- `scripts/run_stepD_aie_head_sweep.py`

Copied or normalized into:
- `_stepd/aie_scores_<COND>.csv`
- `_stepd/sampled_trials_<COND>.json`
- `_stepd/trial_metrics_<COND>.jsonl`
- `_stepd/stepd_meta_<COND>.json`

### Stage 2. Top-Head Selection

Builds:
- per-condition top heads
- reference sets such as `AAA_ref`
- union reference sets when enabled

Outputs:
- `_top_heads/top_heads_<COND>.json`
- `_top_heads/sets/*.json`
- `_top_heads/diagnostics/*.json`

### Stage 3. Vector Extraction

Outputs:
- `_vectors/trial_vectors_AAA_ref_<COND>.npy`
- `_vectors/trial_vectors_union_ref_<COND>.npy`
- `_vectors/trial_vectors_cond_topk_<COND>.npy`
- `_vectors/trial_vectors_capture_headwise_<COND>.npy`
- `_vectors/vector_extraction_meta.json`

Important current rule:
- summed vectors remain the main runtime artifact
- but capture-headwise vectors are also saved so later post-hoc top-k sweeps can be rebuilt without rerunning model forwards, as long as the needed heads are inside the saved capture set

### Stage 4. PCA

Runner:
- `scripts/run_condition_common_pca.py`

Outputs:
- `_pca_common/<ref_mode>/pca_points.csv`
- `_pca_common/<ref_mode>/pca_centroids.csv`
- `_pca_common/<ref_mode>/distance_summary.json`
- plots and interactive HTML

### Stage 5. FV Build

Outputs:
- `_fv/fv_<COND>.npy`
- `_fv/fv_meta.json`

### Stage 6. Optional Injection

Runner:
- `scripts/run_zero_shot_injection.py`

Outputs:
- `_inject/...` when enabled

Important rule:
- FV-build trials and injection-eval trials should be treated as logically split datasets, not the same sample reused without guardrails.

## Why This Pipeline Matters

This is not just another runtime pipeline.

It is the upstream source for:
- multi-feature reweighting
- inside/outside analysis
- state intervention analysis
- movement and local-tangent summaries

## Sync Behavior

This pipeline can run scratch-first and sync selected artifacts back home.

Important implication:
- home may be a mirror
- scratch may hold the fuller scientific tree

## Historical Source Docs

The old execution-plan layer for this pipeline has been archived.
The main source ideas came from:
- `PLAN_CONDITION_QWISE_PCA_INJECTION_V3.md`

That file is now historical context, not the main entrypoint.
