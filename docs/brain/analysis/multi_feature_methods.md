# Multi-Feature Methods

## Purpose

This document explains how the multi-feature analysis branch is constructed from upstream condition-qwise outputs.

It is the method-side companion to:
- `docs/brain/analysis/multi_feature_theory.md`

## Upstream Dependency

The analysis pipeline depends on condition-qwise artifacts.

Minimum upstream layers:
- `_trials`
- `_stepd`
- `_top_heads`
- `_vectors`
- `_pca_common`

The most important direct input for the stepwise analysis branch is:
- per-q vector artifacts from condition-qwise

## Stage Order

### Stage 1. Condition-Qwise Runtime

Entry:
- `scripts/run_condition_qwise_pipeline.py`

What it contributes:
- condition-specific trials
- StepD score tables
- top-head selections
- vector arrays
- PCA outputs
- FV artifacts

Important outputs:
- `_vectors/trial_vectors_AAA_ref_*.npy`
- `_vectors/trial_vectors_union_ref_*.npy`
- `_top_heads/sets/top_heads_ref_AAA.json`
- `_top_heads/sets/top_heads_ref_union.json`

### Stage 2. Stepwise A-State Extraction

Entry:
- `scripts/extract_stepwise_a_states.py`

Why it exists:
- many earlier artifacts only captured final-query states
- stepwise analysis needs matched A states across the prompt

Core idea:
- do not rerun StepD ranking
- reuse a fixed head pool
- run fresh forwards to extract states at matched A positions

Primary matched slot set:
- `A_demo_1`
- `A_demo_2`
- `A_demo_3`
- `A_demo_4`
- `A_query`

Primary conditions:
- `AAA`
- `BABA`
- `DADA`

Typical outputs:
- `stepwise_a_states_AAA_ref.npz`
- `stepwise_a_states_union_ref.npz`
- `stepwise_a_states_meta.json`
- `stepwise_a_states_manifest.csv`

### Stage 3. Reweighting Metrics

Entry:
- `scripts/compute_stepwise_reweighting_metrics.py`

Main purpose:
- build an A-only basis
- project matched states into that basis
- compute coefficient drift and retention-style metrics

Core objects:
- `G_A`
- `c_t^A`
- `c_t^{BAB}`
- `c_t^{DAD}`
- `Δc_t`

Typical outputs:
- reweighting summary CSVs
- packed NPZ arrays
- meta JSON

Observed artifact family:
- `_analysis_stepwise_reweighting`

Current practical extension:
- a top30 rerun of the same stage is now used when a wider but still readable inside-A feature basis is needed for `Q1`
- this keeps the original exploratory top-5 style outputs intact while exposing a larger top-k coefficient space
- observed artifact family:
  - `_analysis_stepwise_reweighting_top30`

### Stage 4. Endpoint-Aligned Contribution

Main purpose:
- connect feature-level coefficient change to intended B or D endpoint directions

Core objects:
- projected endpoint directions in A-space
- signed total alignment
- feature-level signed contributions

This stage is conceptually layered on top of the stepwise reweighting outputs.

Current practical extension:
- the endpoint-alignment stage can now emit a per-feature-step summary table for top30 runs
- this makes it possible to inspect, for each matched slot, which `g_k` axes actually support or oppose the intended endpoint shift
- observed artifact family:
  - `_analysis_stepwise_endpoint_alignment_top30`

### Stage 4B. g_k Co-Movement

Entry:
- `scripts/compute_stepwise_gk_co_movement.py`

Main purpose:
- summarize which inside-A feature coefficients move together during context-driven reweighting

Core objects:
- top-k `Δc` arrays from the reweighting stage
- flattened trial-step feature correlation
- per-step trial-wise feature correlation

Typical outputs:
- flattened correlation matrices
- per-step correlation arrays
- feature-partner summary CSVs
- top30 vs full-rank coverage summaries

Observed artifact family:
- `_analysis_stepwise_gk_co_movement_top30`

### Stage 4C. Four-Module Drift View

Entry:
- `scripts/compute_stepwise_gk_modules.py`

Main purpose:
- compress the top30 inside-A feature view into a small number of co-moving modules that are easier to interpret mechanistically than individual PCA axes

Core objects:
- feature signature tables built from:
  - endpoint alignment
  - stepwise signed contribution
  - timing summaries
  - branch-specific co-movement
- module score tables
- hard module assignments with confidence margins

Typical outputs:
- feature-signature CSVs
- module-score CSVs
- module-assignment CSVs
- module-level stepwise contribution summaries
- module coupling summaries

Current practical rule:
- the first module-analysis pass is Q1-focused
- the target is an interpretable 4-module decomposition rather than a claim that the latent space has exactly four uniquely natural clusters

Observed artifact family:
- `_analysis_stepwise_gk_modules_top30`

### Stage 5. Inside / Outside Endpoint Decomposition

Entry:
- `scripts/compute_stepwise_inside_outside_endpoint_alignment.py`

Main purpose:
- split change into inside-A and outside-A components
- measure alignment, selectivity, and contribution separately

Typical outputs:
- summary CSVs
- packed NPZ arrays
- meta JSON

Observed artifact family:
- `_analysis_stepwise_inside_outside`

### Stage 6. Inside / Outside Joint Decomposition

Entry:
- `scripts/compute_stepwise_inside_outside_joint.py`

Main purpose:
- resolve intended vs cross components separately inside and outside the A-space

Observed artifact family:
- `_analysis_stepwise_inside_outside_joint`

### Stage 7. Movement / Geometry

Entry:
- `scripts/compute_condition_movement_qwise.py`

Main purpose:
- summarize mean movement geometry across conditions

Typical outputs:
- movement summary CSV
- condition mean NPZ
- manifests

Observed artifact family:
- `_analysis_multi_root`

### Stage 8. Local Tangent / Curvature

Main purpose:
- check whether conclusions depend on a global linear basis

Observed artifact family:
- `_analysis_local_tangent_curvature`

### Stage 9. State Intervention

Entries:
- `scripts/prepare_inside_outside_intervention_vectors.py`
- `scripts/run_q1_inside_outside_intervention.py`

Main purpose:
- manipulate inside/outside components in the extracted state space itself

Important scope boundary:
- this is a strict same-state intervention branch
- it is not the same as low-shot PT steering or generic residual portability tests

Observed artifact family:
- `_analysis_state_intervention`

## Reference Choices

### `AAA_ref`

Use:
- primary interpretation reference
- cleaner A-only baseline

### `union_ref`

Use:
- secondary reference for cross-condition geometry
- useful for PCA and comparison views

### Current Practical Rule

For the main multi-feature claim, `AAA_ref` is the primary reference.
`union_ref` is more naturally a supplementary or robustness view.

## Canonical Artifact Roots

In practice, the main derived analysis roots currently live under scratch:
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting_top30`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_endpoint_alignment_top30`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_inside_outside`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_inside_outside_joint`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_state_intervention`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_multi_root`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_local_tangent_curvature`

These should be treated as derived analysis roots, not as the original condition-runtime source tree.

## Minimal Reading Order

If you need to reconstruct this branch quickly:

1. condition-qwise pipeline
2. stepwise A-state extraction
3. reweighting metrics
4. endpoint-aligned contribution
5. top30 g_k co-movement if a wider interpretable basis view is needed
6. four-module drift view if feature bundles are more interpretable than single axes
7. inside/outside endpoint decomposition
8. movement and local tangent summaries
9. state intervention branch

## Source Specs

The detailed source-spec layer remains:
- `docs/multi_feature_reweighting/TECH_SPEC_STEPWISE_A_STATE_EXTRACTION.md`
- `docs/multi_feature_reweighting/TECH_SPEC_STEPWISE_REWEIGHTING_METRICS.md`
- `docs/multi_feature_reweighting/TECH_SPEC_STEPWISE_ENDPOINT_ALIGNED_CONTRIBUTIONS.md`
- `docs/multi_feature_reweighting/TECH_SPEC_INSIDE_OUTSIDE_INTERVENTION_ROLE_SPLIT.md`
