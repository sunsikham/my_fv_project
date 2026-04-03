# Multi-Feature Reweighting

## Core Question

The main claim in this analysis track is:
- condition effects do not simply replace A with one B-only or D-only axis
- instead, multiple A-related features remain active while their weights are rebalanced stepwise

## Read Order

Use this order:
1. `multi_feature_theory.md`
2. `multi_feature_methods.md`
3. this overview page

This page is the short map.
The theory page states the current interpretation.
The methods page states how the analysis artifacts are constructed.

## Current Safest Interpretation

The strongest current reading is:
- a substantial part of the original A-structure remains
- context changes many feature weights rather than collapsing representation to one axis
- the resulting signed changes become increasingly aligned with the intended B or D endpoint
- inside-A structure appears to be the main carrier
- outside-A change exists, but is better read as a smaller selective support component than as the main mechanism

Current Q1-focused extension:
- the wider top30 inside-A view suggests that the mechanistically useful unit may be a co-moving module of features rather than a single PCA axis
- in that reading, context reweights latent inside-A module priority, not just isolated feature coefficients

In short:
- not one-axis replacement
- not pure noise outside A
- mostly structured multi-feature reweighting with endpoint-aligned accumulation

For the more explicit interpretation statement, see:
- `docs/brain/analysis/multi_feature_theory.md`

## Primary Source Notes

Main note set:
- `docs/multi_feature_reweighting/`

Important source documents:
- `PLAN_MULTI_FEATURE_REWEIGHTING.md`
- `SUMMARY_WHAT_WE_HAVE_SO_FAR.md`
- `TECH_SPEC_STEPWISE_A_STATE_EXTRACTION.md`
- `TECH_SPEC_STEPWISE_REWEIGHTING_METRICS.md`

## Upstream Dependency

This analysis depends on condition q-wise outputs, especially:
- `_vectors`
- `_pca_common`
- `_stepwise_a_states`

The practical dependency chain is:
- condition q-wise runtime builds q-level vector artifacts
- stepwise extraction rebuilds A-state trajectories
- reweighting and inside/outside scripts derive analysis arrays from those trajectories

For the stage-by-stage construction path, see:
- `docs/brain/analysis/multi_feature_methods.md`

## Main Script Chain

Typical chain:
- `scripts/extract_stepwise_a_states.py`
- `scripts/compute_stepwise_reweighting_metrics.py`
- `scripts/compute_stepwise_endpoint_aligned_contribs.py`
- `scripts/compute_stepwise_gk_co_movement.py`
- `scripts/compute_stepwise_inside_outside_endpoint_alignment.py`
- `scripts/compute_stepwise_inside_outside_joint.py`
- `scripts/compute_condition_movement_qwise.py`

Related intervention/report scripts extend this chain further.

## Main Analysis Layers

### 1. Stepwise State Extraction

Question:
- what are the matched A states across demos and final query?

Typical outputs:
- `stepwise_a_states_*.npz`
- stepwise manifests
- extraction metadata

### 2. Reweighting Metrics

Question:
- how do coefficients change relative to the A-only basis?

Typical outputs:
- reweighting summary CSVs
- packed coefficient and projection arrays

### 3. Inside / Outside Decomposition

Question:
- how much of the change is inside the A-space vs outside it?
- how aligned is each component with intended endpoints?

Typical outputs:
- endpoint-alignment summaries
- inside/outside NPZ arrays
- joint decomposition summaries

### 3B. Wider g_k View

Question:
- if top-5 is too narrow, which larger inside-A feature set carries the endpoint-aligned shift?
- which `g_k` axes move together as a bundle under context?

Typical outputs:
- top30 reweighting reruns
- per-feature-step endpoint contribution tables
- top30 coefficient co-movement matrices
- top30 vs full-rank coverage summaries

### 3C. Four-Module View

Question:
- are the co-moving top30 `g_k` axes better described as a small set of latent inside-A modules?
- does module-level priority reallocation explain the drift more cleanly than axis-by-axis reading?

Typical outputs:
- feature signature tables
- module score tables
- hard module assignments
- module-level stepwise contribution summaries
- module coupling summaries

### 4. Movement / Geometry

Question:
- how do mixed-condition means move in geometric terms?
- does residual structure survive local tangent or curvature checks?

Typical outputs:
- movement summary CSVs
- local tangent summary CSVs
- figure artifacts

### 5. State Intervention

Question:
- what happens if inside/outside components are added or removed at the state level?

Typical outputs:
- intervention vectors
- synthetic state bundles
- intervention summary CSVs
- markdown reports

## Main Artifact Roots

Current practical analysis roots:
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting_top30`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_endpoint_alignment_top30`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_inside_outside`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_inside_outside_joint`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_multi_root`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_local_tangent_curvature`

## Typical Outputs

- summary CSVs
- analysis meta JSONs
- packed NPZ arrays
- movement or curvature figures
- intervention reports

Typical scratch analysis families currently observed:
- `_analysis_stepwise_reweighting`
- `_analysis_stepwise_reweighting_top30`
- `_analysis_stepwise_endpoint_alignment_top30`
- `_analysis_stepwise_gk_co_movement_top30`
- `_analysis_stepwise_gk_modules_top30`
- `_analysis_stepwise_inside_outside`
- `_analysis_stepwise_inside_outside_joint`
- `_analysis_state_intervention`
- `_analysis_multi_root`
- `_analysis_local_tangent_curvature`

## Interpretation Rule

Treat these outputs as derived analysis built on top of canonical condition-runtime artifacts.

## Primary Source Corpus

If you need detailed theory and method background, the current source corpus is still:
- `docs/multi_feature_reweighting/PLAN_MULTI_FEATURE_REWEIGHTING.md`
- `docs/multi_feature_reweighting/SUMMARY_WHAT_WE_HAVE_SO_FAR.md`
- `docs/multi_feature_reweighting/TECH_SPEC_STEPWISE_A_STATE_EXTRACTION.md`
- `docs/multi_feature_reweighting/TECH_SPEC_STEPWISE_REWEIGHTING_METRICS.md`
- `docs/multi_feature_reweighting/TECH_SPEC_STEPWISE_ENDPOINT_ALIGNED_CONTRIBUTIONS.md`
- `docs/multi_feature_reweighting/TECH_SPEC_INSIDE_OUTSIDE_INTERVENTION_ROLE_SPLIT.md`
