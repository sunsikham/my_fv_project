# Project Map

## What This Repository Is

This repository started as a `src` vs `fv` function-vector parity effort, but it is no longer only that.

The active project now has five connected tracks:
- `core/parity`: prove that `fv` matches the reference `src` path on fixed trials
- `fv-only experiment pipelines`: run q-wise and condition-wise function-vector experiments
- `multi-feature analysis`: analyze how representations move, reweight, and split inside/outside the A-space
- `PT/context-drift evaluation`: run product-test style relation scoring and bootstrap summaries
- `reporting/presentation`: generate plots, HTML summaries, and analysis reports

## Active Roots

Current active roots are:
- code root: `/home/sunsik/my_fv_project`
- large-result root: `/scratch/sunsik/my_fv_project`

This second brain intentionally excludes `/mnt/ebs` from the active map.

## Main Project Backbone

The high-level dependency chain is:

1. `src` defines reference behavior.
2. `fv` re-implements and extends that behavior.
3. `scripts/` orchestrates runnable experiment pipelines.
4. `results/`, `results_fv/`, and `pt_analysis/` store outputs.
5. downstream analysis builds derived artifacts from those outputs.

## Current Tracks

### 1. Core Parity

Purpose:
- freeze semantics for prompts, slots, function vectors, and injection

Main docs:
- `plan.md`
- `spec.md`
- `TECH_SPEC_M0.md` to `TECH_SPEC_M4.md`

Main scripts:
- `scripts/run_m1_golden_artifacts.py`
- `scripts/run_parity_suite.py`

### 2. Relation Q-Wise Runtime

Purpose:
- run q-wise StepD -> StepE -> Step6 FV experiments on relation data

Main docs:
- `TECH_SPEC_RELATION_QWISE.md`

Main scripts:
- `scripts/run_relation_qwise_pipeline.py`
- `scripts/run_qwise_super_fv_pipeline.py`

### 3. Condition Q-Wise Runtime

Purpose:
- run condition-specific trial generation, StepD, vector extraction, PCA, FV build, and optional injection

Main scripts:
- `scripts/run_condition_qwise_pipeline.py`

This track also feeds the main representation-analysis branch.

### 4. Multi-Feature Reweighting

Purpose:
- explain condition effects as structured, stepwise reweighting rather than one-axis replacement

Main doc source:
- `docs/multi_feature_reweighting/`

Main outputs:
- scratch-side `_analysis_*` directories under `results_fv/relation_condition_qwise`

### 5. PT / Context Drift / Unified PT

Purpose:
- score relation families with shot sweeps, bootstrap PT statistics, and generate human-readable reports

Main outputs:
- `/scratch/sunsik/my_fv_project/pt_analysis`

## Working Rule

When documenting this project, do not collapse all work into a single "FV project" bucket.

The minimum stable partition is:
- reference/parity
- runtime pipelines
- representation analysis
- PT evaluation
- storage/reporting

