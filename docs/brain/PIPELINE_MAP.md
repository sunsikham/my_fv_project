# Pipeline Map

## Overview

The repository has four main execution families:
- parity
- relation q-wise runtime
- condition q-wise runtime
- PT evaluation

The condition q-wise family also feeds the main representation-analysis branch.

## 1. Parity Family

Entry points:
- `scripts/run_m1_golden_artifacts.py`
- `scripts/run_parity_suite.py`

Purpose:
- confirm `fv` matches `src` on fixed trials

Main outputs:
- golden artifacts in `results/...`
- parity reports in `results_fv/antonym/...`

## 2. Relation Q-Wise Family

Entry point:
- `scripts/run_relation_qwise_pipeline.py`

Stage flow:
- StepD head sweep
- heatmap
- StepE top-k FV build
- zero-shot snapshot
- Step6 all-layer injection sweep

Main outputs:
- q-wise trial and Step6 artifacts
- step6 layer summaries
- aggregate comparison and plotting folders

## 3. Condition Q-Wise Family

Entry point:
- `scripts/run_condition_qwise_pipeline.py`

Stage flow:
- stage0 trials
- stage1 StepD per condition
- stage2 top-head selection
- stage3 vector extraction
- stage4 PCA
- stage5 FV build
- stage6 optional injection

Main outputs:
- `_trials`
- `_stepd`
- `_top_heads`
- `_vectors`
- `_pca_common`
- `_fv`
- `_inject`
- `_status`

## 4. Multi-Feature Analysis Family

Upstream dependency:
- condition q-wise outputs

Typical flow:
- extract stepwise A states
- compute reweighting metrics
- compute inside/outside endpoint alignment
- compute joint inside/outside decomposition
- compute movement or local tangent summaries
- generate intervention reports and figures

Main outputs:
- scratch-side `_analysis_*` directories

## 5. PT Family

Subfamilies:
- baseline PT
- context-drift PT
- unified PT

Typical flow:
- scorer
- bootstrap summary
- plot or HTML report

Main outputs:
- scratch-side `pt_analysis/<run_id>/...`

## Practical Reading Order

If you are new to the project, read in this order:

1. `pipelines/parity.md`
2. `pipelines/condition_qwise.md`
3. `analysis/multi_feature_reweighting.md`
4. `pipelines/pt.md`
5. `ARTIFACT_ROOTS.md`

