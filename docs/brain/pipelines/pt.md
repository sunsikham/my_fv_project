# PT Pipelines

## Purpose

The PT family evaluates relation behavior with shot sweeps, bootstrap statistics, lexical traces, and human-readable reports.

This is a separate evaluation track from the main FV runtime pipelines.

## Canonical Storage

Current practical canonical root:
- `/scratch/sunsik/my_fv_project/pt_analysis`

## PT Subfamilies

### 1. Baseline PT

Runner:
- `scripts/run_pt_llama70b.sh`

Core scorer:
- `scripts/score_cross_relation_target_logit.py`

Post-processing:
- `scripts/compute_product_test_bootstrap.py`
- `scripts/plot_product_test_summary.py`

Typical outputs:
- `pt_5edge_shot_sweep.csv`
- `pt_bootstrap_summary.csv`
- `pt_bootstrap_summary.png`
- optional edge top-k traces

Current practical note:
- baseline PT trial plans now sample a demo bundle of size `max(shot_list)` once per trial and then use shot-specific prefixes from that shared ordering
- this replaces the older practical assumption of a fixed 10-demo bundle, so qids with only 9 valid demo rows can still be recovered when the run uses `shot_list` whose maximum is 9
- a dedicated recovery wrapper exists at `scripts/run_pt_q6_9shot_topk_recovery.sh` for the known `Q6` 9-shot candidate collection case
- when building manual selected-target review inputs for `C` queries, it is valid to save lexical top-k traces for all five edges `AB, AC, AD, BC, BD`
- `scripts/build_pt_valid_answer_scaffold.py` can now consume 5-edge top-k JSONL directly in addition to unified PT top-k JSONL
- the scaffold builder also supports an optional seed selected-target artifact so previously reviewed units such as `B/D` can be prefilled while leaving new units such as `C` pending review
- candidate-collection runs with different maximum shot values should not be mixed directly for final PT comparison; after target selection, final PT numbers should come from a rerun with a unified shot list

### 2. Context-Drift PT

Runner:
- `scripts/run_pt_context_drift_llama70b.sh`

Core scorer:
- `scripts/score_cross_relation_context_drift_logit.py`

Post-processing:
- `scripts/compute_product_test_bootstrap_context_drift.py`
- `scripts/plot_product_test_context_drift_summary.py`

Typical outputs:
- `pt_context_drift_shot_sweep.csv`
- `pt_context_drift_bootstrap_summary.csv`
- `pt_context_drift_summary.png`
- optional edge top-k traces

Main standard regimes are:
- `ABABAB_B`
- `ADADAD_D`
- `BDBDBD_D`

Important rule:
- baseline PT outputs remain separate and should not be overwritten by context-drift runs

### 3. Unified PT

Runner:
- `scripts/run_pt_unified_drift_control_llama70b.sh`

Core scorer:
- `scripts/score_cross_relation_unified_drift_control.py`

Post-processing:
- `scripts/compute_product_test_bootstrap_unified.py`
- `scripts/build_pt_unified_human_report.py`

Typical outputs:
- `pt_unified_shot_sweep.csv`
- `pt_unified_family_eligibility.csv`
- `pt_unified_bootstrap_summary.csv`
- `human_report/index.html`
- `pt_unified_edge_topk.jsonl`

Observed family structure includes combined baseline/context-control groupings such as:
- `A_ONLY`
- `BASE_ABD`
- `CTX_ABD`
- `ZERO_CTRL`

Unified PT also supports a two-stage selected-target workflow for reviewed B/D units:
- Stage 1: cache-build inference run over `BASE_ABD` and `CTX_ABD` writes `pt_unified_edge_topk.jsonl`
- The edge-topk cache is scratch-canonical and stores lexical candidate token ids plus candidate scores
- In selected-target cache mode, the same edge-topk rows also store forced selected-target first-token scores even when the selected target is outside lexical top-k
- Stage 2: an offline recompute step rebuilds selected-target `pt_unified_shot_sweep.csv` rows from the cached edge-topk artifact without rerunning model inference
- In this workflow, selected-target analysis is currently scoped to review-covered positive-shot B/D units and the six base/context regimes

Current practical default for the candidate cache workflow:
- lexical candidate top-k width: `20`
- scoring semantics: first continuation token only
- offline recompute lookup order: lexical top-k first, forced selected-target fallback second
- report stage must receive both the source edge-topk JSONL and the source family eligibility CSV

### 4. BD Shuffle Comparison PT

Runner:
- `scripts/run_pt_bd_shuffle_compare_llama70b.sh`

Core scorer:
- `scripts/score_bd_shuffle_behavior.py`

Post-processing:
- `scripts/build_bd_shuffle_behavior_summary.py`

Human-view report builder:
- `scripts/build_bd_shuffle_singlefile_report.py`

Purpose:
- compare regular alternating BD layouts against five shuffled layouts per query side
- keep `q_id`, query row, demo multiset, and B/D counts fixed while changing only order
- emphasize target-probability and target-behavior comparison for `BDBDBD_D` and `DBDBDB_B` families

Typical canonical outputs:
- `bd_shuffle_shot_sweep.csv`
- `bd_shuffle_regime_metrics.csv`
- `bd_shuffle_case_deltas.csv`
- `bd_shuffle_side_aggregate.csv`
- `bd_shuffle_summary.md`

Typical human-view derivative:
- repo-local single-file HTML report under `reports/`
- this HTML is a presentation layer derived from a canonical scratch run, not the canonical scientific output itself

## PT Run Structure

A typical scratch PT run contains:
- main sweep CSV
- bootstrap summary CSV
- optional edge-topk traces
- `_resume/` raw row state
- `human_report/` HTML views
- `run.log`

## Classification

- sweep CSVs: canonical scientific outputs
- bootstrap CSVs: derived summaries
- `_resume/`: control state
- `human_report/`: presentation layer
- repo-local single-file BD shuffle HTML reports: presentation layer derived from canonical scratch artifacts

## Historical Source Docs

The old design-plan layer for PT extensions has been archived.
The main source ideas came from:
- `PLAN_PT_CONTEXT_DRIFT.md`
- `PLAN_PT_EDGE_TOPK_TRACE.md`
- `PLAN_PT_RESUME_PROGRESS_LOGGING.md`
