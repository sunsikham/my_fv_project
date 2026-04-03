# Relation Q-Wise Pipeline

## Purpose

Run q-wise FV experiments on relation data:
- StepD head sweep
- StepE FV build
- Step6 all-layer injection sweep

## Main Entry Point

- `scripts/run_relation_qwise_pipeline.py`

## Inputs

Main data:
- `datasets/relation/relation*.csv`

Main runtime parameters:
- model
- model spec
- q list
- number of demos
- number of trials per q

## Stage Flow

### StepD

Entry:
- `scripts/run_stepD_aie_head_sweep.py`

Typical outputs:
- `sampled_trials.json`
- `aie_scores.csv`
- `trial_metrics.jsonl`

### Heatmap

Entry:
- `scripts/plot_stepD_aie_heatmap.py`

Typical outputs:
- q-wise head heatmaps

### StepE

Entry:
- `scripts/run_stepE_topk_fv_and_eval.py`

Typical outputs:
- `top_heads.json`
- `fv_global_resid.pt`
- `fv_by_layer.pt`
- StepE metadata

### Zero-Shot Snapshot

Built from StepD trials.

Typical output:
- `sampled_trials_zeroshot.json`

### Step6 Layer Sweep

Entry:
- `scripts/run_step6_fv_injection_eval.py`

Run once per candidate injection layer.

Typical outputs:
- `step6/layer_*/step6_results_*.json`
- `step6/layer_*/eval_summary.json`
- `step6/layer_*/eval_trials.jsonl`
- `step6/step6_all_layers_summary.json`

## Status and Resume

The orchestrator maintains per-q status files and checks completion before skipping work.

Typical control files:
- `qid_status.json`

## Aggregate and Derived Outputs

Repo-side retained folders currently emphasize:
- `_analysis`
- `_plots_step6_metrics`
- `_super_fv`

Treat these as downstream aggregate layers, not the raw core runtime itself.

