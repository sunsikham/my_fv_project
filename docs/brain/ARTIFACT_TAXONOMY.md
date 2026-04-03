# Artifact Taxonomy

## Classification Levels

Every artifact should be classified as one of:
- canonical scientific output
- control or resume state
- derived analysis
- presentation or report
- mirror copy

## Directory Patterns

### `_trials`

Meaning:
- frozen trial payloads used as stage inputs

Typical files:
- `condition_*.json`

Class:
- canonical scientific output

### `_stepd`

Meaning:
- head-sweep stage outputs

Typical files:
- `aie_scores_*.csv`
- `sampled_trials_*.json`
- `trial_metrics_*.jsonl`
- `stepd_meta_*.json`
- heatmaps

Class:
- canonical scientific output

### `_top_heads`

Meaning:
- selected head sets and diagnostics

Typical files:
- `top_heads_*.json`
- `sets/*.json`
- diagnostics json

Class:
- canonical scientific output

### `_vectors`

Meaning:
- trial-level extracted vector arrays and vector metadata

Typical files:
- `trial_vectors_*.npy`
- `trial_vectors_*_headwise_*.npy`
- `vector_extraction_meta.json`

Class:
- canonical scientific output

### `_pca_common`

Meaning:
- PCA projections and geometric summaries

Typical files:
- `pca_points.csv`
- `pca_centroids.csv`
- `distance_summary.json`
- `scatter*.png`
- `scatter_3d_interactive.html`

Class:
- derived analysis

### `_fv`

Meaning:
- built FV vectors and metadata

Typical files:
- `fv_*.npy`
- `fv_meta.json`

Class:
- canonical scientific output

### `_inject`

Meaning:
- zero-shot injection evaluation outputs

Class:
- canonical scientific output when injection is part of the run

### `_status`

Meaning:
- state machine, fingerprints, lock, run status

Typical files:
- `qid_status.json`
- `run_summary.json`
- fingerprints
- `home_sync_status.json`

Class:
- control or resume state

### `logs/`

Meaning:
- command traces and stage logs

Class:
- control or debug state

### `_resume`

Meaning:
- resume bookkeeping and raw per-unit accumulation

Typical files:
- resume state json
- trial plans
- raw rows jsonl
- raw edge-topk jsonl

Class:
- control or resume state

### `human_report/`

Meaning:
- HTML views for humans

Class:
- presentation or report

### `_analysis_*`

Meaning:
- downstream analysis products derived from canonical condition/runtime outputs

Examples:
- `_analysis_stepwise_reweighting`
- `_analysis_stepwise_inside_outside`
- `_analysis_state_intervention`
- `_analysis_multi_root`
- `_analysis_local_tangent_curvature`

Class:
- derived analysis

## File Type Hints

- `.json`: metadata, summaries, status, diagnostics
- `.jsonl`: raw row streams, per-trial traces, resume accumulators
- `.csv`: tabular summaries, metrics, manifests
- `.npy`: vector arrays
- `.npz`: packed analysis arrays
- `.pt`: torch tensors or tensor bundles
- `.png` / `.pdf`: plots
- `.html`: interactive or human-readable reports
- `.log`: execution traces
- `.md`: narrative reports

## Canonical vs Derived Shortcut

Use this shortcut:

- if a file is required to rerun the next scientific stage, it is usually canonical
- if a file summarizes, visualizes, or aggregates prior results, it is usually derived
- if a file only helps resume, debug, or sync, it is control state
