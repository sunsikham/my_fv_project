# TECH_SPEC_ANTONYM_LAYER_MEAN_DELTA_P_PLOT

## 0. Scope

This spec defines a plotting utility for Step6 layer-wise results:
- x-axis: `layer`
- y-axis: `mean_delta_p`

Target datasets:
- antonym run outputs
- relation q-wise outputs (selected `q_id`s in one combined plot, optional per-`q_id` plots)

Primary goal is to provide reproducible layer-wise curves from existing Step6 outputs.

Out of scope:
- Re-running StepD/StepE/Step6
- Cross-dataset comparison
- Statistical significance testing

Implementation status:
- Pending (`scripts/plot_layer_mean_delta_p.py` is not yet implemented in this repository)

---

## 1. Source of Truth

Use Step6 per-layer summaries.

Antonym canonical input root:
- `results_fv/antonym_t25_llama31/artifacts/step6/`

Relation q-wise canonical input root:
- `results_fv/relation_qwise/<relation_name>/<QID>/artifacts/step6/`

Legacy/smoke roots (optional, non-canonical):
- `results_fv/relation_qwise_smoke/<relation_name>/<QID>/artifacts/step6/`
- `results_fv/relation_qwise_smoke_ultra/<relation_name>/<QID>/artifacts/step6/`

Required per-layer file:
- `layer_<L>/eval_summary.json`

Required metric key:
- `mean_delta_p`

Examples:
- `results_fv/antonym_t25_llama31/artifacts/step6/layer_14/eval_summary.json`
- `results_fv/relation_qwise/relationB_ex/Q6/artifacts/step6/layer_1/eval_summary.json`

This file includes at least:
- `mean_delta_p`
- `std_delta_p`
- `mean_delta_logprob`
- `mean_delta_logit`
- accuracy fields

---

## 2. Functional Requirements

### 2.1 Single Run Mode

1. Discover all directories matching `layer_<int>` under Step6 root.
2. Read each `eval_summary.json`.
3. Extract:
   - `layer` (from directory name)
   - `mean_delta_p` (from JSON)
4. Sort by ascending `layer`.
5. Plot a line chart with markers:
   - x: `layer`
   - y: `mean_delta_p`
6. Save figure and tabular artifact.

Optional:
- Draw horizontal baseline `y=0`.
- Annotate best layer (`argmax(mean_delta_p)`).

### 2.2 Q-wise Batch Mode

When input is relation q-wise root:
1. Discover all `Q*` directories under `<relation_name>/`.
2. For each `Q*`, resolve `artifacts/step6/`.
3. Run the same single-run extraction/plot logic.
4. Select target QIDs:
   - explicit `--q_filter` if provided
   - otherwise all discovered QIDs with valid Step6 summaries
5. Render one combined figure with multiple lines (one line per selected `q_id`).
6. Optionally emit per-`q_id` plot + CSV (+metadata).
7. Emit batch summary CSV with:
   - `q_id`
   - `num_layers_loaded`
   - `best_layer`
   - `best_mean_delta_p`
   - `status` (`ok`/`skipped`/`failed`)

---

## 3. CLI Contract (Proposed)

Script:
- `scripts/plot_layer_mean_delta_p.py`

Arguments:
- `--mode` (required, choices: `single`, `qwise`)

Single mode:
- `--step6_dir` (required when `--mode single`)
  - e.g. `results_fv/antonym_t25_llama31/artifacts/step6`
- `--out_dir` (required)
- `--out_name` (default: `layer_mean_delta_p.png`)
- `--csv_name` (default: `layer_mean_delta_p.csv`)

Q-wise mode:
- `--qwise_root` (required when `--mode qwise`)
  - e.g. `results_fv/relation_qwise/relationB_ex`
- `--out_dir` (required)
  - e.g. `results_fv/relation_qwise/relationB_ex/_plots_layer_mean_delta_p`
- `--q_filter` (optional, comma-separated QIDs; default all discovered)
- `--q_plot_mode` (default: `combined`, choices: `combined`, `per_q`, `both`)
- `--combined_out_name` (default: `qwise_layer_mean_delta_p_combined.png`)
- `--combined_csv_name` (default: `qwise_layer_mean_delta_p_combined.csv`)
- `--batch_csv_name` (default: `qwise_layer_mean_delta_p_summary.csv`)

Common:
- `--title` (optional)
- `--dpi` (default: `180`)

---

## 4. Output Contract

### 4.1 Single Mode Output

In `--out_dir`:

1. Plot image
   - `<out_name>` (PNG)
2. Tabular export
   - `<csv_name>` with schema:
     - `layer` (int)
     - `mean_delta_p` (float)
3. Metadata JSON (recommended)
   - `<out_name_without_ext>.meta.json`
   - Contains:
     - `step6_dir`
     - `num_layers_found`
     - `num_layers_loaded`
     - `missing_layers` (if any)
     - `best_layer`
     - `best_mean_delta_p`
     - `generated_at` (UTC ISO8601)

### 4.2 Q-wise Mode Output

Combined outputs (`q_plot_mode=combined|both`):
- `<combined_out_name>` (one figure; multiple Q lines)
- `<combined_csv_name>` with schema:
  - `q_id` (string)
  - `layer` (int)
  - `mean_delta_p` (float)
- `qwise_layer_mean_delta_p_combined.meta.json`

Per-Q outputs (`q_plot_mode=per_q|both`), in `--out_dir`:
- `Q6.layer_mean_delta_p.png`
- `Q6.layer_mean_delta_p.csv`
- `Q6.layer_mean_delta_p.meta.json`

Batch-level:
- `<batch_csv_name>`
- `qwise_layer_mean_delta_p_batch.meta.json`

---

## 5. Error Handling and Validation

Hard failures:
- invalid mode arguments
- required root path does not exist
- single mode: no `layer_*` directories found
- single mode: no valid `eval_summary.json` with `mean_delta_p`
- q-wise mode: all selected/discovered QIDs have no valid layers (`num_layers_loaded=0` for all)

Soft handling:
- If some layers are missing or malformed, skip them and record in metadata.
- If at least one valid layer exists, produce plot + csv.
- In q-wise mode, failed QIDs are recorded and processing continues for remaining QIDs.
- In combined mode, QIDs with no valid data are excluded from the figure and marked in batch metadata.

Validation rules:
- `layer` must parse as integer from `layer_<L>`.
- `mean_delta_p` must be numeric.

---

## 6. Plot Style Requirements

- Matplotlib backend: non-interactive (`Agg`)
- Figure size: `(8, 4.5)` minimum
- Line + point marker (`-o`)
- Grid enabled
- Axis labels:
  - x: `Layer`
  - y: `Mean Delta p`
- Title default:
  - `Step6: Layer vs Mean Delta p`

Recommended visual cues:
- `ax.axhline(0.0, linestyle="--", linewidth=1)`
- Highlight best layer point in distinct color
- In combined q-wise plot, enable legend with `q_id` labels and place it outside the main plot area when Q count is large.

---

## 7. Determinism and Reproducibility

- Input is file-based; no model inference is run.
- Sorting by integer `layer` guarantees deterministic ordering.
- Same inputs must yield identical CSV and equivalent plot (except metadata timestamp).

---

## 8. Acceptance Criteria

1. Single mode on antonym Step6 root produces plot + csv.
2. CSV contains one row per valid layer and is sorted by `layer`.
3. Y values in CSV exactly match `eval_summary.json` `mean_delta_p` values.
4. Plot x-axis is layer index and y-axis is mean delta p.
5. Q-wise mode on `results_fv/relation_qwise/relationB_ex` with selected QIDs produces one combined multi-line plot (x=layer, y=mean_delta_p).
6. Combined CSV includes all plotted points with `(q_id, layer, mean_delta_p)`.
7. Q-wise batch summary CSV includes one row per processed `Q*` with status and best-layer stats.

---

## 9. Non-Goals / Notes

- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0` is parity-validation output and not a layer sweep source.
- Layer-wise plotting should use Step6 outputs only.
- This spec does not define aggregation across different `q_id` curves into a single statistical model.
