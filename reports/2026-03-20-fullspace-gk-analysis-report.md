# Final Report Template

## Status

- Status: `success`
- Date: `2026-03-20`

## Summary

`Implemented and ran a local top30 g_k analysis workflow for Q1. The run produced new top30 reweighting outputs, endpoint-aligned per-feature-step summaries, branch-specific g_k co-movement artifacts, and top30-vs-full-rank coverage summaries. The run completed without retry.`

## Source Documents

- Plan: `plans/2026-03-20-fullspace-gk-analysis-plan.md`
- Tech Spec: `plans/2026-03-20-fullspace-gk-analysis-tech-spec.md`

## Execution Settings

- Launcher: `local`
- Compute Mode: `local`
- Time limit: `n/a`
- Day-based duration if relevant: `n/a`
- GPU options: `none`
- CPU count: `default local`
- Memory: `default local`
- Partition or queue: `n/a`
- Job name: `n/a`
- Environment setup: `/home/sunsik/.venvs/pt442/bin/python`
- Extra launcher flags: `none`

## Commands Run

1. `mkdir -p logs reports && { echo '# Pre-execution baseline'; echo 'date=2026-03-20'; echo 'cwd=/home/sunsik/my_fv_project'; echo 'python=/home/sunsik/.venvs/pt442/bin/python'; echo; echo '## git status --short'; git status --short; } > logs/2026-03-20-fullspace-gk-analysis-baseline.log`
2. `/home/sunsik/.venvs/pt442/bin/python -m py_compile scripts/compute_stepwise_endpoint_aligned_contribs.py scripts/compute_stepwise_gk_co_movement.py scripts/run_stepwise_top30_gk_analysis.py`
3. `/home/sunsik/.venvs/pt442/bin/python scripts/run_stepwise_top30_gk_analysis.py --q_list Q1`
4. Validation reads against the generated scratch artifacts using `/home/sunsik/.venvs/pt442/bin/python`

## Files Executed

- `/home/sunsik/my_fv_project/scripts/run_stepwise_top30_gk_analysis.py`
- `/home/sunsik/my_fv_project/scripts/compute_stepwise_reweighting_metrics.py`
- `/home/sunsik/my_fv_project/scripts/compute_stepwise_endpoint_aligned_contribs.py`
- `/home/sunsik/my_fv_project/scripts/compute_stepwise_gk_co_movement.py`

## Files Changed

- Modified: `/home/sunsik/my_fv_project/plans/2026-03-20-fullspace-gk-analysis-plan.md`
- Modified: `/home/sunsik/my_fv_project/plans/2026-03-20-fullspace-gk-analysis-tech-spec.md`
- Modified: `/home/sunsik/my_fv_project/scripts/compute_stepwise_endpoint_aligned_contribs.py`
- Created: `/home/sunsik/my_fv_project/scripts/compute_stepwise_gk_co_movement.py`
- Created: `/home/sunsik/my_fv_project/scripts/run_stepwise_top30_gk_analysis.py`
- Modified: `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_methods.md`
- Modified: `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_reweighting.md`
- Created: `/home/sunsik/my_fv_project/reports/2026-03-20-fullspace-gk-analysis-report.md`

## Output Artifacts

- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting_top30/stepwise_reweighting_AAA_ref.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting_top30/stepwise_reweighting_arrays_AAA_ref.npz`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_endpoint_alignment_top30/stepwise_endpoint_alignment_summary.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_endpoint_alignment_top30/stepwise_endpoint_alignment_features.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_endpoint_alignment_top30/stepwise_endpoint_alignment_feature_steps.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/stepwise_gk_co_movement_feature_summary.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/stepwise_gk_co_movement_flat_corr_BAB.npy`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/stepwise_gk_co_movement_flat_corr_DAD.npy`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/stepwise_gk_co_movement_step_corr_BAB.npy`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/stepwise_gk_co_movement_step_corr_DAD.npy`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/top30_fullrank_coverage_by_step.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/top30_fullrank_coverage_summary.json`

## Log Paths

- `logs/2026-03-20-fullspace-gk-analysis-baseline.log`
- `logs/2026-03-20-fullspace-gk-analysis-run.log`

## Validation Results

- Pre-run syntax check passed for the modified and new Python scripts.
- `stepwise_endpoint_alignment_feature_steps.csv` row count is `300`, matching `Q1 x 2 basis scopes x 5 slots x 30 features`.
- `stepwise_gk_co_movement_flat_corr_BAB.npy` has shape `(30, 30)`.
- `stepwise_gk_co_movement_step_corr_BAB.npy` has shape `(5, 30, 30)`.
- Top30 cumulative explained variance:
  - `matched`: `0.8767250776290894`
  - `all`: `0.8783077597618103`
- Top30 retention relative to full-rank retention at `A_query`:
  - `matched BAB`: `0.7842749920476655`
  - `matched DAD`: `0.7776098075693233`
  - `all BAB`: `0.7197487843190983`
  - `all DAD`: `0.7219876593846881`
- `matched / A_query` top intended-contribution features:
  - `BAB`: `g2`, `g4`, `g3`, `g6`
  - `DAD`: `g2`, `g0`, `g1`, `g3`
- `matched` flattened co-movement highlights:
  - `BAB`: `g0` pairs strongly with `g1` and opposes `g2`
  - `DAD`: `g0` pairs strongly with `g1` and opposes `g2`

## Brain Updates

- Required: `yes`
- Updated files: `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_methods.md`, `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_reweighting.md`
- Why: `The run established a stable top30 g_k analysis path, including new artifact families for top30 reweighting reruns, per-feature-step endpoint contribution tables, and g_k co-movement outputs.`

## Result Explanation

- The new workflow keeps the existing top-5 exploratory branch intact and adds a wider `top30` inside-A analysis branch for `Q1`.
- The endpoint-alignment stage now emits a feature-step table, so each `g_k` can be inspected by slot for alignment, coefficient drift, and signed endpoint contribution without opening raw NPZ arrays.
- The new co-movement stage uses `Δc` rather than raw coefficients, so the resulting correlations track context-driven reweighting rather than baseline A loading.
- Top30 explains about `87.7%` to `87.8%` of Q1 basis variance, which supports using it as the main readable space while still leaving a measurable tail for supplementary checks.
- The retention ratios show that top30 captures most, but not all, of the full-rank retained-A structure, especially at later steps where the ratio stays around `0.72` to `0.78` at `A_query`.
- Remaining risk:
  - tail features outside top30 may still matter for some interpretations
  - co-movement is currently centered on the `matched` basis and `Δc`; alternative definitions remain possible if the user wants a second pass

## Retry Record

- Retry attempted: `no`
- Reason: `n/a`

## Failure Details

- Failure point: `n/a`
- Error summary: `n/a`
