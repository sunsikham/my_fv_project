# Tech Spec Template

## Title

`Top30 g_k endpoint alignment, coefficient co-movement, and signed contribution analysis with full-rank supplementary checks`

## Metadata

- Date: `2026-03-20`
- Slug: `fullspace-gk-analysis`
- Source Plan: `plans/2026-03-20-fullspace-gk-analysis-plan.md`
- Approval Status: `approved`

## Scope

- Re-run the stepwise reweighting analysis for `Q1` with `k_a=30` in a new scratch output root.
- Reuse the resulting top30 basis to compute per-`g_k` endpoint alignment and signed contribution outputs.
- Add an explicit per-feature-step summary table so the signed contribution pattern is readable without manually opening NPZ arrays.
- Add a new coefficient co-movement analysis stage for top30 `g_k`, using branch-specific `Δc` arrays from the top30 run.
- Produce supplementary full-rank comparison summaries that quantify how much of the basis variance and retained-energy behavior top30 captures relative to the existing full-rank space.

## Out Of Scope

- Re-running model inference, StepD, or stepwise A-state extraction.
- Extending the stepwise branch beyond `Q1` in this pass.
- Making full-rank feature tables the primary user-facing output.
- Changing the current theoretical interpretation in `docs/brain/` before the analysis is executed and validated.
- PT or behavioral steering experiments.

## Implementation Design

`The implementation will treat top30 as the new main analysis basis and full-rank as a supplementary comparison target. The existing reweighting script already supports arbitrary top-k via --k_a, so the top30 reweighting stage can be produced by re-running that script into a new scratch output directory. The existing endpoint-aligned contribution script will be extended so that, in addition to its current summary and feature-alignment outputs, it also writes a per-feature-step table containing mean coefficient drift and mean signed contribution values for each g_k at each matched slot. A new co-movement script will read the top30 delta_c arrays and compute branch-specific coefficient co-movement in two forms: a main flattened trial-step Pearson correlation matrix over observations (trial, step), and a secondary per-step trial-wise correlation matrix. The flattened matrix will be used for stable feature-partner summaries, while the per-step matrices will preserve stepwise structure. A lightweight wrapper will orchestrate the top30 run, endpoint contribution stage, co-movement stage, and supplementary full-rank coverage summary into one reproducible local command.`

### Top30 Reweighting Stage

- Input: existing `Q1` canonical stepwise A-state artifacts
- Entry: existing `scripts/compute_stepwise_reweighting_metrics.py`
- Run mode:
  - `--k_a 30`
  - `--base_root /home/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc`
  - `--q_list Q1`
  - `--out_dir /scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting_top30`
- Expected result:
  - top30 `G_A_topk`
  - top30 `delta_c_*_trials_topk`
  - full-rank `G_A_full`
  - explained variance ratios and trial-level retention outputs

### Endpoint Alignment And Signed Contribution Stage

- Input:
  - top30 reweighting arrays NPZ
  - top30 reweighting meta JSON
- Entry: modified `scripts/compute_stepwise_endpoint_aligned_contribs.py`
- Main behavior:
  - keep existing summary CSV, feature-alignment CSV, NPZ, and meta JSON outputs
  - add a new feature-step CSV with one row per:
    - `q_id`
    - `basis_scope`
    - `slot_name`
    - `feature_index`
- New feature-step fields:
  - `align_B`, `align_D`
  - `mean_delta_c_BAB`, `mean_delta_c_DAD`
  - `mean_abs_delta_c_BAB`, `mean_abs_delta_c_DAD`
  - `mean_contrib_BAB_B`, `mean_contrib_BAB_D`
  - `mean_contrib_DAD_B`, `mean_contrib_DAD_D`
  - `mean_abs_contrib_BAB_B`, `mean_abs_contrib_DAD_D`
  - `intended_contrib_rank_BAB`, `intended_contrib_rank_DAD`
- Main interpretation rule:
  - for `BABABA`, intended signed contribution is `contrib_k^B(t)`
  - for `DADADA`, intended signed contribution is `contrib_k^D(t)`

### Co-Movement Stage

- Entry: new `scripts/compute_stepwise_gk_co_movement.py`
- Input:
  - top30 reweighting arrays NPZ
  - top30 reweighting meta JSON
- Main coefficient object:
  - use `Δc` rather than raw `c`
  - reason: the user’s mechanistic question is about context-driven reweighting, not baseline feature loading
- Main co-movement definition:
  - branch-specific Pearson correlation between feature pairs over flattened observations `(trial, step)`
  - separately for `BAB` and `DAD`
- Secondary co-movement definition:
  - per-step Pearson correlation over trials for each feature pair
  - separately for `BAB` and `DAD`
- Main outputs:
  - `stepwise_gk_co_movement_flat_corr_BAB.npy`
  - `stepwise_gk_co_movement_flat_corr_DAD.npy`
  - `stepwise_gk_co_movement_step_corr_BAB.npy`
  - `stepwise_gk_co_movement_step_corr_DAD.npy`
  - `stepwise_gk_co_movement_feature_summary.csv`
  - `stepwise_gk_co_movement_meta.json`
- Feature summary CSV fields:
  - `feature_index`
  - `mean_abs_corr_flat_BAB`, `mean_abs_corr_flat_DAD`
  - `top_pos_partner_BAB`, `top_pos_partner_corr_BAB`
  - `top_neg_partner_BAB`, `top_neg_partner_corr_BAB`
  - `top_pos_partner_DAD`, `top_pos_partner_corr_DAD`
  - `top_neg_partner_DAD`, `top_neg_partner_corr_DAD`
- Optional stable ordering:
  - keep native feature index order as canonical
  - if visualization is added later, allow a derived sorted order but do not rewrite the canonical indexing

### Full-Rank Supplementary Coverage Summary

- Entry: new wrapper `scripts/run_stepwise_top30_gk_analysis.py`
- Inputs:
  - top30 reweighting summary CSV
  - top30 reweighting arrays NPZ
- Supplementary outputs:
  - `top30_fullrank_coverage_summary.json`
  - `top30_fullrank_coverage_by_step.csv`
- Required summary fields:
  - `basis_scope`
  - `rank_A_full`
  - `k_a_main`
  - `cumulative_evr_top30`
  - `tail_evr_after_top30`
  - per-step `R_BAB_top30`, `R_BAB_full`, `R_DAD_top30`, `R_DAD_full`
  - per-step retention ratios:
    - `R_BAB_top30_over_full`
    - `R_DAD_top30_over_full`

### Output Roots

- Reweighting:
  - `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting_top30`
- Endpoint alignment:
  - `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_endpoint_alignment_top30`
- Co-movement:
  - `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30`
- Logs:
  - `/home/sunsik/my_fv_project/logs/2026-03-20-fullspace-gk-analysis-*.log`

## Expected File Changes

- Modify: `/home/sunsik/my_fv_project/scripts/compute_stepwise_endpoint_aligned_contribs.py`
- Create: `/home/sunsik/my_fv_project/scripts/compute_stepwise_gk_co_movement.py`
- Create: `/home/sunsik/my_fv_project/scripts/run_stepwise_top30_gk_analysis.py`
- Create: `/home/sunsik/my_fv_project/plans/2026-03-20-fullspace-gk-analysis-tech-spec.md`
- Modify after successful execution if required: `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_methods.md`
- Modify after successful execution if required: `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_reweighting.md`

## Ordered Implementation Steps

1. Update `scripts/compute_stepwise_endpoint_aligned_contribs.py` so it can write a per-feature-step summary CSV without breaking the current summary outputs.
2. Implement `scripts/compute_stepwise_gk_co_movement.py` to consume top30 `Δc` arrays and write flattened and per-step correlation artifacts plus a feature-level partner summary CSV.
3. Implement `scripts/run_stepwise_top30_gk_analysis.py` to orchestrate:
   - top30 reweighting
   - top30 endpoint alignment
   - top30 co-movement
   - top30 vs full-rank coverage summary
4. Validate script argument handling and output path conventions locally without starting any heavy upstream reruns.
5. After tech spec approval and runtime confirmation, execute the wrapper locally against `Q1`.
6. Verify artifact counts, matrix dimensions, coverage numbers, and branch/step indexing.
7. If the run is successful and method behavior is stable, update the relevant `docs/brain/` files.

## Validation Plan

- Pre-run checks:
  - verify `/home/sunsik/.venvs/pt442/bin/python` can import required packages
  - verify all required `Q1` stepwise and vector artifacts exist
- Reweighting stage checks:
  - confirm `k_a=30` is recorded in the output metadata
  - confirm `K_A_eff=30` for `Q1` unless rank is lower than 30
  - confirm explained variance cumulative share at 30 is reported for `matched` and `all`
- Endpoint contribution checks:
  - confirm the new feature-step CSV exists
  - confirm row counts match `q_count x basis_scope_count x slot_count x k_a`
  - confirm `align_B`, `align_D`, and contribution columns are finite for non-degenerate features
- Co-movement checks:
  - confirm flattened correlation matrices are `30 x 30`
  - confirm per-step correlation arrays are `5 x 30 x 30`
  - confirm diagonal values are 1 or numerically close where variance is nonzero
- Coverage checks:
  - confirm `cumulative_evr_top30` matches the saved explained variance ratio arrays
  - confirm `R_top30_over_full` values lie in `[0, 1]` when full retention is nonzero
- End-to-end acceptance:
  - all planned artifacts are produced in the new scratch roots without overwriting the existing top-5 roots

## Expected Outputs

- `plans/2026-03-20-fullspace-gk-analysis-tech-spec.md`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting_top30/stepwise_reweighting_AAA_ref.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting_top30/stepwise_reweighting_arrays_AAA_ref.npz`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_endpoint_alignment_top30/stepwise_endpoint_alignment_summary.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_endpoint_alignment_top30/stepwise_endpoint_alignment_features.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_endpoint_alignment_top30/stepwise_endpoint_alignment_feature_steps.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/stepwise_gk_co_movement_feature_summary.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/stepwise_gk_co_movement_meta.json`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/top30_fullrank_coverage_summary.json`
- A final execution report under `reports/`

## Brain Docs To Update

- `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_methods.md`
- `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_reweighting.md`

## Recommended Execution Strategy

- Launcher: `local`
- Compute Mode: `local`
- Reason: `This is a derived analysis over saved Q1 arrays. No model loading or GPU inference is required. A local CPU execution through the existing Python environment is the simplest and least risky option.`

## User Execution Settings Required Before Run

- Launcher choice: `local` recommended
- Time limit: `not required for local`
- Day-based duration if relevant: `n/a`
- GPU options: `none`
- CPU count: `default local`
- Memory: `default local`
- Partition or queue: `n/a`
- Job name: `n/a`
- Log path: `use repo-local logs/ with dated filenames`
- Environment setup: `confirm use of /home/sunsik/.venvs/pt442/bin/python`
- Extra launcher flags: `none expected`

## Approval Note

Implementation and execution have not started. Waiting for Korean approval.
