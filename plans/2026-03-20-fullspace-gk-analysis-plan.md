# Plan Template

## Title

`Top30 g_k endpoint alignment, coefficient co-movement, and signed contribution analysis with full-rank supplementary checks`

## Metadata

- Date: `2026-03-20`
- Slug: `fullspace-gk-analysis`
- Approval Status: `approved`

## Objective

`Extend the current stepwise multi-feature analysis so that Q1 can be analyzed in a larger and still interpretable A-only latent basis using top30 g_k as the main view, with explicit outputs for per-g_k endpoint alignment, per-g_k coefficient change patterns, and cross-g_k coefficient co-movement, while also producing supplementary full-rank comparison checks.`

## Current Context

`The current Q1 stepwise branch already has canonical scratch outputs for stepwise A-state extraction, stepwise reweighting, endpoint-aligned contribution analysis, inside/outside decomposition, local tangent checks, and strict same-state intervention. However, the feature-level interpretation currently exposed to the user is effectively limited to the top-K A-basis setting with k_a=5. The current scripts already preserve the full A-space basis for retention and inside/outside calculations, but the coefficient-drift and endpoint-contribution outputs are still organized around the top-k coefficient arrays. The saved Q1 explained-variance profiles show that top30 already captures about 87.67 percent of the matched basis variance and about 87.83 percent of the all basis variance, so top30 is a strong main-analysis compromise between interpretability and coverage.`

## Assumptions

- `Q1` remains the only fully prepared stepwise case in canonical scratch for this analysis pass.
- The user wants an additive larger-basis analysis path rather than overwriting the existing top-5 exploratory outputs.
- The user prefers top30 as the main analysis space rather than full-rank as the primary reporting space.
- Full-space coefficient analysis can run locally because it consumes saved NPZ/JSON/CSV artifacts rather than re-running model inference.
- The basis vectors `g_k` are PCA/SVD latent axes, so “relation between g_k vectors” should be operationalized through alignment weights, coefficient co-movement, and signed contribution structure rather than raw basis-angle inspection alone.

## Inputs And Dependencies

- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_stepwise_a_states/stepwise_a_states_AAA_ref.npz`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_stepwise_a_states/stepwise_a_states_meta.json`
- `/home/sunsik/my_fv_project/scripts/compute_stepwise_reweighting_metrics.py`
- `/home/sunsik/my_fv_project/scripts/compute_stepwise_endpoint_aligned_contribs.py`
- Existing canonical scratch analysis roots under `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/`
- Local Python environment capable of running the analysis scripts with NumPy and existing repo dependencies

## Proposed Steps

1. Confirm the exact reuse path in the current analysis scripts, and define an additive output layout for top30 main runs plus full-rank supplementary checks so the existing top-5 outputs remain untouched.
2. Implement the minimum code changes needed to support top30 g-level analysis artifacts cleanly, likely including a dedicated coefficient co-movement stage and small script updates required to make the chosen analysis rank and the supplementary full-rank comparisons explicit.
3. Write a tech spec that fixes the exact output schema for three top30 views: per-g_k endpoint alignment, per-g_k coefficient drift/co-movement, and per-g_k signed contribution summaries, plus a compact full-rank comparison section.
4. After approval, execute the analysis in `local` mode against Q1 canonical scratch artifacts, producing separate full-space analysis outputs and logs.
5. Validate that the generated outputs are internally consistent, interpretable, and sufficient to inspect which latent A-features move together and which ones drive endpoint-directed movement, while also quantifying what top30 leaves in the tail.
6. If the resulting method becomes the new stable analysis path, update the relevant `docs/brain/` method documentation.

## Risks And Blockers

- Even top30 is much larger than the current top-5 view, so naive outputs may still become too large or too hard to inspect unless summaries and filtering are designed carefully.
- Top30 may miss some lower-variance but behaviorally or geometrically important tail features, so the supplementary full-rank checks need to quantify this risk explicitly.
- The current endpoint contribution script assumes the “topk” coefficient arrays as its input interface, so full-space reuse needs to be handled carefully to avoid misleading naming or accidental overwrite of prior artifacts.
- Co-movement can be defined in multiple plausible ways, such as correlation over raw coefficients, correlation over `Δc`, or branch-specific covariance; the tech spec must pin this down clearly before execution.
- If the local Python environment is incomplete, execution could fail before analysis starts; this does not block the planning phase but needs to be checked before run approval.

## Recommended Compute Mode

- Mode: `local`
- Why: `This work is a derived analysis over existing saved arrays for Q1 and does not require model loading, forward passes, or GPU inference. The main cost is matrix computation and artifact writing, which is appropriate for local CPU execution.`

## Expected Outputs

- `plans/2026-03-20-fullspace-gk-analysis-tech-spec.md`
- A dedicated top30 reweighting output root under scratch
- A dedicated top30 endpoint-alignment output root under scratch
- A dedicated top30 coefficient co-movement output root under scratch
- CSV/NPZ/JSON artifacts that expose per-g_k alignment weights, per-g_k coefficient statistics, and per-g_k signed contribution summaries for top30
- Compact supplementary artifacts comparing top30 against the corresponding full-rank basis coverage
- A final execution report under `reports/`

## Success Criteria

- The approved implementation can generate a top30 `A-space` analysis for `Q1` without overwriting the current top-5 exploratory outputs.
- The resulting artifacts expose per-`g_k` endpoint alignment weights for the top30 `matched` and `all` A-basis views used in the run.
- The resulting artifacts expose stepwise coefficient drift and coefficient co-movement summaries for the top30 basis actually used in the run.
- The resulting artifacts expose signed contribution summaries that let the user identify which top30 `g_k` axes support or oppose B/D endpoint movement.
- The resulting artifacts include a supplementary comparison that reports how much variance top30 covers and how much mass remains outside top30 for the basis views used in the run.
- Validation confirms that artifact dimensions, rank metadata, and branch/slot indexing are consistent with the underlying `Q1` stepwise inputs.

## Brain Impact

- Brain impact: `update required`
- Why: `If this full-space analysis path is implemented and run successfully, it changes the stable description of the multi-feature analysis method and artifact family, so the relevant method-side brain documentation should be updated.`

## Approval Note

Substantial execution has not started. Waiting for Korean approval.
