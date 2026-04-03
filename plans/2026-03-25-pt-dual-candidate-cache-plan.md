# Plan Template

## Title

`Unified PT dual-layer candidate cache: shot-wise observed candidates plus high-shot canonical bank`

## Metadata

- Date: `2026-03-25`
- Slug: `pt-dual-candidate-cache`
- Approval Status: `pending`

## Objective

`Extend the current Unified PT candidate-cache workflow so it preserves per-shot observed lexical candidates for HTML and drift analysis, while also introducing a canonical scored candidate bank derived from high shots 7 and 9 and scored across every shot for stable offline PT recompute.`

## Current Context

`The repo already has an approved and partially implemented PT candidate-logit cache workflow under scripts/score_cross_relation_unified_drift_control.py, scripts/recompute_pt_unified_from_edge_cache.py, and docs/brain/pipelines/pt.md. The current design stores shot-specific lexical top-k candidates plus a forced selected-target fallback. This supports HTML candidate inspection and offline recompute for one approved target, but it still fails the user's broader goal of being able to switch to another meaningful candidate such as grain when that token is absent from low-shot lexical top-k rows. The current full cache run at /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_candidate_cache_full_20260324_231056 confirms this gap: Q8::B grain appears strongly at shots 7 and 9 but is absent from many shot-1 rows, so low-shot rescoring cannot be reconstructed from lexical top-k alone.`

## Assumptions

- `The intended analysis scope remains the reviewed q_ids Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18 and the review-covered BASE_ABD plus CTX_ABD regimes.`
- `The first-token PT semantics remain unchanged; both shot-wise observed candidates and canonical bank candidates are scored on the first continuation token only.`
- `Candidate discovery for the canonical bank should be driven by high-shot evidence from shot 7 and shot 9, not by full semantic automation.`
- `The existing shot-wise lexical top-k cache must remain because the user needs HTML views of how candidates drift across shots.`
- `A future execution for this design will require new GPU cache-build work because the current cache does not store per-row scores for a high-shot canonical bank across all shots.`

## Inputs And Dependencies

- `/home/sunsik/my_fv_project/scripts/score_cross_relation_target_logit.py`
- `/home/sunsik/my_fv_project/scripts/score_cross_relation_unified_drift_control.py`
- `/home/sunsik/my_fv_project/scripts/recompute_pt_unified_from_edge_cache.py`
- `/home/sunsik/my_fv_project/scripts/build_pt_valid_answer_scaffold.py`
- `/home/sunsik/my_fv_project/scripts/build_pt_unified_human_report.py`
- `/home/sunsik/my_fv_project/scripts/run_pt_unified_drift_control_llama70b.sh`
- `/home/sunsik/my_fv_project/fv/pt_selected_targets.py`
- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_candidate_cache_full_20260324_231056/pt_unified_edge_topk.jsonl`
- `/scratch/sunsik/my_fv_project/pt_analysis/selected_targets/2026-03-24-selected_targets_all_requested_qs.json`
- `The existing approved PT candidate-cache plan and tech spec from 2026-03-24 as the immediate baseline design to revise`

## Proposed Steps

1. `Define a dual-layer candidate-cache contract: Layer A remains shot-wise observed lexical top-k storage for every row, while Layer B adds a canonical candidate bank per unit built from high-shot 7 and 9 evidence.`
2. `Specify how the canonical bank is built: aggregate high-shot lexical candidates for each reviewed unit, rank or filter them into a stable bank of 20 candidates, and record the provenance that these candidates came from shots 7 and 9.`
3. `Design the cache-build scorer change so every row still writes shot-wise lexical top-k outputs, but also writes scores for the unit's canonical high-shot candidate bank across all shots and trials.`
4. `Design the offline recompute rule so selected-target PT uses the canonical bank score path first, with shot-wise lexical candidates preserved for HTML and exploratory inspection rather than acting as the sole recovery source.`
5. `Define HTML/report expectations so shot-wise candidate drift continues to be visible, while canonical bank candidates and selected-target scoring provenance are also inspectable in the report.`
6. `Plan the validation strategy: confirm the canonical bank contains the intended high-shot candidates, confirm low-shot rows now store their scores even when not in lexical top-k, and confirm offline PT recompute works for alternative targets like Q8 grain without requiring another model rerun.`

## Risks And Blockers

- `Canonical bank construction could drift from user intent if the high-shot aggregation rule is underspecified, especially when shot 7 and 9 disagree or when one surface form dominates only one regime.`
- `The schema will become meaningfully more complex because each row must now preserve both shot-wise observed candidates and canonical-bank candidate scores without confusing downstream readers or HTML builders.`
- `If canonical bank membership is derived from text strings instead of tokenizer-consistent first-token ids, first-token PT semantics can silently misalign across cache-build and offline recompute.`
- `Existing HTML/report code may assume only one candidate layer and may need design changes so drift views and canonical-bank views do not become ambiguous or mislabeled.`
- `This is a stable pipeline behavior change, so stale assumptions in docs/brain/pipelines/pt.md or related runbooks would become misleading if they are not updated together.`

## Recommended Compute Mode

- Mode: `gpu`
- Why: `The design itself can be planned locally, but the first real validation of this dual-layer cache requires a new inference-backed cache-build run so that canonical high-shot candidate bank scores are actually written for all shots.`

## Expected Outputs

- `/home/sunsik/my_fv_project/plans/2026-03-25-pt-dual-candidate-cache-plan.md`
- `/home/sunsik/my_fv_project/plans/2026-03-25-pt-dual-candidate-cache-tech-spec.md`
- `A revised PT cache schema proposal that separates shot-wise observed candidates from canonical high-shot candidate bank scores`
- `If later approved for execution, a new scratch-first Unified PT cache-build run and an offline recompute run that can change selected targets using canonical bank scores without another model rerun`

## Success Criteria

- `The approved design clearly preserves shot-wise observed candidates for HTML drift views while adding a canonical high-shot candidate bank that can be scored across all shots.`
- `The plan identifies a concrete path for rescoring low-shot rows with alternative targets like Q8 grain without relying on those targets being present in low-shot lexical top-k.`
- `The future implementation path is concrete enough to produce a follow-on tech spec without reopening the design question of whether one or two candidate layers are needed.`

## Brain Impact

- Brain impact: `update required`
- Why: `This would change the stable PT cache artifact contract, the meaning of what is stored per row, and the recommended offline recompute workflow, so docs/brain/pipelines/pt.md would need to reflect the new two-layer design if execution is later approved.`

## Approval Note

Substantial execution has not started. Waiting for Korean approval.
