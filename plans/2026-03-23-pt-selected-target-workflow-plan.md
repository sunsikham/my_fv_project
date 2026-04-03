# Plan Template

## Title

`PT human-selected target workflow`

## Metadata

- Date: `2026-03-23`
- Slug: `pt-selected-target-workflow`
- Approval Status: `approved`

## Objective

`Introduce an explicit PT target-selection layer so each Q/query unit is scored against a human-selected relation-valid candidate from the observed top-candidate pool, instead of always scoring the dataset's fixed gold target string.`

## Current Context

`The current PT scorers still bind scoring directly to the dataset query output. In both /home/sunsik/my_fv_project/scripts/score_cross_relation_target_logit.py and /home/sunsik/my_fv_project/scripts/score_cross_relation_unified_drift_control.py, the scored token id and emitted target fields are derived from query["output"], so PT remains coupled to the original target_str for each row. At the same time, the repo already has a partial manual-review layer for Unified PT: /home/sunsik/my_fv_project/scripts/build_pt_valid_answer_scaffold.py builds a scaffold from pt_unified_edge_topk.jsonl, and /home/sunsik/my_fv_project/pt_analysis/valid_answer_sets/llama31_70b_unified_pt_20260311_071325.scaffold.json stores per-unit candidate suggestions plus pending valid_answers review fields. That means the missing piece is not candidate discovery itself but the end-to-end contract that turns reviewed candidates into the canonical scoring target used by PT.`

## Assumptions

- `The intended workflow is human-in-the-loop: a person reviews top candidates per PT unit and explicitly selects one canonical target or a relation-valid answer set before PT rescoring runs.`
- `The first safe end-to-end implementation should target Unified PT because the repo already has edge-topk traces and a scaffold generator there.`
- `Baseline PT and context-drift PT should be kept compatible with the new target-selection contract, but they do not need to block the initial Unified PT implementation if shared utilities are designed cleanly.`
- `Selected-target artifacts should follow the scratch-first storage policy, with scratch as canonical and repo-local copies treated as mirrors when synced.`
- `The selected target must remain traceable back to the original gold target, candidate pool, and reviewer decision so later analyses can explain why PT scores changed.`
- `The rescoring contract should support both a single selected target and, if needed, a best-valid scoring mode over an approved valid-answer set.`
- `Full PT reruns at the project's current model scale still require GPU-backed inference.`

## Inputs And Dependencies

- `Existing Unified PT scorer at /home/sunsik/my_fv_project/scripts/score_cross_relation_unified_drift_control.py`
- `Existing baseline PT scorer at /home/sunsik/my_fv_project/scripts/score_cross_relation_target_logit.py`
- `Existing context-drift PT scorer at /home/sunsik/my_fv_project/scripts/score_cross_relation_context_drift_logit.py`
- `Existing scaffold builder at /home/sunsik/my_fv_project/scripts/build_pt_valid_answer_scaffold.py`
- `Existing scaffold example at /home/sunsik/my_fv_project/pt_analysis/valid_answer_sets/llama31_70b_unified_pt_20260311_071325.scaffold.json`
- `PT pipeline reference at /home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`
- `Project storage policy at /home/sunsik/my_fv_project/docs/brain/ops/storage_and_sync.md`
- `Scratch canonical PT root /scratch/sunsik/my_fv_project/pt_analysis`
- `User approval on the target-selection artifact format and the initial implementation scope`

## Proposed Steps

1. `Define the canonical reviewed-target contract for PT units, including stable unit identity, original gold target fields, reviewer-selected target fields, optional valid-answer-set fields, review status, and metadata that records canonical_root plus any sync metadata.`
2. `Extend or adapt the existing Unified PT valid-answer scaffold flow so candidate review can produce a canonical selected-target artifact rather than only a pending valid_answers scaffold seeded from gold_target.`
3. `Implement a shared loader or resolver that maps each scored PT unit to its approved selected target or approved valid-answer set, with clear failure behavior for missing, pending, or inconsistent reviews.`
4. `Modify PT scorers so scoring no longer defaults to raw query["output"] when a reviewed target artifact is supplied; instead they should score the selected target token or the best-valid candidate, while still emitting the original gold target for traceability.`
5. `Update PT outputs and metadata so sweep rows, raw JSONL traces, and status metadata make the scoring basis explicit, including whether the row used gold_target fallback, single selected target, or best-valid set scoring.`
6. `Validate the new workflow on a small approved subset by checking artifact parsing, unit alignment, scorer behavior, emitted columns, and expected failure modes before any full PT rerun.`
7. `After implementation, update stable PT brain docs if the new reviewed-target layer becomes the supported workflow and record the execution in a final report.`

## Risks And Blockers

- `If unit identity is not defined consistently across scaffold generation and scorer consumption, reviewed targets could be attached to the wrong PT rows.`
- `Single-token target scoring is currently embedded in scorer logic, so multi-token selected answers may require an explicit limitation or a separate scoring rule in the first implementation.`
- `If some units remain review_status=pending, the scorer needs a strict policy for whether to fail hard, skip those units, or fall back to gold_target; choosing the wrong default could silently contaminate results.`
- `Existing historical PT outputs were generated against gold_target, so downstream comparisons must clearly distinguish old and new scoring regimes.`
- `If the initial implementation tries to generalize all PT families at once, scope may expand beyond a safe first pass; keeping Unified PT first is lower risk.`
- `Because the canonical artifacts belong on scratch, storage and sync metadata must be explicit or repo-local mirrors may be mistaken for complete canonical state.`

## Recommended Compute Mode

- Mode: `gpu`
- Why: `The workflow change itself is a code-and-data-contract change, but proving it end to end requires running PT scoring against the project's current large-model setup, which is a GPU-backed path rather than a practical local CPU-only run.`

## Expected Outputs

- `Approved plan artifact at /home/sunsik/my_fv_project/plans/2026-03-23-pt-selected-target-workflow-plan.md`
- `Follow-on tech spec after plan approval`
- `If later executed: a canonical reviewed-target artifact under /scratch/sunsik/my_fv_project/pt_analysis/... that records selected targets or approved valid-answer sets per PT unit`
- `If later executed: scorer updates that consume reviewed targets and emit explicit scoring-basis metadata`
- `If later executed: validation outputs showing selected-target PT rescoring on an approved subset and any resulting updated PT sweep artifacts`
- `If later executed: stable PT documentation updates under /home/sunsik/my_fv_project/docs/brain/ if the reviewed-target workflow becomes the supported path`
- `Final execution report under /home/sunsik/my_fv_project/reports/`

## Success Criteria

- `PT no longer has to rely only on the dataset's original target_str when an approved reviewed-target artifact is provided.`
- `Each PT unit can be matched deterministically to a reviewed selection record without ambiguity.`
- `The candidate-review layer produces a canonical artifact that captures pending vs approved status and preserves traceability to the original gold target and candidate pool.`
- `The scorer makes its scoring basis explicit in outputs, so downstream analyses can distinguish gold-target runs from selected-target runs.`
- `A small validation run demonstrates that human-selected targets are actually being scored end to end rather than only stored as annotations.`
- `The implementation scope remains controlled by delivering one safe end-to-end path first, preferably Unified PT, without blocking later reuse in the other PT families.`

## Brain Impact

- Brain impact: `update required`
- Why: `If implemented, this changes stable PT pipeline behavior, scorer semantics, and PT artifact contracts, which belongs in current docs/brain knowledge rather than only in a dated report.`

## Approval Note

Substantial execution has not started. Waiting for Korean approval.
