# Plan Template

## Title

`Unified PT candidate-logit cache build and offline recompute workflow`

## Metadata

- Date: `2026-03-24`
- Slug: `pt-candidate-logit-cache`
- Approval Status: `approved`

## Objective

`Change the Unified PT workflow so the expensive GPU run stores top-k candidate-level logits/logprobs/probs per scored row, and later PT rescoring can be recomputed offline from the cached candidate pool without rerunning the model backbone when the user changes the selected target within that pool.`

## Current Context

`The current Unified PT scorer already computes next-token logits and writes pt_unified_edge_topk.jsonl, but that edge cache stores lexical candidate token ids, logprobs, and probs without candidate-level logits or ranks. The recently added selected-target workflow can resolve a chosen target and rescore Unified PT directly, but that still requires a fresh model run each time the selected target changes. The user wants the workflow shifted so one GPU cache-building run is followed by human target selection and then offline PT recomputation from cached candidate scores only. Existing reviewed selected-target artifacts under /scratch/sunsik/my_fv_project/pt_analysis/selected_targets should remain usable as selection inputs, but future target changes should not require another full inference run as long as the target stays inside the stored candidate pool.`

## Assumptions

- `The offline recompute step only needs to support targets whose first continuation token is present in the stored top-k candidate pool for each row.`
- `First-token scoring semantics remain unchanged from current PT behavior; this project is not expanding to multi-token exact-target scoring in this pass.`
- `The current Unified PT edge-topk cache remains the canonical scratch-first artifact to extend rather than introducing a separate hidden-state cache format.`
- `The user-approved selected targets already collected for Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18 should be preserved, but the new design must allow alternate later selections from the same candidate cache.`

## Inputs And Dependencies

- `scripts/score_cross_relation_target_logit.py` for shared lexical top-k extraction helpers
- `scripts/score_cross_relation_unified_drift_control.py` as the current Unified PT scorer and edge-topk writer
- `scripts/run_pt_unified_drift_control_llama70b.sh` as the current Unified PT runner wrapper
- `scripts/compute_product_test_bootstrap_unified.py` for offline PT summary generation after recompute
- `scripts/build_pt_unified_human_report.py` for downstream HTML generation from recomputed outputs
- `fv/pt_selected_targets.py` and the existing selected-target artifacts under `/scratch/sunsik/my_fv_project/pt_analysis/selected_targets`
- Current canonical PT storage root `/scratch/sunsik/my_fv_project/pt_analysis`
- Stable PT pipeline documentation in `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`

## Proposed Steps

1. `Define the new workflow boundary clearly: the GPU stage becomes a candidate-logit cache-building run, while selected-target PT values are later reconstructed offline from the cached edge-topk artifact plus a selected-target artifact.`
2. `Extend the lexical top-k cache schema so each row stores candidate-level logits and ranks in addition to the already stored token ids, candidate strings, logprobs, probs, and canonical forms.`
3. `Design an offline Unified PT recompute path that reads the cached edge-topk rows, resolves the selected target per unit, maps the selected target to its first-token id with the active tokenizer, and reconstructs the sweep CSV without model inference.`
4. `Keep bootstrap and HTML generation as downstream consumers of the recomputed sweep CSV, so the expensive rerun is removed only from the scoring stage and not from the existing reporting stack.`
5. `Validate the design on a small slice by comparing offline recomputed rows against direct scorer-produced rows for the same selected target, then prepare the full cache-build run settings only after the tech spec is approved.`
6. `Evaluate whether stable PT documentation in docs/brain must be updated to reflect the new cache-build plus offline-recompute workflow, artifact semantics, and run entrypoint behavior.`

## Risks And Blockers

- `If the stored top-k pool is too small, a later user-selected target may be absent for some rows, making offline recompute impossible for that selection; k sizing and fail-fast validation are critical.`
- `Because PT still uses first-token semantics, surface strings like tadpole or whiskers may map to a first token that differs from naive string matching; the offline path must match by first-token id, not just candidate text.`
- `Existing downstream readers currently assume target_* fields describe the scored target row consistently; the offline recompute path must preserve that schema contract or update readers in lockstep.`
- `The runner currently couples score, bootstrap, and HTML generation in one wrapper. Splitting cache-build from offline recompute without breaking run metadata and artifact expectations requires careful runner design.`
- `The stable docs/brain PT pipeline entry may become stale if the new two-stage workflow lands without a documentation update.`

## Recommended Compute Mode

- Mode: `gpu`
- Why: `The key approved execution after planning is a cache-building Unified PT inference run over many q/regime/shot/trial rows, which still requires loading the Llama 70B model and writing enlarged top-k artifacts. The later offline recompute and bootstrap/reporting stages are lightweight, but the defining new workflow still depends on a GPU-backed inference stage.`

## Expected Outputs

- `A new approved plan markdown file at /home/sunsik/my_fv_project/plans/2026-03-24-pt-candidate-logit-cache-plan.md`
- `A follow-up tech spec markdown file, after plan approval, defining the concrete schema and execution path`
- `An updated Unified PT edge-topk schema that includes candidate-level logits and ranks`
- `A new offline recompute entrypoint or utility for rebuilding selected-target PT sweep CSVs from cached edge-topk artifacts`
- `If execution is later approved, scratch-first cache-build run artifacts under /scratch/sunsik/my_fv_project/pt_analysis and a final execution report under /home/sunsik/my_fv_project/reports`

## Success Criteria

- `The approved design clearly separates GPU cache-building from offline selected-target PT recomputation.`
- `The plan identifies the exact existing scorer, runner, cache, and downstream components that must participate in the workflow change.`
- `The later tech spec can be implemented without revisiting whether the canonical cache artifact is pt_unified_edge_topk.jsonl or whether selected-target changes should trigger another full model run.`
- `The eventual implementation will be able to recompute PT rows offline for any user-selected target whose first token appears in the stored top-k candidate cache, with bootstrap and HTML generated from that recomputed sweep.`

## Brain Impact

- Brain impact: `update required`
- Why: `If this workflow is implemented, the stable Unified PT pipeline description in docs/brain/pipelines/pt.md will change materially: the expensive stage becomes candidate-logit cache building, selected-target scoring becomes offline recompute from edge-topk cache, and the semantics of pt_unified_edge_topk.jsonl become part of the stable run contract.`

## Approval Note

Substantial execution has not started. Waiting for Korean approval.
