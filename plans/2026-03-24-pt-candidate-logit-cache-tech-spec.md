# Tech Spec Template

## Title

`Unified PT candidate-logit cache build and offline recompute workflow`

## Metadata

- Date: `2026-03-24`
- Slug: `pt-candidate-logit-cache`
- Source Plan: `plans/2026-03-24-pt-candidate-logit-cache-plan.md`
- Approval Status: `approved`

## Scope

- `Extend the Unified PT lexical top-k cache so each scored row stores lexical candidate token ids, candidate texts, canonical forms, logits, logprobs, probs, and vocab ranks for the top 20 lexical candidates.`
- `Keep current first-token PT semantics unchanged: all scoring and offline recompute operate on the selected target's first continuation token only.`
- `Add a forced selected-target cache path so each scored row also stores the approved selected target's first-token score even when that token is not present in the lexical top-20 candidate cache.`
- `Add an offline recompute entrypoint that rebuilds selected-target Unified PT sweep rows from a cached pt_unified_edge_topk.jsonl plus a selected-target artifact, with no model inference.`
- `Reuse the existing bootstrap and human-report layers on top of the recomputed sweep CSV so selected-target PT statistics and HTML can be generated offline.`
- `Preserve the existing reviewed selected-target artifact contract in fv/pt_selected_targets.py and the selected target artifacts already created for Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18.`
- `Treat lexical filtering as the candidate discovery rule for this pass; the system will not auto-decide relation-valid targets and will continue to rely on human selection from the cached lexical candidate pool.`
- `Restrict the approved selected-target analysis scope to q_ids Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18 and to the review-covered BASE_ABD and CTX_ABD B/D-query units only.`
- `Restrict the approved regime set to the six intended base/context regimes only: BASE_AB, BASE_AD, BASE_BD, CTX_ABABAB_B, CTX_ADADAD_D, and CTX_BDBDBD_D.`

## Out Of Scope

- `Multi-token exact-target scoring beyond the current first-token semantics.`
- `Guaranteeing offline recompute for targets that are outside the stored lexical top-20 candidate pool for a row.`
- `Eliminating lexical top-k caching; the forced selected-target cache is additive, not a replacement.`
- `Automatic semantic or relation-validity classification of candidates; candidate review remains a human or later-rule layer.`
- `Changing baseline PT or context-drift PT standalone scorers outside the shared lexical top-k helper unless needed for shared helper compatibility.`
- `Replacing pt_unified_edge_topk.jsonl with a hidden-state cache or full-vocabulary logits cache.`
- `Revisiting the already approved selected-target choices; this work is about preserving flexibility for later alternative choices without rerunning the model.`
- `Offline selected-target recompute for q_ids or unit families that are not covered by the current merged selected-target artifact.`
- `ZERO_CTRL, A_ONLY, A-query selected-target analysis, and any expansion to Q12,Q13,Q14,Q17 without a broader reviewed artifact.`
- `Cross-model or cross-tokenizer recompute; this pass assumes the cache-build tokenizer/model is the authoritative token-id space.`

## Implementation Design

`The design introduces an explicit two-stage contract. Stage 1 is a GPU-backed cache-build run using the existing Unified PT scorer. That scorer already has access to next_logits and next_logprobs at each prediction position, so the lexical cache schema change remains to extend _collect_edge_topk() so the lexical candidate records include candidate-level logits and vocab ranks in addition to token ids, texts, canonical forms, logprobs, and probs. The lexical filter remains the current word-start, single-token, alphabetic, non-function-word filter with canonical-form deduplication; no new semantic filter is introduced in this pass. In addition to lexical top-20 storage, the cache-build scorer must also resolve the approved selected target for each in-scope unit and store its first-token score unconditionally in the edge-topk row, even if that selected target token is outside the lexical top-20 list. This forced selected-target cache should include at least the selected target string, first-token id, token string, logit, logprob, prob, and vocab rank. The Unified PT scorer continues to write pt_unified_edge_topk.jsonl, but that file now becomes the canonical candidate-logit cache artifact with both lexical-candidate and forced-selected-target score data.`

`Stage 2 is an offline recompute path. A new script will read the cached pt_unified_edge_topk.jsonl, the selected-target artifact, and cache-build metadata from the source run. The source run metadata is authoritative for model and tokenizer identity; the recompute path must reject any mismatch between the cache-build model/model_spec/tokenizer assumptions and the recompute invocation. For each cached row inside the approved q/family/regime scope, the script will reconstruct the unit id from q_id, query_source, query_input, and gold target, resolve the selected target for that unit, compute the selected target's first continuation token id under the same first-token rules as the cache-build scorer, and then try two lookup paths in order. First it will look for that token id inside the cached lexical_candidate_token_ids list. If found, it will use the lexical candidate arrays. If not found, it will require a matching forced selected-target score record in the same edge-topk row and use that data instead. Only if both lexical-candidate lookup and forced-selected-target lookup fail should the recompute path error. On success it will emit a new sweep row whose target_* fields refer to the selected target, whose gold_target_* fields keep the original dataset target, and whose target_logit, target_logprob_raw, target_prob_raw, and target_rank_in_vocab come from either the lexical candidate arrays or the forced selected-target cache, with metadata recording which path was used. After all rows are reconstructed, the script will recompute target_s_norm offline using the same q_id-family normalization rule as the live scorer.`

`Bootstrap remains unchanged and will consume the recomputed sweep CSV. Human-report generation also remains downstream of the recomputed run, but the report builder must receive both the source edge-topk cache and the source family eligibility CSV so that recompute-only run directories remain reportable. The runner layer should support these two stages cleanly: one inference-backed cache-build stage and one inference-free recompute stage. The first execution pass should fix lexical candidate k at 20 for the cache-building run because the user explicitly approved top 20 as the practical candidate budget. The first implementation and execution scope is not full Unified PT; it is the currently approved Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18 BASE_ABD+CTX_ABD subset only.`

## Expected File Changes

- Modify: `/home/sunsik/my_fv_project/scripts/score_cross_relation_target_logit.py`
- Modify: `/home/sunsik/my_fv_project/scripts/score_cross_relation_unified_drift_control.py`
- Modify: `/home/sunsik/my_fv_project/scripts/run_pt_unified_drift_control_llama70b.sh`
- Modify: `/home/sunsik/my_fv_project/scripts/build_pt_valid_answer_scaffold.py`
- Modify: `/home/sunsik/my_fv_project/scripts/build_pt_unified_human_report.py`
- Modify: `/home/sunsik/my_fv_project/fv/pt_selected_targets.py`
- Create: `/home/sunsik/my_fv_project/scripts/recompute_pt_unified_from_edge_cache.py`
- Optional Modify: `/home/sunsik/my_fv_project/scripts/build_pt_unified_standalone_html.py`
- Update if execution lands: `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`

## Ordered Implementation Steps

1. `Extend the shared lexical top-k helper in score_cross_relation_target_logit.py so _collect_edge_topk() returns lexical_candidate_logits and lexical_candidate_ranks alongside the existing candidate token ids, texts, canonicals, logprobs, and probs. Keep raw_top_* outputs unchanged.`
2. `Update score_cross_relation_unified_drift_control.py so the enriched lexical candidate fields flow into pt_unified_edge_topk.jsonl for every cached row. In selected-target mode, it must also resolve the approved selected target for the current unit and store forced_selected_target_* score fields for that row even when the selected target is not part of lexical top-20.`
3. `Set the cache-building default candidate width to 20 in the Unified PT runner path used for this workflow, and make sure run metadata records the effective edge_topk_k value plus authoritative cache-build model/model_spec information so later offline runs know the candidate budget and token-id space that were cached.`
4. `Update build_pt_valid_answer_scaffold.py so review scaffolds can expose the richer cached candidate summary if useful, especially candidate logits/ranks aggregated from the new edge cache, while still preserving the existing selected_target review fields.`
5. `Update fv/pt_selected_targets.py if needed so cache-build scoring can resolve selected targets cheaply and consistently while building forced selected-target score fields inside the edge-topk rows.`
6. `Create recompute_pt_unified_from_edge_cache.py. It should accept --topk_jsonl, --selected_targets_json, --source_run_dir or explicit cache-build metadata inputs, --out_csv, and optional qid/shot/family filters. It should default to the review-covered q/family/regime subset already approved for this workflow. For each cached row in scope it should: build the unit id, resolve the selected target, compute the selected target first-token id, try lexical-candidate lookup first, fall back to forced_selected_target_* fields if lexical lookup misses, emit a recomputed row with target_* bound to the selected target, and fail hard only if neither lexical nor forced selected-target score data is available. It must also fail hard if the recompute invocation does not match the cache-build model/model_spec/tokenizer assumptions recorded in metadata.`
7. `Inside recompute_pt_unified_from_edge_cache.py, recompute target_s_norm with the same robust min-max q_id-family normalization currently used in the live scorer so compute_product_test_bootstrap_unified.py can be reused unchanged.`
8. `Add a wrapper path for offline recompute. The minimal acceptable version is an extension to run_pt_unified_drift_control_llama70b.sh with a stage or mode switch that can run bootstrap and human-report generation from a recomputed sweep CSV without launching model inference. A separate dedicated wrapper is also acceptable if that yields a cleaner separation. The wrapper must pass or mirror the source pt_unified_family_eligibility.csv into the report stage.`
9. `Update build_pt_unified_human_report.py, and build_pt_unified_standalone_html.py if necessary, so offline recompute runs can point to the source edge-topk cache and source eligibility CSV explicitly and still generate a consistent HTML report whose labels come from the recomputed selected-target sweep rows.`
10. `Run a slice validation: produce a small cache-build run, then recompute with a selected-target artifact whose choices are already known, and compare offline-recomputed target_logit/logprob/prob and target_s_norm rows against the live scorer output for the same selection. The validation must include at least one row where the selected target is outside lexical top-20 so the forced selected-target fallback is exercised.`
11. `After validation passes and if the execution phase is approved, perform the approved cache-build run with k=20 for q_ids Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18 and family_ids BASE_ABD,CTX_ABD only, then use the existing merged selected-target artifact to perform offline recompute, bootstrap, and HTML generation for that same review-covered subset. Any later expansion beyond that subset requires a broader reviewed selected-target artifact first.`

## Validation Plan

- `Schema validation: confirm pt_unified_edge_topk.jsonl rows contain lexical_candidate_logits and lexical_candidate_ranks with the same list length as lexical_candidate_token_ids, lexical_candidate_logprobs, and lexical_candidate_probs.`
- `Forced-target schema validation: confirm pt_unified_edge_topk.jsonl rows produced in selected-target cache-build mode contain forced_selected_target_* score fields for every in-scope row, regardless of whether the selected target is in lexical top-20.`
- `Semantic validation: confirm candidate extraction still uses lexical filtering only and does not silently change the existing raw_top_* fields or target first-token semantics.`
- `Offline recompute correctness: for a small q subset, compare the offline recomputed sweep rows against direct live-scored selected-target rows and require exact equality for target_first_token_id, target_logit, target_logprob_raw, target_prob_raw, and target_rank_in_vocab.`
- `Normalization validation: confirm recomputed target_s_norm matches the live scorer for the same selected target on the validation subset.`
- `Forced-fallback validation: confirm the offline recompute script succeeds when a selected target's first token is absent from lexical top-20 but present in forced_selected_target_* cache fields, and records that fallback path in row metadata.`
- `Missing-candidate validation: confirm the offline recompute script fails clearly only when a selected target's first token is absent from both lexical top-20 and forced selected-target cache fields.`
- `Scope validation: confirm the wrapper and offline recompute stage are restricted to q_ids Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18 with family_ids BASE_ABD,CTX_ABD and the six approved base/context regimes, rather than silently attempting full Unified coverage.`
- `Tokenizer-guard validation: confirm recompute fails fast when cache-build metadata and recompute model/model_spec assumptions do not match.`
- `Bootstrap compatibility validation: run compute_product_test_bootstrap_unified.py on the recomputed sweep CSV and confirm it completes without schema changes.`
- `Report validation: generate HTML from a recomputed run and confirm target labels, gold target labels, top-k candidate tables, and family eligibility rendering remain aligned when the source eligibility CSV is provided explicitly.`

## Expected Outputs

- `/home/sunsik/my_fv_project/plans/2026-03-24-pt-candidate-logit-cache-tech-spec.md`
- `An enriched scratch-first /scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/pt_unified_edge_topk.jsonl with lexical candidate logits and ranks`
- `Forced selected-target score fields inside the same pt_unified_edge_topk.jsonl rows for selected-target cache-build runs`
- `A new offline recompute sweep CSV such as /scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/pt_unified_shot_sweep.csv generated without model inference`
- `A bootstrap summary CSV from the recomputed sweep`
- `A human_report/ HTML tree generated from the recomputed run`
- `A recompute run directory or wrapper output that records the source edge-topk cache path and source eligibility CSV path used for reporting`
- `If execution is later approved, a final execution report under /home/sunsik/my_fv_project/reports`

## Brain Docs To Update

- `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`

## Recommended Execution Strategy

- Launcher: `srun`
- Compute Mode: `gpu`
- Reason: `The first approved execution after the tech spec should include an interactive validation slice before any long full-run cache build. srun is the most pragmatic launcher for that because it keeps the workflow inspectable while still providing the required GPU resources for the Llama 70B inference-backed cache-build stage. The offline recompute, bootstrap, and HTML stages are lightweight, but the gating execution still depends on a GPU run to create the candidate-logit cache.`

## User Execution Settings Required Before Run

- Launcher choice: `srun` or `sbatch`, unless the user explicitly accepts the recommendation above
- Time limit: `required`
- Day-based duration if relevant: `optional`
- GPU options: `required`
- CPU count: `required`
- Memory: `required`
- Partition or queue: `required`
- Job name: `required`
- Log path: `required`
- Environment setup: `required`
- Extra launcher flags: `optional`

## Approval Note

Implementation and execution have not started. Waiting for Korean approval.
