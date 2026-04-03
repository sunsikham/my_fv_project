# Tech Spec Template

## Title

`Unified PT dual-layer candidate cache with shot-wise observed candidates and high-shot canonical bank scoring`

## Metadata

- Date: `2026-03-25`
- Slug: `pt-dual-candidate-cache`
- Source Plan: `plans/2026-03-25-pt-dual-candidate-cache-plan.md`
- Approval Status: `pending`

## Scope

- `Preserve the current shot-wise lexical top-k cache exactly as the observed-candidate layer so each shot / trial / regime row still records what the model actually surfaced at that shot.`
- `Pin the observed-cache build width to the canonical-bank budget: Stage 1 observed-cache discovery for this workflow must persist at least top 20 lexical candidates per row, not the historical top 10 default.`
- `Add a unit-level canonical candidate bank artifact built from high-shot evidence at shots 7 and 9 only, using the review-covered q_ids Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18 and the review-covered BASE_ABD plus CTX_ABD regimes.`
- `Score the canonical candidate bank across every shot row so low-shot rows store logits/logprobs/probs/ranks for the same bank candidates even when those candidates are not present in the observed lexical top-k for that low-shot row.`
- `Keep first-token PT semantics unchanged: both observed-candidate fields and canonical-bank fields refer to first continuation token scores only.`
- `Retain the current HTML/report ability to show shot-wise candidate drift from the observed lexical layer.`
- `Extend offline recompute so selected-target PT can resolve through the canonical bank layer first, then compatibility fallbacks, without another full model rerun when the target is in the canonical bank.`
- `Support the already approved selected-target scope for B/D-query units only, not A-query units or ZERO_CTRL families.`

## Out Of Scope

- `Replacing or removing the current observed lexical top-k cache.`
- `Full semantic relation-validity automation; candidate approval and final target choice remain human-driven or later-rule-driven.`
- `Exact multi-token PT scoring beyond the existing first-token contract.`
- `Guaranteeing rescoring for arbitrary targets that are absent from the canonical bank and absent from any forced selected-target cache.`
- `Changing the current review-covered q scope beyond Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18 without a separate artifact expansion step.`
- `Expanding this design to ZERO_CTRL, A_ONLY, or A-query selected-target analysis in this pass.`
- `Replacing scratch-first PT storage rules or changing canonical PT roots.`

## Implementation Design

`The design becomes explicitly two-layer and multi-artifact. Layer A is the observed shot-wise lexical cache. This remains the current pt_unified_edge_topk.jsonl content: for each shot / trial / regime row, store the actual lexical top-k candidates the model surfaced at that row, including token ids, texts, canonicals, logits, logprobs, probs, and ranks. Layer A exists for HTML drift views and exploratory analysis, not as the only recovery source for later retargeting. Because Stage 2 canonical-bank construction can only promote candidates that Stage 1 actually persisted, the observed-cache discovery stage for this workflow must run with edge_topk_k=20, matching the intended canonical-bank budget.`

`Layer B is a canonical candidate bank built per reviewed unit from high-shot evidence only. The bank-builder reads a completed observed cache run and inspects only shot 7 and shot 9 rows for the relevant unit regimes. For a B-query unit this means the B-side regimes only, and for a D-query unit this means the D-side regimes only. It aggregates lexical candidates across those high-shot rows, computes support statistics such as coverage_rows, top1_count, best_rank, mean_logprob, mean_prob, and source_regimes_seen, and produces a stable bank of up to 20 candidates per unit. Candidate selection should be deterministic and tokenizer-consistent. The recommended ranking rule for the first pass is coverage_rows desc, then top1_count desc, then mean_logprob desc, then candidate text asc. The bank artifact must record that it was derived from shots 7 and 9 so later users can audit why a candidate is in the bank. Because the bank is token-id keyed, the artifact must also carry authoritative tokenizer identity sourced from the cache-build run metadata, at minimum source_model, source_model_spec, and source_run_dir/run_meta provenance sufficient to reject cross-tokenizer reuse.`

`The scoring path then gains a dedicated canonical-bank scoring mode. This mode takes a canonical candidate bank artifact plus the exact source run metadata and trial plan used for the observed cache build. It must not resample trials or reconstruct prompts loosely from q scope alone. Instead, it must replay the source run's authoritative prompt plan from source_run_dir metadata and source sweep rows, including the exact demo_row_ids_used/demo_ids_used or an explicit derived trial-plan artifact. For every shot / trial / regime row, it computes scores for the bank token ids directly from next_logits and next_logprobs, even if those candidates are not present in the observed lexical top-k for that row. The simplest stable schema is to keep the existing observed lexical fields unchanged and add a second set of row fields named canonical_bank_candidates, canonical_bank_candidate_canonicals, canonical_bank_candidate_token_ids, canonical_bank_candidate_logits, canonical_bank_candidate_logprobs, canonical_bank_candidate_probs, canonical_bank_candidate_ranks, canonical_bank_source_shots, and canonical_bank_size. In other words, the final row carries both what was actually observed at that shot and what the high-shot canonical bank scored at that shot. The rescoring refresh must merge back onto the observed cache with a strict 1:1 key check on (family_id, q_id, trial_index, shot, edge), and it must fail if any source row is missing or extra after replay.`

`Because the canonical bank can only be known after inspecting high-shot observed candidates, the practical workflow becomes a staged dual-pass cache workflow. Stage 1 is an observed-cache discovery run that writes the current shot-wise lexical cache with edge_topk_k fixed to 20 for this workflow. Stage 2 is an offline bank-construction step that reads Stage 1 and emits a canonical bank artifact from shots 7 and 9. Stage 3 is a GPU-backed bank-score refresh run that replays the Stage 1 trial plan and scores those canonical bank candidates across all shots. The Stage 3 output becomes the canonical dual-layer cache artifact. This can be implemented either as a new run directory that writes a fresh enriched pt_unified_edge_topk.jsonl containing both observed lexical and canonical-bank fields, or as a merge step that reads the Stage 1 observed cache and writes an enriched copy while attaching the Stage 3 bank-score arrays. The second option is preferred because it avoids losing observed lexical provenance and keeps one final dual-layer cache file per row.`

`Offline recompute then changes lookup priority. For alternative target rescoring the primary lookup path should be the canonical bank arrays, because those are the high-shot-derived candidates intentionally scored across all shots. If the selected target is absent from the canonical bank for a row, the recompute layer may still use the existing forced selected-target fields if available, and finally may use observed lexical top-k as a legacy compatibility path when the token is present there. But the design goal is that future candidate switching should primarily happen within the canonical bank, not depend on low-shot observed lexical inclusion. Both canonical-bank scoring and offline recompute must reject artifacts whose tokenizer/model provenance does not match the authoritative source run metadata.`

`Human-report generation must keep using the observed lexical layer for shot-drift views. That is the point of retaining Layer A. In addition, report generation should surface the canonical bank explicitly, ideally with one per-unit summary table showing the bank candidates, their high-shot provenance statistics, and which selected target is currently active. When rendering a row or aggregate that uses a selected target, the report should also record which lookup layer supplied the score: canonical_bank, forced_selected_target, or observed_lexical. This prevents ambiguity when a selected target is absent from the observed shot-specific top-k but still has a valid canonical-bank score. The staged runner defaults used for this workflow must be narrowed explicitly to family_ids=BASE_ABD,CTX_ABD and positive shots only, rather than inheriting the broader Unified PT defaults that include ZERO_CTRL, A_ONLY, and shot 0.`

`This design supersedes the narrower 2026-03-24 forced-selected-target-only cache design as the long-term preferred PT cache contract. The forced selected-target cache can remain as a compatibility bridge, but the primary retargeting story should move to the high-shot canonical candidate bank.`

## Expected File Changes

- Modify: `/home/sunsik/my_fv_project/scripts/score_cross_relation_target_logit.py`
- Modify: `/home/sunsik/my_fv_project/scripts/score_cross_relation_unified_drift_control.py`
- Modify: `/home/sunsik/my_fv_project/scripts/recompute_pt_unified_from_edge_cache.py`
- Modify: `/home/sunsik/my_fv_project/scripts/build_pt_valid_answer_scaffold.py`
- Modify: `/home/sunsik/my_fv_project/scripts/build_pt_unified_human_report.py`
- Modify: `/home/sunsik/my_fv_project/scripts/run_pt_unified_drift_control_llama70b.sh`
- Modify: `/home/sunsik/my_fv_project/fv/pt_selected_targets.py`
- Create: `/home/sunsik/my_fv_project/scripts/build_pt_unified_canonical_candidate_bank.py`
- Create: `/home/sunsik/my_fv_project/scripts/merge_pt_unified_dual_candidate_cache.py`
- Optional Modify: `/home/sunsik/my_fv_project/scripts/build_pt_unified_standalone_html.py`
- Update if execution is later approved: `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`

## Ordered Implementation Steps

1. `Keep the existing observed lexical top-k row schema stable and rename nothing that downstream tools already depend on. Treat those lexical_* fields as the official Layer A observed-candidate cache.`
2. `Pin Stage 1 observed-cache discovery to edge_topk_k=20 for this workflow and make the staged runner override the current broad Unified defaults with family_ids=BASE_ABD,CTX_ABD and shot_list=1,3,5,7,9.`
3. `Create build_pt_unified_canonical_candidate_bank.py. It should read an observed cache pt_unified_edge_topk.jsonl, restrict to approved q_ids/families/regimes, keep only shot 7 and 9 rows, aggregate lexical candidates per unit, compute deterministic support statistics, and emit a canonical bank artifact with up to 20 candidates per unit plus provenance metadata.`
4. `Define the unit-level canonical bank schema. Each unit record must include q_id, query_source, query_input, gold_target, source_shots=[7,9], source_run_dir, source_topk_jsonl, source_model, source_model_spec, and a candidates list containing at least candidate_text, candidate_canonical, candidate_token_id, coverage_rows, top1_count, best_rank, mean_logprob, mean_prob, and source_regimes_seen.`
5. `Extend score_cross_relation_unified_drift_control.py with a canonical-bank scoring mode or a merge-friendly refresh mode. Given a canonical bank artifact plus the authoritative source sweep/trial metadata, it must replay the exact source prompts and score every bank token id for every in-scope row, then write row-level canonical_bank_* arrays without removing the existing observed lexical arrays.`
6. `Implement merge_pt_unified_dual_candidate_cache.py so a Stage 1 observed cache and a Stage 3 canonical-bank score refresh can be combined into one enriched pt_unified_edge_topk.jsonl whose rows carry both Layer A observed lexical fields and Layer B canonical bank fields, with a strict 1:1 key match check on (family_id, q_id, trial_index, shot, edge).`
7. `Update recompute_pt_unified_from_edge_cache.py so selected-target lookup order becomes canonical_bank first, then forced_selected_target if present, then observed lexical as legacy compatibility. It must record which lookup path was used in selected_target_lookup_source and reject canonical bank artifacts whose model/model_spec/tokenizer provenance does not match the source run metadata.`
8. `Update build_pt_valid_answer_scaffold.py so review scaffolds can still summarize observed shot-wise lexical candidates, but may also expose canonical-bank summaries for the unit when present. This keeps manual review grounded in both actual shot drift and the high-shot bank.`
9. `Update build_pt_unified_human_report.py, and build_pt_unified_standalone_html.py if needed, so HTML keeps the current shot-drift tables from Layer A and adds a canonical-bank section that explains the stable candidate bank used for rescoring.`
10. `Extend run_pt_unified_drift_control_llama70b.sh with explicit staged support for observed-cache build, canonical-bank build, canonical-bank scoring refresh, dual-cache merge, and offline recompute/report generation. The wrapper must not inherit broad Unified defaults for families or shots when invoked in this workflow.`
11. `Run a slice validation on a small q subset that includes Q8. Build observed cache at k=20, build a canonical bank from shots 7 and 9, replay the source trial plan to score that bank across all shots, merge the dual cache, and verify that grain is absent from some low-shot observed lexical top-k rows but present in canonical-bank score arrays for those same rows.`
12. `Run offline recompute on that slice with Q8::B switched from seeds to grain and verify the recompute succeeds without another full model rerun, while the HTML still shows the original shot-wise candidate drift from Layer A.`
13. `After validation passes and only if execution is approved, perform the full reviewed-scope dual-layer cache workflow for q_ids Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18 and family_ids BASE_ABD,CTX_ABD, then regenerate offline PT sweep, bootstrap, and HTML from the enriched dual cache.`

## Validation Plan

- `Observed-layer preservation: confirm the enriched dual cache still contains the current lexical_* arrays and that HTML shot-drift tables remain unchanged when reading only those fields.`
- `Observed-cache budget validation: confirm Stage 1 used edge_topk_k=20 and that bank-builder inputs are not silently truncated to the historical top-10 default.`
- `Canonical-bank correctness: confirm build_pt_unified_canonical_candidate_bank.py only uses shot 7 and 9 rows and produces deterministic up-to-20 candidate banks per unit.`
- `Canonical-bank coverage: for Q8::B, confirm grain enters the canonical bank because of high-shot evidence even though it is absent from many shot-1 observed lexical top-k rows.`
- `Replay-contract validation: confirm the canonical-bank scoring refresh consumes the authoritative source sweep/trial metadata and merges back onto the observed cache with a 1:1 key match and no missing or extra rows.`
- `Row-level score coverage: confirm low-shot Q8::B rows in the enriched dual cache contain canonical_bank_* scores for grain even when lexical_candidate_* fields do not include grain.`
- `Offline recompute correctness: switch Q8::B from seeds to grain and confirm recompute succeeds on all 500 relevant rows without another full model rerun.`
- `Lookup provenance: confirm selected_target_lookup_source distinguishes canonical_bank from forced_selected_target and observed_lexical.`
- `Compatibility validation: confirm already approved targets that remain in lexical or forced caches still recompute correctly after the lookup-order change.`
- `Bootstrap compatibility: run compute_product_test_bootstrap_unified.py on the recomputed sweep CSV and confirm no schema break.`
- `Report validation: generate HTML and confirm shot-wise candidate drift is still visible while canonical-bank contents and selected-target lookup provenance are also rendered.`
- `Scope-default validation: confirm staged runner invocations for this workflow use BASE_ABD,CTX_ABD and positive shots only, and do not trip preflight_review_scope with ZERO_CTRL, A_ONLY, or shot 0 defaults.`
- `Tokenizer-guard validation: confirm canonical-bank scoring and recompute fail fast if the canonical bank artifact provenance does not match the source run model/model_spec/tokenizer identity.`
- `Slice-to-full consistency: confirm the same dual-layer logic used for a Q8 slice scales unchanged to the reviewed full q scope.`

## Expected Outputs

- `/home/sunsik/my_fv_project/plans/2026-03-25-pt-dual-candidate-cache-tech-spec.md`
- `A shot-wise observed cache run directory under /scratch/sunsik/my_fv_project/pt_analysis`
- `A canonical bank artifact such as /scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/pt_unified_canonical_candidate_bank.json`
- `An enriched dual-layer cache artifact whose rows contain both observed lexical and canonical-bank score fields`
- `A recomputed PT sweep CSV that can change selected targets using canonical-bank scores`
- `A bootstrap summary CSV and HTML report generated from that recomputed run`
- `If execution is later approved, a final execution report under /home/sunsik/my_fv_project/reports`

## Brain Docs To Update

- `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`

## Recommended Execution Strategy

- Launcher: `srun`
- Compute Mode: `gpu`
- Reason: `This design introduces a new staged cache workflow with at least one small validation slice and one additional targeted GPU refresh step before any full rerun. srun is the most practical launcher for inspecting those stages interactively, especially because the user may want to confirm canonical-bank behavior on Q8 before committing to the full reviewed-scope run.`

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
