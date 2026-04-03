# Plan Template

## Title

`Q1 BABABA vs BBBBBB behavior comparison`

## Metadata

- Date: `2026-03-21`
- Slug: `bababa-vs-bbbbbb-behavior`
- Approval Status: `pending`

## Objective

`Establish a Q1-focused same-query behavior-summary analysis that treats AAA vs BABABA as the primary fair comparison and uses BBBBBB only as a reference endpoint, using existing stored artifacts only.`

## Current Context

`The current multi-feature branch already shows that Q1 BABABA drifts toward the B endpoint in state space and inside-A feature space. Inspection of the stored condition trials shows an important fairness constraint for behavior work: AAA and BABA share the same query by trial_id and query_source_index, but BBB uses the paired B-side query at the same source slot rather than the same lexical query. That means a direct raw behavior comparison between BABA and BBB is not a same-query comparison. This revised plan therefore stays narrower and fairer: it will summarize same-query AAA vs BABA behavior first, and use BBB only as a clean B-side reference endpoint, without adding a new raw natural-behavior evaluation stage or a representation-to-behavior synthesis stage.`

## Assumptions

- `The first pass should stay Q1-focused because Q1 already has the richest existing interpretive context and aligned condition artifacts for AAA, BBB, and BABA.`
- `Existing AAA and BABA trial definitions are sufficient to align a fair same-query shared-trial comparison through trial_id and query_source_index.`
- `Existing BBB trials can be used only as a source-index-aligned clean B reference, not as a same-query natural behavior comparator.`
- `The existing stored BBB and BABA artifacts are sufficient for a scoped first-pass comparison, even if they do not fully answer the stronger raw natural-behavior question.`
- `Because this revised scope excludes new model evaluation, a local CPU run should be sufficient for the planned work.`

## Inputs And Dependencies

- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_trials/condition_BBB.json`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_trials/condition_BABA.json`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_trials/condition_AAA.json`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_stepd/sampled_trials_BBB.json`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_stepd/sampled_trials_BABA.json`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_stepd/sampled_trials_AAA.json`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_stepd/trial_metrics_BBB.jsonl`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_stepd/trial_metrics_BABA.jsonl`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_stepd/trial_metrics_AAA.jsonl`
- `Relevant current interpretation docs under docs/multi_feature_reweighting/ and docs/brain/analysis/`
- `Model/runtime environment used by the condition-qwise branch if new raw behavior evaluation must be executed`

## Proposed Steps

1. `Audit the available Q1 AAA, BABA, and BBB artifacts to verify the comparison contract: AAA vs BABA will be matched by identical trial_id, query_source_index, and query anchor, while BBB will be treated separately as a source-index-aligned clean B reference rather than a same-query comparator.`
2. `Define the behavior readouts for the primary same-query AAA vs BABA comparison, with priority on target logprob, target probability, rank, top-1 match, top-candidate identity and overlap, and a same-query candidate-distribution comparison on aligned trials.`
3. `Design a derived analysis stage that computes two linked but distinct summaries from existing stored artifacts only: (a) a primary same-query AAA vs BABA behavior summary, and (b) a secondary BBB reference table that reports the clean B-side endpoint outputs at the same source slots without pretending that this is a same-query comparison.`
4. `If the stored metrics turn out to be too limited for a strong same-query conclusion, stop at an explicit limitation statement rather than extending scope to a new raw evaluation stage or a new trial-generation stage in this task.`
5. `If the stored-artifact comparison is clear enough to stand on its own, summarize the result in the final run report; if not, record the limitation and explicitly note that a true clean-B same-query control would require a new condition such as all-B demos with the same A query in a later task.`

## Risks And Blockers

- `The existing StepD trial metrics are intervention or patch metrics, so this reduced-scope plan may only support a weaker behavior-adjacent conclusion rather than a full raw natural-behavior claim.`
- `A direct BBB vs BABA behavior comparison is query-confounded because BBB changes the query itself; if that distinction is ignored, the conclusion would be methodologically weak.`
- `Shared AAA/BABA trial alignment may be narrower than expected if the stored trial-id overlap is smaller than the nominal 25-trial condition size.`
- `A behavior result may show only partial convergence, which would complicate the simple reading "BABABA behaves like BBBBBB" and require more careful phrasing.`
- `Because raw evaluation is out of scope in this revision, the task may end with a clear limitation report instead of a strong yes or no answer.`

## Recommended Compute Mode

- Mode: `local`
- Why: `This revised scope is limited to auditing and summarizing existing stored artifacts. No new model evaluation is planned, so a local CPU run is the appropriate recommendation.`

## Expected Outputs

- `plans/2026-03-21-bababa-vs-bbbbbb-behavior-tech-spec.md`
- `A new scratch analysis root for Q1 same-query AAA-vs-BABA behavior comparison outputs`
- `A paired trial-level behavior comparison artifact for Q1 AAA vs BABA`
- `A secondary BBB reference artifact keyed by the same source slots`
- `An aggregate summary artifact that states what can and cannot be concluded about AAA vs BABA behavior and about BBB as a non-same-query reference`
- `A compact top-candidate comparison view for the same-query AAA vs BABA comparison`
- `A final execution report under reports/ if the later run is approved and executed`

## Success Criteria

- `The primary comparison uses an explicitly defined aligned trial set for Q1 AAA and BABA with the same query anchor rather than mixing query-mismatched examples.`
- `The resulting analysis reports at least target probability, target logprob, rank-style behavior, and top-candidate comparison for the same-query AAA vs BABA comparison.`
- `Any BBB readout is clearly labeled as a source-index-aligned clean B reference rather than a same-query behavior comparison.`
- `The final summary clearly distinguishes direct findings from limitations caused by relying only on stored artifacts.`
- `The plan does not expand into new raw evaluation or broader representation-synthesis work unless separately approved later.`

## Brain Impact

- Brain impact: `none`
- Why: `This revised scope is intentionally narrower and may end in a limitation-aware summary rather than a stable upgrade of project knowledge. Unless the stored-artifact comparison is unexpectedly decisive, the result should stay in the dated run outputs rather than changing the stable brain docs.`

## Approval Note

Substantial execution has not started. Waiting for Korean approval.
