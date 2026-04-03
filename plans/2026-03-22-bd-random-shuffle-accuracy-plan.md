# Plan Template

## Title

`BD alternating vs random-shuffled accuracy comparison`

## Metadata

- Date: `2026-03-22`
- Slug: `bd-random-shuffle-accuracy`
- Approval Status: `approved`

## Objective

`Create an execution plan to compare existing alternating BD mixed regimes against multiple newly added random-shuffled BD mixed regimes while keeping query identity and trial alignment fixed, then measure top-1 accuracy, target logit/logprob, and competing candidate behavior under the same evaluation protocol.`

## Current Context

`The repo already contains a BD mixed-condition generation path and scratch outputs for Q1 under relation_condition_qwise, including alternating mixed conditions BDBDBD_D and DBDBDB_B. Existing scratch evidence shows that BDBDBD_D already has unified-PT accuracy-style summaries, while DBDBDB_B already exists as a condition/trial/StepD artifact on scratch but does not currently appear to have a matching precomputed top-1 accuracy summary in the same PT summary format, so that baseline likely needs to be generated. The user clarified that the shuffle manipulation should not preserve the old alternating-count structure at each prefix. Instead, it should preserve only the total B/D counts and the same query, while allowing much more irregular and even strongly clumped layouts such as BBBBDDDDB-like patterns. The shuffled comparison should also avoid depending on only one random realization, so the current desired design is to create about five deterministic shuffled cases per query side rather than a single shuffled control.`

## Assumptions

- `Random-shuffled BD regimes must preserve the original query side and query row so that each alternating regime is compared against a same-query shuffled counterpart.`
- `Random-shuffled BD regimes must preserve the original B/D shot counts for each alternating regime, changing only the presentation order and not the relation inventory.`
- `The preferred shuffle control should preserve the exact concrete mixed demo set from the alternating trial and only permute demo order, rather than remapping source identity onto different row indices.`
- `Using multiple shuffled cases is scientifically better than using a single shuffled case because it reduces the chance that one accidental permutation dominates the conclusion.`
- `A practical first design is five deterministic shuffled cases for the D-query side and five deterministic shuffled cases for the B-query side, with each case defined by a stable shuffle seed family.`
- `The shuffled cases should deliberately span different irregularity strengths, from mild local alternation breaks to heavily clumped layouts, because the behavioral question is whether strict regularity itself is carrying the effect.`
- `The scientifically relevant comparison should use the same summary protocol across all compared regimes: top-1 accuracy, target logit/logprob, target rank, and leading competing candidates.`
- `Running the needed evaluation at the intended model scale will require GPU-backed execution rather than a lightweight local CPU-only pass.`

## Inputs And Dependencies

- `Existing BD mixed-condition precedent in /home/sunsik/my_fv_project/scripts/run_bd_interleave_pca.py`
- `Existing PT scoring and summary scripts under /home/sunsik/my_fv_project/scripts/`
- `Scratch canonical results root /scratch/sunsik/my_fv_project`
- `Existing alternating BD artifacts under /scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1`
- `Existing unified PT summary artifacts under /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_unified_pt_20260311_071325`
- `Model/runtime availability for the approved launcher, likely using /scratch/sunsik/models/Llama-3.1-70B`

## Planned Shuffled Case Layouts

`The planned shuffled cases preserve only the total source counts and the same query side. They do not preserve the old prefix-wise alternation balance. This is intentional: the goal is to test whether the model depends on strict alternation regularity, so the shuffled family must include both mildly irregular and strongly clumped layouts. The cases below are fixed, deterministic comparison layouts for the planning stage.`

- `Regular D-query baseline: BDBDBDBDB`
- `Shuffled D-case 1 (mild local break): BDDBBDBDB`
- `Shuffled D-case 2 (paired blocks): BBDDBDDBB`
- `Shuffled D-case 3 (front-loaded B cluster): BBBDDBDDB`
- `Shuffled D-case 4 (strong clump): BBBBDDDDB`
- `Shuffled D-case 5 (opposite-start irregular): DBBBDDBBD`
- `Regular B-query baseline: DBDBDBDBD`
- `Shuffled B-case 1 (mild local break): DBBDDBDBD`
- `Shuffled B-case 2 (paired blocks): DDBBDBBDD`
- `Shuffled B-case 3 (front-loaded D cluster): DDDBBDBBD`
- `Shuffled B-case 4 (strong clump): DDDDBBBBD`
- `Shuffled B-case 5 (opposite-start irregular): BDDDBBDDB`

`The B-query cases are selected as structurally comparable B<->D mirrors of the D-query cases so the two query sides remain interpretable as a pair. The important property is not exact symmetry at each prefix, but that both sides cover a similar range from mild irregularity to heavy clumping while keeping total counts fixed.`

## Proposed Steps

1. `Audit the existing alternating BD generation and evaluation paths to lock the comparison contract: exact regime naming, trial alignment fields, query source behavior, summary outputs, and the gap between BDBDBD_D and DBDBDB_B baseline accuracy coverage.`
2. `Design a deterministic random-shuffled BD trial builder that starts from each existing alternating trial, keeps the same query and the same concrete mixed demos, and applies only a seeded permutation to demo order. The shuffle must preserve source counts and must reject perfectly alternating layouts so the result is genuinely irregular.`
3. `Materialize approximately five shuffled case families for the D-query side and five shuffled case families for the B-query side, each one representing a distinct deterministic shuffle seed family built from the same underlying baseline trials.`
4. `Treat BDBDBD_D and DBDBDB_B as the regular baseline regimes. Reuse their existing scratch condition/trial artifacts where valid, and explicitly generate the missing DBDBDB_B baseline accuracy summary if that summary is not already present in a comparable PT format.`
5. `Implement evaluation support so that both alternating and all shuffled BD regimes can be scored under one aligned protocol, producing comparable outputs for top-1 accuracy, target logit/logprob, target rank, and top competing candidates.`
6. `Run the approved evaluation to obtain results for BDBDBD_D, DBDBDB_B, shuffled_BD_D_case1-5, and shuffled_BD_B_case1-5, reusing existing artifacts where safe and only executing the missing paths needed for a fair comparison.`
7. `Aggregate the results into a concise comparison artifact that reports both per-case and aggregated shuffled-vs-regular changes, showing how accuracy and target/candidate score behavior move when strict alternation is replaced by random shuffle, and determine whether any stable docs/brain updates are required after execution.`

## Risks And Blockers

- `DBDBDB_B does not appear to have the same precomputed PT top-1 summary coverage as BDBDBD_D, so baseline generation may need to be added before the regular-vs-shuffled comparison is complete.`
- `If current PT scripts hard-code only CTX_BDBDBD_D and do not generalize cleanly to DBDBDB_B or new shuffled regimes, implementation scope may expand from a small extension into a reusable regime-generalization patch.`
- `A naive random shuffle could accidentally preserve too much of the alternating structure or even reproduce a perfectly alternating layout; the design must explicitly reject that outcome.`
- `If the chosen shuffle method changes concrete demo content instead of only order, the comparison would be confounded; the implementation must keep the exact mixed demo multiset fixed.`
- `Using five shuffled cases per query side increases execution volume materially; runtime and storage planning must treat this as a small sweep rather than a one-off comparison.`
- `If different shuffled cases vary widely, the result may require reporting not just a mean but also spread, worst-case, and best-case behavior to avoid overstating stability.`
- `Because the intended model path is large, runtime and launcher settings matter; using the wrong execution mode could block or invalidate the approved run.`
- `If the comparison requires changes to stable regime naming or pipeline entrypoints, docs/brain updates may become necessary rather than optional.`

## Recommended Compute Mode

- Mode: `gpu`
- Why: `The comparison depends on model-backed PT or equivalent scoring at the existing project scale, and the established artifacts and scripts are tied to large-model inference workflows that are not realistically replaceable with a local CPU-only run.`

## Expected Outputs

- `Approved planning artifact at /home/sunsik/my_fv_project/plans/2026-03-22-bd-random-shuffle-accuracy-plan.md`
- `Follow-on tech spec after plan approval`
- `If later executed: one newly generated DBDBDB_B baseline accuracy summary in the same protocol family as the alternating comparison`
- `If later executed: shuffled BD regime artifacts under scratch for about five D-query shuffled cases and five B-query shuffled cases`
- `If later executed: aligned scoring outputs and a comparison summary covering top-1 accuracy plus target/candidate score changes, both per shuffled case and in aggregated form`
- `If later executed: final run report under /home/sunsik/my_fv_project/reports/`

## Success Criteria

- `The plan clearly defines a same-query, same-template comparison between alternating BD regimes and shuffled BD regimes.`
- `The shuffled control is defined as an order-only permutation of the original mixed demo set, not as a content-changing rebuild.`
- `The approved execution path will produce directly comparable results for BDBDBD_D, DBDBDB_B, shuffled_BD_D_case1-5, and shuffled_BD_B_case1-5 using one common metric protocol.`
- `The plan explicitly includes generation of the missing DBDBDB_B baseline accuracy summary before or alongside the shuffled comparison.`
- `The work will explicitly quantify both top-1 accuracy changes and target/candidate score changes rather than relying only on PCA or representation summaries.`
- `The work will report not just one shuffled outcome but a small distribution over shuffled cases so the conclusion is not tied to a single permutation.`
- `The plan identifies whether the current environment already contains reusable baselines and where missing baseline coverage must be filled in.`

## Brain Impact

- Brain impact: `update required`
- Why: `If this work is implemented as intended, it likely adds a stable new interpretation and a stable new regime/evaluation path for BD mixed-condition validation, which belongs in current project knowledge rather than only in a dated run report.`

## Approval Note

Substantial execution has not started. Waiting for Korean approval.
