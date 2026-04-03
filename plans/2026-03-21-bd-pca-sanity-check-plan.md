# Plan Template

## Title

`BD PCA sanity check for common-relation vs alternation`

## Metadata

- Date: `2026-03-21`
- Slug: `bd-pca-sanity-check`
- Approval Status: `approved`

## Objective

`Plan a minimal, high-signal Q1-only workflow to test whether BD interleave behavior is better explained by a shared common relation or by learned alternation structure, using a shared BD-specific head reference and common PCA before any heavier downstream analysis.`

## Current Context

`This repo's stable analysis backbone is condition-qwise -> vector extraction -> common PCA -> downstream multi-feature analysis. The current stable condition pipeline supports AAA/BBB/BABA, with a separate D extension for DDD/DADA. PT already includes a BD alternating regime (BDBDBD_D), but the representation-analysis branch does not yet have a stable BD-specific PCA path or a BD-specific shared top-head reference. For Q1 specifically, BBB/DDD StepD artifacts already exist under scratch, so the planned extension can be narrowed to reuse those existing pure-condition rankings and only add the missing mixed-BD conditions. Current AAA_ref is useful for A-centered interpretation, but it is not a neutral measurement basis for comparing BBB/DDD/BDBDBD/DBDBDB conditions. Canonical large outputs should live under /scratch/sunsik/my_fv_project.`

## Assumptions

- `Q1` is the only target q in this first sanity-check pass.
- `AAAAAA`-anchored head sets are not sufficient as the only measurement basis for BD-focused geometry.
- The first question is qualitative-but-rigorous geometry confirmation, not yet full behavioral or stepwise mechanistic proof.
- Existing `BBB` and `DDD` StepD outputs for `Q1` are reusable as pure-anchor ranking inputs.
- New StepD-based head selection is only expected for the missing mixed BD condition family.

## Inputs And Dependencies

- `docs/brain/pipelines/condition_qwise.md`
- `docs/brain/pipelines/pt.md`
- `scripts/run_condition_qwise_pipeline.py`
- `scripts/run_d_extension_stepd_pca.py`
- `scripts/run_condition_common_pca.py`
- `fv/condition_trials.py`
- `fv/head_vector_extract.py`
- existing per-q artifacts under `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise`

## Proposed Steps

1. `Fix the Q1-only BD PCA comparison set and success readout: BBB, DDD, BDBDBD_D, DBDBDB_B, with one fixed shared reference pool used for all compared conditions.`
2. `Reuse the existing Q1 BBB/DDD StepD results as pure-anchor ranking inputs, and add only the missing mixed-BD fixed-trial JSON generation needed for BDBDBD_D and DBDBDB_B.`
3. `Run the existing StepD runner on those new mixed-BD trial JSON files rather than designing a new StepD method; the intended reuse pattern is trial generation extension plus existing StepD execution.`
4. `Build one shared BD reference set from BBB, DDD, BDBDBD_D, and DBDBDB_B ranking information without reusing AAA_ref as the sole measurement basis.`
5. `Extract per-trial vectors for all four BD comparison conditions under that one shared reference and run common PCA to produce 2D/3D plots, centroids, and pairwise distance summaries.`
6. `Use the PCA readout only as a sanity check: if the two mixed BD conditions collapse toward a shared middle region, the common-relation hypothesis remains viable; if they split toward pure B-only vs D-only anchors, alternation becomes the leading explanation.`
7. `Decide after PCA whether to proceed to stronger follow-up analysis such as raw-space distance quantification, shuffled-order controls, or stepwise state extraction.`

## Risks And Blockers

- `If each condition uses its own top-head pool, geometry becomes non-comparable and the PCA result is not interpretable.`
- `Current condition-qwise code is locked to AAA/BBB/BABA, so adding BDBDBD_D and DBDBDB_B may require a parallel extension rather than a trivial flag change.`
- `PT currently exposes BDBDBD_D but not the symmetric DBDBDB_B path in the stable representation-analysis flow, so trial-generation support may be incomplete.`
- `Even though BBB/DDD StepD already exists, pure-anchor vectors may still need re-extraction under the new BD shared reference; old AAA_ref/union_ref vectors are not automatically reusable for a new measurement basis.`
- `A purely qualitative PCA result can be suggestive but not decisive; it must be treated as a gate for deeper follow-up, not final proof.`
- `Fresh model forwards and StepD sweeps likely require GPU and scratch-first storage.`

## Recommended Compute Mode

- Mode: `gpu`
- Why: `This plan is for a representation-analysis sanity check that likely requires new StepD scores, new vector extraction forwards, and common PCA artifacts for new BD conditions. CPU-only local execution is likely to be too slow or operationally brittle for the intended workflow.`

## Expected Outputs

- `plans/2026-03-21-bd-pca-sanity-check-plan.md`
- `plans/2026-03-21-bd-pca-sanity-check-tech-spec.md`
- scratch-side BD comparison artifacts under `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/...`
- shared-reference top-head set for BD PCA comparison
- Q1-only vector artifacts for `BBB`, `DDD`, `BDBDBD_D`, and `DBDBDB_B` under the new shared reference
- PCA outputs including `pca_points.csv`, `pca_centroids.csv`, `distance_summary.json`, and figure files for the BD condition set

## Success Criteria

- `A concrete implementation path is defined for generating a shared BD reference instead of relying only on AAA_ref.`
- `The plan reuses existing Q1 BBB/DDD StepD outputs rather than rerunning unnecessary pure-condition ranking work.`
- `The planned comparison set is fixed and symmetric enough to distinguish shared-middle vs endpoint-split behavior.`
- `The first execution target is limited enough to run as a sanity check rather than an open-ended pipeline rewrite.`
- `The resulting PCA artifacts would let us make a clear next-step decision: continue to stronger controls or stop and revise the hypothesis.`

## Brain Impact

- Brain impact: `update required`
- Why: `If this plan is executed, it will likely add a stable new BD-focused PCA workflow or at minimum a stable new open-question/resolution path for representation analysis, which belongs in current project knowledge rather than in one-off scratch notes.`

## Approval Note

Substantial execution has not started. Waiting for Korean approval.
