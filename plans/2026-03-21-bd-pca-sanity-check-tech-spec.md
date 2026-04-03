# Tech Spec Template

## Title

`Q1 BD-ref PCA sanity check implementation`

## Metadata

- Date: `2026-03-21`
- Slug: `bd-pca-sanity-check`
- Source Plan: `plans/2026-03-21-bd-pca-sanity-check-plan.md`
- Approval Status: `approved`

## Scope

- `Implement a Q1-only BD PCA sanity-check path that uses a shared BD_ref measurement basis.`
- `Reuse existing Q1 BBB/DDD trial and StepD artifacts instead of rerunning those pure-anchor conditions.`
- `Generate only the missing mixed-BD fixed-trial payloads for BDBDBD_D and DBDBDB_B.`
- `Run the existing StepD runner on those new mixed-BD payloads.`
- `Build a shared BD_ref from BBB, DDD, BDBDBD_D, and DBDBDB_B StepD rankings.`
- `Extract vectors for all four comparison conditions under BD_ref and run common PCA on that shared basis.`
- `Write artifacts scratch-first under the existing Q1 runtime tree.`

## Out Of Scope

- `No multi-q rollout beyond Q1.`
- `No shuffled-order BD control in this pass.`
- `No stepwise state extraction or stepwise PCA in this pass.`
- `No endpoint-distance or raw-space metric suite beyond the PCA artifacts and centroid-distance summary already produced by the common PCA runner.`
- `No rewrite of the stable AAA/BBB/BABA condition pipeline into a fully generic arbitrary-condition framework.`

## Implementation Design

`The implementation will follow a reuse-first extension pattern rather than a pipeline rewrite. A new focused runner will operate on one existing q_dir, expected first target Q1. It will validate that existing BBB and DDD artifacts are present, then reuse those existing trial/StepD outputs as pure-anchor inputs. The runner will generate only the missing mixed-BD fixed-trial JSON payloads using the same trial schema already used by condition_BBB.json and condition_DDD.json. The mixed payload builder will copy trial_id, trial_idx, query_source_index, and demo_source_indices from an existing pure-condition template so the new conditions stay trial-aligned with the reused anchor conditions.`

`For StepD, the runner will reuse scripts/run_stepD_aie_head_sweep.py exactly as-is. The only new work before StepD is producing fixed_trials-style JSONs for BDBDBD_D and DBDBDB_B. That preserves the existing scoring logic and avoids introducing a second StepD implementation.`

`The mixed-BD layout rule will mirror the existing condition builders. The implementation will inherit n_demos from the reused Q1 template rather than inventing a new shot count. For Q1 this currently means 9 demos plus 1 query. BDBDBD_D will therefore use demo slots in the order B,D,B,D,B,D,B,D,B and then a D query. DBDBDB_B will use D,B,D,B,D,B,D,B,D and then a B query. This preserves the existing trial count and index alignment while making the total source counts balanced across demo+query: each final prompt contains 5 examples from one source in demos and 4 from the other, with the query coming from the minority side so the total becomes 5 vs 5 across the full prompt.`

`For head selection, the runner will build a new shared BD_ref by loading StepD score CSVs for BBB, DDD, BDBDBD_D, and DBDBDB_B, converting them to rank maps, and using the same mean-rank aggregation idea already used for union_ref. The resulting shared set will be saved as a new reference artifact, for example _top_heads/sets/top_heads_ref_BD.json.`

`For vector extraction, the runner will reuse fv.head_vector_extract.extract_condition_trial_vectors rather than inventing new activation-capture code. The implementation will pass the BD_ref head set as the comparison reference and save the resulting arrays as trial_vectors_BD_ref_<COND>.npy for BBB, DDD, BDBDBD_D, and DBDBDB_B. Existing BBB/DDD vectors under AAA_ref or union_ref will not be treated as reusable for this purpose because they were measured under a different head basis.`

`For PCA, scripts/run_condition_common_pca.py will be minimally generalized so --ref_mode can accept BD_ref instead of being hardcoded to AAA_ref/union_ref only. Then the new runner can call it with conditions BBB,DDD,BDBDBD_D,DBDBDB_B and an isolated out_subdir such as BD_ref_BD_compare. This preserves the existing PCA artifact format and makes the BD sanity check comparable to existing condition-qwise PCA outputs.`

`The implementation design intentionally keeps BD-specific logic in a focused new runner instead of widening fv/condition_trials.py into a generic arbitrary-condition system. That choice minimizes blast radius, preserves reuse of existing Q1 outputs, and matches the current task scope.`

## Expected File Changes

- Modify: `scripts/run_condition_common_pca.py`
- Create: `scripts/run_bd_interleave_pca.py`
- Modify: `plans/2026-03-21-bd-pca-sanity-check-plan.md`
- Create: `plans/2026-03-21-bd-pca-sanity-check-tech-spec.md`
- Update at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_trials/*`
- Update at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_stepd/*`
- Update at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_top_heads/sets/*`
- Update at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_vectors/*`
- Update at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/*`

## Ordered Implementation Steps

1. `Generalize scripts/run_condition_common_pca.py so ref_mode is not limited to AAA_ref and union_ref. The loader should continue to resolve files by trial_vectors_<ref_mode>_<condition>.npy with no behavior change for existing refs.`
2. `Create scripts/run_bd_interleave_pca.py as a focused one-q runner that accepts q_dir, q_id, score_key, topk, model-loading options, and scratch output settings.`
3. `Inside that runner, validate reuse prerequisites: existing Q1 condition_BBB.json, condition_DDD.json, aie_scores_BBB.csv, and aie_scores_DDD.csv must exist. Also validate that the pure-anchor trial templates are mutually consistent enough to share trial_id, query_source_index, and demo_source_indices.`
4. `Implement mixed-BD payload generation for BDBDBD_D and DBDBDB_B by reusing the existing trial JSON schema and prompt-construction conventions already used by the current condition trial files. The intended Q1 layout is 9 alternating demos plus 1 query: B,D,B,D,B,D,B,D,B then D-query for BDBDBD_D, and D,B,D,B,D,B,D,B,D then B-query for DBDBDB_B. Save these payloads under _trials/condition_BDBDBD_D.json and _trials/condition_DBDBDB_B.json.`
5. `Call scripts/run_stepD_aie_head_sweep.py for BDBDBD_D and DBDBDB_B only. Reuse existing BBB/DDD StepD score CSVs without rerunning them. Copy or record the resulting outputs under the standard _stepd naming pattern.`
6. `Build the shared BD_ref from BBB, DDD, BDBDBD_D, and DBDBDB_B StepD rankings using mean-rank aggregation. Save the resulting ref set under _top_heads/sets/top_heads_ref_BD.json and record metadata about source conditions, topk, and score_key.`
7. `Run vector extraction for BBB, DDD, BDBDBD_D, and DBDBDB_B under BD_ref using the existing extract_condition_trial_vectors utility. Save outputs as trial_vectors_BD_ref_<COND>.npy and update vector_extraction_meta.json with trial_ids and seq_token_indices for the new mixed conditions plus BD_ref extraction metadata if needed.`
8. `Run scripts/run_condition_common_pca.py with --ref_mode BD_ref, --conditions BBB,DDD,BDBDBD_D,DBDBDB_B, and a dedicated out_subdir for the BD sanity-check outputs.`
9. `Validate that the PCA outputs exist and that the centroid summary is sufficient to answer the immediate sanity-check question before stopping. Do not extend to shuffled controls or stepwise analysis in this pass.`

## Validation Plan

- `Pre-run validation: confirm Q1 pure-anchor prerequisites exist under scratch before any new work starts.`
- `Schema validation: confirm the new mixed-BD trial JSONs match the key structure used by existing condition_BBB.json / condition_DDD.json.`
- `Reuse validation: confirm BBB and DDD StepD are not rerun when their score CSVs already exist.`
- `Head-set validation: confirm top_heads_ref_BD.json exists, is non-empty, records the four source conditions, and has the requested top-k size.`
- `Vector validation: confirm trial_vectors_BD_ref_BBB.npy, trial_vectors_BD_ref_DDD.npy, trial_vectors_BD_ref_BDBDBD_D.npy, and trial_vectors_BD_ref_DBDBDB_B.npy all exist and share the same feature dimension.`
- `PCA validation: confirm pca_points.csv, pca_centroids.csv, distance_summary.json, and scatter plots are generated in the BD-specific PCA output directory.`
- `Interpretation gate: verify that the outputs let us directly inspect whether mixed BD conditions collapse together or split toward the pure anchors.`

## Expected Outputs

- `plans/2026-03-21-bd-pca-sanity-check-tech-spec.md`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_trials/condition_BDBDBD_D.json`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_trials/condition_DBDBDB_B.json`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_stepd/aie_scores_BDBDBD_D.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_stepd/aie_scores_DBDBDB_B.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_top_heads/sets/top_heads_ref_BD.json`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_vectors/trial_vectors_BD_ref_BBB.npy`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_vectors/trial_vectors_BD_ref_DDD.npy`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_vectors/trial_vectors_BD_ref_BDBDBD_D.npy`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_vectors/trial_vectors_BD_ref_DBDBDB_B.npy`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/BD_ref_BD_compare/pca_points.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/BD_ref_BD_compare/pca_centroids.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/BD_ref_BD_compare/distance_summary.json`

## Brain Docs To Update

- `docs/brain/pipelines/condition_qwise.md` `if the BD-ref PCA runner becomes a stable supported extension path`
- `docs/brain/OPEN_QUESTIONS.md` `if the execution resolves or sharpens the BD common-relation vs alternation question in a way that should remain current`

## Recommended Execution Strategy

- Launcher: `srun`
- Compute Mode: `gpu`
- Reason: `This is a single-q bounded run with reuse of existing artifacts, but it still requires model loading, mixed-condition StepD, fresh vector extraction, and PCA artifact generation. An interactive or near-interactive GPU launch is the most practical recommendation for fast validation and log inspection. If you prefer unattended cluster execution, the same tech spec can be launched via sbatch after approval.`

## User Execution Settings Required Before Run

- Launcher choice: `srun recommended; sbatch acceptable; local not recommended`
- Time limit: `user to specify; recommend at least 2-4 hours`
- Day-based duration if relevant: `not required unless using long sbatch scheduling`
- GPU options: `user to specify; recommend 1 GPU`
- CPU count: `user to specify; recommend 8+ CPUs`
- Memory: `user to specify; recommend 48G-64G`
- Partition or queue: `user cluster choice required`
- Job name: `recommend q1_bd_pca`
- Log path: `user to specify; recommend a repo log under logs/ plus scratch-side runner logs`
- Environment setup: `user to specify if non-default module/venv activation is required`
- Extra launcher flags: `user optional`

## Approval Note

Implementation and execution have not started. Waiting for Korean approval.
