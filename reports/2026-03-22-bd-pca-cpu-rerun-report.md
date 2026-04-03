# Final Report Template

## Status

- Status: `success`
- Date: `2026-03-22`

## Summary

`Cancelled the pending topk=12 GPU rerun job and reran only the common PCA step locally on CPU using the already existing BD_ref vectors. The CPU rerun completed successfully and produced artifacts identical to the original BD_ref PCA summary.`

## Source Documents

- Plan: `plans/2026-03-21-bd-pca-sanity-check-plan.md`
- Tech Spec: `plans/2026-03-21-bd-pca-sanity-check-tech-spec.md`

## Execution Settings

- Launcher: `local`
- Compute Mode: `local`
- Time limit: `n/a`
- Day-based duration if relevant: `n/a`
- GPU options: `none`
- CPU count: `default local`
- Memory: `default local`
- Partition or queue: `n/a`
- Job name: `q1_bd_pca_k12` cancelled before start
- Environment setup: `source /home/sunsik/.venvs/pt442/bin/activate`
- Extra launcher flags: `n/a`

## Commands Run

1. `scancel 8827106`
2. `sacct -j 8827106 --format=JobID,JobName%24,State,ExitCode,Elapsed,Timelimit -n -P`
3. `/home/sunsik/.venvs/pt442/bin/python scripts/run_condition_common_pca.py --q_dir /scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1 --ref_mode BD_ref --conditions BBB,DDD,BDBDBD_D,DBDBDB_B --n_components 3 --balance_trials 1 --seed 0 --out_subdir BD_ref_BD_compare_cpu_rerun`
4. `diff -u .../BD_ref_BD_compare/distance_summary.json .../BD_ref_BD_compare_cpu_rerun/distance_summary.json`
5. `diff -u .../BD_ref_BD_compare/pca_centroids.csv .../BD_ref_BD_compare_cpu_rerun/pca_centroids.csv`

## Files Executed

- `scripts/run_condition_common_pca.py`

## Files Changed

- Created: `reports/2026-03-22-bd-pca-cpu-rerun-report.md`
- Created at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/BD_ref_BD_compare_cpu_rerun/*`
- Cancelled job: `8827106`

## Output Artifacts

- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/BD_ref_BD_compare_cpu_rerun/pca_points.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/BD_ref_BD_compare_cpu_rerun/pca_centroids.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/BD_ref_BD_compare_cpu_rerun/distance_summary.json`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/BD_ref_BD_compare_cpu_rerun/scatter.png`

## Log Paths

- `n/a for local rerun beyond terminal invocation`

## Validation Results

- `Job 8827106 was cancelled before start: sacct reports CANCELLED by user with elapsed 00:00:00.`
- `CPU rerun completed successfully.`
- `distance_summary.json` is byte-identical to the original BD_ref PCA output.`
- `pca_centroids.csv` is byte-identical to the original BD_ref PCA output.`

## Brain Updates

- Required: `no`
- Updated files: `none`
- Why: `This was an execution-control change plus a duplicate PCA rerun with identical outputs, so no stable project knowledge changed.`

## Result Explanation

- `What changed: the pending topk=12 GPU rerun was stopped and replaced with a PCA-only local rerun on existing BD_ref vectors.`
- `What was run: only scripts/run_condition_common_pca.py on the precomputed BD_ref vectors for BBB, DDD, BDBDBD_D, and DBDBDB_B.`
- `What outputs mean: the rerun confirms PCA itself does not require GPU when the vectors already exist, and it reproduces the original BD_ref PCA exactly.`
- `Remaining risks or next steps: if you want a true top12-head PCA, that still requires either per-head saved contributions or a fresh vector re-extraction under the top12 ref.`

## Retry Record

- Retry attempted: `no`
- Reason: `n/a`

## Failure Details

- Failure point: `n/a`
- Error summary: `n/a`
