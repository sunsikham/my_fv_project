# Final Report Template

## Status

- Status: `success`
- Date: `2026-03-22`

## Summary

`Implemented the approved Q1-only BD alternating-vs-shuffled behavior comparison path, validated the new scripts for syntax and trial-plan invariants, created an sbatch launcher using the same cluster pattern as prior successful Q1 GPU jobs, submitted an initial all-shot job as 8830003, canceled it after the user changed the scope to shot 9 only, updated the tech spec and launcher accordingly, revalidated the launcher, and submitted the current shot-9-only job successfully as 8830094. Cluster execution itself has not completed yet; the scheduler currently shows the replacement job as pending by priority, so scientific output artifacts are not available yet.`

## Source Documents

- Plan: `plans/2026-03-22-bd-random-shuffle-accuracy-plan.md`
- Tech Spec: `plans/2026-03-22-bd-random-shuffle-accuracy-tech-spec.md`

## Execution Settings

- Launcher: `sbatch`
- Compute Mode: `gpu`
- Time limit: `08:00:00`
- Day-based duration if relevant: `none`
- Shot selector: `9`
- GPU options: `--gres=gpu:h100:1`
- CPU count: `--cpus-per-task=8`
- Memory: `--mem=128G`
- Partition or queue: `gpubase_bygpu_b4`
- Job name: `q1_bd_shuffle_compare`
- Environment setup: `source /home/sunsik/.venvs/pt442/bin/activate && module load cuda/12.9 && export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:${LD_LIBRARY_PATH:-} && export BNB_CUDA_VERSION=122 && unset CUDA_HOME && export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}`
- Extra launcher flags: `-A def-twwebb_gpu -o /home/sunsik/my_fv_project/logs/%x_%j.out`

## Commands Run

1. `git -C /home/sunsik/my_fv_project status --short`
2. `python -m py_compile /home/sunsik/my_fv_project/scripts/score_bd_shuffle_behavior.py /home/sunsik/my_fv_project/scripts/build_bd_shuffle_behavior_summary.py`
3. `bash -n /home/sunsik/my_fv_project/scripts/run_pt_bd_shuffle_compare_llama70b.sh /home/sunsik/my_fv_project/scripts/slurm/run_q1_bd_shuffle_compare.sbatch`
4. `/home/sunsik/.venvs/pt442/bin/python - <<'PY' ... mod._validate_layouts() ... expected_total_q1 ... PY`
5. `/home/sunsik/.venvs/pt442/bin/python - <<'PY' ... _build_trial_plan(qids=['Q1'], n_trials=2, ...) ... PY`
6. `sbatch /home/sunsik/my_fv_project/scripts/slurm/run_q1_bd_shuffle_compare.sbatch`
7. `squeue -j 8830003 -o '%i|%j|%T|%M|%l|%R'`
8. `sacct -j 8830003 --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit -n -P`
9. `scancel 8830003`
10. `bash -n /home/sunsik/my_fv_project/scripts/slurm/run_q1_bd_shuffle_compare.sbatch /home/sunsik/my_fv_project/scripts/run_pt_bd_shuffle_compare_llama70b.sh`
11. `python -m py_compile /home/sunsik/my_fv_project/scripts/score_bd_shuffle_behavior.py /home/sunsik/my_fv_project/scripts/build_bd_shuffle_behavior_summary.py`
12. `sbatch /home/sunsik/my_fv_project/scripts/slurm/run_q1_bd_shuffle_compare.sbatch`
13. `squeue -j 8830094 -o '%i|%j|%T|%M|%l|%R'`
14. `sacct -j 8830094 --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit -n -P`

## Files Executed

- `python -m py_compile`
- `scripts/score_bd_shuffle_behavior.py` `import/validation only`
- `scripts/build_bd_shuffle_behavior_summary.py` `syntax validation only`
- `scripts/slurm/run_q1_bd_shuffle_compare.sbatch` `submitted to scheduler twice; first job canceled after scope change, second job is the current pending run`

## Files Changed

- Modified: `plans/2026-03-22-bd-random-shuffle-accuracy-plan.md`
- Modified: `plans/2026-03-22-bd-random-shuffle-accuracy-tech-spec.md`
- Created: `scripts/score_bd_shuffle_behavior.py`
- Created: `scripts/build_bd_shuffle_behavior_summary.py`
- Created: `scripts/run_pt_bd_shuffle_compare_llama70b.sh`
- Created: `scripts/slurm/run_q1_bd_shuffle_compare.sbatch`
- Modified: `scripts/slurm/run_q1_bd_shuffle_compare.sbatch` `shot selector changed from 1,3,5,7,9 to 9 after user scope change`
- Created: `reports/2026-03-22-bd-random-shuffle-accuracy-report.md`

## Output Artifacts

- `/home/sunsik/my_fv_project/plans/2026-03-22-bd-random-shuffle-accuracy-plan.md`
- `/home/sunsik/my_fv_project/plans/2026-03-22-bd-random-shuffle-accuracy-tech-spec.md`
- `/home/sunsik/my_fv_project/scripts/score_bd_shuffle_behavior.py`
- `/home/sunsik/my_fv_project/scripts/build_bd_shuffle_behavior_summary.py`
- `/home/sunsik/my_fv_project/scripts/run_pt_bd_shuffle_compare_llama70b.sh`
- `/home/sunsik/my_fv_project/scripts/slurm/run_q1_bd_shuffle_compare.sbatch`
- `/home/sunsik/my_fv_project/logs/q1_bd_shuffle_compare_8830003.out` `created by the canceled first submission`
- `Expected after job start: /home/sunsik/my_fv_project/logs/q1_bd_shuffle_compare_8830094.out`
- `Expected after job start: /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_bd_shuffle_compare_<timestamp>_8830094/`
- `Expected after job completion: bd_shuffle_shot_sweep.csv, bd_shuffle_edge_topk.jsonl, bd_shuffle_regime_metrics.csv, bd_shuffle_case_deltas.csv, bd_shuffle_side_aggregate.csv, bd_shuffle_edge_topk_trial_agg.csv, bd_shuffle_summary.md, progress_status.json`

## Log Paths

- `/home/sunsik/my_fv_project/logs/q1_bd_shuffle_compare_8830003.out` `from the canceled first submission`
- `/home/sunsik/my_fv_project/logs/q1_bd_shuffle_compare_8830094.out` `not created yet because the replacement job has not started`
- `Expected after job start: /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_bd_shuffle_compare_<timestamp>_8830094/run.log`
- `Expected after job start: /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_bd_shuffle_compare_<timestamp>_8830094/progress_status.json`

## Validation Results

- `New Python scripts passed py_compile.`
- `New shell and sbatch scripts passed bash -n.`
- `Layout validation passed: 12 regimes total.`
- `Initial all-shot dry check passed with expected Q1 eval count = 3000.`
- `Deterministic trial-plan dry check passed for Q1 with two sample rows.`
- `Initial scheduler submission succeeded: sbatch returned job id 8830003.`
- `User then changed scope to shot 9 only; the initial job was canceled successfully.`
- `Launcher was updated to SHOT_LIST=9 and revalidated before resubmission.`
- `Replacement scheduler submission succeeded: sbatch returned job id 8830094.`
- `Current scheduler state check passed: squeue and sacct both report 8830094 as PENDING with time limit 08:00:00.`
- `Scientific metric validation has not run yet because the cluster job has not started/completed.`

## Brain Updates

- Required: `no`
- Updated files: `none`
- Why: `The current turn established and submitted a new experiment path, but the run has not completed yet, so there is no stable result or validated operational knowledge to promote into docs/brain.`

## Result Explanation

- `What changed: a dedicated Q1 BD shuffle scorer, a BD-specific summary builder, a run wrapper, and an sbatch launcher were added so the alternating-vs-shuffled behavioral test can run without widening the generic unified PT stack.`
- `What was run: local validation plus two scheduler actions. The first sbatch submission was canceled after the user narrowed scope to shot 9 only. The current active submission is job 8830094.`
- `What outputs mean: there are still no scientific outputs yet because the replacement cluster job is pending. The important completed outcome in this turn is that the implementation now matches the approved shot-9-only tech spec and the corrected job has been accepted by Slurm.`
- `Remaining risks or next steps: wait for job 8830094 to start and finish, then inspect the scratch run directory, run.log, progress_status.json, and the summary CSV/Markdown artifacts. After completion, compare BDBDBD_D vs shuffled D cases and DBDBDB_B vs shuffled B cases at shot 9 only.`

## Retry Record

- Retry attempted: `no`
- Reason: `n/a`

## Failure Details

- Failure point: `n/a`
- Error summary: `n/a`
