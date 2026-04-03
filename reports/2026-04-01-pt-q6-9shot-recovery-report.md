# Final Report Template

## Status

- Status: `success`
- Date: `2026-04-01`

## Summary

`Q6가 fixed 10-demo bundle 제약 때문에 11-Q candidate run에서 빠지는 문제를 max-shot bundle semantics로 수정했고, Q6 only 9-shot recovery run을 srun으로 실행해 canonical scratch output을 생성했다. 결과적으로 Q6는 shots 1,3,5,7,9 전부에서 AB/AC/AD/BC/BD 5-edge top-k candidate evidence를 정상 생성했다.`

## Source Documents

- Plan: `/home/sunsik/my_fv_project/plans/2026-04-01-pt-q6-9shot-recovery-plan.md`
- Tech Spec: `/home/sunsik/my_fv_project/plans/2026-04-01-pt-q6-9shot-recovery-tech-spec.md`

## Execution Settings

- Launcher: `srun`
- Compute Mode: `gpu`
- Time limit: `01:00:00`
- Day-based duration if relevant: `n/a`
- GPU options: `--gres=gpu:h100:1`
- CPU count: `8`
- Memory: `128G`
- Partition or queue: `not specified; scheduler selected gpubase_i`
- Job name: `pt_q6_9shot_recovery`
- Environment setup: `source /home/sunsik/.venvs/pt442/bin/activate && module load cuda/12.9 && export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH && export BNB_CUDA_VERSION=122 && unset CUDA_HOME`
- Extra launcher flags: `-A def-twwebb_gpu`

## Commands Run

1. `git -C /home/sunsik/my_fv_project status --short`
2. `rg -n "A10_row_ids|B10_row_ids|demo_ids_10|max\\(shots\\)|shot_list includes value > 10|demo_ids_bundle|A10|B10" scripts/score_cross_relation_target_logit.py scripts/run_pt_llama70b.sh scripts/build_pt_valid_answer_scaffold.py docs/brain/pipelines/pt.md`
3. `/home/sunsik/.venvs/pt442/bin/python -m py_compile /home/sunsik/my_fv_project/scripts/score_cross_relation_target_logit.py`
4. `bash -n /home/sunsik/my_fv_project/scripts/run_pt_q6_9shot_topk_recovery.sh`
5. `bash -n /home/sunsik/my_fv_project/scripts/slurm/run_pt_q6_9shot_topk_recovery.sbatch`
6. `/home/sunsik/.venvs/pt442/bin/python - <<'PY' ... _build_trial_plan_rows(qid=Q6, shots=1,3,5,7,9) and _build_trial_plan_rows(qid=Q1, shots=1,3,5,7,10) ... PY`
7. `srun -A def-twwebb_gpu --gres=gpu:h100:1 --cpus-per-task=8 --mem=128G --time=01:00:00 --job-name=pt_q6_9shot_recovery bash -lc 'cd /home/sunsik/my_fv_project && source /home/sunsik/.venvs/pt442/bin/activate && module load cuda/12.9 && export LD_LIBRARY_PATH="$EBROOTCUDA/lib64:${LD_LIBRARY_PATH:-}" && export BNB_CUDA_VERSION=122 && unset CUDA_HOME && export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" && export PY="/home/sunsik/.venvs/pt442/bin/python" && export RUN_ID="$(date +%Y%m%d_%H%M%S)_srun" && export MODEL_TAG="llama31_70b_pt_q6_9shot_recovery" && bash /home/sunsik/my_fv_project/scripts/run_pt_q6_9shot_topk_recovery.sh'`
8. `python - <<'PY' ... validate Q6 row counts in pt_5edge_shot_sweep.csv ... PY`
9. `python - <<'PY' ... validate Q6 top-k rows, shot coverage, and edge coverage in pt_edge_topk.jsonl ... PY`
10. `cat /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_q6_9shot_recovery_20260401_014648_srun/run_status.json`
11. `cat /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_q6_9shot_recovery_20260401_014648_srun/pt_bootstrap_summary.csv`

## Files Executed

- `/home/sunsik/my_fv_project/scripts/score_cross_relation_target_logit.py`
- `/home/sunsik/my_fv_project/scripts/run_pt_q6_9shot_topk_recovery.sh`
- `/home/sunsik/my_fv_project/scripts/run_pt_llama70b.sh`
- `/home/sunsik/my_fv_project/scripts/compute_product_test_bootstrap.py`
- `/home/sunsik/my_fv_project/scripts/plot_product_test_summary.py`
- `/home/sunsik/.venvs/pt442/bin/python`

## Files Changed

- Modified: `/home/sunsik/my_fv_project/scripts/score_cross_relation_target_logit.py`
- Modified: `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`
- Modified: `/home/sunsik/my_fv_project/plans/2026-04-01-pt-q6-9shot-recovery-plan.md`
- Modified: `/home/sunsik/my_fv_project/plans/2026-04-01-pt-q6-9shot-recovery-tech-spec.md`
- Created: `/home/sunsik/my_fv_project/scripts/run_pt_q6_9shot_topk_recovery.sh`
- Created: `/home/sunsik/my_fv_project/scripts/slurm/run_pt_q6_9shot_topk_recovery.sbatch`
- Created: `/home/sunsik/my_fv_project/logs/2026-04-01-pt-q6-9shot-recovery-execution.log`
- Created: `/home/sunsik/my_fv_project/reports/2026-04-01-pt-q6-9shot-recovery-report.md`

## Output Artifacts

- Canonical run root: `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_q6_9shot_recovery_20260401_014648_srun`
- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_q6_9shot_recovery_20260401_014648_srun/pt_5edge_shot_sweep.csv`
- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_q6_9shot_recovery_20260401_014648_srun/pt_bootstrap_summary.csv`
- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_q6_9shot_recovery_20260401_014648_srun/pt_bootstrap_summary.png`
- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_q6_9shot_recovery_20260401_014648_srun/pt_edge_topk.jsonl`
- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_q6_9shot_recovery_20260401_014648_srun/pt_edge_topk_change_summary.csv`
- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_q6_9shot_recovery_20260401_014648_srun/run_meta.json`
- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_q6_9shot_recovery_20260401_014648_srun/run_status.json`

## Log Paths

- Repo execution log: `/home/sunsik/my_fv_project/logs/2026-04-01-pt-q6-9shot-recovery-execution.log`
- Canonical run log: `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_q6_9shot_recovery_20260401_014648_srun/run.log`

## Validation Results

- `py_compile passed for scripts/score_cross_relation_target_logit.py`
- `bash -n passed for scripts/run_pt_q6_9shot_topk_recovery.sh and scripts/slurm/run_pt_q6_9shot_topk_recovery.sbatch`
- `planner smoke passed: Q6 with shots 1,3,5,7,9 produced 3/3 plan rows with bundle_size=9 and B_len=9`
- `backward-compat smoke passed: Q1 with shots 1,3,5,7,10 produced 3/3 plan rows with bundle_size=10`
- `run_status.json reports completed exit_code=0`
- `pt_5edge_shot_sweep.csv contains exactly 1250 Q6 rows`
- `Q6 shot coverage is complete: shots 1,3,5,7,9 each have 250 rows`
- `Q6 edge coverage is complete: AB, AC, AD, BC, BD each have 250 rows`
- `pt_edge_topk.jsonl contains exactly 1250 Q6 rows with shots 1,3,5,7,9 and edges AB, AC, AD, BC, BD`
- `pt_bootstrap_summary.csv contains 5 Q6 rows, one per shot`

## Brain Updates

- Required: `yes`
- Updated files: `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`
- Why: `baseline PT의 demo-bundle semantics가 fixed-10에서 max(shot_list) 기반으로 바뀌었고, 이 변화가 Q6 같은 9-shot recovery workflow를 직접 가능하게 하므로 stable pipeline knowledge로 반영했다.`

## Result Explanation

- `Q6 skip의 직접 원인은 fixed 10-demo bundle 가정이었다. 이 제약을 max(shot_list) 기반 bundle로 일반화해, 9-shot run에서는 Q6가 bundle_size=9로 정상 planning 되도록 바꿨다.`
- `실제 Q6 recovery run은 shots 1,3,5,7,9 에서 50 trials x 5 edges = 1250 score rows를 전부 생성했다.`
- `candidate collection 관점에서 필요한 AB/AC/AD/BC/BD top-k traces도 모두 생성됐으므로, 다음 단계에서 Q6의 C target review input으로 바로 사용할 수 있다.`
- `이 run의 bootstrap summary도 생성됐지만, 이는 Q6 candidate recovery의 부산물이다. 최종 12-Q PT 비교는 shot semantics를 통일한 selected-target-aware rerun에서 다시 계산해야 한다.`

## Retry Record

- Retry attempted: `no`
- Reason: `n/a`

## Failure Details

- Failure point: `n/a`
- Error summary: `n/a`
