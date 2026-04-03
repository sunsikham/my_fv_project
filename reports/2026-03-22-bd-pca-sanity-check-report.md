# Final Report Template

## Status

- Status: `success`
- Date: `2026-03-22`

## Summary

`Executed the approved Q1-only BD-ref PCA sanity check successfully. The run reused existing Q1 BBB/DDD trial and StepD artifacts, generated only the missing mixed-BD trial payloads, ran StepD for BDBDBD_D and DBDBDB_B, built a shared BD_ref, re-extracted four-condition vectors under BD_ref, and produced common PCA artifacts. The resulting PCA summary shows the two mixed BD conditions are much closer to each other than either is to pure BBB or pure DDD, so the current BD-ref PCA does not support a simple pure-endpoint split reading.`

## Source Documents

- Plan: `plans/2026-03-21-bd-pca-sanity-check-plan.md`
- Tech Spec: `plans/2026-03-21-bd-pca-sanity-check-tech-spec.md`

## Execution Settings

- Launcher: `sbatch`
- Compute Mode: `gpu`
- Time limit: `1-00:00:00`
- Day-based duration if relevant: `1 day`
- GPU options: `--gres=gpu:h100:1`
- CPU count: `--cpus-per-task=8`
- Memory: `--mem=120G`
- Partition or queue: `gpubase_bygpu_b4`
- Job name: `q1_bd_pca`
- Environment setup: `source /home/sunsik/.venvs/pt442/bin/activate && module load cuda/12.9 && export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:${LD_LIBRARY_PATH:-} && export BNB_CUDA_VERSION=122 && unset CUDA_HOME && export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}`
- Extra launcher flags: `-A def-twwebb_gpu -o /home/sunsik/my_fv_project/logs/%x_%j.out`

## Commands Run

1. `python -m py_compile scripts/run_condition_common_pca.py scripts/run_bd_interleave_pca.py`
2. `bash -n scripts/slurm/run_q1_bd_pca.sbatch`
3. `sbatch scripts/slurm/run_q1_bd_pca.sbatch`
4. `python scripts/run_bd_interleave_pca.py --q_dir /scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1 --relation_b_csv /home/sunsik/my_fv_project/datasets/relation/relationB_ex.csv --relation_d_csv /home/sunsik/my_fv_project/datasets/relation/relationD_ex.csv --model /scratch/sunsik/models/Llama-3.1-70B --model_spec llama3 --device cuda --dtype bf16 --quant 4bit --topk 20 --score_key mean_delta_p --stepd_layers 16-55 --resume 1 --stop_on_error 1 --pca_out_subdir BD_ref_BD_compare`
5. `/home/sunsik/.venvs/pt442/bin/python /home/sunsik/my_fv_project/scripts/run_stepD_aie_head_sweep.py --model /scratch/sunsik/models/Llama-3.1-70B --model_spec llama3 --device cuda --quant 4bit --layers 16-55 --heads all --n_trials 25 --n_icl_examples 9 --score_key mean_delta_p --seed 0 --fixed_trials_path /scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_trials/condition_BDBDBD_D.json --fixed_out_dir /scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_stepd/run_BDBDBD_D --dtype bf16`
6. `/home/sunsik/.venvs/pt442/bin/python /home/sunsik/my_fv_project/scripts/run_stepD_aie_head_sweep.py --model /scratch/sunsik/models/Llama-3.1-70B --model_spec llama3 --device cuda --quant 4bit --layers 16-55 --heads all --n_trials 25 --n_icl_examples 9 --score_key mean_delta_p --seed 0 --fixed_trials_path /scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_trials/condition_DBDBDB_B.json --fixed_out_dir /scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_stepd/run_DBDBDB_B --dtype bf16`
7. `/home/sunsik/.venvs/pt442/bin/python /home/sunsik/my_fv_project/scripts/run_condition_common_pca.py --q_dir /scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1 --ref_mode BD_ref --conditions BBB,DDD,BDBDBD_D,DBDBDB_B --n_components 3 --balance_trials 1 --seed 0 --out_subdir BD_ref_BD_compare`

## Files Executed

- `scripts/run_bd_interleave_pca.py`
- `scripts/run_condition_common_pca.py`
- `scripts/run_stepD_aie_head_sweep.py`
- `scripts/slurm/run_q1_bd_pca.sbatch`

## Files Changed

- Modified: `plans/2026-03-21-bd-pca-sanity-check-plan.md`
- Modified: `plans/2026-03-21-bd-pca-sanity-check-tech-spec.md`
- Modified: `scripts/run_condition_common_pca.py`
- Created: `scripts/run_bd_interleave_pca.py`
- Created: `scripts/slurm/run_q1_bd_pca.sbatch`
- Created: `reports/2026-03-22-bd-pca-sanity-check-report.md`
- Created at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_trials/condition_BDBDBD_D.json`
- Created at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_trials/condition_DBDBDB_B.json`
- Created at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_stepd/aie_scores_BDBDBD_D.csv`
- Created at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_stepd/aie_scores_DBDBDB_B.csv`
- Created at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_top_heads/sets/top_heads_ref_BD.json`
- Created at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_vectors/trial_vectors_BD_ref_BBB.npy`
- Created at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_vectors/trial_vectors_BD_ref_DDD.npy`
- Created at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_vectors/trial_vectors_BD_ref_BDBDBD_D.npy`
- Created at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_vectors/trial_vectors_BD_ref_DBDBDB_B.npy`
- Created at runtime: `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/BD_ref_BD_compare/*`

## Output Artifacts

- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_status/bd_interleave_pca_status.json`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_top_heads/sets/top_heads_ref_BD.json`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_vectors/trial_vectors_BD_ref_BBB.npy`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_vectors/trial_vectors_BD_ref_DDD.npy`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_vectors/trial_vectors_BD_ref_BDBDBD_D.npy`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_vectors/trial_vectors_BD_ref_DBDBDB_B.npy`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/BD_ref_BD_compare/pca_points.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/BD_ref_BD_compare/pca_centroids.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/BD_ref_BD_compare/distance_summary.json`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/BD_ref_BD_compare/scatter.png`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/BD_ref_BD_compare/scatter_3d.png`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/_pca_common/BD_ref_BD_compare/scatter_3d_interactive.html`

## Log Paths

- `/home/sunsik/my_fv_project/logs/q1_bd_pca_8709240.out`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/logs/bd_interleave_pca_orchestrator.log`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/logs/stepD_BDBDBD_D.log`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/logs/stepD_DBDBDB_B.log`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1/logs/pca_BD_ref_BD_compare.log`

## Validation Results

- `python -m py_compile scripts/run_condition_common_pca.py scripts/run_bd_interleave_pca.py` passed before submission.
- `bash -n scripts/slurm/run_q1_bd_pca.sbatch` passed before submission.
- `sacct -j 8709240` reports `COMPLETED` with exit code `0:0`.
- `bd_interleave_pca_status.json` reports `mixed_trials_done=true`, `mixed_stepd_done=true`, `bd_ref_done=true`, `bd_vectors_done=true`, `pca_done=true`.
- `pca_model_meta.json` confirms PCA artifacts were written with `n_samples=100`, `n_features=8192`, `n_components=3`, and all scatter outputs enabled.
- `distance_summary.json` shows:
  - `BDBDBD_D__DBDBDB_B = 0.5335`
  - `BBB__DDD = 4.7966`
  - `BBB__BDBDBD_D = 3.2631`
  - `BBB__DBDBDB_B = 3.7899`
  - `BDBDBD_D__DDD = 2.7467`
  - `DBDBDB_B__DDD = 2.5653`
- `Interpretation from the PCA summary: the two mixed BD conditions are very close to each other and are not simply splitting to pure BBB vs pure DDD endpoints in this BD_ref view.`

## Brain Updates

- Required: `no`
- Updated files: `none`
- Why: `This execution added a focused one-q experimental runner and produced run-specific artifacts, but it did not yet establish stable project knowledge that should replace current docs/brain summaries. The result is better treated as experiment output plus follow-up interpretation work before changing stable brain docs.`

## Result Explanation

- `What changed: a new BD-ref sanity-check runner and sbatch entrypoint were added, and the Q1 scratch tree gained mixed-BD trials, StepD outputs, a shared BD_ref, BD_ref vectors, and BD-ref PCA artifacts.`
- `What was run: only the missing mixed-BD path was executed. Existing BBB/DDD StepD results were reused; BDBDBD_D and DBDBDB_B received new trial payloads and StepD runs; then one shared BD_ref vector extraction plus one common PCA run was executed.`
- `What outputs mean: in the current BD_ref PCA summary, BDBDBD_D and DBDBDB_B cluster much closer to each other than either does to pure BBB or pure DDD. This does not support a simple pure-endpoint split story in the PCA view that was run here.`
- `Remaining risks or next steps: this is still only the PCA sanity-check layer. It does not by itself resolve whether the mixed-BD closeness reflects a subtle common relation, a dual-relation maintenance pattern, or another structure. The next decision point is how to interpret the finished BD_ref PCA against the main A-anchor multi-feature story, and whether shuffled-order controls should be added.`

## Retry Record

- Retry attempted: `no`
- Reason: `n/a`

## Failure Details

- Failure point: `n/a`
- Error summary: `n/a`
