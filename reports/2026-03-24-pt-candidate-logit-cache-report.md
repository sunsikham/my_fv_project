# Final Report Template

## Status

- Status: `submitted`
- Date: `2026-03-24`

## Summary

`Unified PT candidate-logit cache plus offline-recompute workflow was implemented and then hardened twice during cluster validation. First, the merged selected-target artifact was resynced to the current dataset gold targets to fix the Q5 live-validation failure. Second, the cache-build path was extended to store forced selected-target scores so selected targets like Q11 uniform can be recovered even when they fall outside lexical top-20. The latest failure was only a floating-point parity check issue at ~7e-09 on target_prob_raw, so a dedicated parity validator with tolerance-based float comparison was added and validated locally against the previously failed live/recompute validation pair. A new full sbatch job 8998207 was then submitted with the corrected parity validator and is now the active retry.`

## Source Documents

- Plan: `plans/2026-03-24-pt-candidate-logit-cache-plan.md`
- Tech Spec: `plans/2026-03-24-pt-candidate-logit-cache-tech-spec.md`

## Execution Settings

- Launcher: `sbatch`
- Compute Mode: `gpu`
- Time limit: `1-00:00:00`
- Day-based duration if relevant: `1 day`
- GPU options: `--gres=gpu:h100:1`
- CPU count: `8`
- Memory: `128G`
- Partition or queue: `gpubase_bygpu_b4`
- Job name: `pt_candidate_logit_cache_full`
- Environment setup: `source /home/sunsik/.venvs/pt442/bin/activate && module load cuda/12.9`
- Extra launcher flags: `-A def-twwebb_gpu -o /home/sunsik/my_fv_project/logs/%x_%j.out`

## Commands Run

1. `git status --short`
2. `source /home/sunsik/.venvs/pt442/bin/activate && python -m py_compile scripts/score_cross_relation_target_logit.py scripts/build_pt_valid_answer_scaffold.py scripts/recompute_pt_unified_from_edge_cache.py scripts/score_cross_relation_unified_drift_control.py`
3. `bash -n scripts/run_pt_unified_drift_control_llama70b.sh`
4. `source /home/sunsik/.venvs/pt442/bin/activate && python scripts/recompute_pt_unified_from_edge_cache.py --source_run_dir <smoke_run_dir> --selected_targets_json /scratch/sunsik/my_fv_project/pt_analysis/selected_targets/2026-03-24-selected_targets_all_requested_qs.json --out_csv <smoke_out_csv> --family_ids BASE_ABD,CTX_ABD --shot_list 1,3,5,7,9 --qid Q1`
5. `source /home/sunsik/.venvs/pt442/bin/activate && python scripts/recompute_pt_unified_from_edge_cache.py --source_run_dir <smoke_run_dir_q4> --selected_targets_json /scratch/sunsik/my_fv_project/pt_analysis/selected_targets/2026-03-24-selected_targets_all_requested_qs.json --out_csv <smoke_out_csv_q4> --family_ids BASE_ABD,CTX_ABD --shot_list 1,3,5,7,9 --qid Q4`
6. `sbatch --parsable <<'EOF' ... EOF`
7. `source /home/sunsik/.venvs/pt442/bin/activate && python scripts/validate_pt_unified_recompute_parity.py --live_csv /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_candidate_live_validate_20260324_120713/pt_unified_shot_sweep.csv --recompute_csv /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_candidate_recompute_validate_20260324_120713/pt_unified_shot_sweep.csv`
8. `sbatch --parsable <<'EOF' ... EOF` `resubmitted as job 8998207 with tolerance-based parity validator`

## Files Executed

- `/home/sunsik/my_fv_project/scripts/recompute_pt_unified_from_edge_cache.py`
- `/home/sunsik/my_fv_project/scripts/run_pt_unified_drift_control_llama70b.sh`
- `/home/sunsik/my_fv_project/scripts/score_cross_relation_unified_drift_control.py` `submitted inside sbatch job`
- `/home/sunsik/my_fv_project/scripts/compute_product_test_bootstrap_unified.py` `submitted inside sbatch job`
- `/home/sunsik/my_fv_project/scripts/build_pt_unified_human_report.py` `submitted inside sbatch job`
- `/home/sunsik/my_fv_project/scripts/validate_pt_unified_recompute_parity.py`

## Files Changed

- Modified: `/home/sunsik/my_fv_project/scripts/score_cross_relation_target_logit.py`
- Modified: `/home/sunsik/my_fv_project/scripts/build_pt_valid_answer_scaffold.py`
- Modified: `/home/sunsik/my_fv_project/scripts/run_pt_unified_drift_control_llama70b.sh`
- Modified: `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`
- Modified: `/home/sunsik/my_fv_project/plans/2026-03-24-pt-candidate-logit-cache-plan.md`
- Modified: `/home/sunsik/my_fv_project/plans/2026-03-24-pt-candidate-logit-cache-tech-spec.md`
- Created: `/home/sunsik/my_fv_project/scripts/recompute_pt_unified_from_edge_cache.py`
- Created: `/home/sunsik/my_fv_project/scripts/validate_pt_unified_recompute_parity.py`
- Created: `/home/sunsik/my_fv_project/reports/2026-03-24-pt-candidate-logit-cache-report.md`

## Output Artifacts

- `/scratch/sunsik/my_fv_project/pt_analysis/pt_recompute_smoke_6fh1638j/recomputed.csv`
- `/scratch/sunsik/my_fv_project/pt_analysis/pt_recompute_smoke_q4_ynv3xqtw/recomputed.csv`
- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_candidate_live_validate_20260324_120713/pt_unified_shot_sweep.csv`
- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_candidate_recompute_validate_20260324_120713/pt_unified_shot_sweep.csv`
- `Submitted and failed sbatch jobs: 8966465, 8985286, 8994632`
- `Submitted active sbatch job: 8998207`

## Log Paths

- `/home/sunsik/my_fv_project/logs/pt_candidate_logit_cache_full_8966465.out`
- `/home/sunsik/my_fv_project/logs/pt_candidate_logit_cache_full_8985286.out`
- `/home/sunsik/my_fv_project/logs/pt_candidate_logit_cache_full_8994632.out`
- `/home/sunsik/my_fv_project/logs/pt_candidate_logit_cache_full_8998207.out`

## Validation Results

- `Python syntax validation passed for the modified scorer/helper/recompute scripts.`
- `Shell syntax validation passed for scripts/run_pt_unified_drift_control_llama70b.sh.`
- `Offline recompute smoke test passed for a Q1 row where selected target and gold target are the same.`
- `Offline recompute smoke test passed for a Q4 row where selected target surface is tadpole but the cached lexical candidate is tad, confirming first-token lookup behavior.`
- `Forced selected-target fallback smoke test passed for a synthetic Q11 uniform row where lexical top-k lookup is absent but forced-selected-target cache is present.`
- `Tolerance-based parity validator passed on the previously failed live/recompute validation pair with abs_tol=1e-6 and rel_tol=1e-6.`

## Brain Updates

- Required: `yes`
- Updated files: `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`
- Why: `The stable Unified PT workflow now includes a candidate-logit cache build stage, an offline selected-target recompute stage, top-k width guidance, and explicit report-stage dependency on source topk and eligibility artifacts.`

## Result Explanation

- `The lexical candidate cache now records candidate-level logits and vocab ranks in addition to token ids, logprobs, and probs.`
- `Manual-review scaffolds can now surface aggregated candidate logit and rank statistics from the enriched edge cache.`
- `A new offline recompute entrypoint can rebuild selected-target Unified PT sweep rows from a cached edge-topk artifact plus the selected-target artifact, without rerunning model inference.`
- `The Unified PT runner now supports three stages: standard_full, cache_build_only, and offline_recompute.`
- `The submitted sbatch job was designed to run a validation slice first and then the approved full q subset (Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18) for BASE_ABD and CTX_ABD only.`
- `The selected-target artifact was resynced to the current dataset-backed gold targets for Q5 B, Q6 B, Q8 B, Q9 D, and Q11 D.`
- `Cache-build rows now store both lexical top-20 candidate scores and forced selected-target scores, so top-k misses for approved targets no longer block offline recompute.`
- `Live/offline parity is now checked by a dedicated validator that keeps exact comparison for discrete fields and uses tolerance-based comparison for floating-point fields.`

## Retry Record

- Retry attempted: `yes`
- Reason: `The workflow was resubmitted multiple times as blockers were resolved in sequence: 8966452 was cancelled pre-start after a validation-path expansion bug; 8966465 failed on stale selected-target artifact keys; 8985286 failed before full run on the same artifact contract issue; 8994632 failed only on a ~7e-09 floating-point parity mismatch after the structural issues were fixed; 8998207 is the current retry with the new tolerance-based parity validator.`

## Failure Details

- Failure point: `most recent finished job 8994632, validation slice parity check`
- Error summary: `validation mismatch key=('BASE_ABD', 'Q11', '0', '1', 'BASE_AB') field=target_prob_raw live=0.10361314564943314 recomp=0.10361315310001373`
