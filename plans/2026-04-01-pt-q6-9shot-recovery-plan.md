# Plan Template

## Title

`Q6 PT 9-Shot Recovery For Candidate Collection`

## Metadata

- Date: `2026-04-01`
- Slug: `pt-q6-9shot-recovery`
- Approval Status: `approved`

## Objective

`현재 11-Q 5-edge top-k run에서 빠진 Q6를 복구하기 위해, PT scorer의 demo-bundle 가정을 9-shot 친화적으로 정리하고, Q6 전용 candidate collection run을 다시 설계한다. 목표는 Q6의 AC/BC lexical candidate evidence를 확보해 이후 C target 선택과 최종 12-Q selected-target PT 재계산에 연결 가능한 상태를 만드는 것이다.`

## Current Context

`현재 /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt5edge_q12_topk_srun_20260331_235612_srun 에서 11-Q(실제로는 Q1,Q3,Q4,Q5,Q7,Q8,Q9,Q10,Q11,Q16,Q18) 5-edge top-k 수집 run이 진행 중이다. Q6는 datasets/relation/relationB_ex.csv 에 demo가 9개뿐인데, scripts/score_cross_relation_target_logit.py 가 shot과 무관하게 trial마다 A10/B10을 먼저 샘플하는 구조라 plan 단계에서 제외되었다. 따라서 Q6를 살리려면 단순히 Q6만 다시 돌리는 것이 아니라, max(shots) 기반 demo-bundle 규칙으로 scorer를 조정하고 Q6를 1,3,5,7,9 shot 조건에서 별도 실행해야 한다.`

## Assumptions

- `Q6를 포함하는 별도 recovery run은 현재 진행 중인 11-Q srun을 중단하지 않고, 완료 후 독립적으로 수행하는 것이 안전하다.`
- `Q6의 A demo는 10개, B demo는 9개이므로 max_shot=9 규칙이면 candidate collection이 가능하다.`
- `Q6 candidate collection의 직접 목적은 C target review input 확보이며, 이것만으로 최종 selected-target-aware PT 비교가 완료되지는 않는다.`
- `최종 12-Q 비교에 shot 목록 일관성이 필요하다면, 후속 selected-target-aware rerun은 1,3,5,7,9 기준으로 다시 정렬될 가능성이 있다.`
- `Q6 recovery 작업이 완료되면 PT pipeline의 stable semantics가 바뀌므로 docs/brain/pipelines/pt.md 업데이트가 필요할 가능성이 높다.`

## Inputs And Dependencies

- `/home/sunsik/my_fv_project/scripts/score_cross_relation_target_logit.py`
- `/home/sunsik/my_fv_project/scripts/run_pt_llama70b.sh`
- `/home/sunsik/my_fv_project/scripts/slurm/run_pt_5edge_q12_topk_candidates.sbatch`
- `/home/sunsik/my_fv_project/scripts/build_pt_valid_answer_scaffold.py`
- `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`
- `/home/sunsik/my_fv_project/datasets/relation/relationA_ex.csv`
- `/home/sunsik/my_fv_project/datasets/relation/relationB_ex.csv`
- `/home/sunsik/my_fv_project/datasets/relation/icl_B_data.csv`
- `/home/sunsik/my_fv_project/datasets/relation/icl_C_data.csv`
- `/home/sunsik/my_fv_project/datasets/relation/icl_D_data.csv`
- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt5edge_q12_topk_srun_20260331_235612_srun/`
- `Llama-3.1-70B inference 가능한 GPU runtime`

## Proposed Steps

1. `Q6 skip 원인을 fixed 10-demo bundle 가정으로 고정하고, max_shot 기반 sampler로 바꿀 최소 수정 범위를 정리한다.`
2. `score_cross_relation_target_logit.py 와 run_pt_llama70b.sh 기준으로, shot_list가 1,3,5,7,9일 때 9-demo bundle을 쓰도록 설계를 구체화한다.`
3. `Q6 only recovery launcher(srun 또는 sbatch)를 별도로 설계해, 현재 11-Q run과 충돌 없이 Q6 candidate collection만 다시 실행할 수 있게 한다.`
4. `Q6 recovery run의 expected outputs를 pt_edge_topk.jsonl 중심으로 정의하고, 이후 build_pt_valid_answer_scaffold.py 로 C review scaffold에 합치는 경로를 준비한다.`
5. `후속 단계에서 11-Q run과 Q6 recovery run을 어떻게 함께 사용해 C target 선택 및 최종 selected-target-aware rerun으로 연결할지 기술명세에 명시한다.`

## Risks And Blockers

- `현재 11-Q run은 shot 10을 포함하고 Q6 recovery는 shot 9 기준이므로, candidate collection과 최종 PT 비교 단계를 분리해서 생각하지 않으면 shot semantics가 섞일 수 있다.`
- `score_cross_relation_target_logit.py 의 sampler semantics를 바꾸면 기존 10-shot workflow와의 호환성을 검증해야 한다.`
- `Q6 only recovery를 서둘러 실행하면 현재 진행 중인 run과 로그/아티팩트 해석이 섞일 수 있다.`
- `Q6 recovery 후에도 최종 12-Q selected-target-aware PT는 별도 rerun이 필요할 가능성이 높다.`

## Recommended Compute Mode

- Mode: `gpu`
- Why: `Q6 recovery 자체는 단일 q_id 재실행이지만, Llama-3.1-70B inference가 필요하고 top-k candidate evidence를 실제로 다시 생성해야 하므로 GPU runtime이 현실적이다. 코드 수정과 smoke 검증은 local에서 가능하지만, canonical candidate collection run은 GPU가 필요하다.`

## Expected Outputs

- `/home/sunsik/my_fv_project/plans/2026-04-01-pt-q6-9shot-recovery-plan.md`
- `/home/sunsik/my_fv_project/plans/2026-04-01-pt-q6-9shot-recovery-tech-spec.md`
- `Q6 9-shot candidate collection launcher`
- `/scratch/sunsik/my_fv_project/pt_analysis/<q6-recovery-run>/pt_edge_topk.jsonl`
- `/scratch/sunsik/my_fv_project/pt_analysis/<q6-recovery-run>/pt_5edge_shot_sweep.csv`
- `후속 C review scaffold에 합칠 수 있는 Q6 candidate evidence`

## Success Criteria

- `Q6가 fixed 10-demo bundle 때문에 빠졌다는 원인이 문서와 코드 기준으로 명확히 정리된다.`
- `승인 후 구현 단계에서 Q6를 1,3,5,7,9 shot 조건으로 실제 실행할 수 있는 설계가 준비된다.`
- `Q6 recovery run이 AC/BC candidate evidence를 생성해 이후 C target 수동 선택에 사용할 수 있다.`
- `후속 최종 rerun에서 shot semantics를 어떻게 맞출지 계획이 문서에 남는다.`

## Brain Impact

- Brain impact: `update required`
- Why: `이 작업이 완료되면 PT 5-edge scorer의 demo-bundle semantics가 fixed-10에서 max-shot 기반으로 바뀌거나, 최소한 Q6 recovery runbook이 stable knowledge가 된다. 이는 docs/brain/pipelines/pt.md 에 반영할 가치가 있다.`

## Approval Note

Substantial execution has not started. Waiting for Korean approval.
