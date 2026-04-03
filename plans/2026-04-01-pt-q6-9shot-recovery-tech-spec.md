# Tech Spec Template

## Title

`Q6 PT 9-Shot Recovery For Candidate Collection`

## Metadata

- Date: `2026-04-01`
- Slug: `pt-q6-9shot-recovery`
- Source Plan: `plans/2026-04-01-pt-q6-9shot-recovery-plan.md`
- Approval Status: `approved`

## Scope

- `Q6가 fixed 10-demo bundle 제약 때문에 skip되는 현재 5-edge scorer 로직을 max(shots) 기반 bundle 로직으로 조정한다.`
- `shot_list=1,3,5,7,9 조건에서 Q6가 trial plan에 포함되도록 score_cross_relation_target_logit.py 와 runner 경로를 정리한다.`
- `Q6 only candidate collection launcher를 추가하거나 기존 runner를 Q6 recovery 용도로 재사용할 수 있게 설정을 고정한다.`
- `Q6 recovery run이 pt_edge_topk.jsonl 과 pt_5edge_shot_sweep.csv 를 canonical scratch 경로에 남기도록 실행 설계를 준비한다.`
- `Q6 recovery output을 이후 C review scaffold 와 최종 selected-target-aware rerun에 연결하는 사용 규칙을 문서화한다.`

## Out Of Scope

- `현재 진행 중인 11-Q srun의 중단, 수정, 재시작`
- `Q6 외 다른 q_id의 shot semantics 변경`
- `최종 12-Q selected-target-aware PT rerun 자체 실행`
- `C target 수동 선택 또는 finalize artifact 생성`
- `논문 초안, PCA 분석, figure 수정`

## Implementation Design

`핵심 설계는 5-edge scorer의 demo sampling semantics를 shot-aware 로 바꾸는 것이다. 현재 코드는 trial마다 A10/B10을 먼저 샘플하고 이후 shot별로 prefix만 잘라 쓰기 때문에, max shot이 9여도 B pool이 10 미만이면 q_id가 제외된다. 이를 max_bundle = max(shots) 기반으로 일반화하면, shot_list=1,3,5,7,9 인 recovery run에서는 A9/B9를 미리 샘플하고 각 shot에서 앞부분만 사용하게 된다. 이렇게 하면 기존 “한 trial 안에서 shot이 커질수록 같은 randomized demo ordering의 prefix를 공유한다”는 의미는 유지하면서, Q6처럼 demo가 9개뿐인 문항도 합법적으로 포함시킬 수 있다.` 

`구현은 세 층으로 나뉜다. 첫째, score_cross_relation_target_logit.py 의 trial plan 생성부와 q-level pool validation을 max_bundle 기준으로 바꾼다. 둘째, run_pt_llama70b.sh 가 SHOT_LIST 에 따라 이 semantics를 자연스럽게 전달하도록 유지하되, 기본 10-shot workflow와 충돌하지 않도록 metadata 에 shot_list 와 bundle 의미가 남게 한다. 셋째, Q6 recovery 전용 launcher를 추가해 qid=Q6, shot_list=1,3,5,7,9, save_edge_topk=1 을 고정하고 canonical scratch root 로 기록한다. 이 launcher는 추천상 srun 으로 실행하지만, 같은 env export 를 sbatch wrapper 에도 재사용할 수 있게 설계한다.` 

`후속 사용 규칙도 중요하다. Q6 recovery run의 pt_edge_topk.jsonl 은 11-Q 기존 run 결과와 합쳐 C review scaffold 를 만드는 입력으로 쓰인다. 다만 shot 10 이 섞인 기존 11-Q candidate collection 과 shot 9 기반 Q6 recovery 는 최종 PT 수치 비교에 바로 합쳐 쓰지 않는다. 최종 비교는 C target selection 이 끝난 뒤, shot semantics 를 통일한 selected-target-aware rerun 에서 다시 계산한다. 따라서 이번 작업의 산출물은 “Q6 candidate evidence 보강”으로 한정된다.` 

## Expected File Changes

- Modify: `/home/sunsik/my_fv_project/scripts/score_cross_relation_target_logit.py`
- Modify: `/home/sunsik/my_fv_project/scripts/run_pt_llama70b.sh`
- Modify: `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`
- Create: `/home/sunsik/my_fv_project/scripts/run_pt_q6_9shot_topk_recovery.sh`
- Create: `/home/sunsik/my_fv_project/scripts/slurm/run_pt_q6_9shot_topk_recovery.sbatch`
- Create: `/home/sunsik/my_fv_project/logs/2026-04-01-pt-q6-9shot-recovery-*.log`
- Create: `/scratch/sunsik/my_fv_project/pt_analysis/<q6-recovery-run>/...`

## Ordered Implementation Steps

1. `score_cross_relation_target_logit.py 에서 A10/B10 고정 가정을 제거하고 max_bundle=max(shots) 기반 sampling 으로 일반화한다. plan 생성, pool 크기 체크, metadata/demo_ids 필드가 이 변경을 일관되게 반영하도록 수정한다.`
2. `기본 10-shot workflow 를 깨지 않도록, shot_list=1,3,5,7,10 인 기존 run 에서는 동작이 동일하고 shot_list=1,3,5,7,9 인 경우에만 Q6가 새로 포함되는지 로컬 smoke 검증을 추가한다.`
3. `run_pt_llama70b.sh 를 점검해 QID comma-list, SHOT_LIST, EDGE_TOPK 저장 동작이 Q6 recovery 에 그대로 재사용되도록 유지하고, 필요하면 run metadata 설명을 보강한다.`
4. `Q6 only recovery 용 shell wrapper(run_pt_q6_9shot_topk_recovery.sh)를 추가해 model/env/out_root/qid/shot_list/save_edge_topk 값을 안전하게 고정한다.`
5. `같은 설정을 Slurm 에서 쉽게 재사용할 수 있도록 run_pt_q6_9shot_topk_recovery.sbatch 를 추가한다.`
6. `로컬 문법 및 planner-level 검증 후, 승인된 launcher 설정에 따라 Q6 recovery run 을 실제로 실행한다.`
7. `성공 시 산출물의 pt_edge_topk.jsonl 과 pt_5edge_shot_sweep.csv 를 확인하고, docs/brain/pipelines/pt.md 에 max-shot bundle semantics 와 Q6 recovery usage note 를 반영한다.`

## Validation Plan

- `py_compile 로 수정된 Python 스크립트 문법을 검사한다.`
- `shot_list=1,3,5,7,9 와 qid=Q6 로 scorer planning 을 돌렸을 때 Q6가 skip되지 않는지 확인한다.`
- `shot_list=1,3,5,7,10 와 qid=Q1 같은 기존 문항으로 smoke test 했을 때 기존 planner semantics 가 유지되는지 확인한다.`
- `Q6 recovery run 의 run_meta.json 에 shot_list 와 canonical_root 가 올바르게 기록되는지 확인한다.`
- `Q6 recovery run output 에서 pt_edge_topk.jsonl 과 pt_5edge_shot_sweep.csv 가 생성되고 q_id=Q6 row 가 존재하는지 확인한다.`
- `Q6 recovery top-k row 들에 AC 와 BC edge 가 포함되는지 확인한다.`
- `docs/brain/pipelines/pt.md 업데이트 시 fixed-10 과 max-shot semantics 차이가 명시되는지 확인한다.`

## Expected Outputs

- `Q6 recovery tech spec: /home/sunsik/my_fv_project/plans/2026-04-01-pt-q6-9shot-recovery-tech-spec.md`
- `Q6 recovery launcher: /home/sunsik/my_fv_project/scripts/run_pt_q6_9shot_topk_recovery.sh`
- `Optional sbatch wrapper: /home/sunsik/my_fv_project/scripts/slurm/run_pt_q6_9shot_topk_recovery.sbatch`
- `Q6 recovery run directory under /scratch/sunsik/my_fv_project/pt_analysis/`
- `Q6 recovery pt_edge_topk.jsonl`
- `Q6 recovery pt_5edge_shot_sweep.csv`
- `필요 시 Q6 candidate evidence 를 기존 11-Q 결과와 함께 읽는 scaffold input note`

## Brain Docs To Update

- `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`

## Recommended Execution Strategy

- Launcher: `srun`
- Compute Mode: `gpu`
- Reason: `Q6 recovery 는 단일 q_id inference 재실행이라 대형 12-Q batch 보다 작고, 사용자가 진행 상태를 직접 보며 candidate evidence 생성 여부를 확인하기 쉽다. 다만 70B inference 이므로 local 은 비현실적이고 GPU 런처가 필요하다. queue 안정성이 더 중요하면 같은 설정으로 sbatch 도 가능하지만, 이번 recovery 성격에는 srun 이 기본 추천이다.`

## User Execution Settings Required Before Run

- Launcher choice:
- Time limit:
- Day-based duration if relevant:
- GPU options:
- CPU count:
- Memory:
- Partition or queue:
- Job name:
- Log path:
- Environment setup:
- Extra launcher flags:

## Approval Note

Implementation and execution have not started. Waiting for Korean approval.
