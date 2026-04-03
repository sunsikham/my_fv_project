# Tech Spec Template

## Title

`Q12 PT ABC/ABD Shotwise Comparison With C Selected-Target Review`

## Metadata

- Date: `2026-03-31`
- Slug: `pt-q12-abc-abd-selected-targets`
- Source Plan: `plans/2026-03-31-pt-q12-abc-abd-selected-targets-plan.md`
- Approval Status: `approved`

## Scope

- `12개 Q(Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18)에 대해 C query selected-target 리뷰 경로를 추가한다.`
- `5-edge PT scorer가 B/C/D selected target artifact를 받아 AB, AC, AD, BC, BD를 같은 normalization pool에서 다시 점수화하도록 확장한다.`
- `shot별 ABC, ABD, delta를 내는 canonical bootstrap summary를 다시 산출한다.`
- `기존 B/D selected-target artifact를 재사용해 C review 부담을 최소화한다.`

## Out Of Scope

- `AAA/BBB/BABA/DADA/DDD PCA 분석 자체 변경`
- `unified PT family 정의(BASE_ABD/CTX_ABD) 자체 재설계`
- `BD shuffle comparison workflow 변경`
- `논문 본문/figure 직접 수정`
- `C selected-target의 semantic correctness를 모델 자동 판단으로 대체하는 일`

## Implementation Design

`핵심 원칙은 old 5-edge baseline의 normalization semantics를 유지한 채 selected-target review 범위를 C까지 넓히는 것이다. 따라서 최종 ABC vs ABD 비교는 unified PT가 아니라 5-edge PT를 기준으로 산출한다.`

`설계는 두 단계로 나뉜다. 첫 단계는 review data 확보 단계다. 5-edge scorer를 review/cache-build 모드로 다시 실행해 AC/BC를 포함한 lexical top-k evidence를 저장한다. 이때 edge-topk 저장 범위는 최소 AB, AC, AD, BC, BD 전체로 넓힌다. 이후 scaffold builder는 이 5-edge top-k JSONL을 읽어 B/C/D query units를 unit_id 단위로 묶고, 기존 12-Q B/D selected-target artifact를 seed로 주입해 B/D는 자동 승인 상태로 채우고 C units만 pending review로 남긴다.`

`두 번째 단계는 selected-target-aware 재점수화 단계다. reviewer가 C units를 승인해 finalized selected-target artifact를 만들면, 5-edge scorer는 이 artifact를 받아 각 edge의 scored target을 다음처럼 결정한다: AB는 B selected target, AC/BC는 C selected target, AD/BD는 D selected target. 그런 다음 row-level raw logprob를 다시 계산하고, q_id 안에서 AB/AC/AD/BC/BD 전체와 모든 shot을 함께 모아 기존 5-edge와 동일한 robust min-max normalization(qid_all_edges_all_shots_this_run)을 적용한다. 마지막으로 compute_product_test_bootstrap.py를 그대로 사용해 PT_ABC, PT_ABD, delta를 q_id/shot 단위로 계산한다.`

`기존 코드 재사용 전략은 다음과 같다. finalize_pt_selected_targets.py는 unit schema가 generic하므로 그대로 재사용한다. build_pt_valid_answer_scaffold.py는 unified 전용 입력 가정을 완화해 5-edge top-k row도 받을 수 있게 일반화하거나, 최소 수정으로 seed selected-target artifact를 받을 수 있게 확장한다. recompute_pt_unified_from_edge_cache.py는 family-normalized unified semantics에 묶여 있으므로 이번 작업의 주경로로는 재사용하지 않는다.`

## Expected File Changes

- Modify: `/home/sunsik/my_fv_project/scripts/score_cross_relation_target_logit.py`
- Modify: `/home/sunsik/my_fv_project/scripts/build_pt_valid_answer_scaffold.py`
- Modify: `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`
- Create: `/home/sunsik/my_fv_project/scripts/run_pt_5edge_selected_targets_llama70b.sh`
- Create: `/home/sunsik/my_fv_project/logs/2026-03-31-pt-q12-abc-abd-selected-targets-*.log`
- Create: `/scratch/sunsik/my_fv_project/pt_analysis/<new-5edge-cache-run>/...`
- Create: `/scratch/sunsik/my_fv_project/pt_analysis/<new-5edge-selected-run>/...`

## Ordered Implementation Steps

1. `score_cross_relation_target_logit.py에 selected_targets_json 옵션과 5-edge 전 edge top-k 저장 경로를 추가한다. selected target 적용 시 gold_target과 scored_target을 모두 기록해 후속 검증이 가능하도록 row schema를 확장한다.`
2. `build_pt_valid_answer_scaffold.py를 5-edge top-k JSONL에도 동작하도록 일반화하고, 선택적으로 기존 selected-target artifact를 seed로 넣어 matching B/D units를 자동 승인 상태로 채울 수 있게 한다.`
3. `기존 12-Q B/D artifact를 seed로 사용해 5-edge review scaffold를 만들고, C query units만 수동 검토 대상으로 남기는 실행 경로를 준비한다.`
4. `finalize_pt_selected_targets.py로 B/C/D 통합 artifact를 생성한다.`
5. `새 shell wrapper(run_pt_5edge_selected_targets_llama70b.sh)를 추가해 12-Q 대상 5-edge selected-target-aware inference run을 scratch canonical root에 기록한다.`
6. `compute_product_test_bootstrap.py를 selected-target-aware 5-edge sweep CSV에 적용해 shot별 ABC, ABD, delta summary를 만든다.`
7. `산출물 검증 후, 필요하면 docs/brain/pipelines/pt.md에 C 포함 5-edge selected-target workflow를 stable knowledge로 반영한다.`

## Validation Plan

- `새 5-edge cache-build run의 edge-topk JSONL에 AB, AC, AD, BC, BD가 모두 존재하는지 확인한다.`
- `review scaffold에 12개 Q의 C units가 모두 포함되는지 확인한다.`
- `seed artifact 주입 후 B/D units가 기존 선택값과 일치하는지 spot-check 한다.`
- `finalized selected-target artifact가 B 12 + C 12 + D 12 단위를 모두 포함하는지 확인한다.`
- `selected-target-aware 5-edge sweep CSV의 norm_scope가 qid_all_edges_all_shots_this_run인지 확인한다.`
- `bootstrap summary에서 12개 Q x shot 목록 전체에 대해 pt_abc_mean, pt_abd_mean, delta가 채워지는지 확인한다.`
- `old gold-target 5-edge 결과와 새 selected-target-aware 결과를 비교해 값이 비정상적으로 붕괴한 q_id가 없는지 sanity check 한다.`

## Expected Outputs

- `5-edge review/cache-build run directory under /scratch/sunsik/my_fv_project/pt_analysis/`
- `C review scaffold JSON/MD`
- `12-Q B/C/D combined selected-target artifact JSON/MD`
- `selected-target-aware pt_5edge_shot_sweep.csv`
- `selected-target-aware pt_bootstrap_summary.csv`
- `필요 시 q별 shotwise 비교용 markdown/table artifact`

## Brain Docs To Update

- `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`

## Recommended Execution Strategy

- Launcher: `sbatch`
- Compute Mode: `gpu`
- Reason: `70B inference를 이용한 5-edge cache-build와 selected-target-aware rerun이 둘 다 필요할 가능성이 높고, 12개 Q x 다중 shot x 5 edge 구조는 장시간 GPU job에 가깝다. canonical scratch 산출물과 로그를 남기려면 interactive local보다 batch job이 안전하다. smoke나 짧은 schema 검증은 srun/local로 가능하지만, 본 실행은 sbatch가 적합하다.`

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
