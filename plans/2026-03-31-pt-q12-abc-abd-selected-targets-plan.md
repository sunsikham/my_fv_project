# Plan Template

## Title

`Q12 PT ABC/ABD Shotwise Comparison With C Selected-Target Review`

## Metadata

- Date: `2026-03-31`
- Slug: `pt-q12-abc-abd-selected-targets`
- Approval Status: `approved`

## Objective

`12개 Q에 대해 shot별 ABC와 ABD product-test 값을 같은 비교 체계 안에서 산출하고, 최종 비교에 쓸 수 있는 canonical CSV/summary를 만든다. 이를 위해 C query의 selected target 리뷰 경로를 추가하고, 5-edge 기반 재계산 흐름을 정리한다.`

## Current Context

`이 저장소의 canonical PT root는 /scratch/sunsik/my_fv_project/pt_analysis 이다. 현재 old 5-edge baseline PT는 ABC와 ABD를 모두 계산하지만 gold C target을 전제로 하며, unified selected-target workflow는 review-covered B/D units와 BASE_ABD/CTX_ABD 여섯 regime에만 제한되어 있다. 최신 12-Q behavioral 비교에 사용 중인 unified recompute run은 ABD만 제공하고, 12개 Q는 Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18이다.`

## Assumptions

- `최종 비교 대상 12개 Q는 최신 full q8grain PT run의 qid 목록을 기준으로 고정한다.`
- `ABC vs ABD를 공정하게 비교하려면 AB, AC, AD, BC, BD를 같은 5-edge normalization pool에서 다시 계산해야 한다.`
- `C query도 selected target 리뷰가 필요하며, 문항당 C unit 하나를 정하면 AC와 BC에 공통 적용할 수 있다.`
- `기존 old 5-edge top-k 캐시는 AC/BC 리뷰 재료로 충분하지 않으므로, 5-edge scorer 또는 cache workflow 보강이 필요하다.`
- `다만 selected-target 리뷰/확정 자체는 기존 수동 검토 코드(build_pt_valid_answer_scaffold.py, finalize_pt_selected_targets.py)를 최대한 재사용하는 방향이 가능하다.`

## Inputs And Dependencies

- `/home/sunsik/my_fv_project/scripts/score_cross_relation_target_logit.py`
- `/home/sunsik/my_fv_project/scripts/compute_product_test_bootstrap.py`
- `/home/sunsik/my_fv_project/scripts/score_cross_relation_unified_drift_control.py`
- `/home/sunsik/my_fv_project/scripts/build_pt_valid_answer_scaffold.py`
- `/home/sunsik/my_fv_project/scripts/finalize_pt_selected_targets.py`
- `/home/sunsik/my_fv_project/fv/pt_selected_targets.py`
- `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`
- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_candidate_recompute_full_q8grain_20260325_033335/`
- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_20260310_104803/`
- `Llama-3.1-70B local inference availability and GPU-capable runtime`

## Proposed Steps

1. `현재 PT 파이프라인과 selected-target 제약을 정리해, ABC/ABD 최종 비교에 필요한 최소 변경점을 확정한다.`
2. `12개 Q에 대한 C query review scaffold 설계를 정하고, B/C/D selected target을 함께 다룰 수 있는 5-edge selected-target workflow를 기술명세 수준으로 구체화한다.`
3. `5-edge scorer 또는 cache 경로를 보강해 AC/BC lexical candidate evidence를 확보하고, 기존 scaffold/finalize 리뷰 코드를 재사용할 수 있는 입력 형태를 맞춘다.`
4. `그 다음 5-edge scorer 또는 후처리 경로를 보강해 AB/AC/AD/BC/BD를 같은 normalization pool에서 selected-target aware로 산출할 구현 계획을 확정한다.`
5. `shot별 ABC, ABD, delta를 내는 canonical output 구조와 검증 방식까지 포함한 실행 명세를 준비한다.`
6. `승인 후 tech spec에서 실제 수정 파일, 실행 엔트리포인트, launcher 권장안, 예상 산출물 경로를 확정한다.`

## Risks And Blockers

- `C selected target 리뷰 데이터가 아직 없어, 실행 전 review scaffold 생성과 사용자 승인 루프가 필요할 수 있다.`
- `기존 unified selected-target workflow는 B/D만 지원하므로, 이를 억지로 재사용하면 로직이 꼬일 수 있다.`
- `AC/BC lexical evidence 확보를 위해 5-edge inference 재실행이 필요할 가능성이 높다.`
- `정규화 scope를 잘못 섞으면 old 5-edge ABC와 unified ABD를 동일 척도로 해석할 수 없게 된다.`
- `GPU inference volume이 커질 수 있어 local 실행은 위험하다.`

## Recommended Compute Mode

- Mode: `gpu`
- Why: `C selected-target 리뷰 재료를 확보하고 5-edge selected-target aware 재계산까지 하려면 Llama-3.1-70B inference가 다시 필요할 가능성이 높고, 12개 Q x 다중 shot x 5 edge 구조는 CPU-only local 실행에 비해 GPU launcher가 현실적이다.`

## Expected Outputs

- `/home/sunsik/my_fv_project/plans/2026-03-31-pt-q12-abc-abd-selected-targets-plan.md`
- `/home/sunsik/my_fv_project/plans/2026-03-31-pt-q12-abc-abd-selected-targets-tech-spec.md`
- `12개 Q용 C selected-target scaffold 또는 finalized artifact`
- `selected-target aware 5-edge sweep CSV`
- `shot별 ABC/ABD/bootstrap summary CSV`
- `필요 시 updated human report 또는 비교용 summary table`

## Success Criteria

- `12개 Q에 대해 shot별 ABC, ABD, delta를 한 canonical CSV에서 확인할 수 있다.`
- `ABC와 ABD가 같은 5-edge normalization pool에서 계산되었다는 점이 코드와 산출물 메타에 명확히 남는다.`
- `C selected target 결정 방식이 artifact로 기록되고 재현 가능하다.`
- `최종 비교에 사용할 q 목록과 shot 목록이 문서와 산출물에서 일치한다.`

## Brain Impact

- Brain impact: `update required`
- Why: `이 작업이 완료되면 PT selected-target workflow의 범위가 B/D-only에서 C 포함 5-edge 비교로 확장되거나, 최소한 새로운 stable runbook과 해석 규칙이 생기므로 docs/brain/pipelines/pt.md 수준의 stable knowledge 갱신이 필요할 가능성이 높다.`

## Approval Note

Substantial execution has not started. Waiting for Korean approval.
