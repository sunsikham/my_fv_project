# PLAN: Q-Subset Super FV Injection

## 1. 목표
- 사용자가 선택한 q subset으로 `super FV`를 1회 생성한다.
- 기존 Step6 주입/평가 로직은 유지하고, FV만 `super FV`로 교체한다.
- q/layer 단위로 baseline 대비 metric 변화(`acc`, `logprob`, `delta`)를 정량 비교한다.

## 2. 입력 고정
- relation: `relationB_ex` (또는 실행 인자로 지정)
- q subset: 실행 인자로 명시(예: `Q1,Q3,Q10`)
- top-k: `20` (고정)
- seed: baseline과 동일
- model/spec/device/dtype/quant: 기존 qwise와 동일
- Step6 trial snapshot: 각 q의 `artifacts/sampled_trials_zeroshot.json` 재사용

## 3. 단계별 실행
### P1. Preflight
- q subset의 필수 산출물 존재 확인:
  - StepD/StepE 관련 입력(통합 가능한 layer-head 정보)
  - `sampled_trials_zeroshot.json`
- shape/레이어/헤드 호환성 검증
- 누락 시 실패 처리(기본)

### P2. Super FV 생성
- q subset의 layer-head 값/점수를 수집
- q 축 평균으로 전역 layer-head 점수 계산
- 전역 top-20 head 선택
- 선택 head로 super FV 산출:
  - `super_fv_global_resid.pt`
  - `super_fv_by_layer.pt`
  - `super_top_heads.json`
  - `super_fv_meta.json`

### P3. Step6 재실행 (기존 로직 재사용)
- 대상 q별, 레이어별(`layer_<LAYER>`)로 Step6 실행
- 기존 Step6 출력 유지:
  - `eval_summary.json`
  - `eval_trials.jsonl`
  - `step6.log`
- q별 `step6_all_layers_summary.json` 생성

### P4. Baseline 비교 리포트
- 동일 `q_id`, 동일 `layer`로 baseline/super 조인
- 출력:
  - `comparison_vs_baseline.csv`
  - `comparison_vs_baseline.json`
- 필수 비교 필드:
  - `q_id`, `layer`, `baseline_layer`, `super_layer`
  - `acc_base_*`, `acc_with_*`, `delta_acc_*`
  - `mean_delta_logprob_*`, `mean_delta_p_*`, `mean_delta_logit_*`
  - `diff_*` (super - baseline)

### P5. 완료 검증
- q/layer 전부 `eval_summary.json` 생성 확인
- missing/failed layer 0 확인
- 비교 리포트 행 수 검증(`n_q * n_layers`)
- 실행 메타/파라미터 기록 확인

## 4. 저장 경로 계약
- 루트:
  - `results_fv/relation_qwise/<relation_name>/_super_fv/<run_id>/`
- 하위:
  - `super_fv_global_resid.pt`
  - `super_fv_by_layer.pt`
  - `super_top_heads.json`
  - `super_fv_meta.json`
  - `eval_by_q/<QID>/layer_<LAYER>/eval_summary.json`
  - `eval_by_q/<QID>/layer_<LAYER>/eval_trials.jsonl`
  - `eval_by_q/<QID>/layer_<LAYER>/step6.log`
  - `eval_by_q/<QID>/step6_all_layers_summary.json`
  - `comparison_vs_baseline.csv`
  - `comparison_vs_baseline.json`

## 5. 구현 순서
1. `build_super_fv_from_q_subset` 유틸 구현
2. `run_qwise_super_fv_pipeline` 오케스트레이터 구현
3. Step6 호출 어댑터(기존 Step6 인자 체계 유지) 연결
4. baseline 비교 리포트 생성기 구현
5. dry-run + 소규모 q subset으로 검증
6. 전체 대상 실행

## 6. 리스크/대응
- 리스크: q subset 편향으로 일부 q 성능 저하
  - 대응: subset 버전별 결과 분리 저장, diff 리포트로 영향 추적
- 리스크: layer/head shape mismatch
  - 대응: P1에서 즉시 실패 처리
- 리스크: baseline/super 입력 불일치
  - 대응: 동일 snapshot 경로/seed를 메타에 강제 기록

