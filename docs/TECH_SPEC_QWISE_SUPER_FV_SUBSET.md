# Tech Spec: Q-Subset Super FV Injection (SSOT)

## 1. 목적
- `qwise` 실험에서 q별 FV 대신, 사용자가 선택한 q subset으로부터 하나의 공용 FV(`super FV`)를 만든다.
- 기존 Step6 주입/평가 로직은 유지하고, 주입 벡터(FV)만 교체했을 때 지표 변화(`acc`, `logprob`, `delta`)를 측정한다.
- 본 문서는 실험 계약(SSOT)이다. 항목이 바뀌면 동일 실험으로 간주하지 않는다.

## 2. 핵심 정의
- `q subset`: super FV를 만들 때 사용할 q_id 목록(사용자 지정).
- `super FV`: q subset의 StepD 계열 정보를 통합해 만든 단일 FV.
- `평균 규칙`: q subset 전체에서 layer-head 값을 평균한 집계값 사용.
- `top-k 규칙`: 평균 점수 기준으로 전역 `top 20 head` 선택.
- `주입 규칙`: 선택된 top-20 기반으로 만든 super FV를 각 q 평가에 공통 주입.

## 3. In Scope
- q subset 기반 super FV 생성
- 기존 Step6 injection/eval 경로 재사용
- q별 비교 지표 산출:
  - `acc_base`, `acc_with`, `delta_acc`
  - `mean_delta_logprob`
  - `mean_delta_p`
  - `mean_delta_logit`

## 4. Out of Scope
- StepD/StepE/Step6 핵심 계산 로직 변경
- prompt 형식, trial 샘플링 규칙 변경
- 모델/토크나이저/spec 변경 실험
- q subset 외 q를 super FV 생성에 포함

## 5. 고정 계약 (변경 금지)
- 모델/디바이스/양자화 설정은 기존 qwise 런과 동일 프로필 사용
- Step6 평가 입력(`sampled_trials_zeroshot.json` 등)은 기존 방식 유지
- top-k는 항상 `20`
- super FV 생성 시 q별 가중치는 동일(단순 평균)
- 동일 seed/동일 입력이면 동일 결과가 재현되어야 함

## 6. 입력 계약
### 6.1 필수 입력
- relation run root: `results_fv/relation_qwise/<relation_name>/`
- q subset 목록(예: `Q1,Q3,Q10`)
- 각 q의 StepD/StepE 산출물(통합 가능한 layer-head 정보)
- Step6 평가 trial snapshot:
  - 기본: `results_fv/relation_qwise/<relation_name>/<QID>/artifacts/sampled_trials_zeroshot.json`
  - baseline/super 비교 시 동일 snapshot 사용(동일 q, 동일 seed)

### 6.2 유효성 조건
- q subset의 각 q는 필수 산출물이 존재해야 함
- 누락 q가 있으면 기본 정책은 실패 처리(부분 성공 허용 여부는 별도 플래그로만 허용)
- layer/head shape가 서로 호환되어야 함

## 7. 처리 계약
1. q subset의 layer-head 값을 수집한다.
2. q 축으로 평균하여 전역 layer-head 점수를 만든다.
3. 전역 점수 기준으로 top-20 head를 선택한다.
4. 선택 head로 super FV(`global`)를 생성한다.
5. 기존 Step6 주입 루틴에서 q별 FV 대신 super FV를 사용한다.
6. 기존과 동일한 eval summary/trial 지표를 산출한다.

## 8. 산출물 계약
### 8.1 super FV 생성 산출물
- `.../_super_fv/<run_id>/super_fv_global_resid.pt`
- `.../_super_fv/<run_id>/super_fv_by_layer.pt`
- `.../_super_fv/<run_id>/super_top_heads.json`
- `.../_super_fv/<run_id>/super_fv_meta.json`

### 8.2 q별 주입/평가 산출물
- `.../_super_fv/<run_id>/eval_by_q/<QID>/layer_<LAYER>/eval_summary.json`
- `.../_super_fv/<run_id>/eval_by_q/<QID>/layer_<LAYER>/eval_trials.jsonl`
- `.../_super_fv/<run_id>/eval_by_q/<QID>/layer_<LAYER>/step6.log`
- `.../_super_fv/<run_id>/eval_by_q/<QID>/step6_all_layers_summary.json`
- 경로/파일명 규칙은 기존 Step6 레이어 단위 구조를 재사용한다.

### 8.3 비교 리포트
- `.../_super_fv/<run_id>/comparison_vs_baseline.csv`
- `.../_super_fv/<run_id>/comparison_vs_baseline.json`
- 포함 컬럼:
  - `q_id`
  - `layer`
  - `baseline_layer`, `super_layer` (고정 layer 비교면 두 값 동일)
  - `acc_base_baseline`, `acc_with_baseline`, `delta_acc_baseline`
  - `acc_base_super`, `acc_with_super`, `delta_acc_super`
  - `mean_delta_logprob_baseline`, `mean_delta_logprob_super`
  - `mean_delta_p_baseline`, `mean_delta_p_super`
  - `mean_delta_logit_baseline`, `mean_delta_logit_super`
  - `diff_*` (super - baseline)

## 9. 판정 기준 (Pass/Fail)
- Pass:
  - q subset 전부에서 super FV 생성 성공
  - 모든 평가 대상 q/layer에서 eval_summary 생성 성공
  - 비교 리포트 생성 성공
- Fail:
  - 필수 입력 누락
  - shape 불일치
  - top-20 선택 불가
  - eval 단계 중단/요약 미생성

## 10. 리스크 및 가정
- 일부 q는 기존 단일-q FV 대비 성능 하락 가능
- q subset 구성에 따라 super FV 성격이 크게 변함
- baseline과 super 비교는 동일 평가 샘플/동일 seed에서만 유효

## 11. 변경 관리 규칙
- 아래 변경 시 새 실험 세트로 분리:
  - q subset 구성
  - top-k 값(20 외)
  - 평균 규칙(단순 평균 외 가중 평균 등)
  - 모델/spec/device/dtype/quant
  - 평가 입력 샘플/seed
