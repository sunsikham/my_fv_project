# PLAN: relation_qwise Runtime를 antonym으로 고정해 src/fv 차이 추적

## 1) 목표
- `relation_qwise runtime` 코드를 그대로 사용한다.
- 데이터만 `antonym`으로 고정한다.
- `src/fv parity 경로` 대비 어디서 first divergence가 생기는지 찾는다.

## 2) 핵심 아이디어
- 이미 `M1~M3`에서 `src == fv` 동치 검증은 완료된 상태다.
- 이제 필요한 것은 "코드 경로 차이" 검증이다.
- 따라서 `relation_qwise runtime`에 동일 antonym 입력을 넣고 parity 기준선과 직접 비교한다.

## 3) SSOT (실험 계약)
- model: `/mnt/ebs/.hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- dataset: `antonym`
- seed: `0`
- max_trials: `5` (빠른 디버그)
- edit_layer: `9`
- n_top_heads: `10`
- token_class_idx: `-1`
- 공통 trial 키: `(trial_id, trial_idx)` 동시 사용

## 4) 입력/출력 계약
### 입력 고정
- parity 기준 trial: `datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json`
- relation runtime 입력은 위 trial과 동일 샘플을 사용하도록 맞춘다.

### 필수 산출물
- parity 경로:
  - `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/injection_parity_trials.csv`
  - `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/injection_parity_report.json`
- relation runtime 경로:
  - `.../artifacts/step6/layer_*/step6_results_*.json`
  - `.../artifacts/step6/layer_*/eval_summary.json`
  - `.../artifacts/step6/layer_*/eval_trials.jsonl`
  - `.../artifacts/step6/layer_*/step6.log`
- 비교 리포트:
  - `relation_runtime_vs_parity_diff_report.json`
  - `relation_runtime_vs_parity_diff_report.md`

## 5) 마일스톤
### R0. Freeze
- 파라미터/입력 파일/레이어를 고정한다.
- 변동값이 생기지 않도록 실행 스크립트 인자값을 명시한다.

### R1. Baseline 재확인 (parity 경로)
- 기존 M2 산출물을 재확인한다.
- `mismatch_count == 0`, `delta_logprob` 값이 존재하는지 확인한다.

### R2. relation runtime on antonym 실행
- relation_qwise 런타임 코드를 사용하되 antonym 입력으로 step6를 생성한다.
- `residual hook calls` 로그와 per-trial delta 산출 여부를 확인한다.

### R3. 정규화 비교
- 두 경로 산출물을 공통 스키마로 맞춘다.
- 비교 키: `trial_id`, `trial_idx`, `target_id`
- 비교 값:
  - clean_logprob
  - with_logprob
  - delta_logprob
  - (가능하면) clean/with logits max abs diff

### R4. First Divergence 판정
- stage 순서로 first divergence를 고정한다.
- stage 순서:
  1. prompt/token boundary
  2. target_id/slot
  3. fv(top_heads/fv norm)
  4. injection(clean/with/delta)
  5. summary aggregation

### R5. 패치 루프 (필요 시)
- first divergence stage만 최소 수정한다.
- 같은 입력으로 재실행 후 diff 리포트 갱신한다.

## 6) 완료 기준 (DoD)
- antonym 고정 조건에서 relation runtime vs parity 비교 리포트가 생성됨
- first divergence가 없거나, 있으면 stage/file/key 단위로 명확히 특정됨
- 최종적으로 목표 상태는:
  - parity 동치 유지
  - runtime 경로에서도 기대하는 delta/summary 재현 가능

## 7) 즉시 실행 우선순위
1. R0 Freeze
2. R1 Baseline 재확인
3. R2 relation runtime on antonym
4. R3/R4 diff report 생성

