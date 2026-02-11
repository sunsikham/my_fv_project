# Tech Spec: relation_qwise Runtime with antonym (SSOT)

## 1. 목적
- `relation_qwise runtime` 코드 경로를 유지한 채, 입력 데이터를 `antonym`으로 고정한다.
- `src/fv parity` 기준선과 `relation runtime` 산출물을 동일 조건에서 비교해 first divergence를 찾는다.
- 본 문서는 실험 계약(SSOT)이다. 항목이 바뀌면 동일 실험으로 간주하지 않는다.

## 2. 고정 실험 프로필 (변경 금지)
- Model path: `/mnt/ebs/.hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- Dataset: `antonym`
- Fixed trials file: `datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json`
- Fixed trials ID: `fixed_trials_antonym_t10_s10_seed0_llama31_8b`
- Seed: `0`
- Device: `cuda`
- n_top_heads: `10`
- token_class_idx: `-1`
- edit_layer: `9`
- max_trials: `5` (debug profile)

## 3. 비교 범위 (In Scope)
- `src/fv parity` 경로 산출물 vs `relation runtime on antonym` 경로 산출물 비교
- trial-level 지표 비교:
  - `clean_logprob`
  - `with_logprob`
  - `delta_logprob`
- summary-level 지표 비교:
  - `mean_delta_logprob`
  - `mean_delta_p`
  - `mean_delta_logit`

## 4. 비범위 (Out of Scope)
- relation dataset 자체 품질 판정
- 대규모 sweep (`max_trials > 5`) 결과 일반화
- model/spec 변경 실험

## 5. 산출물 계약
### 5.1 parity 기준 산출물
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/parity_suite_report.json`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/injection_parity_trials.csv`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/injection_parity_report.json`

### 5.2 relation runtime (antonym 입력) 산출물
- `.../artifacts/step6/layer_*/step6_results_*.json`
- `.../artifacts/step6/layer_*/eval_summary.json`
- `.../artifacts/step6/layer_*/eval_trials.jsonl`
- `.../artifacts/step6/layer_*/step6.log`

### 5.3 diff 산출물
- `relation_runtime_vs_parity_diff_report.json`
- `relation_runtime_vs_parity_diff_report.md`

## 6. 정렬/매칭 계약
- trial-level 비교 키는 `(trial_id, trial_idx)` 복합키를 기본으로 사용한다.
- `trial_id`만 단독 매칭하지 않는다.
- `target_id` 불일치 시 해당 trial은 즉시 mismatch로 판정한다.

## 7. 판정 기준 (Pass/Fail)
- parity suite: `status == PASS`
- injection parity: `mismatch_count == 0`
- runtime step6: hook/shape/target_id 오류 없음
- diff report: first divergence가 없거나, 있으면 stage/file/key 단위로 특정

## 8. 실행 순서 규약
1. parity 기준선 확인 (M2/M3)
2. relation runtime on antonym 실행
3. trial-level/summary-level 정규화 비교
4. first divergence 판정

## 9. 변경 관리 규칙
- 아래 항목 변경 시 새 실험 세트로 분리한다:
  - model path
  - fixed trials file/ID
  - seed, max_trials
  - edit_layer, n_top_heads, token_class_idx
  - device/dtype/quant

