# Tech Spec: Llama src-vs-fv Experiment Contract (SSOT)

## 1. 목적
- `antonym`에서 `src`와 `fv`의 차이를 0으로 맞춘 뒤, 같은 로직을 `relation_qwise`로 전이한다.
- 이 문서는 실험 계약(SSOT) 고정 문서다. 여기 값이 바뀌면 같은 실험으로 간주하지 않는다.

## 2. 고정 실험 프로필 (변경 금지)
- Model path: `/mnt/ebs/.hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- Tokenizer path: 모델 경로와 동일
- Dataset (antonym): `datasets/antonym/raw/antonym.json`
- Fixed trials file: `datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json`
- Fixed trials ID: `fixed_trials_antonym_t10_s10_seed0_llama31_8b`
- Seed: `0`
- n_shots: `10`
- n_trials: `10` (본 계약 고정값)
- Device: `cuda`
- Prompt template: 기본 `Q:/A:` 템플릿

## 3. 산출물 파일 계약

### 3.1 M1 Golden (src 기준선)
- `results/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/antonym_mean_head_activations_FIXED.pt`
- `results/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/antonym_dummy_labels.json`
- `results/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/antonym_indirect_effect.pt`

### 3.2 Parity Gate 산출물
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/parity_suite_report.json`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/parity_suite_stages.csv`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/parity_suite.log`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/injection_parity_trials.csv`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/injection_parity_report.json`

### 3.3 antonym runtime 재현 산출물 계약
- `mean_activations.pt`
- `mean_activations_meta.json`
- `top_heads.json`
- `fv_by_layer.pt`
- `fv_global_resid.pt`
- `fv_global_resid_meta.json`
- `step6_results_*.json` (최소 1개)
- `eval_summary.json`
- `eval_trials.jsonl`
- `eval_meta.json`

## 4. 정렬 키/매칭 계약
- trial-level 비교는 `(trial_id, trial_idx)` 복합키로만 수행한다.
- 단일 `trial_id`만으로 매칭하지 않는다.
- 비교 테이블에 아래 키가 모두 있어야 한다:
  - `trial_id`
  - `trial_idx`
  - `target_id`
  - `base_logprob`
  - `with_logprob`
  - `delta_logprob`

## 5. 판정 기준 (Pass/Fail)
- Prompt parity: `mismatch_count == 0`
- Slot parity: `mismatch_count == 0`
- Parity suite 전체: `mismatch_count == 0`
- Injection parity trial-level: `mismatch_count == 0`
- Stage diff: first divergence 없음

## 6. 범위 규칙
- M0~M6: `antonym only`
- M7: `relation_qwise transfer`에서만 relation 산출물 사용
- M0~M6 단계에서 relation 지표로 품질 판정 금지

## 7. 변경 관리 규칙
- 아래 항목 중 하나라도 바뀌면 새로운 실험 세트로 분리:
  - model/tokenizer path
  - fixed_trials 파일/ID
  - seed, n_shots, n_trials
  - prompt template
  - device/dtype/quant

