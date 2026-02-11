# Implementation Run Plan (Immediate Execution)

## Goal
- Llama 기준으로 `src vs fv` 정합성을 `antonym`에서 먼저 0으로 맞춘다.
- 이후에만 `relation_qwise` 전이 검증으로 넘어간다.

## SSOT (Do Not Change)
- `MODEL=/mnt/ebs/.hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- `FT=datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json`
- `FT_ID=fixed_trials_antonym_t10_s10_seed0_llama31_8b`
- `seed=0`
- `n_shots=10`
- `n_trials=10`
- `device=cuda`
- `dataset=datasets/antonym/raw/antonym.json`

## M0: Fixed Trials Build + Validation
1. Llama tokenizer 기준 fixed_trials 생성
2. Prompt parity 확인
3. Slot parity 확인
4. `src/make_fixed_trials.py --verify` 확인
5. Gate: `mismatch_count == 0` 아니면 즉시 중단

## M1: src Golden Build
1. `scripts/run_m1_golden_artifacts.py` 실행
2. Golden 3종 존재 확인
- `antonym_mean_head_activations_FIXED.pt`
- `antonym_dummy_labels.json`
- `antonym_indirect_effect.pt`
3. Shape 확인
- mean rank=4
- indirect_effect rank=3
4. Gate: 누락/shape 불일치 시 중단

## M2: Parity Gate
1. `scripts/run_parity_suite.py` 실행
2. `parity_suite_report.json`에서 전체 PASS 확인
3. stage별 `mismatch_count == 0` 확인
4. Gate: FAIL이면 M3+ 진행 금지, 실패 stage부터 수정

## M3-M6: antonym only
- M3: src reference dump 저장 (`trial_id`, `trial_idx` 복합키)
- M4: antonym runtime 산출물 수집
- M5: stage diff (`prompt/token -> slot -> mean -> top_heads -> fv -> injection`)
- M6: patch + 재검증 루프
- Exit: suite + trial-level mismatch 모두 0

## M7: relation_qwise Transfer
1. 단일 qid smoke
2. qid 확대
3. all-qid 실행
4. Gate: antonym에서 해결한 divergence가 relation에서 재발하면 중단

## Acceptance
- `M0~M6` 완료 시: antonym 기준 `src vs fv mismatch_count == 0`
- `M7` 완료 시: relation_qwise에서도 동일 정합성 유지
