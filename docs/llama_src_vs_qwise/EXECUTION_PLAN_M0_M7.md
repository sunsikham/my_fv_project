# Execution Plan: M0-M7 (Based on SSOT)

## 공통 준비
```bash
cd /mnt/ebs/my_fv_project
PY=/mnt/ebs/venv/bin/python
MODEL=/mnt/ebs/.hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
FT=datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json
FT_ID=fixed_trials_antonym_t10_s10_seed0_llama31_8b
```

## M0. Llama fixed_trials 생성/검증
목표:
- fixed_trials가 src 기대 포맷과 일치하는지 확인

실행:
```bash
env PYTHONPATH=. $PY src/make_fixed_trials.py \
  --dataset_json datasets/antonym/raw/antonym.json \
  --out_path "$FT" \
  --n_trials 10 \
  --n_shots 10 \
  --seed 0 \
  --model_name_for_tokenizer "$MODEL" \
  --model_prepend_bos true \
  --prepend_bos_token_used false

env PYTHONPATH=. $PY scripts/verify_prompt_parity.py \
  --fixed_trials_path "$FT" \
  --max_trials 10 \
  --model_name_for_tokenizer "$MODEL"

env PYTHONPATH=. $PY scripts/verify_slot_parity_against_src.py \
  --fixed_trials_path "$FT" \
  --max_trials 10 \
  --mode corrupted \
  --tokenizer_name "$MODEL" \
  --assert_zero

env PYTHONPATH=. $PY src/make_fixed_trials.py \
  --verify true \
  --out_path "$FT" \
  --verify_n 5
```

완료 조건:
- prompt/slot `mismatch_count == 0`
- verify에서 token id 일치

## M1. src golden 생성
목표:
- src 기준선 3종 생성

실행:
```bash
env PYTHONPATH=. $PY scripts/run_m1_golden_artifacts.py \
  --dataset_name antonym \
  --fixed_trials_path "$FT" \
  --model_name "$MODEL" \
  --device cuda \
  --seed 0 \
  --n_shots 10 \
  --n_trials 10 \
  --save_path_root results \
  --python_bin /mnt/ebs/venv/bin/python
```

완료 조건:
- golden 3개 파일 생성
- mean rank=4, indirect_effect rank=3

## M2. parity suite gate
목표:
- src vs fv core parity 0 확인

실행:
```bash
env PYTHONPATH=. $PY scripts/run_parity_suite.py \
  --dataset_name antonym \
  --fixed_trials_path "$FT" \
  --fixed_trials_id "$FT_ID" \
  --model_name "$MODEL" \
  --device cuda \
  --seed 0 \
  --n_top_heads 10 \
  --token_class_idx -1 \
  --edit_layer 9 \
  --max_trials 5
```

완료 조건:
- parity suite PASS
- stage별 `mismatch_count == 0`

중단 규칙:
- FAIL이면 M3 이후 진행 금지. FAIL stage부터 수정.

## M3. src reference dump
목표:
- stage-wise diff용 src 기준 레퍼런스 저장

작업:
1. src 기준 dump 스크립트 준비/실행
2. `(trial_id, trial_idx)` 복합키로 정렬 저장
3. top_heads/FV/injection trial table 저장

최소 산출물:
- `src_reference_dump.json`
- `src_fv.pt`
- `top_heads_src.json`

완료 조건:
- `(trial_id, trial_idx)` 누락/중복 없음

## M4. antonym runtime 재현 (relation 제외)
목표:
- antonym에서 fv runtime 결과 확보

작업:
1. antonym 입력으로 runtime 실행
2. 아래 산출물 수집

필수 산출물:
- `mean_activations.pt`
- `mean_activations_meta.json`
- `top_heads.json`
- `fv_by_layer.pt`
- `fv_global_resid.pt`
- `fv_global_resid_meta.json`
- `step6_results_*.json`
- `eval_summary.json`
- `eval_trials.jsonl`
- `eval_meta.json`

완료 조건:
- 필수 산출물 100% 존재
- target_id/shape/hook 오류 없음

## M5. stage-by-stage diff (src vs fv on antonym)
목표:
- first divergence 특정

비교 순서:
1. prompt/token boundary
2. slot map
3. mean activations
4. top heads
5. fv vector
6. injection outputs

산출물:
- `src_vs_fv_diff_report.json`
- `src_vs_fv_diff_report.md`

완료 조건:
- first divergence가 stage/파일/키 수준으로 명시됨

## M6. patch + 재검증 루프
목표:
- first divergence 제거

루프:
1. patch
2. 해당 stage 재실행
3. M5 diff 재생성
4. mismatch 해소 확인

완료 조건:
- `mismatch_count == 0` (suite + trial-level)
- first divergence 없음

## M7. relation_qwise 전이 검증
목표:
- antonym에서 맞춘 로직이 relation_qwise에서도 유지되는지 확인

작업:
1. relation 단일 qid smoke
2. qid 확대
3. all-qid run

완료 조건:
- relation 전이에서도 동일 정합성 유지

## 최종 체크리스트
- M0 PASS
- M1 PASS
- M2 PASS
- M3 완료
- M4 완료
- M5 보고서 작성 완료
- M6에서 mismatch 0
- M7 전이 검증 완료

