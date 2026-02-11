# FINAL PLAN: Llama src vs fv (Antonym First, Relation Transfer Later)

## 0. Final Objective
- 최종 목표는 `relation_qwise`에서 `src`와 달라지는 지점을 찾는 것이다.
- 단, 지금 단계에서는 relation 판단을 하지 않고 `antonym`에서 `src vs fv` 오차를 0으로 만든다.
- `M0~M6`은 **antonym only**, `M7`에서만 relation 전이를 수행한다.

## 1. Phase Rules (Critical)
- `M0~M6`: antonym only
- `M7`: relation_qwise transfer only
- `M0~M6`에서 relation 산출물(`qid_status`, `step6_all_layers_summary`)로 품질 판정하지 않는다.
- 체크키는 아래 고정:
1. prompt/token boundary
2. slot map
3. mean activations
4. top heads
5. fv vector
6. injection output

## 2. Global Lock Profile (SSOT)
- Model: `/mnt/ebs/.hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- Dataset: `datasets/antonym/raw/antonym.json`
- Seed: `0`
- n_shots: `10`
- n_trials: `10` (본 계획 고정값)
- Device: `cuda`
- Prompt template: `Q:/A:` default
- Target token rule: boundary recompute (`target_first_token_id`)

Go/No-Go:
- 프로필 변경(run 중 model/tokenizer/seed 변경) 시 해당 결과는 비교에서 제외한다.
- `25+` 검증은 별도 후속 계획으로 분리한다. 본 문서 범위는 `n_trials=10`만 허용한다.

Common shell vars:
```bash
cd /mnt/ebs/my_fv_project
PY=/mnt/ebs/venv/bin/python
MODEL=/mnt/ebs/.hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
FT=datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json
```

## 3. Milestone M0: Llama Fixed Trials Canonicalization (antonym)
Goal:
- fixed_trials가 `src` 기대 포맷과 완전히 일치하는지 확정한다.

Actions:
1. Llama tokenizer 기준 fixed_trials 생성
2. prompt parity 검사
3. slot parity 검사
4. `src/make_fixed_trials.py --verify`로 token id 일치 검사

Deliverables:
- `datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json`
- parity 로그(`mismatch_count` 포함)

Pass criteria:
- prompt parity `mismatch_count == 0`
- slot parity `mismatch_count == 0`
- verify에서 `first_token_id == answer_ids_first`

Canonical command:
```bash
env PYTHONPATH=. /bin/bash -lc '
set -euo pipefail
cd /mnt/ebs/my_fv_project
PY=/mnt/ebs/venv/bin/python
MODEL=/mnt/ebs/.hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
FT=datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json

$PY src/make_fixed_trials.py \
  --dataset_json datasets/antonym/raw/antonym.json \
  --out_path "$FT" \
  --n_trials 10 \
  --n_shots 10 \
  --seed 0 \
  --model_name_for_tokenizer "$MODEL" \
  --model_prepend_bos true \
  --prepend_bos_token_used false

$PY scripts/verify_prompt_parity.py \
  --fixed_trials_path "$FT" \
  --max_trials 10 \
  --model_name_for_tokenizer "$MODEL"

$PY scripts/verify_slot_parity_against_src.py \
  --fixed_trials_path "$FT" \
  --max_trials 10 \
  --mode corrupted \
  --tokenizer_name "$MODEL" \
  --assert_zero

$PY src/make_fixed_trials.py \
  --verify true \
  --out_path "$FT" \
  --verify_n 5
'
```

## 4. Milestone M1: src Golden Artifact Build (antonym)
Goal:
- `src` 기준선(golden) 3종을 생성한다.

Actions:
1. `scripts/run_m1_golden_artifacts.py` 실행 (Llama + fixed_trials)
2. canonical 경로 정규화
3. shape sanity check

Deliverables:
- `results/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/antonym_mean_head_activations_FIXED.pt`
- `results/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/antonym_dummy_labels.json`
- `results/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/antonym_indirect_effect.pt`

Pass criteria:
- 3개 파일 존재
- mean rank=4
- indirect_effect rank=3 (`last_token_only`)

Canonical command:
```bash
env PYTHONPATH=. /bin/bash -lc '
set -euo pipefail
cd /mnt/ebs/my_fv_project
PY=/mnt/ebs/venv/bin/python
MODEL=/mnt/ebs/.hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
FT=datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json

$PY scripts/run_m1_golden_artifacts.py \
  --dataset_name antonym \
  --fixed_trials_path "$FT" \
  --model_name "$MODEL" \
  --device cuda \
  --seed 0 \
  --n_shots 10 \
  --n_trials 10 \
  --save_path_root results \
  --python_bin /mnt/ebs/venv/bin/python
'
```

## 5. Milestone M2: Core Parity Gate (antonym)
Goal:
- relation 이전에 `src vs fv` core parity를 0으로 맞춘다.

Actions:
1. `scripts/run_parity_suite.py` 실행
2. stage별 mismatch 및 earliest fail stage 확인

Deliverables:
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/parity_suite_report.json`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/parity_suite_stages.csv`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/parity_suite.log`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/injection_parity_trials.csv`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/injection_parity_report.json`

Pass criteria:
- suite PASS
- stage별 `mismatch_count == 0`

Decision rule:
- FAIL이면 M3/M4로 넘어가지 않고 FAIL stage부터 수정한다.

Canonical command:
```bash
env PYTHONPATH=. /bin/bash -lc '
set -euo pipefail
cd /mnt/ebs/my_fv_project
PY=/mnt/ebs/venv/bin/python
MODEL=/mnt/ebs/.hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
FT=datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_llama31_8b.json

$PY scripts/run_parity_suite.py \
  --dataset_name antonym \
  --fixed_trials_path "$FT" \
  --fixed_trials_id fixed_trials_antonym_t10_s10_seed0_llama31_8b \
  --model_name "$MODEL" \
  --device cuda \
  --seed 0 \
  --n_top_heads 10 \
  --token_class_idx -1 \
  --edit_layer 9 \
  --max_trials 5
'
```

## 6. Milestone M3: src Reference Dump (antonym)
Goal:
- 단계별 diff가 가능하도록 `src` 레퍼런스 텐서/메타를 저장한다.

Actions:
1. src 기준 값을 `(trial_id, trial_idx)` 복합키 정렬로 dump
2. 최소 항목 저장

Required fields:
- `trial_id`
- `trial_idx`
- mean shape + `QUERY_PRED` slot index
- indirect_effect summary (top-k)
- `top_heads_src.json`
- `src_fv.pt` (+norm)
- injection trial table: `base_logprob`, `with_logprob`, `delta_logprob`, `target_id`

Deliverables:
- `results/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/src_reference_dump.json`
- `results/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/src_fv.pt`
- `results/antonym/fixed_trials_antonym_t10_s10_seed0_llama31_8b/top_heads_src.json`

Pass criteria:
- `(trial_id, trial_idx)` 누락/중복 없음
- prompt boundary/target_id 정보 포함

## 7. Milestone M4: antonym fv Runtime Reproduction (no relation)
Goal:
- relation 경로를 섞지 않고 antonym에서 fv runtime을 재현한다.

Actions:
1. antonym fixed/sampled trials로 fv runtime 실행
2. mean/top_heads/fv/injection 산출물 수집

Required outputs:
- `mean_activations.pt`
- `mean_activations_meta.json`
- `top_heads.json`
- `fv_by_layer.pt`
- `fv_global_resid.pt`
- `fv_global_resid_meta.json`
- `step6_results_*.json` (at least one)
- `eval_summary.json`
- `eval_trials.jsonl`
- `eval_meta.json`

Pass criteria:
- 필수 산출물 생성
- 로그상 target_id/shape/hook 오류 없음

## 8. Milestone M5: Stage-by-Stage Diff (src vs fv on antonym)
Goal:
- 첫 divergence stage를 기계적으로 특정한다.

Strict comparison order:
1. Prompt/Token boundary
2. Slot map
3. Mean activations
4. Top heads
5. FV vector
6. Injection outputs

Deliverables:
- `results_fv/antonym/src_vs_fv_diff_report.json`
- `results_fv/antonym/src_vs_fv_diff_report.md`

Pass criteria:
- first divergence가 stage/파일/키 단위로 명시됨
- 재현 명령 1줄 포함

## 9. Milestone M6: Patch and Re-verify (antonym)
Goal:
- M5에서 찾은 first divergence를 제거한다.

Loop:
1. patch
2. 해당 stage 재실행
3. diff 재생성
4. mismatch 해소 여부 확인

Exit criteria:
- `mismatch_count == 0` (suite + trial-level)
- first divergence 없음
- 동일 프로필 재실행 안정

## 10. Milestone M7: relation_qwise Transfer Validation
Goal:
- antonym에서 0으로 맞춘 로직이 relation_qwise에서도 유지되는지 확인한다.

Actions:
1. relation 단일 qid smoke
2. 문제 없으면 qid 확대
3. 최종 all-qid

Deliverables:
- `qid_status.json` 완료
- `step6_all_layers_summary.json` 생성
- 실패 qid가 있으면 원인 분류 리포트

Pass criteria:
- antonym에서 확정한 divergence fix가 relation에서도 유지됨

## 11. Final Acceptance Criteria
- `M0~M6` 완료 시: antonym 기준 `src vs fv` mismatch 0
- `M7` 완료 시: relation_qwise 전이에서도 동일 정합성 유지

## 12. Execution Priority
1. M0
2. M1
3. M2
4. M3
5. M4
6. M5
7. M6
8. M7
