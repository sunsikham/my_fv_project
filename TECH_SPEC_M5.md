# TECH_SPEC_M5.md

## Scope (M5 only)

### In
- 모델별 차이를 `ModelSpec` 계층으로 분리.
- GPT-2 parity 완료 상태를 유지한 채 추가 모델(예: LLaMA 계열) 호환 확장.
- 동일 파이프라인(M1~M4 검증 경로)에서 모델 교체가 가능하도록 인터페이스 정리.

### Out
- parity 알고리즘 자체 변경(M2/M3 의미론 수정)
- relation 실험 수행(M6)
- 성능 최적화/대규모 추론 튜닝

## Goal

- core FV/injection 로직은 모델 불문 동일하게 유지.
- 모델별 차이는 `fv/model_spec.py` + adapter resolution로만 처리.

## Current Baseline

- GPT-2 기준 M0~M4 parity suite PASS 상태.
- 기준 스크립트:
- `scripts/run_parity_suite.py`

## Target Architecture

모델 의존 코드는 아래 경계 안으로 제한:

1. Model specification
- `fv/model_spec.py`
- `blocks_path`
- `attn_path_in_block`
- `out_proj_path_in_attn`
- `prepend_bos`
- optional static dims (`n_layers`, `n_heads`, `head_dim`, `hidden_size`)

2. Adapter resolution
- `fv/adapters.py`
- `resolve_blocks`
- `resolve_attn`
- `resolve_out_proj`

3. Loader/model config bridge
- `fv/hf_loader.py`
- `fv/model_config.py`

core logic(`fv/fv.py`, `fv/intervene.py`)는 model-name 분기 최소화/제거가 원칙.

## Must-Match Compatibility Requirements

- Hook target equivalence:
- `src`의 attention out projection과 동일 모듈을 spec로 해석 가능해야 함.

- Injection hook point equivalence:
- parity 모드에서 layer block output edit 경로 유지 (`TraceDict + layer_hook_names + edit_output`).

- Tokenization/BOS policy:
- `prepend_bos`/`add_special_tokens` 정책이 모델별로 명확히 정의되어야 함.

- Dtype/device contract:
- parity는 CPU/fp32 기준.
- non-parity 실험 경로에서도 spec가 dtype/device 결정에 일관되게 관여해야 함.

## M5 Implementation Tasks

1. ModelSpec 확장
- `fv/model_spec.py`에 대상 모델 스펙 추가.
- 필수 경로 필드 검증(누락 시 초기 실패하도록).

2. Adapter smoke 강화
- `scripts/smoke_resolve_spec_outproj.py`로 신규 모델에서
- blocks/attn/out_proj 해석 가능 여부 확인.

3. Loader diagnostics 정비
- `fv/hf_loader.py` diagnostics에 out_proj class/path 확인 정보 유지.

4. Parity suite 파라미터화
- `scripts/run_parity_suite.py --model_name ...` 경로로 모델 교체 가능 상태 유지.
- 모델별 tokenizer/BOS 차이로 실패 시 원인 로그를 남김.

## Verification Plan (M5)

## 1) Spec resolution smoke
```bash
PYTHONPATH=. .venv/bin/python scripts/smoke_resolve_spec_outproj.py \
  --model <NEW_MODEL_ID> \
  --model_spec <NEW_SPEC_NAME> \
  --device cpu \
  --dtype fp32 \
  --quant none
```

## 2) Minimal forward/hook smoke
신규 모델에서 forward + hook 호출 성공 여부 확인.

## 3) Reduced parity run (sanity)
```bash
PYTHONPATH=. .venv/bin/python scripts/run_parity_suite.py \
  --dataset_name antonym \
  --fixed_trials_path datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json \
  --fixed_trials_id fixed_trials_antonym_t10_s10_seed0 \
  --model_name <NEW_MODEL_ID_OR_ALIAS> \
  --max_trials 1 \
  --n_top_heads 5 \
  --edit_layer <VALID_LAYER> \
  --seed 42 \
  --fail_fast 1
```

참고:
- M5 DoD는 “완전 parity PASS”가 아니라 “추가 모델에서 end-to-end 호환 경로가 에러 없이 동작”이 우선.
- 완전 parity 수치 기준은 모델별 안정화 후 별도 게이트로 승격.

## Reference Config Normalization Policy

- `src` 참조 경로의 모델명 분기 누락은 parity runner에서만 reference config 정규화로 보정 가능.
- GPT-2의 `c_proj` 강제 매핑 정책은 유지.
- 신규 모델 확장 시에도 이 정책은 “검증 스크립트 한정”이며 `fv` core semantics 변경 수단으로 사용하면 안 됨.

## DoD (M5)

- 신규 모델용 `ModelSpec` 추가 완료.
- out_proj resolution smoke PASS.
- 신규 모델에서 최소 forward/hook smoke PASS.
- reduced parity run이 크래시 없이 완료되고 stage 결과 파일 생성.
- 변경사항이 문서/로그로 재현 가능.

## Troubleshooting

1. blocks_path 불일치
- Symptom: layer block 해석 실패.
- Check: 실제 HF module tree와 spec `blocks_path` 일치 여부 확인.

2. out_proj_path 불일치
- Symptom: out_proj hook 등록 실패.
- Check: `attn_path_in_block + out_proj_path_in_attn` 경로 점검.

3. tokenizer/BOS 정책 불일치
- Symptom: prompt/slot parity에서 즉시 mismatch.
- Check: spec `prepend_bos`와 tokenizer `add_special_tokens` 사용 규칙 검토.

4. dtype/quant 조합 문제
- Symptom: 모델 로드 실패 또는 수치 불안정.
- Check: CPU parity에선 fp32/quant none 고정 후 재검증.

5. layer index 범위 오류
- Symptom: injection/edit layer invalid.
- Check: 모델 실 layer 수를 diagnostics에서 확인하고 `edit_layer` 재설정.

## Expected Files Tree (M5)

```text
results_fv/
  antonym/
    fixed_trials_antonym_t10_s10_seed0/
      parity_suite_report.json
      parity_suite_stages.csv
      parity_suite.log
```
