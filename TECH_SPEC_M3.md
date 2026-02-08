# TECH_SPEC_M3.md

## Scope (M3 only)

### In
- `fv`의 FV injection 로직을 `src`와 동등하게 구현.
- 고정된 FV/입력 프롬프트에서 `src` vs `fv` intervention 결과 parity 검증.
- 최소 지표 parity:
- clean target logprob
- intervention target logprob
- delta logprob (`with - clean`)

### Out
- FV construction parity (`M2`) 재작업
- parity suite 통합 (`M4+`)
- 모델 확장 (`M5+`)
- relation 실험 (`M6+`)

## Preconditions

- M1 golden artifacts 준비:
- `results/antonym/fixed_trials_antonym_t10_s10_seed0/antonym_mean_head_activations_FIXED.pt`
- `results/antonym/fixed_trials_antonym_t10_s10_seed0/antonym_indirect_effect.pt`
- M2 parity 가능한 FV builder 준비 (`fv.compute_function_vector`).
- parity runtime contract:
- CPU, float32, `model.eval()`, fixed seed, `torch.set_grad_enabled(False)`.

## Reference Semantics (src ground truth)

Reference functions:
- `src/utils/intervention_utils.py:100` `add_function_vector(...)`
- `src/utils/intervention_utils.py:126` `function_vector_intervention(...)`

Must-match rules:
- Hook point (mandatory):
- `TraceDict(model, layers=model_config["layer_hook_names"], edit_output=...)`
- reference: `src/utils/intervention_utils.py:173`
- Not allowed for M3 parity:
- attention out_proj pre-hook injection (`c_proj/o_proj pre-hook`) 경로.
- Injection operation:
- `edit_layer`에서만 hidden state에 FV를 더한다.
- default index `idx = -1` (single-token next-token mode).
- NLL mode index rule: `idx = -1 - target_len` (`src/utils/intervention_utils.py:160`).
- Output comparison unit:
- last-token logits 기반 target token logprob.
- parity primary metric: `delta_logprob = intervention_logprob - clean_logprob`.

## Reference Config Normalization (runner-only)

- `src` config branch coverage can vary by exact model name.
- M3 parity runner may normalize **reference config only** to keep `src` call path canonical.
- GPT-2 parity rule:
- force reference `src` out_proj mapping to `c_proj`.
- Scope restriction:
- normalization is allowed only in verification scripts and must not alter `fv` core behavior.

## Target Implementation in fv

Primary implementation targets:
- `fv`에 `add_function_vector(...)` 등가 함수 추가/정비
- `fv`에 `function_vector_intervention(...)` 등가 함수 추가/정비

Implementation requirements:
- `TraceDict` 기반 `layer_hook_names` 편집 사용.
- tuple output(`(hidden, ...)`)과 tensor output 둘 다 처리.
- injection vector shape은 `(1, resid_dim)` 또는 broadcast 가능한 동일 의미 shape.
- target token logprob 계산 로직을 `src`와 동일 기준으로 구현.

## M3 Verification Plan

## 1) Deterministic environment
```bash
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=""
```

## 2) FV 준비 (M2 결과 재사용)
```bash
PYTHONPATH=. .venv/bin/python scripts/verify_fv_parity.py \
  --dataset_name antonym \
  --fixed_trials_id fixed_trials_antonym_t10_s10_seed0 \
  --model_name gpt2 \
  --n_top_heads 10 \
  --token_class_idx -1 \
  --assert_zero
```

## 3) Injection parity 실행 (M3)
권장 신규 러너:
- `scripts/verify_injection_parity.py`

Expected command:
```bash
PYTHONPATH=. .venv/bin/python scripts/verify_injection_parity.py \
  --dataset_name antonym \
  --fixed_trials_path datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json \
  --fixed_trials_id fixed_trials_antonym_t10_s10_seed0 \
  --model_name gpt2 \
  --n_top_heads 10 \
  --max_trials 5 \
  --edit_layer 9 \
  --assert_zero
```

Required checks inside runner:
- `src` clean target logprob vs `fv` clean target logprob exact match.
- `src` intervention target logprob vs `fv` intervention target logprob exact match.
- `src` delta logprob vs `fv` delta logprob exact match.
- mismatch aggregate:
- `mismatch_count == 0` when `--assert_zero`.

## Canonical Output for M3

- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0/injection_parity_report.json`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0/injection_parity_trials.csv`

## DoD (M3)

- `fv` injection 함수가 `TraceDict + layer_hook_names + edit_output`로 동작.
- 동일 trial 집합에서 아래가 `src`와 exact match:
- clean target logprob
- intervention target logprob
- delta logprob
- parity runner 종료 코드 `0` and `mismatch_count: 0`.
- 동일 명령 2회 재실행 시 동일 결과 재현.

## Troubleshooting

1. Wrong hook point
- Symptom: clean은 맞지만 intervention 값이 지속적으로 다름.
- Check: out_proj pre-hook이 아니라 layer block output edit인지 확인.

2. Wrong token index policy
- Symptom: NLL 모드에서만 불일치.
- Check: `idx = -1 - target_len` 규칙 적용 여부 확인.

3. Edit layer mismatch
- Symptom: 델타 크기/방향이 src와 다름.
- Check: 동일 `edit_layer`를 src/fv 모두에 사용했는지 확인.

4. Tensor shape/broadcast mismatch
- Symptom: 런타임 shape 에러 또는 조용한 수치 오차.
- Check: FV shape을 `(1, resid_dim)`로 normalize 후 더하기.

5. Runtime contract drift
- Symptom: 작은 수치 차이가 남음.
- Check: CPU/fp32/eval/no-grad/fixed-seed 강제 여부 확인.

## Expected Files Tree (M3)

```text
results_fv/
  antonym/
    fixed_trials_antonym_t10_s10_seed0/
      injection_parity_report.json
      injection_parity_trials.csv
```
