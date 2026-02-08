# TECH_SPEC_M2.md

## Scope (M2 only)

### In
- `fv`에서 Function Vector(FV) 구성 로직을 `src`와 동등하게 구현.
- Fixed trials 기반 golden(`M1`)을 입력으로 FV parity 검증.
- `top_heads`와 `function_vector` 동시 parity 체크.

### Out
- FV injection parity (`M3+`)
- parity suite 통합 (`M4+`)
- 모델 확장 (`M5+`)
- relation 실험 (`M6+`)

## Inputs and Preconditions

- Canonical golden run_dir:
- `results/antonym/fixed_trials_antonym_t10_s10_seed0/`
- Required inputs:
- `antonym_mean_head_activations_FIXED.pt`
- `antonym_indirect_effect.pt`
- Runtime parity contract:
- CPU, float32, `model.eval()`, fixed seed, `torch.set_grad_enabled(False)`.

## Reference Semantics (must match src exactly)

Reference function:
- `src/utils/extract_utils.py:415` `compute_function_vector(...)`

Must-match rules:
- Indirect effect aggregation:
- If shape is `(N, L, H)` and `token_class_idx == -1`, use `indirect_effect.mean(dim=0)` (`src/utils/extract_utils.py:438` to `src/utils/extract_utils.py:439`).
- Else expect `(N, L, H, C)` and use `indirect_effect[:,:,:,token_class_idx].mean(dim=0)` (`src/utils/extract_utils.py:440` to `src/utils/extract_utils.py:442`).
- Top-head selection:
- Flatten `mean_indirect_effect` then `torch.topk(..., largest=True)` and unravel back to `(layer, head)` preserving ordering (`src/utils/extract_utils.py:445` to `src/utils/extract_utils.py:448`).
- `top_heads` score formatting:
- Round score to 4 decimals (`round(x.item(), 4)`) (`src/utils/extract_utils.py:447`).
- FV synthesis:
- `T = -1` token slot from mean activations (`src/utils/extract_utils.py:452`).
- Build residual vector `x` with head slice insertion (`src/utils/extract_utils.py:464` to `src/utils/extract_utils.py:465`).
- Pass through per-layer attention out projection and sum (`src/utils/extract_utils.py:466` to `src/utils/extract_utils.py:468`).
- Output shape/type:
- final reshape `(1, resid_dim)` and cast to `model.dtype` (`src/utils/extract_utils.py:470` to `src/utils/extract_utils.py:472`).

## Target Implementation in fv

Primary code target:
- Add `compute_function_vector(...)` in `fv` (new module or `fv/fv.py` extension).

Recommended signature:
```python
def compute_function_vector(
    mean_activations,
    indirect_effect,
    model,
    model_config,
    n_top_heads: int = 10,
    token_class_idx: int = -1,
):
    ...
    return function_vector, top_heads
```

Implementation notes:
- Reuse `fv.adapters.resolve_out_proj(...)` to resolve out projection module by model spec (`fv/adapters.py:75`).
- Keep `top_heads` type as list of `(layer:int, head:int, score:float)` with 4-decimal score.
- Keep runtime tensor placement/casting order aligned with `src`.

## Reference Config Normalization (runner-only)

- `src` may have model-name-specific branch gaps in config-dependent paths.
- Therefore M2 parity runner may normalize **reference config only** before invoking `src`.
- GPT-2 parity rule:
- force `src` out_proj path to `c_proj` branch (canonical mapping).
- Scope restriction:
- this mapping must exist only in verification scripts (e.g., `scripts/verify_fv_parity.py`).
- `fv` core logic must remain unaffected by this normalization policy.

## Parity Verification Plan

## 1) Deterministic setup
```bash
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=""
```

## 2) Golden inputs check
```bash
test -f results/antonym/fixed_trials_antonym_t10_s10_seed0/antonym_mean_head_activations_FIXED.pt
test -f results/antonym/fixed_trials_antonym_t10_s10_seed0/antonym_indirect_effect.pt
```

## 3) FV parity execution (M2)
Recommended runner:
- `scripts/verify_fv_parity.py` (add in M2 implementation if missing)

Expected command:
```bash
PYTHONPATH=. .venv/bin/python scripts/verify_fv_parity.py \
  --dataset_name antonym \
  --fixed_trials_id fixed_trials_antonym_t10_s10_seed0 \
  --model_name gpt2 \
  --n_top_heads 10 \
  --token_class_idx -1 \
  --assert_zero
```

Minimum checks in runner:
- `top_heads` exact equality:
- same length, same tuple order, same rounded score.
- `function_vector` exact equality:
- `shape == (1, resid_dim)`
- `dtype` matches parity contract (float32 runtime on CPU)
- `max_abs_diff == 0`

## DoD (M2)

- `fv.compute_function_vector` exists and is callable in parity path.
- Under fixed trials + parity runtime contract:
- `top_heads` exact match with `src` (ordering + 4-decimal scores).
- `function_vector` exact match (`max_abs_diff == 0`).
- Re-run stability:
- same command run 2 times gives identical `top_heads` and `max_abs_diff`.
- Comparison logs/reports are saved for traceability.

## Suggested Artifacts (results_fv)

- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0/fv_function_vector.pt`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0/fv_top_heads.json`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0/fv_parity_report.json`

## Troubleshooting

1. Top-head order mismatch
- Symptom: same heads but different order.
- Check: `topk` before unravel, and preserve returned index order.

2. Score formatting mismatch
- Symptom: numeric close but head tuples mismatch.
- Check: apply `round(score, 4)` before emitting/comparing.

3. Wrong token slot
- Symptom: FV norm/value 크게 다름.
- Check: ensure `T = -1` is used for head slice extraction.

4. Wrong out_proj module
- Symptom: per-head projected vector mismatch.
- Check: resolve attention out projection path by model spec (not hardcoded single-model path).

5. Device/dtype drift
- Symptom: tiny numeric diffs remain.
- Check: CPU + float32 + eval + no grad; ensure cast/order mirrors `src`.

## Expected Files Tree (M2)

```text
results_fv/
  antonym/
    fixed_trials_antonym_t10_s10_seed0/
      fv_function_vector.pt
      fv_top_heads.json
      fv_parity_report.json
```
