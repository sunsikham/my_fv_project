# PLAN.md — Function Vector Parity → Relation Experiments

## Project Goal
1) **Parity phase (verification):** Re-implement the Function Vector (FV) build/injection/eval logic in `fv/` so it is *semantically identical* to the reference implementation in `src/`, and prove this with fixed trials (parity checks).  
2) **Experiment phase:** After parity is complete, run the same pipeline (head sweep → FV build → FV injection → eval) on **relation data**, using **only `fv/`** (no `src` imports in the final workflow).  
3) **Model generalization:** Start with GPT-2 for fast iteration, then extend to other models (e.g., LLaMA) by isolating model-specific details behind a spec layer.

---

## Global Rules (Parity Contract)
For parity comparisons, the following conditions are **mandatory**:

- Device: **CPU**
- dtype: **float32**
- Model mode: **`model.eval()`** (forced in parity runners)
- RNG: fixed seed (stable across `src` vs `fv` calls inside the same run)
- Grad: `torch.set_grad_enabled(False)`
- Same tokenizer and prompt templates
- Fixed trials must be used consistently (see **Clean vs Corrupted regimes** below)

**Acceptance target**
- Preferred: **bitwise identical** outputs (`max_abs_diff == 0`)
- Fallback: `torch.allclose(atol=0, rtol=0)` only if the root cause is known and documented

---

## Clean vs Corrupted Regimes (Fixed Trials)
Fixed-trials parity requires **consistent data regimes**:

- **Mean activations** (golden): computed from **`prompt_data_clean`**
- **Indirect effect** (golden): computed from **`prompt_data_corrupted`**

If these are mixed, parity will fail.

---

## Artifact Conventions
- `results/` : **Golden artifacts produced by `src`** (read-only during parity runs)
- `results_fv/` : **Outputs produced by `fv`** (comparison targets)
- Parity runners must not overwrite `results/` golden files.
- Unless explicitly overridden, `src` may save fixed-trials outputs under:
  - `results/<fixed_trials_id>/` (e.g., `results/fixed_trials_antonym_t10_s10_seed0/`)
- If outputs must be forced under `results/<dataset_name>/`, the CLI **must** pass `--dataset_name <name>` explicitly (documented in RUNBOOK).

---

# Milestones

## M0. Freeze runtime + determinism
**Purpose:** Ensure parity comparisons are reproducible and do not drift.

**Tasks**
- Establish a “parity runtime profile”: GPT-2 + CPU + fp32 + eval + fixed seed.
- Run fixed-trials sanity checks:
  - `scripts/verify_prompt_parity.py`
  - `scripts/verify_slot_parity_against_src.py`

**DoD**
- Both verify scripts PASS on fixed trials.
- Re-running the same inputs produces identical outputs (prompt/tokenization/slot mapping).

---

## M1. Produce and freeze `src` golden artifacts
**Purpose:** Create stable reference artifacts so `fv/` can be compared against a single ground truth.

**Tasks**
- Using fixed trials, generate and save from `src` (and **document the exact output folder**):
  - `results/<fixed_trials_id>/<dataset>_mean_head_activations_FIXED.pt`
    - computed from **`prompt_data_clean`**
  - `results/<fixed_trials_id>/<dataset>_dummy_labels.json`
    - dummy-label/slot alignment evidence
  - `results/<fixed_trials_id>/<dataset>_indirect_effect.pt`
    - computed from **`prompt_data_corrupted`**
  - (Optional) a small-debug version (e.g., `*_indirect_effect_small.pt`) with very small `n_trials`
- Record exact commands and the expected folder naming behavior in `docs/RUNBOOK.md`.
  - Note: passing only `--fixed_trials_path` may default outputs to `results/<fixed_trials_id>/`.
  - If a stable `results/<dataset_name>/` layout is required, explicitly pass `--dataset_name <name>`.

**DoD**
- Golden files exist and paths are reproducible.
- Re-running under the parity contract produces the same shapes and consistent contents.
- Tensor specs are documented and match reality.

---

## M2. Re-implement FV construction in `fv/` + FV parity PASS (core)
**Purpose:** Make `fv.compute_function_vector` **exactly equivalent** to `src/utils/extract_utils.py::compute_function_vector`.

**Tasks**
- Implement (or extend existing) `fv.compute_function_vector(...)` with *identical semantics*:
  - `top_heads` selection: flatten → `topk` → unravel (same ordering)
  - mean-activation token index usage: same rule as `src` (e.g., `T = -1`)
  - head-slot insertion → residual vector → **pass through attention out_proj** → sum across heads
  - match dtype/device casting order
  - **match `top_heads` score formatting**: `src` rounds scores to **4 decimal places**; `fv` must apply the same rounding when producing/comparing `top_heads`
- Add optional “intermediate diff” logging in the parity runner:
  - `mean_indirect_effect` (L,H)
  - `topk` flat indices
  - per-head projection input `x` and projection output `d_out`
  - partial sums (first 1–2 heads)

**DoD**
- `top_heads` matches `src` exactly (including order and 4-decimal rounding of scores).
- `function_vector` matches `src`:
  - Preferred: `max_abs_diff == 0`
  - Fallback: `allclose(atol=0, rtol=0)` with documented cause

---

## M3. Re-implement FV injection in `fv/` + injection parity PASS
**Purpose:** Make `fv` interventions behave exactly like `src/utils/intervention_utils.py`.

**Key requirement**
- Injection parity must use the same intervention point as `src`:
  - **add FV to the *layer block output hidden states*** via `TraceDict(..., edit_output=...)` on `layer_hook_names`.
  - Using an **attention out_proj pre-hook** (or any different hook point) is not acceptable for parity.

**Tasks**
- Implement `fv` equivalents:
  - `add_function_vector(edit_layer, fv_vector, idx=...)`
  - `function_vector_intervention(...)` using the same hooking approach (e.g., `TraceDict` on `layer_hook_names` with `edit_output`)
- Compare minimal metrics first:
  - For the same prompt/target:
    - clean target logprob
    - intervention target logprob
    - delta (Δ) logprob
  - (Optional) full logits equality if feasible

**DoD**
- Clean outputs match `src`.
- Intervention outputs match `src`.
- Specifically: target-token Δ logprob matches exactly.

**Notes**
- Index policy must match `src`:
  - default `idx = -1`
  - NLL mode: `idx = -1 - target_len`

---

## M4. End-to-end parity suite + regression harness
**Purpose:** Ensure future refactors do not silently break parity.

**Tasks**
- Add a single-command parity suite, e.g.:
  - `python scripts/run_parity_suite.py --dataset antonym --model gpt2 --fixed_trials ...`
- Minimum coverage:
  - prompt parity
  - slot parity
  - FV parity (M2)
  - injection parity (M3; small sample)
- Make failure diagnostics explicit:
  - stage (M0/M2/M3), `max_abs_diff`, first mismatch index (layer/head/token)

**DoD**
- One command yields a clear PASS/FAIL.
- On FAIL, logs identify the earliest divergence point.

---

## M5. Model spec separation/extension (GPT-2 → LLaMA, etc.)
**Purpose:** Keep core logic model-agnostic; support new models by swapping specs only.

**Principle**
- Core FV logic remains unchanged.
- Model-specific paths live exclusively in a `ModelSpec` layer.

**Tasks**
- Extend `fv/model_spec.py` to mirror `src` model branching 1:1:
  - block module path (layer blocks)
  - attention out_proj path (`c_proj` / `out_proj` / `o_proj` / `dense`)
  - hook name generation (`layer_hook_names`, `attn_hook_names`)
  - `resid_dim` / `n_heads` extraction
  - tuple output handling (if needed)
- Smoke test on one additional model:
  - forward + hooks
  - mean activations collection
  - FV build
  - injection

**DoD**
- One non-GPT2 model runs end-to-end without errors.
- Performance improvements are not required at this stage (compatibility only).

---

## M6. Run experiments on relation data (final)
**Purpose:** Execute the pipeline on relation data using `fv/` only.

**Tasks**
- Implement relation data loader + prompt/target extraction as a task layer.
- Run head sweep → FV build → injection eval on relation data.
- Standardize outputs using `fv/io.py` or a consistent `results/` layout.

**DoD**
- End-to-end run on relation data succeeds.
- Outputs (head sweep, FV artifacts, eval metrics) are reproducible and saved consistently.

---

## Execution Order
**M0 → M1 → M2 → M3 → M4** (Parity complete)  
Then **M5 (model expansion) → M6 (relation experiments)**

---

## Non-goals (for now)
- Optimizing large-model (e.g., LLaMA 8B) sweep speed during the parity phase
- Claiming performance gains before parity is proven
- Shipping a final workflow that imports `src` (allowed only as a reference during parity)
