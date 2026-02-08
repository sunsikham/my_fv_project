# SPEC.md — Parity Contract for Function Vectors (src ↔ fv)

This document defines the **non-negotiable contract** for parity between the reference implementation in `src/` and the re-implementation in `fv/`.  
All `fv/` implementations, runners, and tests must conform to this spec.

---

## 0) Scope

### In scope (Parity certification)
- Fixed-trials parity for:
  - prompt construction and tokenization
  - slot/dummy-label alignment
  - mean head activations
  - indirect effect (AIE/CIE) tensors
  - Function Vector (FV) construction
  - FV injection (intervention) and minimal eval metrics

### Out of scope (for parity)
- Performance optimization
- Large-model (e.g., LLaMA 8B) sweep efficiency tuning
- Final experiment results on relation data (covered after parity)

---

## 1) Determinism / Runtime Contract (Mandatory for Parity)

Parity comparisons MUST be run with:

- **Device:** CPU
- **dtype:** float32
- **Model mode:** `model.eval()` (forced by the parity runner)
- **Seed:** fixed seed set once per run (stable across `src` vs `fv` calls in the same run)
- **Grad:** disabled (`torch.set_grad_enabled(False)`)
- **Tokenizer:** identical tokenizer for the same model name (no mixing versions/revisions)
- **No stochastic generation** during parity (use deterministic scoring/logprobs)

---

## 2) Acceptance Criteria

### 2.1 Parity mode (mandatory)
Under the determinism contract (CPU + fp32 + eval + fixed seed), require **exact equality**:
- `max_abs_diff == 0`
- `torch.equal(...) == True` where applicable

### 2.2 Non-parity mode (optional)
If running on GPU and/or reduced precision (fp16/bf16):
- allow a documented tolerance (example):
  - `allclose(atol=1e-6, rtol=0)` (or stricter if needed)
- **This is not valid for parity certification**; it is only for exploratory runs.

---

## 3) Terminology

- `dataset_name`: logical dataset identifier used by prompts (e.g., `antonym`, later `relation`).
- `fixed_trials_id`: identifier derived from the fixed-trials file name/stem  
  (e.g., `fixed_trials_antonym_t10_s10_seed0`).
- `run_dir` (canonical): the directory under which parity artifacts are stored and compared:
  - `results/<dataset_name>/<fixed_trials_id>/`        (src golden, normalized)
  - `results_fv/<dataset_name>/<fixed_trials_id>/`     (fv outputs)

---

## 4) Fixed Trials Regimes (Clean vs Corrupted)

For fixed-trials parity, the input regimes are **strict**:

- Mean activations MUST be computed from **`prompt_data_clean`**
- Indirect effect MUST be computed from **`prompt_data_corrupted`**

Mixing these regimes is a spec violation and will invalidate comparisons.

---

## 5) Artifact Layout and Naming

### 5.1 Canonical layout (required by the parity workflow)

Regardless of where individual `src` scripts write by default, the parity workflow MUST normalize
all golden artifacts into the canonical `run_dir`:

- **Golden (src) canonical location:**
  - `results/<dataset_name>/<fixed_trials_id>/`

- **fv outputs (comparison targets):**
  - `results_fv/<dataset_name>/<fixed_trials_id>/`

Parity runners treat `results/...` golden artifacts as **read-only**.

### 5.2 Script-default output behavior (informational; not canonical)

Different `src` scripts may choose different default output folders. In particular:

- `src/compute_indirect_effect.py`:
  - If only `--fixed_trials_path` is provided, it may default to a folder like:
    - `results/<fixed_trials_id>/`
  - If `--dataset_name <dataset_name>` is provided, it may default to:
    - `results/<dataset_name>/`
- `src/compute_average_activations.py`:
  - Output folder behavior can differ from `compute_indirect_effect.py`.
  - To keep parity bookkeeping consistent, outputs MUST be normalized into `run_dir`.

**Normalization rule (M1 responsibility):**
- After running `src` scripts, copy/move the expected golden files into:
  - `results/<dataset_name>/<fixed_trials_id>/`
- The parity runner then loads golden artifacts from the canonical `run_dir` only.

### 5.3 Required golden artifacts (src reference)

Within `results/<dataset_name>/<fixed_trials_id>/`, the required files are:

- `<dataset_name>_mean_head_activations_FIXED.pt`
- `<dataset_name>_dummy_labels.json`
- `<dataset_name>_indirect_effect.pt`

### 5.4 Optional debug artifact (NOT produced by default)

`<dataset_name>_indirect_effect_small.pt` is **not** produced by default.

If needed for debugging:
- run `compute_indirect_effect.py` with a small `--n_trials`,
- and write to a separate debug location or post-process via copy/rename into:
  - `results/<dataset_name>/<fixed_trials_id>/`

This file is optional and must not be assumed to exist.

---

## 6) Tensor Specifications

### 6.1 mean_activations
- File: `<dataset_name>_mean_head_activations_FIXED.pt`
- Shape: `(n_layers, n_heads, n_slots, head_dim)`
- Stored dtype: float32
- Regime: computed from `prompt_data_clean`

### 6.2 indirect_effect
- File: `<dataset_name>_indirect_effect.pt`
- Shape (common case): `(n_trials, n_layers, n_heads)` for `last_token_only=True`
- Stored dtype: float32
- Regime: computed from `prompt_data_corrupted`

### 6.3 top_heads
- Type: list of tuples `(layer_idx, head_idx, score)`
- Length: `n_top_heads`
- Selection rule:
  1) Compute `mean_indirect_effect` over trials → shape `(n_layers, n_heads)`
  2) Select with `torch.topk(mean_indirect_effect.view(-1), k=n_top_heads)`
  3) Unravel flat indices into `(layer_idx, head_idx)` in the same ordering as `src`
- **Score formatting rule:** when emitting/storing/comparing `top_heads`,
  `score` MUST be rounded to **4 decimal places** (to match `src` behavior).

### 6.4 function_vector
- Shape: `(1, resid_dim)`
- Runtime dtype: must match `model.dtype`
  - Note: parity mode uses float32, so runtime dtype must resolve to float32 in parity runs.
- Runtime device: CPU in parity runs

---

## 7) Function Vector Construction (fv must match src semantics)

Given:
- `mean_activations` and `indirect_effect`
- `model`, `model_config`
- `n_top_heads`, `token_class_idx`

FV construction MUST:

1) Compute mean indirect effect over trials:
   - `mean_indirect_effect` shape `(n_layers, n_heads)`

2) Select `top_heads` using flatten → topk → unravel (preserve ordering).

3) For each `(L, H)` in `top_heads`:
   - Build a residual-space vector `x` of length `resid_dim` initialized to zeros.
   - Insert the head vector into the H-th slice:
     - `x[H*head_dim:(H+1)*head_dim] = mean_activations[L, H, T]`
   - Token index `T` MUST match `src` (commonly `T = -1`).
   - Pass `x` through the layer’s **attention out projection** module (model-specific).
   - Sum the projected vectors across all selected heads.

4) Return:
   - `function_vector` with shape `(1, resid_dim)`
   - `top_heads` with scores rounded to 4 decimals for representation/comparison

Notes:
- `top_heads` scores are used for selection/recordkeeping; FV summation behavior must match `src`.
- dtype/device casting order must match `src` semantics in parity mode.

---

## 8) Injection Semantics (fv must match src semantics)

### 8.1 Hook point (mandatory for parity)
Injection parity MUST occur at the **layer block output hidden states**, matching `src`:
- use `TraceDict(model, layers=model_config["layer_hook_names"], edit_output=...)`

Using attention-internal hooks (e.g., attention out_proj pre-hooks) is **not acceptable** for parity.

### 8.2 Injection operation
- Add FV to hidden states at token index `idx`:
  - Default: `idx = -1`
  - NLL mode: `idx = -1 - target_len` (must match `src` logic)

---

## 9) Required Parity Checks (Minimum)

Parity runners/tests must check, in order:

1) Prompt parity (fixed trials)
2) Slot parity (dummy-label alignment)
3) FV parity:
   - `top_heads` exact match (including ordering and 4-decimal score rounding)
   - `function_vector` exact match (parity mode requires `max_abs_diff == 0`)
4) Injection parity:
   - clean target logprob match
   - intervention target logprob match
   - Δ logprob match

Failure outputs must indicate:
- the earliest divergence stage, and
- minimal diffs (e.g., `max_abs_diff`, mismatch indices, first differing head/layer).

---

## 10) Reference Config Normalization (Parity Runner Policy)

Some `src` paths may have model-name-specific config branching gaps.  
To keep parity checks stable, the parity runner is allowed to normalize **reference config only**.

Rules:
- Parity runner may canonicalize `src` reference config before calling `src` functions.
- For GPT-2 parity, `src` out projection path must be forced to the `c_proj` branch.
- This mapping exists **only** in verification/parity scripts.
- This policy must not change `fv` core implementation semantics.

---
