# TECH_SPEC_RELATION_QWISE

## 0. Scope

This spec defines **Option 1 (q-wise pipeline)** for relation experiments.

Goal per `q_id`:
1. Run StepD independently
2. Build q-specific attention head heatmap
3. Select top-k heads
4. Build FV
5. Run Step6 FV injection in zero-shot mode

Later aggregation across q_ids is out of scope.

---

## 1. Conceptual Flow (Per q_id)

`relationA_ex.csv`
-> StepD (CIE/AIE)
-> Heatmap
-> Top-k head selection
-> FV build
-> Step6 zero-shot injection across all layers

Each `q_id` is a fully independent unit.

---

## 2. Fixed Global Settings

```bash
MODEL=meta-llama/Llama-3.1-8B
MODEL_SPEC=llama3
DEVICE=<env>
DTYPE=<env>
QUANT=<env>
SEED=0

REL=relationA_ex
RELATION_CSV=datasets/relation/relationA_ex.csv

N_TRIALS_PER_Q=25
N_DEMOS=9

SCORE_KEY=mean_delta_p
TOPK=20
ALPHA=1.0
N_EVAL=50
```

### Hard Constraints
- `TOPK` is fixed to `20`.
- Step6 must run all candidate injection layers using an external layer loop.
- Input relation file is fixed to `datasets/relation/relationA_ex.csv`.
- If a `q_id` has fewer than `N_DEMOS + 1 = 10` rows, that `q_id` is skipped.
- Per `q_id`, exactly `N_TRIALS_PER_Q=25` trials are sampled/executed when eligible.
- For each trial, query is sampled randomly from that `q_id`.
- Demo rows are sampled from rows excluding the selected query (no query/demo overlap).

---

## 3. Directory Layout (Per q_id)

For relation `REL=relationA_ex` and q_id `Q`:

```text
results_fv/relation_qwise/REL/Q/
  artifacts/
    sampled_trials.json
    sampled_trials_zeroshot.json
    cie_scores.csv
    aie_scores.csv
    trial_metrics.jsonl

    stepD_mean_acts/
      qid_Q_clean_mean.pt
      global_clean_mean.pt

    top_heads.json
    fv_by_layer.pt
    fv_global_resid.pt
    fv_global_resid_meta.json
    stepE_eval.json

    step6/
      layer_0/
        step6_results_*.json
        eval_summary.json
        eval_trials.jsonl
        eval_meta.json
      layer_1/
        ...
      ...
      layer_{L-1}/
        step6_results_*.json
        eval_summary.json
        eval_trials.jsonl
        eval_meta.json
      step6_all_layers_summary.json

    qid_status.json

  logs/
    stepD_aie.log
```

`qid_status.json` is required for checkpoint/resume safety.

---

## 4. QID Processing Policy (Skip + Persistence)

For each `q_id`:
1. Validate demo availability for `N_DEMOS=9`.
2. If insufficient demos, write skip status and continue to next `q_id`.
3. If eligible, run StepD with `N_TRIALS_PER_Q=25`.
4. Run Heatmap and StepE.
5. Build zero-shot trial snapshot.
6. Run Step6 across all injection layers.
7. Persist final q-level status before moving to next `q_id`.

### Contract
- `sampled_trials.json` is frozen after StepD and reused downstream.
- Each stage writes outputs to disk before next stage starts.
- Before moving to the next `q_id`, `qid_status.json` must be updated to `completed`.
- `expected_layers` must be computed once per q_id from model depth:
  - preferred: `L = len(model blocks from model_spec adapter)`
  - fallback: `L = len(top-level transformer/hf blocks)`
  - `expected_layers = [0, 1, ..., L-1]`
- `completed` requires all of the following:
  - StepD outputs: `sampled_trials.json`, `aie_scores.csv`, `trial_metrics.jsonl`
  - StepE outputs: `top_heads.json`, `fv_global_resid.pt`, `fv_by_layer.pt`
  - Step6 outputs: for every `layer in expected_layers`, `step6/layer_{layer}/step6_results_*.json` and `step6/layer_{layer}/eval_summary.json` must exist
  - q-level aggregate: `step6_all_layers_summary.json`
- On restart, completed `q_id`s are not re-run; only pending/failed `q_id`s continue.

---

## 5. StepD (q-wise CIE/AIE)

One `q_id` per run:

```bash
Q=Q3
RUN_BASE=results_fv/relation_qwise/${REL}/${Q}

python scripts/run_stepD_aie_head_sweep.py \
  --model ${MODEL} \
  --model_spec ${MODEL_SPEC} \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --quant ${QUANT} \
  --layers all \
  --heads all \
  --relation_csv_path ${RELATION_CSV} \
  --relation_q_list ${Q} \
  --relation_n_trials_per_q ${N_TRIALS_PER_Q} \
  --relation_n_demos ${N_DEMOS} \
  --seed ${SEED} \
  --score_key ${SCORE_KEY} \
  --out_base_dir ${RUN_BASE}
```

If StepD cannot satisfy demo/trial constraints, mark `q_id` skipped and continue.

Pre-check rule:
- A q_id is eligible only if CSV count for that q_id is `>= 10` (1 query + 9 demos).

Trial generation rule (q-random):
1. For each trial, sample one query pair for that `q_id`.
2. Sample 9 demo pairs from rows excluding the sampled query.
3. Build clean/corrupted prompts from that trial-specific query and demos.
4. Repeat to generate 25 trials per `q_id`.
5. Save trial metadata in `sampled_trials.json` (`query_source_index`, `demo_source_indices`).

---

## 6. Heatmap (Per q_id)

```bash
python scripts/plot_stepD_aie_heatmap.py \
  --in_dir ${RUN_BASE}/artifacts \
  --metric ${SCORE_KEY} \
  --topk ${TOPK} \
  --out_name aie_heatmap_${Q}.png
```

---

## 7. StepE (FV Construction, Top-k=20)

```bash
python scripts/run_stepE_topk_fv_and_eval.py \
  --stepd_artifacts_dir ${RUN_BASE}/artifacts \
  --sampled_trials_path ${RUN_BASE}/artifacts/sampled_trials.json \
  --k ${TOPK} \
  --score_key ${SCORE_KEY} \
  --model ${MODEL} \
  --model_spec ${MODEL_SPEC} \
  --device ${DEVICE} \
  --dtype ${DTYPE} \
  --quant ${QUANT} \
  --alpha ${ALPHA} \
  --out_dir ${RUN_BASE}/artifacts
```

Expected outputs:
- `top_heads.json`
- `fv_global_resid.pt`
- `fv_by_layer.pt`

---

## 8. Zero-shot Trial Snapshot (Required)

Create `sampled_trials_zeroshot.json` from `sampled_trials.json`:
- Remove demos
- Keep `target_id`
- Prompt format: `Q: <query>\nA: `
- Use each trial's original query from StepD snapshot (no additional query resampling at Step6 input build time).

---

## 9. Step6 (Zero-shot FV Injection Across All Layers)

`run_step6_fv_injection_eval.py` accepts only integer `--edit_layer`.
So all-layer evaluation must be done by looping layers externally:
1. Get model layer count `L` (same rule as `expected_layers` in Contract).
2. For each `layer in [0, L-1]`, run Step6 once.
3. Save each layer output into `step6/layer_{layer}/` to avoid overwrite.
4. Build `step6_all_layers_summary.json` after all layers finish.

Per-layer command template:

```bash
python scripts/run_step6_fv_injection_eval.py \
  --model ${MODEL} \
  --model_spec ${MODEL_SPEC} \
  --fv_global_path ${RUN_BASE}/artifacts/fv_global_resid.pt \
  --sampled_trials_path ${RUN_BASE}/artifacts/sampled_trials_zeroshot.json \
  --edit_layer ${LAYER} \
  --alpha ${ALPHA} \
  --n_eval ${N_EVAL} \
  --out_dir ${RUN_BASE}/artifacts/step6/layer_${LAYER}
```

Required outputs:
- per-layer results in `step6/layer_*/`
- q-level aggregate file: `step6_all_layers_summary.json`

---

## 10. Failure Safety

To avoid losing progress on interruption:
- Persist every stage output immediately.
- Persist q-level status transitions (`pending -> running -> completed/failed/skipped`).
- Never advance to next `q_id` before current `q_id` outputs and status are flushed.
- Resume mode must detect and continue from saved state.
