# RUNBOOK: D Extension (DDD/DADA StepD + 5-Condition PCA)

## 1) Goal
- Keep existing `AAA/BBB/BABA` outputs as-is.
- Add only `DDD` and `DADA` condition processing.
- Merge for PCA visualization with 5 conditions:
  - `AAA,BBB,BABA,DDD,DADA`

## 2) Condition Mapping
- `DDD`:
  - Same logic as existing `BBB`
  - Data source changed from `relationB_ex` to `relationD_ex`
- `DADA`:
  - Same logic as existing `BABA`
  - B-side data changed from `relationB_ex` to `relationD_ex`

## 3) Scope
- In scope:
  - D-only trial generation (`DDD`, `DADA`)
  - D-only StepD (`aie_scores_DDD.csv`, `aie_scores_DADA.csv`)
  - D-only vector extraction for `AAA_ref`
  - 5-condition common PCA output
- Out of scope:
  - Re-running completed `AAA/BBB/BABA` stages
  - Injection extension for D conditions

## 4) Required Inputs
- Base relation run root (existing):
  - actual run root under either:
    - `/home/sunsik/my_fv_project/results_fv/relation_condition_qwise/<relation_name>/`
    - `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/<relation_name>/`
- Relation files:
  - `datasets/relation/relationA_ex.csv`
  - `datasets/relation/relationD_ex.csv`
- Per-Q existing artifacts (must already exist):
  - `_trials/condition_BBB.json`
  - `_trials/condition_BABA.json`
  - `_vectors/trial_vectors_AAA_ref_AAA.npy`
  - `_vectors/trial_vectors_AAA_ref_BBB.npy`
  - `_vectors/trial_vectors_AAA_ref_BABA.npy`

## 5) Hard Preconditions
- `relationD_ex.csv` must be non-empty and valid CSV (`id,ex_A,ex_B`).
- Q list is always user-provided at runtime:
  - Example: `Q11,Q13,Q16,Q17,Q18`

If `relationD_ex.csv` is empty, stop immediately and print a fix message.

## 6) Non-Destructive Output Policy
- Keep existing artifacts unchanged.
- Add new D artifacts under each Q:
  - `_trials/condition_DDD.json`
  - `_trials/condition_DADA.json`
  - `_stepd/aie_scores_DDD.csv`
  - `_stepd/aie_scores_DADA.csv`
  - `_vectors/trial_vectors_AAA_ref_DDD.npy`
  - `_vectors/trial_vectors_AAA_ref_DADA.npy`
- Save 5-condition PCA to a separate folder:
  - `_pca_common/AAA_ref_with_D/`

## 7) Execution Design (Per Q)
1. Load existing `condition_BBB.json` and `condition_BABA.json`.
2. Build D trials with same `trial_id` alignment:
   - `DDD` from BBB template + relation D pairs.
   - `DADA` from BABA template + relation D pairs.
3. Run StepD for `DDD` and `DADA` only (fixed trials mode).
4. Build `AAA_ref` vectors for `DDD` and `DADA`.
5. Run common PCA with:
   - `--conditions AAA,BBB,BABA,DDD,DADA`
   - output to `_pca_common/AAA_ref_with_D`.

## 8) Planned CLI Shape
Target script:
- `scripts/run_d_extension_stepd_pca.py`

Expected command pattern:

```bash
ROOT=/home/sunsik/my_fv_project
PY=${PY:-/home/sunsik/.venvs/pt442/bin/python}
BASE_ROOT=<set-existing-condition-run-root>

"$PY" "$ROOT/scripts/run_d_extension_stepd_pca.py" \
  --base_root "$BASE_ROOT" \
  --relation_a_csv "$ROOT/datasets/relation/relationA_ex.csv" \
  --relation_d_csv "$ROOT/datasets/relation/relationD_ex.csv" \
  --q_list Q11,Q13,Q16,Q17,Q18 \
  --model /scratch/sunsik/models/Llama-3.1-70B \
  --model_spec llama3 \
  --device cuda \
  --dtype bf16 \
  --quant 4bit \
  --resume 1 \
  --stop_on_error 0
```

Use the real source run root for `BASE_ROOT`.
If the canonical run was scratch-first, point `BASE_ROOT` at scratch, not at a partial home mirror.

## 9) Validation Checklist
- For each target Q:
  - `condition_DDD.json` and `condition_DADA.json` exist.
  - `aie_scores_DDD.csv` and `aie_scores_DADA.csv` exist.
  - `trial_vectors_AAA_ref_DDD.npy` and `trial_vectors_AAA_ref_DADA.npy` exist.
  - `_pca_common/AAA_ref_with_D/pca_points.csv` exists.
  - `_pca_common/AAA_ref_with_D/pca_centroids.csv` exists.

## 10) Failure Cases
- Empty `relationD_ex.csv`:
  - hard fail at precheck.
- Missing prior BBB/BABA trial artifacts for a Q:
  - mark Q failed and continue when `--stop_on_error=0`.
- Active lock for the same Q:
  - skip that Q by default to avoid collision with running jobs.
