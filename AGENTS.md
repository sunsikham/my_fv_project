# AGENTS.md

## Result Storage Policy

- Default canonical root for large experiment outputs is `/scratch/sunsik/my_fv_project`.
- Use scratch-first storage for:
  - large GPU experiment runs
  - `results_fv` runtime trees
  - stepwise analysis artifacts
  - PT runs
  - large logs, `jsonl`, `npy`, `npz`, and large plot/report outputs
- Use repo-local storage under `/home/sunsik/my_fv_project` for:
  - code and docs
  - parity goldens and small deterministic reference artifacts
  - selected summaries or mirrors

## Canonical vs Mirror Rule

- If a run writes to scratch and also copies artifacts home, scratch remains canonical and home is a mirror.
- Do not assume repo-local artifacts are complete if the run was scratch-first.
- If a run writes directly to home with sync disabled, home is canonical for that run.

## Condition Q-Wise Sync Rule

- When using condition-qwise pipelines, set storage intent explicitly with:
  - `--out_root`
  - `--sync_home_root`
  - `--sync_mode`
  - `--home_artifact_profile`
- Treat `home_artifact_profile=core` as a partial mirror, not a full copy.

## PT Storage Rule

- Default canonical PT root is `/scratch/sunsik/my_fv_project/pt_analysis`.
- Repo-local `pt_analysis` should be treated as partial unless a run explicitly wrote there as its source root.

## Metadata Rule

- New experiment runs should record `canonical_root`.
- If artifacts are mirrored or synced, also record:
  - `sync_root`
  - `sync_mode`
  - `artifact_profile`
- Store these fields in run metadata or status JSON when practical.

## Reference

- See `docs/brain/ops/storage_and_sync.md` for the explanation behind these rules.
