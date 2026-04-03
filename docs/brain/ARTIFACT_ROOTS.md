# Artifact Roots

## Active Roots

### Code Root

- `/home/sunsik/my_fv_project`

Use for:
- code
- docs
- repo-local outputs

### Repo-Local Output Roots

- `/home/sunsik/my_fv_project/results`
- `/home/sunsik/my_fv_project/results_fv`
- `/home/sunsik/my_fv_project/pt_analysis`

Use for:
- parity goldens
- small or mirrored outputs
- repo-visible summaries

### Scratch Output Roots

- `/scratch/sunsik/my_fv_project/results_fv`
- `/scratch/sunsik/my_fv_project/pt_analysis`

Use for:
- large GPU runs
- multi-feature analysis artifacts
- PT runs and reports

### Model Root

- `/scratch/sunsik/models`

Use for:
- local large-model checkpoints

## Canonical Root Rule

Do not decide canonical storage from directory name alone.

Use this rule:
- the path under the run's actual `out_root` is the source-of-truth output root
- any later copy to another root is a mirror unless the run itself was launched there

## Practical Rules

### Parity

Canonical:
- `results/<dataset>/<fixed_trials_id>/...`
- `results_fv/antonym/<fixed_trials_id>/...`

### Relation Q-Wise

Canonical:
- whichever `out_root` the run used

Current repo view:
- repo-side relation q-wise folders mostly show retained analysis/aggregate products

### Condition Q-Wise

Important:
- this pipeline can run scratch-first and sync selected artifacts back home

If `sync_home_root` is used:
- scratch is usually the source root
- home may be a partial mirror

If `sync_mode` is `none` and `out_root` is home:
- home is the source root

### PT

Current practical rule:
- treat `/scratch/sunsik/my_fv_project/pt_analysis/<run_id>/` as canonical when a scratch run exists

Repo-local `pt_analysis/` should be treated as partial unless confirmed otherwise.

## Condition Q-Wise Sync Notes

The condition pipeline supports:
- `sync_home_root`
- `sync_mode`
- `home_artifact_profile`

`home_artifact_profile=core` means home may receive only selected files, not the full scratch tree.

## Explicit Exclusion

This second brain does not use `/mnt/ebs` as an active root.

