# Storage and Sync

## Why This Matters

This project does not use a single always-canonical artifact root.

Large runs often write to scratch first and only sync selected artifacts back to the repo-visible home tree.

## Active Roots

Primary roots:
- `/home/sunsik/my_fv_project`
- `/scratch/sunsik/my_fv_project`

## Core Rule

The real source-of-truth root for a run is the root passed as that run's actual output root.

Implication:
- if a run wrote to scratch and later copied files home, scratch is canonical
- if a run wrote directly to home with sync disabled, home is canonical

## Condition Q-Wise Sync Model

The condition pipeline supports:
- `--out_root`
- `--sync_home_root`
- `--sync_mode`
- `--home_artifact_profile`

### `sync_mode`

Observed modes include:
- `none`
- `per_q`
- `end`

### `home_artifact_profile`

Observed profiles include:
- `core`
- `full`

`core` means the home copy may include only selected artifacts such as:
- status
- selected logs
- trial payloads
- StepD summaries
- top-heads
- vectors
- PCA outputs
- FV outputs

It may not be a full mirror of everything written under scratch.

`full` means the entire q directory can be copied.

## Practical Reading Rule

When inspecting a run:

1. locate the run root
2. check whether the run used sync
3. determine whether the visible tree is a source tree or a mirror
4. only then decide which artifacts are canonical

## PT Storage Pattern

PT runs are currently scratch-first in practice.

A typical PT canonical run contains:
- main sweep CSV
- bootstrap CSV
- optional lexical top-k traces
- `_resume/`
- `human_report/`
- `run.log`

Repo-local PT should be treated as partial unless explicitly confirmed otherwise.

## Metadata Contract

The preferred storage metadata fields are:
- `canonical_root`
- `sync_root`
- `sync_mode`
- `artifact_profile`

Current status:
- main pipeline runners and PT run wrappers should record these fields
- some lower-level stage producers may still omit them

When present, these fields should be treated as the authoritative storage contract for that run.
