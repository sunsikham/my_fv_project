# Environments

## Active Environment Assumption

The active documentation layer assumes:
- code root: `/home/sunsik/my_fv_project`
- large artifact root: `/scratch/sunsik/my_fv_project`
- large model root: `/scratch/sunsik/models`

This second brain does not treat `/mnt/ebs` as an active environment.

## Common Runtime Modes

### CPU / Deterministic Parity

Used for:
- prompt parity
- slot parity
- FV parity
- injection parity when deterministic reference checks matter

Typical constraints:
- CPU
- float32
- `model.eval()`
- fixed seed

### GPU / Large-Model Runtime

Used for:
- condition q-wise large-model runs
- PT runs
- context-drift and unified PT
- some union-ref or D-extension flows

Typical constraints:
- H100 GPU jobs
- `bitsandbytes` 4bit paths
- CUDA module consistency
- scratch-first output placement

## Storage Behavior

For large runs, do not assume repo-local output is complete.

The practical pattern is:
- compute on scratch
- optionally sync selected artifacts back to home

## Current Operational Caveat

Some older runbooks still mention inactive environment roots.
Use `docs/brain/ARTIFACT_ROOTS.md` first, then consult the runbooks with that adjustment in mind.

