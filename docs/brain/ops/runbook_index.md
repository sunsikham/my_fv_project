# Runbook Index

## Purpose

Runbooks are still useful, but they are no longer the top-level project map.

Use this order:
1. read `docs/brain/`
2. identify the relevant pipeline
3. open the matching runbook only if you need execution details

## Active Runbooks

### M1 Golden

File:
- `docs/RUNBOOK_M1_GOLDEN.md`

Use for:
- fixed-trial generation and parity-gated golden artifact production

Status:
- rewritten against the active root model
- use this when you need the concrete command sequence, not the conceptual map

### D Extension StepD + PCA

File:
- `docs/RUNBOOK_D_EXTENSION_STEPD_PCA.md`

Use for:
- adding `DDD` and `DADA` to an existing condition-qwise run
- producing 5-condition PCA outputs

Status:
- still operationally relevant

### bitsandbytes + CUDA for union_ref

File:
- `docs/RUNBOOK_BNB_CUDA_UNION_REF.md`

Use for:
- GPU environment troubleshooting for union-ref vector/PCA runs

Status:
- active-root-safe, but still an ops troubleshooting note rather than a general environment source-of-truth

## Archived Runbook-Like Material

The following kinds of documents were moved to archive:
- dated handoffs
- debug plans
- implementation rollout plans

See:
- `docs/archive/plans/`
- `docs/archive/handoffs/`
