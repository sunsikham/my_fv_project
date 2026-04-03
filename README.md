# my_fv_project

Function-vector research repository with four active workstreams:
- `src` vs `fv` parity
- relation q-wise runtime
- condition q-wise runtime
- PT / context-drift evaluation

## Read First

Current project-knowledge hub:
- `docs/brain/INDEX.md`

Use `docs/brain/` for:
- current project structure
- pipeline map
- artifact storage rules
- analysis-track map

Use `docs/archive/` for:
- historical plans
- handoffs
- old environment-specific specs

## Active Roots

- code root: `/home/sunsik/my_fv_project`
- large artifact root: `/scratch/sunsik/my_fv_project`
- model root: `/scratch/sunsik/models`

Do not treat `/mnt/ebs` as an active root in current docs.

## Code Layout

- `src/`: reference implementation
- `fv/`: current working implementation and experiment library
- `scripts/`: runnable pipelines, checks, analysis, and reports
- `datasets/`: fixed trials and relation inputs
- `results/`: reference-side and legacy outputs
- `results_fv/`: repo-visible FV outputs
- `pt_analysis/`: repo-visible PT outputs

## Setup

Minimal local setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Project-specific GPU runs in this repo often use:
- `/home/sunsik/.venvs/pt442`

See:
- `docs/brain/ops/environments.md`
- `docs/brain/ops/runbook_index.md`

## Common Entry Points

### Parity

Main docs:
- `docs/brain/pipelines/parity.md`

Main scripts:
- `scripts/run_m1_golden_artifacts.py`
- `scripts/run_parity_suite.py`

### Relation Q-Wise

Main docs:
- `docs/brain/pipelines/relation_qwise.md`

Main script:
- `scripts/run_relation_qwise_pipeline.py`

### Condition Q-Wise

Main docs:
- `docs/brain/pipelines/condition_qwise.md`

Main script:
- `scripts/run_condition_qwise_pipeline.py`

### PT

Main docs:
- `docs/brain/pipelines/pt.md`

Main runners:
- `scripts/run_pt_llama70b.sh`
- `scripts/run_pt_context_drift_llama70b.sh`
- `scripts/run_pt_unified_drift_control_llama70b.sh`

## Sanity Checks

Useful quick checks:

```bash
python scripts/smoke_test.py --model gpt2
python scripts/smoke_resolve_spec_outproj.py --model gpt2 --model_spec gpt2 --layer 0 --device cpu --dtype fp32
python scripts/verify_prompt_parity.py --fixed_trials_path datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json --max_trials 5 --model_name_for_tokenizer gpt2
```

## Practical Rule

If you are unsure where to start:

1. open `docs/brain/INDEX.md`
2. read `PROJECT_MAP.md`
3. open the relevant pipeline doc
4. only then open runbooks or archived plans
