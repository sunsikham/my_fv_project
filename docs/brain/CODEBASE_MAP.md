# Codebase Map

## Ownership By Directory

### `src/`

Role:
- reference implementation
- source of truth for parity behavior

Typical contents:
- prompt construction
- fixed-trial handling
- function-vector synthesis
- intervention logic
- evaluation helpers

Important files:
- `src/utils/extract_utils.py`
- `src/utils/intervention_utils.py`
- `src/utils/prompt_utils.py`
- `src/utils/model_utils.py`
- `src/utils/eval_utils.py`

### `fv/`

Role:
- working library for the current project
- re-implementation plus extension layer

Typical contents:
- model spec and adapter resolution
- unified HF loading
- token/slot alignment
- fixed-trial adapters
- relation and condition trial builders
- vector extraction and artifact IO

Important files:
- `fv/model_spec.py`
- `fv/adapters.py`
- `fv/hf_loader.py`
- `fv/fixed_trials_adapter.py`
- `fv/mean_activations.py`
- `fv/relation_trials.py`
- `fv/condition_trials.py`
- `fv/head_vector_extract.py`
- `fv/intervene.py`
- `fv/io.py`

### `scripts/`

Role:
- runnable entrypoints
- orchestration, verification, analysis, plotting, and reporting

Main subgroups:
- `run_*`: pipeline runners
- `verify_*`: parity checks
- `compute_*`: analysis/statistics
- `plot_*`: figures
- `build_*`: human-readable reports
- `smoke_*`: fast sanity checks

### `datasets/`

Role:
- canonical inputs

Important subfolders:
- `datasets/fixed_trials/`: parity inputs
- `datasets/relation/`: relation and PT inputs
- `datasets/processed/`: canonical antonym pairs

### `docs/`

Role:
- mixed collection of specs, plans, runbooks, handoffs, and analysis notes

Important distinction:
- not every document under `docs/` is current SSOT
- `docs/brain/` is the new stable knowledge layer

### `results/`

Role:
- reference-side or legacy result artifacts
- especially important for parity golden artifacts

### `results_fv/`

Role:
- current repo-side FV outputs
- parity outputs, q-wise outputs, and some derived comparisons

### `pt_analysis/`

Role:
- repo-local PT artifacts

Current state:
- partial compared with scratch
- do not assume repo-local PT is complete when scratch PT exists

### `tests/`

Role:
- small pytest layer

Current state:
- thin coverage
- many practical checks still live in `verify_*` and `smoke_*` scripts

## Mental Model

The stable layering is:

1. `src` = reference semantics
2. `fv` = current reusable library
3. `scripts` = executable workflows
4. `results*` = stored artifacts
5. `docs/brain` = explanation layer over all of the above

