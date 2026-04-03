# Glossary

## Core Objects

### `src`

Reference implementation used as the parity baseline.

### `fv`

Current working implementation and experiment library layer.

### Function Vector (`FV`)

A residual-space vector built from selected attention-head outputs and used for intervention.

## Parity Terms

### fixed trials

Frozen prompt/trial snapshots used to compare `src` and `fv` deterministically.

### golden artifacts

Canonical reference-side outputs used by parity checks, especially:
- mean head activations
- dummy labels
- indirect effect

### prompt parity

Check that reconstructed prompts and tokenized inputs match the fixed-trial reference.

### slot parity

Check that token labels, dummy labels, and slot alignment maps match between `src` and `fv`.

### FV parity

Check that selected top heads and the final function vector match exactly.

### injection parity

Check that the intervention path reproduces the same clean and intervened scores.

## Pipeline Terms

### StepD

Head-sweep stage that computes q- or condition-specific head scores and supporting trial metrics.

Typical outputs:
- `aie_scores.csv`
- `sampled_trials.json`
- `trial_metrics.jsonl`

### StepE

Top-k FV construction stage built on StepD outputs.

Typical outputs:
- `top_heads.json`
- `fv_global_resid.pt`
- `fv_by_layer.pt`

### Step6

FV injection evaluation stage, often run once per candidate injection layer.

### q-wise

Run each `q_id` as an independent experimental unit.

### condition q-wise

Run each `q_id` with several structured conditions such as `AAA`, `BBB`, `BABA`, and optionally D-family extensions.

## Condition Labels

### `AAA`

All demos and query remain in the A-family baseline condition.

### `BBB`

B-family condition aligned to the B relation source.

### `BABA`

Alternating A/B condition intended to induce mixed-context effects.

### `DDD`

D-family analogue of `BBB`.

### `DADA`

D-family analogue of `BABA`.

## Reference Sets

### `AAA_ref`

Reference head or vector set built from the `AAA` condition.

### `union_ref`

Reference set built by aggregating top-head rankings across several conditions.

## Analysis Terms

### multi-feature reweighting

Interpretation that context changes do not collapse representation to one axis, but instead rebalance several features while preserving substantial original structure.

### inside-A / outside-A

Decomposition of change into the component inside the A-basis and the component outside it.

### endpoint alignment

Measure of whether a state or change vector points toward the intended B or D endpoint.

### movement

Geometric summary of how condition means move relative to reference anchors.

### local tangent / curvature

Checks for whether non-one-axis structure survives when analysis is localized around the current state.

## PT Terms

### PT

Product Test evaluation family over relation edges and shot sweeps.

### baseline PT

Original PT family scored from baseline relation prompts.

### context drift PT

Mixed-context PT family using alternating demo regimes such as `ABABAB_B`.

### unified PT

Combined evaluation family that keeps several baseline/context-control groups in one run.

### lexical top-k trace

Filtered next-token top-k record intended for interpretable lexical drift analysis.

## Storage Terms

### canonical output

The source-of-truth artifact written to the run's actual output root.

### mirror

A copied artifact tree or partial sync of canonical outputs into another root.

### `_resume`

Per-run resume and raw accumulation state.

### `human_report`

HTML presentation layer intended for people, not for upstream computation.

