# Inside / Outside Analysis

## Scope

This branch isolates:
- change inside the A-space
- change outside the A-space
- how each component aligns with B or D endpoints

## Main Inputs

Usually built from:
- stepwise A-state arrays
- stepwise reweighting arrays
- endpoint anchor vectors

## Main Scripts

- `scripts/compute_stepwise_inside_outside_endpoint_alignment.py`
- `scripts/compute_stepwise_inside_outside_joint.py`
- `scripts/run_q1_inside_outside_intervention.py`
- `scripts/prepare_inside_outside_intervention_vectors.py`

## Main Artifact Roots

- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_inside_outside`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_inside_outside_joint`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_state_intervention`

## Typical Outputs

- endpoint-alignment summary CSVs
- packed NPZ decomposition arrays
- synthetic intervention state files
- markdown intervention reports

## Relationship To Multi-Feature Reweighting

This is not a separate unrelated project.

It is a focused sub-branch of the larger multi-feature reweighting program.

