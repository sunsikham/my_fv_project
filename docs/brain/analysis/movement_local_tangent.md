# Movement and Local Tangent Analysis

## Scope

This branch summarizes how condition means move geometrically and whether residual structure survives local tangent or curvature checks.

## Main Scripts

- `scripts/compute_condition_movement_qwise.py`
- `scripts/plot_endpoint_movement_figure.py`

Related outputs also connect to local-tangent and curvature summaries.

## Main Artifact Roots

- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_multi_root`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_local_tangent_curvature`

## Typical Outputs

- `movement_qwise.csv`
- `movement_condition_means.npz`
- movement manifests
- endpoint figures
- local tangent summary CSVs
- local tangent NPZ arrays

## Interpretation Role

These artifacts are derived geometric summaries.

They help explain:
- endpoint movement
- off-axis residual structure
- whether non-one-axis structure survives local reparameterization

