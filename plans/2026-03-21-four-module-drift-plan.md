# Plan Template

## Title

`Four-module inside-A drift analysis from top30 g_k outputs`

## Metadata

- Date: `2026-03-21`
- Slug: `four-module-drift`
- Approval Status: `approved`

## Objective

`Build a module-level analysis on top of the existing Q1 top30 g_k outputs so that the current "multi-feature reweighting" story can be sharpened into a four-module interpretation, where the main unit of mechanism is a co-moving inside-A module rather than an individual PCA axis.`

## Current Context

`The previous local top30 run for Q1 already produced canonical scratch outputs for reweighting, per-feature-step endpoint contributions, and g_k co-movement. Those artifacts show that the top5 narrative largely survives, but top30 exposes additional supporting features and nontrivial co-movement structure. The current interpretive gap is that we can now say which individual g_k axes move together, but we do not yet have a stable module-level decomposition that groups them into a small number of mechanistically readable units. The user wants to move from individual g_k axes toward a four-module view, where the modules are the candidate computation units.`

## Assumptions

- The first pass will stay focused on `Q1`, because that is the only fully prepared stepwise case with top30 outputs and prior interpretation context.
- The four-module target is a deliberate modeling choice for interpretability, not a claim that the latent space truly contains exactly four mathematically natural clusters.
- Module discovery should be driven primarily by coefficient co-movement and endpoint-contribution behavior, not by raw basis-vector geometry alone.
- A useful first four-module framing is likely to include roles close to:
  - `B-consistent push`
  - `D-consistent push`
  - `shared scaffold`
  - `competitor suppression / disambiguation`
- The analysis can be executed in `local` mode because it reuses existing top30 scratch outputs and adds only derived clustering or summary stages.

## Inputs And Dependencies

- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting_top30/stepwise_reweighting_AAA_ref.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting_top30/stepwise_reweighting_arrays_AAA_ref.npz`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_endpoint_alignment_top30/stepwise_endpoint_alignment_feature_steps.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/stepwise_gk_co_movement_feature_summary.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/stepwise_gk_co_movement_flat_corr_BAB.npy`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/stepwise_gk_co_movement_flat_corr_DAD.npy`
- `/home/sunsik/my_fv_project/reports/2026-03-20-fullspace-gk-analysis-report.md`
- Existing analysis scripts and any new module-analysis script added in the follow-up implementation

## Proposed Steps

1. Fix the operational definition of a `module` for this project so it is based on co-moving coefficient structure plus consistent endpoint contribution behavior, rather than on arbitrary basis labels.
2. Define a module-signature table for each `g_k` using the current top30 outputs. The signature should combine at least:
   - endpoint alignment
   - branch-specific signed contribution profile across steps
   - branch-specific co-movement profile
   - early vs late drift timing
3. Choose a concrete four-module assignment rule. The preferred options to compare are:
   - constrained clustering into 4 groups from the signature table
   - rule-based grouping from endpoint role plus suppression/sharedness markers
4. Build a module-level summary layer that aggregates per-g_k outputs into per-module outputs:
   - module activation across steps
   - module signed endpoint contribution
   - module co-movement relations
   - module role interpretation
5. Evaluate whether the resulting four modules are stable and interpretable enough to support a stronger claim:
   - `context reweights latent inside-A modules`
   instead of only
   - `context reweights multiple features`
6. If the module decomposition is convincing, update the stable analysis narrative in `docs/brain/`.

## Risks And Blockers

- Forcing exactly four modules may produce visually neat groups that are not actually stable under small analysis choices.
- Some `g_k` axes may be mixed-role or weakly assigned, making hard clustering brittle.
- The module names themselves can become over-interpretive if they are assigned before the quantitative structure is clear.
- Co-movement alone may not separate `shared scaffold` from `competitor suppression`, so the signature design has to include signed endpoint behavior and timing as well.
- Q1 may support a four-module story that does not generalize immediately to other Qs; this first pass should be framed as a Q1-focused mechanistic interpretation.

## Recommended Compute Mode

- Mode: `local`
- Why: `This work is a derived module-analysis layer over the already generated top30 artifacts. It should be CPU-friendly and does not require model loading or GPU inference.`

## Expected Outputs

- `plans/2026-03-21-four-module-drift-tech-spec.md`
- A module-signature summary artifact under scratch
- A four-module assignment artifact under scratch
- Module-level contribution and drift summaries under scratch
- Possibly one compact figure or table showing the four modules and their roles
- A final execution report under `reports/`

## Success Criteria

- The plan leads to an implementation that assigns the top30 Q1 inside-A features into exactly four explicit modules.
- Each module has a quantitatively defined signature, not only a hand-written narrative label.
- The resulting four modules explain stepwise endpoint drift more compactly and more stably than a feature-by-feature list alone.
- The module outputs make it possible to distinguish at least:
  - branch-specific push
  - shared support
  - suppression or disambiguation behavior
- The final interpretation can state whether module-level reweighting is a better mechanistic description than individual-axis reweighting for Q1.

## Brain Impact

- Brain impact: `update required`
- Why: `If the four-module decomposition works, it upgrades the stable interpretation of the multi-feature branch from feature-level reweighting to module-level priority reallocation, which belongs in the current brain-layer analysis docs.`

## Approval Note

Substantial execution has not started. Waiting for Korean approval.
