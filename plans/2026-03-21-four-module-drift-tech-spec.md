# Tech Spec Template

## Title

`Four-module inside-A drift analysis from top30 g_k outputs`

## Metadata

- Date: `2026-03-21`
- Slug: `four-module-drift`
- Source Plan: `plans/2026-03-21-four-module-drift-plan.md`
- Approval Status: `approved`

## Scope

- Build a Q1-focused module-signature analysis on top of the existing top30 `g_k` outputs.
- Assign the 30 inside-A features into exactly 4 explicit modules.
- Aggregate feature-level drift and contribution outputs into module-level summaries.
- Produce enough evidence to judge whether module-level reweighting is a stronger mechanistic description than individual-axis reweighting for Q1.

## Out Of Scope

- Re-running model inference, StepD, FV construction, or stepwise A-state extraction.
- Generalizing the module scheme to other Qs in the same pass.
- Claiming that 4 modules are uniquely correct in a global sense.
- Changing the core representation theory before the Q1 module outputs are validated.
- PT or behavioral execution.

## Implementation Design

`The implementation will reuse the existing Q1 top30 artifacts as inputs and add a new module-analysis layer. The key design choice is to define each g_k by a multi-view signature rather than by a single scalar. That signature will combine endpoint alignment, signed endpoint contribution over the 5 matched steps, early/late drift timing, and branch-specific co-movement profile. A constrained four-module assignment will then be computed from this signature table. The preferred implementation is a semi-structured clustering path: first derive interpretable module scores for B-push, D-push, shared-support, and suppression/disambiguation behavior; then assign each g_k to the dominant module under those scores, while also storing the underlying continuous scores so the hard assignment remains auditable. This avoids a purely aesthetic clustering and keeps the final 4-module result interpretable.`

### Module Signature Design

- Base unit: one row per `g_k` in the top30 `matched` basis for `Q1`
- Inputs combined into each signature:
  - endpoint alignment:
    - `align_B`
    - `align_D`
  - intended signed contribution trajectory:
    - `mean_contrib_BAB_B` across `A_demo_1 ... A_query`
    - `mean_contrib_DAD_D` across `A_demo_1 ... A_query`
  - drift magnitude trajectory:
    - `mean_abs_delta_c_BAB`
    - `mean_abs_delta_c_DAD`
  - timing summaries:
    - onset step for branch-specific intended contribution
    - peak step for branch-specific intended absolute contribution
    - late-minus-early contribution contrast
  - co-movement summaries:
    - flattened mean absolute correlation by branch
    - strongest positive and negative partners by branch

### Four Module Definitions

- The working module labels are:
  - `module_B_push`
  - `module_D_push`
  - `module_shared_scaffold`
  - `module_suppression_disambiguation`
- Operational reading:
  - `module_B_push`:
    - positive B alignment
    - positive BAB intended contribution
    - weak or non-dominant D role
  - `module_D_push`:
    - positive D alignment or anti-competitor pattern that consistently supports D
    - positive DAD intended contribution
  - `module_shared_scaffold`:
    - contributes in both branches with low branch selectivity
    - often moderate alignment and stable co-movement with multiple groups
  - `module_suppression_disambiguation`:
    - supports one branch partly by decreasing competitor-consistent features
    - often identified by negative alignment times negative drift producing positive intended contribution

### Assignment Rule

- Main path:
  - compute a standardized signature table for the 30 features
  - derive module score components from that table
  - assign each `g_k` to the module with the largest interpretable module score
- Required audit outputs:
  - continuous module scores for all four modules
  - hard assignment label
  - assignment confidence margin:
    - top score minus second score
- Important rule:
  - the final hard assignment must remain traceable back to continuous scores
  - do not hide ambiguous features

### Module-Level Summaries

- After assignment, compute module-level outputs by summing or averaging member features:
  - module signed intended contribution across steps
  - module absolute drift across steps
  - module branch selectivity
  - module onset and peak timing
  - module internal cohesion from within-module co-movement
  - module external coupling with other modules
- These outputs should answer:
  - which module turns on early
  - which module dominates the endpoint push
  - which module behaves like support or scaffold
  - which module behaves like suppression/disambiguation

### Preferred Files And Script Layout

- Create:
  - `scripts/compute_stepwise_gk_modules.py`
- Optional small helper reuse:
  - existing top30 artifacts remain read-only inputs
- Main entry behavior:
  - read top30 feature-step CSV
  - read top30 co-movement summary and flat correlation matrices
  - construct signature table
  - assign four modules
  - emit module-level summaries and assignment artifacts

### Expected Output Roots

- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30`

## Expected File Changes

- Create: `/home/sunsik/my_fv_project/scripts/compute_stepwise_gk_modules.py`
- Create: `/home/sunsik/my_fv_project/plans/2026-03-21-four-module-drift-tech-spec.md`
- Modify after successful execution if required: `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_methods.md`
- Modify after successful execution if required: `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_reweighting.md`
- Modify after successful execution if required: `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_theory.md`

## Ordered Implementation Steps

1. Implement `scripts/compute_stepwise_gk_modules.py` so it can build a feature signature table from the existing top30 artifacts.
2. Encode the four module-score definitions and hard assignment rule inside the script.
3. Add outputs for:
   - feature signature table
   - feature module score table
   - hard assignments
   - module-level stepwise contribution summaries
   - module cohesion and coupling summaries
4. Run the script locally against the current Q1 top30 artifacts.
5. Inspect whether the assignments are interpretable and whether ambiguous features are visible through low confidence margins.
6. If the module view is coherent, update the relevant `docs/brain/` interpretation/method files.

## Validation Plan

- Input checks:
  - confirm the required top30 feature-step and co-movement artifacts exist
  - confirm the run is using `Q1` and the `matched` top30 basis
- Signature checks:
  - confirm exactly 30 feature rows enter the signature table
  - confirm all required signature fields are finite or explicitly marked as missing when appropriate
- Assignment checks:
  - confirm exactly 4 module labels are used
  - confirm all 30 features receive a hard assignment
  - confirm module confidence margins are recorded
- Module summary checks:
  - confirm module-level stepwise contribution tables cover all 5 matched slots
  - confirm within-module cohesion values are computable for modules with at least 2 members
- Interpretation checks:
  - verify that at least one module is clearly B-consistent
  - verify that at least one module is clearly D-consistent
  - verify that at least one module is plausibly shared or support-like
  - verify that at least one module captures suppression/disambiguation behavior or explicitly report failure of that interpretation

## Expected Outputs

- `plans/2026-03-21-four-module-drift-tech-spec.md`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30/gk_module_signatures.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30/gk_module_scores.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30/gk_module_assignments.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30/module_stepwise_summary.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30/module_coupling_summary.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30/module_meta.json`
- A final execution report under `reports/`

## Brain Docs To Update

- `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_methods.md`
- `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_reweighting.md`
- `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_theory.md`

## Recommended Execution Strategy

- Launcher: `local`
- Compute Mode: `local`
- Reason: `This is a pure derived analysis stage over the existing top30 scratch outputs. No model loading or GPU inference is required, and the work is most safely done as a local CPU run.`

## User Execution Settings Required Before Run

- Launcher choice: `local` recommended
- Time limit: `not required for local`
- Day-based duration if relevant: `n/a`
- GPU options: `none`
- CPU count: `default local`
- Memory: `default local`
- Partition or queue: `n/a`
- Job name: `n/a`
- Log path: `use repo-local logs/ with dated filenames`
- Environment setup: `confirm use of /home/sunsik/.venvs/pt442/bin/python`
- Extra launcher flags: `none expected`

## Approval Note

Implementation and execution have not started. Waiting for Korean approval.
