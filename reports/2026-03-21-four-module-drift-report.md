# Final Report Template

## Status

- Status: `success`
- Date: `2026-03-21`

## Summary

`Implemented and ran a local four-module inside-A drift analysis for Q1 on top of the existing top30 feature outputs. The run produced feature-signature tables, module scores, hard assignments, module-level stepwise summaries, and module coupling summaries. The result supports a coherent Q1-focused module interpretation with 4 non-empty modules.`

## Source Documents

- Plan: `plans/2026-03-21-four-module-drift-plan.md`
- Tech Spec: `plans/2026-03-21-four-module-drift-tech-spec.md`

## Execution Settings

- Launcher: `local`
- Compute Mode: `local`
- Time limit: `n/a`
- Day-based duration if relevant: `n/a`
- GPU options: `none`
- CPU count: `default local`
- Memory: `default local`
- Partition or queue: `n/a`
- Job name: `n/a`
- Environment setup: `/home/sunsik/.venvs/pt442/bin/python`
- Extra launcher flags: `none`

## Commands Run

1. `mkdir -p logs reports && { echo '# Pre-execution baseline'; echo 'date=2026-03-21'; echo 'cwd=/home/sunsik/my_fv_project'; echo 'python=/home/sunsik/.venvs/pt442/bin/python'; echo; echo '## git status --short'; git status --short; } > logs/2026-03-21-four-module-drift-baseline.log`
2. `/home/sunsik/.venvs/pt442/bin/python -m py_compile scripts/compute_stepwise_gk_modules.py`
3. `/home/sunsik/.venvs/pt442/bin/python scripts/compute_stepwise_gk_modules.py --qid Q1 --basis_scope matched`
4. Validation reads against the generated scratch outputs using `/home/sunsik/.venvs/pt442/bin/python`

## Files Executed

- `/home/sunsik/my_fv_project/scripts/compute_stepwise_gk_modules.py`

## Files Changed

- Modified: `/home/sunsik/my_fv_project/plans/2026-03-21-four-module-drift-plan.md`
- Modified: `/home/sunsik/my_fv_project/plans/2026-03-21-four-module-drift-tech-spec.md`
- Created: `/home/sunsik/my_fv_project/scripts/compute_stepwise_gk_modules.py`
- Modified: `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_methods.md`
- Modified: `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_reweighting.md`
- Modified: `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_theory.md`
- Created: `/home/sunsik/my_fv_project/reports/2026-03-21-four-module-drift-report.md`

## Output Artifacts

- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30/gk_module_signatures.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30/gk_module_scores.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30/gk_module_assignments.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30/module_stepwise_summary.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30/module_coupling_summary.csv`
- `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30/module_meta.json`

## Log Paths

- `logs/2026-03-21-four-module-drift-baseline.log`
- `logs/2026-03-21-four-module-drift-run.log`

## Validation Results

- Pre-run syntax check passed for `scripts/compute_stepwise_gk_modules.py`.
- The run produced all planned module artifacts in the new scratch root.
- Exactly 30 top30 features entered the module analysis.
- Exactly 4 module labels were used, and all 30 features received a hard assignment.
- Module counts:
  - `module_B_push`: `9`
  - `module_D_push`: `4`
  - `module_shared_scaffold`: `7`
  - `module_suppression_disambiguation`: `10`
- The assignments are interpretable against the existing top30 feature reading:
  - `g2`, `g4`, `g6` fall into `module_B_push`
  - `g0`, `g11` fall into `module_D_push`
  - `g15` falls into `module_shared_scaffold`
  - `g1` falls into `module_suppression_disambiguation`
- Module-level stepwise summaries cover all 5 matched slots.
- Module coupling summaries exist for both `BAB` and `DAD` branches.

## Brain Updates

- Required: `yes`
- Updated files:
  - `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_methods.md`
  - `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_reweighting.md`
  - `/home/sunsik/my_fv_project/docs/brain/analysis/multi_feature_theory.md`
- Why: `The run establishes a stable Q1-focused module-analysis layer on top of the top30 feature analysis, which changes the current method and theory description for this branch.`

## Result Explanation

- The new script compresses the top30 inside-A feature view into a four-module interpretation rather than leaving the user with only an axis-by-axis reading.
- The result is Q1-focused and should be read as an interpretable decomposition, not as proof that the latent space has exactly four uniquely natural clusters.
- The four resulting modules behave plausibly:
  - `module_B_push` carries the strongest BAB-directed endpoint contribution and grows toward `A_query`
  - `module_D_push` carries the clearest direct D-directed contribution
  - `module_shared_scaffold` contributes to both branches and is more balanced
  - `module_suppression_disambiguation` contains many features whose contribution pattern is more consistent with branch sharpening or competitor suppression than with direct positive push
- The most useful upgrade over the previous feature-only story is that Q1 can now be summarized as:
  - context reallocates priority across a small number of co-moving inside-A modules
  rather than only
  - context changes many individual coefficients
- Remaining risk:
  - some assignments have small margins and should be treated as weaker than the clearest core members
  - the 4-module decomposition is currently established only for Q1

## Retry Record

- Retry attempted: `no`
- Reason: `n/a`

## Failure Details

- Failure point: `n/a`
- Error summary: `n/a`
