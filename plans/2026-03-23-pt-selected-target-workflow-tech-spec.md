# Tech Spec Template

## Title

`Unified PT selected-target review and rescoring implementation`

## Metadata

- Date: `2026-03-23`
- Slug: `pt-selected-target-workflow`
- Source Plan: `plans/2026-03-23-pt-selected-target-workflow-plan.md`
- Approval Status: `pending`

## Scope

- `Implement an explicit selected-target review path for a review-covered Unified PT subset, starting from existing edge-topk candidate traces.`
- `Extend the current scaffold generator so candidate-review artifacts carry enough metadata to support later canonical selection finalization.`
- `Add a canonical selected-target artifact format that is produced only from reviewed scaffold data and is suitable for scorer consumption.`
- `Add a shared selected-target resolver module so scorer-side target lookup is deterministic and auditable.`
- `Modify Unified PT scoring so it can score one approved human-selected target instead of always scoring the original dataset gold target.`
- `Redefine target_* output fields to mean the actually scored target and emit separate gold_target_* fields for the original dataset value.`
- `Update selected downstream readers so selected-target runs are labeled with the scored target rather than the dataset gold target.`
- `Validate the end-to-end path on a small approved Unified PT subset before any broader rerun.`

## Out Of Scope

- `Do not fully retrofit baseline PT and context-drift PT in the first implementation pass.`
- `Do not run selected-target mode across the full Unified PT family set in the first pass. A_ONLY, ZERO_CTRL, and shot-0 rows remain out of scope until review coverage is extended to those units.`
- `Do not build a GUI or interactive review editor; manual review continues to happen by editing JSON or a mirrored markdown summary.`
- `Do not add automatic relation-valid candidate selection rules in the same pass; the first supported path is human-approved selection.`
- `Do not add valid-answer-set or multi-answer scoring semantics in the first pass. The only supported scoring mode is one human-selected target per unit.`
- `Do not rewrite historical PT outputs to the new scoring basis.`
- `Do not promote docs/brain updates until execution proves the new workflow is stable.`

## Implementation Design

`The implementation will separate candidate review from scorer consumption rather than making the scorer read an editable scaffold directly. The reviewed-target path will therefore have three layers. Layer 1 is the existing candidate scaffold, expanded to retain the candidate token metadata needed for exact later scoring and to expose explicit selection fields. Layer 2 is a finalization step that validates human-reviewed units and writes a canonical selected-target artifact for scorer use. Layer 3 is scorer-side target resolution that reads only the canonical artifact and converts each scored PT unit into an explicit single-target scoring spec.` 

`This split is intentional. The current scaffold JSON is a human work surface and naturally contains pending units, notes, and broad candidate summaries. The scorer should not depend on partially reviewed data. A dedicated finalization step gives one place to reject incomplete review, duplicate unit ids, or malformed selection records before any PT run begins. It also gives a clean scratch-first canonical artifact that can be versioned and mirrored without ambiguity.`

`The first implementation target is Unified PT because that path already produces pt_unified_edge_topk.jsonl and already has a scaffold generator keyed by true evaluation units inside each q_id. The shared resolver module will be written so baseline PT and context-drift PT can adopt the same selected-target contract later, but those scorers are deliberately left out of the first approved execution scope to keep blast radius controlled. Just as importantly, the first selected-target run will be limited to the review-covered Unified subset: positive-shot B/D query units in BASE_ABD and CTX_ABD. A_ONLY, ZERO_CTRL, and shot-0 units are excluded from selected-target mode in this first pass, and the scorer should preflight that scope before execution rather than failing mid-run on missing reviewed units.`

### Artifact Model

- Layer 1 review scaffold:
  - produced from `pt_unified_edge_topk.jsonl`
  - still grouped by `q_id` and `unit_id`
  - enriched so each candidate suggestion carries representative token information needed for later exact scoring
  - includes explicit editable fields such as:
    - `review_status`
    - `selected_target`
    - `selected_target_canonical`
    - `notes`
- Layer 2 canonical selected-target artifact:
  - produced by a new finalization script from the reviewed scaffold
  - contains only validated units ready for scoring
  - records:
    - `format_version`
    - `artifact_kind`
    - `canonical_root`
    - `sync_root`
    - `sync_mode`
    - `artifact_profile`
    - `source_run_dir`
    - `source_topk_jsonl`
    - `source_scaffold_json`
    - `source_model`
    - `source_model_spec`
    - `selection_policy_version`
    - one normalized record per approved unit
- Layer 3 scorer rows:
  - use `target_*` fields for the actually scored target
  - retain original dataset fields separately as gold-target metadata
  - add explicit reviewed-target fields such as:
    - `gold_target_str`
    - `scored_target_str`
    - `scored_target_canonical`
    - `scored_target_token_id`
    - `scoring_basis`
    - `selected_target_artifact`
    - `target_resolution_status`

### Canonical Unit Identity

`The first-pass canonical unit key will remain aligned with the existing scaffold contract: q_id + query_source + query_input + gold_target, rendered as unit_id = <q_id>::<source>::<input>-><gold>. This avoids changing the current scaffold grouping logic and matches the actual evaluation unit identity already used in scripts/build_pt_valid_answer_scaffold.py. The finalization and scorer resolver must both recompute this identity from scorer row context, not from loose q_id-only matching.`

### Selection Mode

- `single_target`
  - exactly one approved selected target is scored
  - this is the only supported mode in the first pass

### First-Token Scoring Semantics

`The current PT pipeline is a first-next-token scorer, and the first implementation will keep that invariant exactly. The selected-target artifact stores the human-approved selected_target string, but the scorer always recomputes the actual first continuation token id at runtime using the active tokenizer for the current run. The artifact is therefore a review-decision artifact, not a tokenizer-validity guarantee artifact. Source model/model_spec metadata should still be recorded for auditability, but correctness comes from runtime first-token resolution with the active model tokenizer.`

### Resolver Behavior

`A new shared resolver module will load the canonical selected-target artifact and return a normalized scoring spec for each unit. The spec will include the approved selected target, canonical form, and enough metadata for row emission. Unified PT will call the resolver immediately before target-id lookup. When no selected-target artifact is provided, existing gold-target behavior remains available. When an artifact is provided, the first-pass policy should be strict within the supported selected-target scope: the run should preflight that only review-covered families and shots are requested, and within that scope any missing unit or invalid reviewed target should fail the run instead of silently falling back. If a softer fallback mode is needed later, it can be added as an explicit CLI option rather than as the default.`

### Unified PT Scorer Integration

`The Unified PT scorer currently derives target_id and emitted target metadata from query["output"]. That logic will be refactored so the scorer first builds the normal prompt prefix, then asks the resolver what selected target should be scored for the current unit. The scorer then computes the first-token id for that selected target with the active tokenizer, scores it exactly as existing PT does, and emits both gold-target metadata and scored-target metadata. target_* fields should mean the actually scored selected target, while dataset-origin fields move to gold_target_* columns. Edge-topk rows should also expose the scored-target basis so later lexical analysis can tell what target definition was active. Selected-target runs should also update downstream readers such as build_pt_unified_human_report.py and plot_pt_edge_target_prob_grid.py so labels and plotted probabilities remain aligned.`

### Validation And Failure Policy

- Scaffold finalization must reject:
  - `review_status != approved`
  - duplicate `unit_id`
  - missing `selected_target`
  - selected target not traceable to the reviewed candidate pool
- Unified PT scorer must reject by default:
  - selected-target mode requested together with unsupported families or shot-0 rows in the first pass
  - missing selected-target unit within the review-covered selected-target scope
  - inconsistent artifact metadata or unsupported artifact version

## Expected File Changes

- Modify: `/home/sunsik/my_fv_project/plans/2026-03-23-pt-selected-target-workflow-plan.md`
- Create: `/home/sunsik/my_fv_project/plans/2026-03-23-pt-selected-target-workflow-tech-spec.md`
- Modify: `/home/sunsik/my_fv_project/scripts/build_pt_valid_answer_scaffold.py`
- Create: `/home/sunsik/my_fv_project/scripts/finalize_pt_selected_targets.py`
- Create: `/home/sunsik/my_fv_project/fv/pt_selected_targets.py`
- Modify: `/home/sunsik/my_fv_project/scripts/score_cross_relation_unified_drift_control.py`
- Modify: `/home/sunsik/my_fv_project/scripts/build_pt_unified_human_report.py`
- Modify: `/home/sunsik/my_fv_project/scripts/plot_pt_edge_target_prob_grid.py`
- Optional modify in a later follow-up, not first pass: `/home/sunsik/my_fv_project/scripts/score_cross_relation_target_logit.py`
- Optional modify in a later follow-up, not first pass: `/home/sunsik/my_fv_project/scripts/score_cross_relation_context_drift_logit.py`
- Modify after successful execution if justified: `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`
- Modify after successful execution if justified: `/home/sunsik/my_fv_project/docs/brain/ops/storage_and_sync.md`
- Modify after successful execution if justified: `/home/sunsik/my_fv_project/docs/brain/INDEX.md`

## Ordered Implementation Steps

1. `Extend scripts/build_pt_valid_answer_scaffold.py so scaffold candidate suggestions retain representative token metadata from the edge-topk source and expose explicit editable selected_target fields needed for manual approval.`
2. `Create fv/pt_selected_targets.py as the shared artifact and resolver module. It should define the canonical unit-id helper, JSON loading and validation helpers, supported-family preflight logic, and scorer-facing resolution APIs.`
3. `Create scripts/finalize_pt_selected_targets.py to read a human-reviewed scaffold JSON, validate approved units, and emit a scratch-first canonical selected-target artifact plus optional repo-local mirror metadata fields and audit provenance.`
4. `Modify scripts/score_cross_relation_unified_drift_control.py to accept an optional selected-target artifact CLI path, preflight the first-pass supported family and shot scope, resolve reviewed targets per unit, and score the selected target's first continuation token instead of raw query["output"] when configured.`
5. `Update Unified PT raw output rows, edge-topk rows, and run metadata so target_* columns always refer to the scored target and gold_target_* columns carry the original dataset target.`
6. `Update scripts/build_pt_unified_human_report.py and scripts/plot_pt_edge_target_prob_grid.py so selected-target runs are labeled from the scored target fields and remain schema-compatible with gold-target runs.`
7. `Run a small approved validation slice, ideally one or a few q_ids with reduced trials and supported families only, to confirm scaffold-finalization parsing, scope preflight, unit alignment, strict failure behavior, and actual scorer-side selected-target evaluation.`
8. `If the validation pass succeeds and the workflow looks stable, update the relevant PT brain docs to reflect the new reviewed-target layer and record the run in the final report.`

## Validation Plan

- `Scaffold schema validation: confirm candidate_suggestions now retain token metadata and editable selection fields for every unit in the reviewed scaffold.`
- `Finalization validation: confirm finalize_pt_selected_targets.py rejects pending units, duplicate unit ids, and malformed selected-target records.`
- `Scope validation: confirm selected-target mode rejects unsupported families, A-query units, and shot-0 rows before execution in the first pass.`
- `Unit-alignment validation: confirm scorer-side unit_id reconstruction matches the finalized artifact records exactly for the approved validation subset.`
- `Single-target validation: confirm at least one unit whose selected target differs from gold_target is scored against the approved selected target rather than the original dataset target.`
- `First-token validation: confirm the scorer recomputes the selected target first token with the active tokenizer at runtime and emits the resulting token id in scored-target fields.`
- `Output-column validation: confirm Unified PT sweep rows and edge-topk rows include both gold-target and scored-target metadata, with target_* reserved for the scored target and a clear scoring_basis field.`
- `Downstream-reader validation: confirm build_pt_unified_human_report.py and plot_pt_edge_target_prob_grid.py label selected-target runs from the scored target rather than the dataset gold target.`
- `Strict-policy validation: confirm the scorer fails clearly when a selected-target artifact is supplied but a unit is missing or invalid within the supported selected-target scope, rather than silently falling back.`
- `Storage validation: confirm the finalized selected-target artifact records canonical_root and any sync metadata in line with the repo storage policy.`

## Expected Outputs

- `/home/sunsik/my_fv_project/plans/2026-03-23-pt-selected-target-workflow-tech-spec.md`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/selected_targets/<name>.json`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/selected_targets/<name>.md` `optional mirrored human-readable summary`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/pt_unified_shot_sweep.csv` `or equivalent selected-target rescored sweep output for the approved validation run`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/pt_unified_edge_topk.jsonl` `with scored-target basis metadata when enabled`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/run_meta.json` `updated to record selected-target artifact usage when applicable`
- `/home/sunsik/my_fv_project/reports/<YYYY-MM-DD>-pt-selected-target-workflow-report.md`

## Brain Docs To Update

- `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`
- `/home/sunsik/my_fv_project/docs/brain/ops/storage_and_sync.md` `if selected-target artifacts introduce a stable PT storage convention beyond the existing generic policy`
- `/home/sunsik/my_fv_project/docs/brain/INDEX.md` `only if a new stable brain document is created`

## Recommended Execution Strategy

- Launcher: `srun`
- Compute Mode: `gpu`
- Reason: `The first approved execution should be a bounded end-to-end Unified PT validation run over the review-covered B/D subset rather than a long unattended full rerun. The work still depends on large-model inference, so GPU is required, but the validation scope is small enough that an interactive or short scheduled srun workflow is the most practical default.`

## User Execution Settings Required Before Run

- Launcher choice: `srun recommended; sbatch acceptable; local not recommended for end-to-end validation`
- Selected-target artifact input: `user-approved reviewed scaffold path or finalized selected-target JSON path`
- Validation subset: `user to confirm q_id list for the first run, for example one or a few q_ids`
- Family scope for first run: `BASE_ABD and CTX_ABD reviewed B/D units only`
- Shot list: `user to confirm, recommend a reduced positive-shot validation list such as 1 or 1,3 before any wider sweep`
- Trial count: `user to confirm, recommend a small n_trials for the first validation run`
- Time limit: `user to specify; recommend at least 1-3 hours for a bounded validation run`
- Day-based duration if relevant: `not required unless using sbatch`
- GPU options: `user to specify; recommend 1 GPU`
- CPU count: `user to specify; recommend 8+ CPUs`
- Memory: `user to specify; recommend 48G-80G depending on model path`
- Partition or queue: `user cluster choice required`
- Job name: `recommend pt_selected_target_validation`
- Log path: `user to specify; recommend repo-local logs/ plus scratch-side run.log`
- Environment setup: `user to confirm model environment, python env, and any required module loads`
- Extra launcher flags: `user optional`

## Approval Note

Implementation and execution have not started. Waiting for Korean approval.
