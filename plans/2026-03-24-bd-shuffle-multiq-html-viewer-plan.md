# Plan Template

## Title

`BD shuffle single-file multi-Q target-probability HTML viewer`

## Metadata

- Date: `2026-03-24`
- Slug: `bd-shuffle-multiq-html-viewer`
- Approval Status: `approved`

## Objective

`Create an implementation plan for a single self-contained home-local HTML file that lets the user inspect, for each q_id, the baseline target probability versus the five shuffled cases in one glance, with one separate chart for the BDBDBD_D family and one separate chart for the DBDBDB_B family, while always showing the full layout strings, the query and target used, and the shuffle-construction context. The delivered report should be usable by opening just that one HTML file without downloading a companion folder.`

## Current Context

`The repo already has a Q1-only BD shuffle HTML path at /home/sunsik/my_fv_project/scripts/build_bd_shuffle_html_report.py and a home-local mirror at /home/sunsik/my_fv_project/bd_shuffle_q1_report.html. The latest broader canonical run is /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_bd_shuffle_compare_selected_20260324_023633_8964412, which contains multi-Q summary artifacts such as bd_shuffle_regime_metrics.csv, bd_shuffle_case_deltas.csv, bd_shuffle_side_aggregate.csv, and bd_shuffle_summary.md. The current HTML builder is oriented around one q only and groups primarily by query_side rather than q_id, so it is not a clean per-Q viewer for the selected run. The user wants a clearer per-Q presentation centered on target probability, not just raw CSV review, and specifically wants the final deliverable to be one downloadable HTML file rather than a page plus companion asset folder.`

## Presentation Options Considered

- `Single self-contained static HTML`: recommended baseline because it is easy to open locally, easy to hand off as one file, does not require a running server, and matches the user's requirement that the report be usable immediately after downloading one HTML file.
- `Jupyter notebook`: possible, but weaker for this use because the user wants a stable, low-friction artifact that can be opened quickly per q_id without rerunning notebook cells or depending on notebook UI state.
- `Streamlit or another local app`: more interactive, but likely too heavy for the current goal because it introduces a serving workflow, runtime dependency surface, and a less durable artifact than a generated report.
- `Markdown plus PNG plots`: simpler than an app, but weaker than single-file HTML because the long layout labels, fixed context blocks, and per-q navigation would be harder to keep readable and searchable.

`The current recommendation is therefore one self-contained static HTML file as the primary deliverable, with internal q navigation such as a sticky selector, anchor links, or collapsible q sections instead of multiple pages.`

## Assumptions

- `The existing selected-run summary artifacts are sufficient to build the desired viewer without rerunning model inference.`
- `The primary visual metric should be mean target probability, with baseline-versus-shuffled comparison readable directly from the bar chart and value labels.`
- `Because the layout strings are long and must be shown in full, horizontal bar charts are more readable than vertical bars.`
- `The viewer should make shuffle construction rules explicit on every q page: same q_id, same query row, same demo multiset, same B/D counts, order only changed.`
- `The relation provenance can be shown from the stable BD shuffle pipeline conventions and supporting files even though the selected run meta does not currently record the relation CSV paths explicitly inside run_meta.json.`
- `A single-file deliverable implies that all required CSS, any lightweight JavaScript, and the reduced report data needed for navigation and rendering should be embedded directly into the HTML rather than copied as sidecar assets.`

## Inputs And Dependencies

- `Existing Q1-only HTML builder at /home/sunsik/my_fv_project/scripts/build_bd_shuffle_html_report.py`
- `Selected canonical run root at /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_bd_shuffle_compare_selected_20260324_023633_8964412`
- `Per-regime metrics at /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_bd_shuffle_compare_selected_20260324_023633_8964412/bd_shuffle_regime_metrics.csv`
- `Per-case deltas at /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_bd_shuffle_compare_selected_20260324_023633_8964412/bd_shuffle_case_deltas.csv`
- `Run metadata and logs at /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_bd_shuffle_compare_selected_20260324_023633_8964412/run_meta.json and /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_bd_shuffle_compare_selected_20260324_023633_8964412/run.log`
- `BD shuffle regime definitions and layout order in /home/sunsik/my_fv_project/scripts/score_bd_shuffle_behavior.py`
- `Storage and mirror rules from /home/sunsik/my_fv_project/AGENTS.md and /home/sunsik/my_fv_project/docs/brain/ops/storage_and_sync.md`

## Proposed Steps

1. `Audit the current BD shuffle HTML builder and the selected multi-Q artifacts to lock the report data contract: q_id grouping, regime order, query and target fields, target probability fields, delta fields, and what fixed metadata can be shown without ambiguity.`
2. `Define the report information architecture as one self-contained HTML file with internal q navigation so the user can move between q_ids without needing multiple files or a companion asset folder.`
3. `Define the per-q section layout around two separate horizontal target-probability bar charts: one for the BDBDBD_D family and one for the DBDBDB_B family, with the chart titles including q_id, query, and target.`
4. `Specify the fixed context blocks that must appear on every q section: baseline family name, full baseline layout string, full shuffled layout strings, query side, query text, target text, relation provenance, demo-count balance, and the shuffle-construction rule that only order changes while content is held fixed.`
5. `Plan the label and annotation scheme so each bar is readable at a glance: regime id plus full layout string on the left, mean target probability at the bar end, and optional delta-versus-baseline text alongside the shuffled bars without letting secondary metrics dominate the visual.`
6. `Define how the single-file artifact will embed all required CSS, lightweight navigation logic, and reduced per-q summary data directly into the HTML so the delivered file remains portable and immediately viewable on its own.`
7. `Define the output structure so canonical inputs remain scratch-side while the generated human-review deliverable is one home-local HTML file at a predictable path suitable for direct sharing or download.`
8. `Define validation checks for the future implementation: exactly one generated HTML file, internal q navigation present, two charts per q section, exactly six bars per chart, titles that include the correct query and target, full layout strings visible, baseline families separated, and no required external asset folder.`
9. `After plan approval, write a tech spec that maps these presentation requirements to the exact script changes, output path, embedded-data strategy, and validation commands without starting implementation yet.`

## Risks And Blockers

- `The current Q1-only HTML builder hardcodes Q1-oriented copy and lookup keys that are too coarse for multi-Q use, so extending it carelessly could produce silently mixed or overwritten q-level content.`
- `The selected run metadata does not explicitly persist the relation CSV inputs in run_meta.json, so the implementation must decide whether to source relation provenance from stable pipeline conventions, the shell wrapper defaults, or the run log.`
- `Full layout labels are long enough that a poor chart orientation or cramped table layout would reduce readability instead of improving it.`
- `If the report tries to show too many secondary metrics on the same page, the primary goal of making target probability easy to read may get diluted.`
- `A single self-contained HTML file can become bulky if it embeds too much unnecessary raw data, so the implementation must include only the summary fields needed for the intended view rather than dumping every artifact into the page.`
- `Because scratch is canonical and home is a mirror, the report path and page text must label clearly that the HTML is a human-view derivative, not the canonical result root itself.`

## Recommended Compute Mode

- Mode: `local`
- Why: `This task is a report-generation and artifact-reading workflow over existing CSV and metadata files. It does not require rerunning model inference or loading the 70B model, so local CPU-side processing is the correct default for the eventual implementation.`

## Expected Outputs

- `/home/sunsik/my_fv_project/plans/2026-03-24-bd-shuffle-multiq-html-viewer-plan.md`
- `Follow-on tech spec after plan approval`
- `If later executed: one home-local self-contained multi-Q BD shuffle HTML file that can be opened directly without a companion folder`
- `If later executed: a clearly documented source path back to the canonical scratch run used to generate the single-file report`

## Success Criteria

- `The plan clearly defines a per-q viewer rather than another aggregate-only report.`
- `The plan explicitly justifies why one self-contained static HTML file is preferred over notebook, app, multi-page HTML, or PNG-only alternatives for this repo and task.`
- `The future viewer is specified to show exactly two baseline-family charts per q_id, not a mixed combined chart.`
- `Every per-q chart is planned to show the full layout string for baseline and all five shuffled cases.`
- `Every per-q section is planned to show the query, target, baseline family, and shuffle-construction context in fixed visible text.`
- `The plan preserves scratch-first canonicality while making the human-review HTML easy to open or share as a single file from the repo root.`
- `The future implementation is planned to require no companion asset directory for normal viewing.`
- `The future implementation can be validated entirely from existing selected-run artifacts without requiring a new model run.`

## Brain Impact

- Brain impact: `update required`
- Why: `If this viewer becomes the stable way to inspect BD shuffle selected-run results, the repo will gain a new persistent human-review entrypoint and reporting convention that should be reflected in the current project knowledge under docs/brain rather than left only as one-off report code.`

## Approval Note

Substantial execution has not started. Waiting for Korean approval.
