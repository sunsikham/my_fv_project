# Tech Spec Template

## Title

`BD shuffle single-file multi-Q target-probability HTML viewer`

## Metadata

- Date: `2026-03-24`
- Slug: `bd-shuffle-multiq-html-viewer`
- Source Plan: `plans/2026-03-24-bd-shuffle-multiq-html-viewer-plan.md`
- Approval Status: `approved`

## Scope

- `Build one self-contained HTML report from an existing BD shuffle selected run without rerunning model inference.`
- `Support multi-Q input from the selected canonical run at /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_bd_shuffle_compare_selected_20260324_023633_8964412.`
- `Render one section per q_id inside a single HTML file, with internal q navigation that works without external assets or a server.`
- `For each q_id, render two separate horizontal target-probability charts: one for the BDBDBD_D family and one for the DBDBDB_B family.`
- `Show all six regimes in each family chart: one baseline plus five shuffled cases.`
- `Show the full layout string for every bar label together with the regime id.`
- `Show fixed context for each q section: q_id, query, target, query side, baseline family, relation provenance, B/D count balance, and the shuffle-construction rule that only order changes while content stays fixed.`
- `Embed the reduced report data, CSS, and any lightweight JavaScript directly into the HTML so the file is portable as a one-file deliverable.`

## Out Of Scope

- `No new model scoring run, no GPU inference, and no regeneration of BD shuffle CSV artifacts.`
- `No Streamlit app, notebook dashboard, or multi-page HTML site.`
- `No companion asset directory required for normal viewing.`
- `No attempt to preserve the existing Q1-only visual layout exactly if that conflicts with the new single-file multi-Q requirement.`
- `No full top-k lexical candidate browser in the first implementation pass unless it can be included without diluting the target-probability-focused view.`
- `No attempt to generalize the viewer to unrelated PT report families in this pass.`

## Implementation Design

`The implementation will favor a dedicated single-file report builder over extending the current Q1-only builder in place. The current script at /home/sunsik/my_fv_project/scripts/build_bd_shuffle_html_report.py is tightly coupled to one-q assumptions, an asset mirror directory, and a wider report surface including tables and top-k sections. The new requirement is narrower in metric scope but stricter in portability: one HTML file, no asset folder, multi-Q aware, and optimized for target probability. A dedicated builder keeps that requirement isolated and avoids breaking the existing Q1 artifact path.`

`The builder will read the selected run's regime metrics CSV as the primary source because it already contains q_id, query_side, regime_id, case_kind, layout_pattern, query_input, target_str, baseline_regime_id, and mean_target_prob. The case-deltas CSV will be read as a secondary source so shuffled bars can optionally display delta-versus-baseline annotations without recomputing them. The run metadata and run log will be read only for page-level provenance and relation-source labeling.`

`Data preparation will group rows first by q_id and then by query_side. Within each side, rows will be ordered by the stable regime order already defined in /home/sunsik/my_fv_project/scripts/score_bd_shuffle_behavior.py so the baseline always appears first and the five shuffled cases follow in a fixed order. Each grouped section will carry both display labels and compact embedded data fields: regime id, layout string, case kind, mean target probability, optional delta, query, target, baseline family id, and explanatory metadata.`

`The HTML itself will be fully self-contained. CSS will be embedded in a style block. The prepared per-q data will be embedded either as inline JSON inside a script tag or already rendered into the page markup. Lightweight client-side JavaScript is acceptable only for internal q navigation, such as a sticky q selector that scrolls to sections or toggles section visibility, but the report must remain readable even if scripting support is limited. No external fonts, libraries, CDN assets, or linked CSV files will be required.`

`Each q section will have the same structure. At the top, a context block will state the q_id, the D-side query and target, the B-side query and target, the two baseline families, the relation provenance, and the shuffle-construction rule. Below that, the section will present two charts: one titled with the D-family baseline plus its query and target, and one titled with the B-family baseline plus its query and target. Bars will be horizontal to preserve legibility for full layout strings. Baseline bars and shuffled bars will use distinct colors, and each bar will show the mean target probability numerically at the bar end. If included, delta annotations will be visually secondary to keep the page focused on target probability.`

`The output path will be a single home-local HTML report under /home/sunsik/my_fv_project/reports so the artifact is easy to open and hand off. The page will still label the canonical scratch run root clearly, consistent with the repo's scratch-first storage policy.`

## Expected File Changes

- Modify: `/home/sunsik/my_fv_project/plans/2026-03-24-bd-shuffle-multiq-html-viewer-plan.md`
- Create: `/home/sunsik/my_fv_project/plans/2026-03-24-bd-shuffle-multiq-html-viewer-tech-spec.md`
- Create: `/home/sunsik/my_fv_project/scripts/build_bd_shuffle_singlefile_report.py`
- Create at runtime: `/home/sunsik/my_fv_project/reports/<generated-single-file-report>.html`
- Update after execution if justified: `/home/sunsik/my_fv_project/docs/brain/INDEX.md`
- Update after execution if justified: `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md`

## Ordered Implementation Steps

1. `Create a dedicated single-file BD shuffle report builder script rather than modifying the existing Q1-only builder first.`
2. `Implement CSV readers for bd_shuffle_regime_metrics.csv and bd_shuffle_case_deltas.csv, plus lightweight readers for run_meta.json and run.log where provenance text is needed.`
3. `Define the stable multi-Q grouping and ordering logic: q_id -> query_side -> regime order, using the same family order as the scorer definitions.`
4. `Build a reduced in-memory report model that contains exactly the fields needed for the final viewer: q_id, family, regime id, layout string, mean target probability, optional delta, query, target, and fixed explanatory metadata.`
5. `Implement the single-file HTML shell with embedded CSS and, if needed, lightweight inline JavaScript for internal q navigation.`
6. `Implement one per-q section renderer that emits the fixed context block plus two horizontal charts, one for the BDBDBD_D family and one for the DBDBDB_B family.`
7. `Implement bar labeling so every row shows the full regime id and full layout string, with baseline bars visually distinct from shuffled bars and numeric target-probability labels visible.`
8. `Implement page-level provenance text that points back to the canonical scratch run and explicitly marks the HTML as a human-view derivative.`
9. `Run local validation on the generated HTML from the selected run and verify that the file is self-contained, readable, and requires no companion directory.`

## Validation Plan

- `Run the builder locally against /scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_bd_shuffle_compare_selected_20260324_023633_8964412 and confirm that exactly one HTML file is produced.`
- `Confirm that the HTML opens directly from disk without broken links to external CSS, JS, CSV, or image assets.`
- `Confirm that every q_id in the selected run appears in the HTML navigation and in a corresponding rendered q section.`
- `Confirm that each q section contains exactly two charts, one for BDBDBD_D and one for DBDBDB_B.`
- `Confirm that each chart contains exactly six bars in stable order: one baseline and five shuffled cases.`
- `Confirm that every bar label shows both regime id and the full layout string.`
- `Confirm that each chart title includes the correct q_id, query, and target derived from the selected run artifacts.`
- `Confirm that the fixed explanatory text states the shuffle rule, relation provenance, and canonical scratch source clearly.`
- `Confirm that baseline and shuffled bars are visually distinct and that mean target probability values are legible without reading raw CSV files.`
- `Confirm that no model execution or scratch artifact mutation occurred during the build.`

## Expected Outputs

- `/home/sunsik/my_fv_project/plans/2026-03-24-bd-shuffle-multiq-html-viewer-tech-spec.md`
- `/home/sunsik/my_fv_project/scripts/build_bd_shuffle_singlefile_report.py`
- `/home/sunsik/my_fv_project/reports/<generated-single-file-report>.html`

## Brain Docs To Update

- `/home/sunsik/my_fv_project/docs/brain/pipelines/pt.md` `if the single-file BD shuffle viewer becomes the stable human-review entrypoint for selected runs`
- `/home/sunsik/my_fv_project/docs/brain/INDEX.md` `if a new stable report entrypoint or report-generation script needs to be indexed`

## Recommended Execution Strategy

- Launcher: `local`
- Compute Mode: `local`
- Reason: `This work is a deterministic report-generation task over existing scratch-side CSV and metadata artifacts. It requires only local file reads and one HTML write, so local execution is the appropriate default and avoids unnecessary cluster usage.`

## User Execution Settings Required Before Run

- Launcher choice: `local recommended`
- Time limit: `not required for local; expected to be short`
- Day-based duration if relevant: `none`
- GPU options: `none`
- CPU count: `default local shell is sufficient`
- Memory: `default local shell is sufficient`
- Partition or queue: `none`
- Job name: `none`
- Log path: `optional; recommend /home/sunsik/my_fv_project/logs/bd_shuffle_singlefile_report.log if the user wants a captured build log`
- Environment setup: `default repo shell; no model runtime environment should be required beyond standard Python used to run the builder`
- Extra launcher flags: `if desired, the user may later specify the exact --run_dir and --out_html paths; otherwise the approved selected run and a repo-local report path can be used as defaults`

## Approval Note

Implementation and execution have not started. Waiting for Korean approval.
