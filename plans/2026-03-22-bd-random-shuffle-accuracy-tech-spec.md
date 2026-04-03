# Tech Spec Template

## Title

`Q1 BD alternating vs random-shuffled behavior comparison implementation`

## Metadata

- Date: `2026-03-22`
- Slug: `bd-random-shuffle-accuracy`
- Source Plan: `plans/2026-03-22-bd-random-shuffle-accuracy-plan.md`
- Approval Status: `approved`

## Scope

- `Implement a dedicated Q1-only BD behavior-scoring path for alternating vs shuffled comparison.`
- `Implement the scorer and wrapper interfaces so that a later user-specified q list can be provided without redesigning the experiment path, even though the first approved execution remains Q1-only.`
- `Score both regular baselines BDBDBD_D and DBDBDB_B under one common run so the final comparison uses one aligned trial plan instead of mixing old and new runs.`
- `Add five fixed shuffled D-query cases and five fixed shuffled B-query cases, using the approved layouts from the plan.`
- `Keep query identity fixed within each regular-vs-shuffled comparison group and keep the exact concrete mixed demo multiset fixed within each trial group.`
- `Measure top-1 accuracy, target logit, target logprob, target rank, and lexical top-k competitors for every regime at shot 9 only in the first execution pass.`
- `Write scratch-first run outputs under /scratch/sunsik/my_fv_project/pt_analysis and produce comparison-ready summary CSV/JSON/Markdown artifacts.`
- `Provide visibly strong progress reporting during execution through unbuffered stdout/stderr logging plus a machine-readable progress status file that is updated throughout the run.`
- `Retain the old scratch BDBDBD_D summary only as an external sanity-check reference, not as the canonical comparator for the new shuffled experiment.`

## Out Of Scope

- `No attempt to retrofit the entire generic unified PT bootstrap/report stack to natively understand all shuffled BD regimes.`
- `No AB or AD shuffled controls in this pass.`
- `No representation-analysis rerun such as PCA, vector extraction, or StepD in this pass.`
- `No multi-q execution rollout beyond Q1 in this pass.`
- `No unconstrained random shuffle family generation beyond the five approved fixed layouts per query side.`
- `No automatic docs/brain rewrite before execution proves the path is stable and worth promoting.`

## Implementation Design

`The implementation will follow the same isolation strategy used by the earlier BD PCA work: create a focused BD-specific path instead of widening the generic pipeline more than necessary. The existing generic unified PT scorer is heavily organized around four families and a fixed three-regime CTX_ABD bundle. Extending that whole stack to natively support one regular BD-B query baseline plus ten shuffled BD cases would increase blast radius across scoring, bootstrap, HTML reporting, and stable summary assumptions. For this task, that is unnecessary. The user wants one direct behavioral test on BD alternation, not a general re-architecture of unified PT.`

`The core new scorer will therefore be a dedicated BD-only script, tentatively scripts/score_bd_shuffle_behavior.py. It will reuse low-level utilities and model-loading behavior already present in the repo, including tokenizer/model loading from fv.hf_loader, prompt construction from fv.prompting.build_prompt_qa, and top-k extraction helpers already used in the target-logit scoring stack. The first approved execution will be explicitly scoped to Q1, but the scorer interface itself will accept a q selector such as --qid or --q_list so that later user-specified q subsets can be run without redesigning the script.`

`Trial construction will be pair-grouped by query side. For the D-query comparison group, one trial plan row will contain: a single D-side query row, five sampled B demo rows, and four sampled D demo rows. For the B-query comparison group, one trial plan row will contain: a single B-side query row, four sampled B demo rows, and five sampled D demo rows. Within each group, the sampled row identities are fixed once per trial and reused across the regular baseline and all five shuffled cases. This guarantees the intended fairness contract: same q_id, same query, same demo row multiset, same source counts, only different order. The internal data structures will be keyed by q_id from the start so that moving from one-q execution to a later user-provided q list is an input change rather than a structural code rewrite.`

`Layouts will be encoded as explicit 9-character source strings rather than inferred by an alternating rule. The approved D-query layouts are BDBDBDBDB, BDDBBDBDB, BBDDBDDBB, BBBDDBDDB, BBBBDDDDB, and DBBBDDBBD. The approved B-query layouts are DBDBDBDBD, DBBDDBDBD, DDBBDBBDD, DDDBBDBBD, DDDDBBBBD, and BDDDBBDDB. A helper will map each layout string to a concrete ordered demo list by consuming the next sampled row from the relevant source-specific list whenever a B or D symbol appears. This makes the layout contract explicit and easy to validate.`

`Scoring will still use the same shot-prefix machinery, but the first execution pass will run shot 9 only. That means each regime is evaluated on its full 9-demo layout rather than on intermediate prefixes. This materially reduces runtime and lets the first pass answer the user's immediate question about full-context behavioral collapse or robustness before spending GPU time on intermediate shot curves. The scorer and wrapper will keep a shot-list interface so later runs can re-enable 1,3,5,7,9 if the user wants the full trajectory.`

`The raw scorer output will be a sweep CSV with one row per q_id / trial / shot / regime. Each row will include query metadata, layout_pattern, query_side, demo row ids used in order, target token id/string, target logprob, target prob, target logit, target rank in vocab, and normalized lexical top-k metadata if enabled. A matching edge-topk JSONL will store lexical competitor traces for later aggregation.`

`Progress visibility is an explicit design requirement in this pass. The scorer will therefore compute the exact expected evaluation count up front. For the default Q1 shot-9-only run with 50 trials and 12 regimes, that is 600 prompt evaluations. For later user-specified q lists or broader shot lists, the same mechanism will recompute totals from the selected q ids and shots rather than relying on Q1-specific constants. The run wrapper and scorer will emit unbuffered progress lines to stdout and run.log in a stable format that includes current q_id, current regime, completed/total evaluations, percentage, elapsed time, ETA, last trial index, current shot, target_logprob, target_prob, target_rank, and current top-1 candidate. In addition, a progress_status.json file inside the scratch run directory will be rewritten periodically so the current run state is inspectable without tailing logs.`

`Each regime will also get explicit lifecycle logging: q-start, regime-start, periodic progress ticks, and regime-end summary lines with rows completed, wall time, mean target_logprob so far, and provisional top-1 accuracy so far. After raw scoring completes, the summary builder will emit stage-start and stage-end log lines for each aggregate artifact it writes. The goal is that the user can tell at a glance both where the job is and whether the numerical outputs look sane while it is still running.`

`A second new script, tentatively scripts/build_bd_shuffle_behavior_summary.py, will aggregate those raw outputs into analysis-ready artifacts. First, it will compute per-regime metrics by q_id and shot: n_trials, top1_accuracy, mean target_logprob, mean target_prob, mean target_logit, mean target_rank, and standard deviations where useful. Second, it will create per-case delta tables against the correct baseline on the same query side: each shuffled D-case versus BDBDBD_D, and each shuffled B-case versus DBDBDB_B. Third, it will create side-level aggregates across the five shuffled cases, reporting mean/min/max/std deltas so the final interpretation is not driven by a single permutation. Fourth, it will aggregate lexical top-k candidates into a compact competitor summary similar in spirit to the existing top-k summary CSVs, but restricted to the BD regimes relevant here.`

`A lightweight run wrapper, tentatively scripts/run_pt_bd_shuffle_compare_llama70b.sh, will orchestrate the scorer and summary builder using the same environment and storage conventions as the existing PT scripts. The wrapper will write its canonical run tree under /scratch/sunsik/my_fv_project/pt_analysis, emit a raw sweep CSV, edge-topk JSONL, regime metrics CSV, shuffled-vs-regular delta CSV, side-aggregate CSV, a short Markdown summary for the run, and a machine-readable progress_status.json file. The wrapper will default to Q1 but expose a q selector argument so that later user-provided q lists can be passed through cleanly. The wrapper will use unbuffered Python execution and tee logging so progress is visible immediately in the launcher log.`

`The design intentionally regenerates both regular baselines inside the same dedicated BD run, even though an older BDBDBD_D scratch summary already exists. That choice avoids cross-run confounds and gives one canonical comparison table across regular D-query, regular B-query, and all shuffled cases. The old BDBDBD_D scratch result will still be checked as a sanity reference to ensure the new run is in the expected qualitative range.`

## Expected File Changes

- Modify: `/home/sunsik/my_fv_project/plans/2026-03-22-bd-random-shuffle-accuracy-plan.md`
- Create: `/home/sunsik/my_fv_project/plans/2026-03-22-bd-random-shuffle-accuracy-tech-spec.md`
- Create: `/home/sunsik/my_fv_project/scripts/score_bd_shuffle_behavior.py`
- Create: `/home/sunsik/my_fv_project/scripts/build_bd_shuffle_behavior_summary.py`
- Create: `/home/sunsik/my_fv_project/scripts/run_pt_bd_shuffle_compare_llama70b.sh`
- Optional create if launcher choice later requires it: `/home/sunsik/my_fv_project/scripts/slurm/run_q1_bd_shuffle_compare.sbatch`
- Update at runtime: `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/*`
- Update at runtime: `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/progress_status.json`
- Update after execution if justified: `/home/sunsik/my_fv_project/docs/brain/OPEN_QUESTIONS.md`
- Update after execution if justified: `/home/sunsik/my_fv_project/docs/brain/INDEX.md`

## Ordered Implementation Steps

1. `Add the approved regular and shuffled BD layout definitions to a dedicated scorer script, with clear regime ids, query side labels, and layout_pattern strings.`
2. `Implement deterministic BD trial-plan generation keyed by q_id, with the first execution path defaulting to Q1. Each plan row must lock query row identity plus the exact sampled B/D demo row ids that will be reused across the regular baseline and the five shuffled cases on that side.`
3. `Expose a q selector argument in the scorer and wrapper so later user-provided q lists can be passed in, while keeping the default execution target fixed to Q1 for the first approved run.`
4. `Implement layout materialization from explicit B/D pattern strings so that every regime consumes the same sampled source-specific demo rows but emits them in a different order.`
5. `Implement prompt scoring for shot 9 in the first execution pass using the existing model/tokenizer loading path and the existing prompt builder conventions. Save a raw sweep CSV with one row per q_id/trial/shot/regime plus lexical top-k JSONL for competitor analysis. Keep the shot-list interface intact for later runs.`
6. `Add explicit progress instrumentation to the scorer: total-eval counting, stable progress lines, q-start/regime-start/regime-end summaries, and periodic rewrites of progress_status.json with done/total, pct, current_q_id, current_regime, current_trial, current_shot, elapsed, eta, and latest score fields.`
7. `Implement a BD-specific summary builder that reads the raw sweep and top-k outputs and writes: per-regime metrics CSV, per-case delta CSV versus the correct baseline, side-level aggregated shuffled summary CSV, lexical top-k aggregate CSV, and a concise Markdown summary.`
8. `Create a small run wrapper that standardizes model/env arguments, scratch-first output roots, unbuffered logging, q-selector passthrough, run metadata, progress-status output, and summary generation so the experiment can be launched cleanly under local shell, srun, or sbatch.`
9. `If the user later chooses sbatch, add a minimal sbatch wrapper that calls the run wrapper without changing experiment logic.`
10. `Validate the new path on the approved Q1-only shot-9-only scope, confirm that DBDBDB_B baseline summary is now present, confirm that the shuffled cases all preserve query identity and demo multisets, and confirm that the comparison artifacts and progress logs are sufficient to answer the alternation hypothesis question directly.`

## Validation Plan

- `Layout validation: for every shuffled regime, confirm the layout string length is 9, the total source counts match the intended side, and the layout is not exactly the alternating baseline.`
- `Trial-alignment validation: for each side-specific comparison group, confirm regular and shuffled regimes share the same q_id, query row id, and demo row-id multiset.`
- `Interface validation: confirm the scorer/wrapper expose a q selector argument and that Q1 remains the default when none is supplied.`
- `Shot-scope validation: confirm the first execution pass writes only shot 9 rows and that no unintended intermediate-shot rows are present.`
- `Baseline validation: confirm the new run produces both BDBDBD_D and DBDBDB_B regular summaries under the same protocol.`
- `Reference sanity-check: compare the newly generated Q1 BDBDBD_D top-1 accuracy curve against the previously observed scratch reference to confirm the new run is in a plausible range, without requiring exact identity.`
- `Raw-output validation: confirm the raw sweep CSV and edge-topk JSONL exist and contain rows for all 12 regimes at shot 9.`
- `Progress validation: confirm that run.log shows immediate unbuffered progress lines, regime-start/regime-end markers, and meaningful ETA fields, and confirm that progress_status.json is updated during the run.`
- `Summary validation: confirm the per-regime metrics CSV, shuffled-vs-regular delta CSV, side-level aggregate CSV, lexical competitor summary CSV, and Markdown summary are all produced and non-empty.`
- `Interpretation validation: confirm the summary artifacts expose, at minimum, top-1 accuracy deltas, target logit/logprob deltas, target-rank deltas, and the main competing candidate changes for every shuffled case.`

## Expected Outputs

- `/home/sunsik/my_fv_project/plans/2026-03-22-bd-random-shuffle-accuracy-tech-spec.md`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/bd_shuffle_shot_sweep.csv`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/bd_shuffle_edge_topk.jsonl`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/bd_shuffle_regime_metrics.csv`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/bd_shuffle_case_deltas.csv`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/bd_shuffle_side_aggregate.csv`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/bd_shuffle_edge_topk_trial_agg.csv`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/bd_shuffle_summary.md`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/progress_status.json`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/run.log`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/run_meta.json`
- `/scratch/sunsik/my_fv_project/pt_analysis/<run_dir>/run_status.json`

## Brain Docs To Update

- `/home/sunsik/my_fv_project/docs/brain/OPEN_QUESTIONS.md` `if the run materially sharpens the current project reading of the alternation-vs-relation question`
- `/home/sunsik/my_fv_project/docs/brain/INDEX.md` `only if a new stable brain document is created after execution`

## Recommended Execution Strategy

- Launcher: `srun`
- Compute Mode: `gpu`
- Reason: `The run is Q1-only and now shot-9-only, so it is more bounded than the previous draft. GPU is still required because the workload uses large-model inference across 12 regimes and 50 trials, but the reduced shot scope makes a focused first-pass sbatch or srun run reasonable. If later runs expand back to 1,3,5,7,9 or multiple q ids, the same wrapper can do that without redesign.`

## User Execution Settings Required Before Run

- Launcher choice: `srun recommended; sbatch acceptable; local not recommended`
- Q selector for this run: `default Q1; later runs may override with a user-provided q list`
- Shot selector for this run: `shot 9 only`
- Time limit: `user to specify; for Q1-only shot-9-only first pass recommend at least 2-4 hours`
- Day-based duration if relevant: `not required unless the user chooses long sbatch scheduling`
- GPU options: `user to specify; recommend 1 GPU`
- CPU count: `user to specify; recommend 8+ CPUs`
- Memory: `user to specify; recommend 48G-64G`
- Partition or queue: `user cluster choice required`
- Job name: `recommend q1_bd_shuffle_compare`
- Log path: `user to specify; recommend logs/q1_bd_shuffle_compare.log plus scratch-side run.log and progress_status.json`
- Environment setup: `user to specify if non-default venv or module activation is required`
- Extra launcher flags: `user optional`

## Approval Note

Implementation and execution have not started. Waiting for Korean approval.
