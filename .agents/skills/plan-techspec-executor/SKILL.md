---
name: plan-techspec-executor
description: "Use when the user wants a gated execution workflow: first write a plan markdown file, self-review it, get Korean approval, then write a tech spec markdown file, self-review it, get Korean approval again, collect user-chosen execution settings for local, srun, or sbatch runs, execute only after approval, stop on unrecoverable errors, and always write a final report with commands, executed files, changed files, outputs, and logs."
---

# Plan Techspec Executor

## Overview

This skill enforces a two-stage approval workflow before implementation or execution begins. It is for tasks where the user wants reusable planning artifacts, explicit Korean approval gates, user-controlled runtime settings, and a final execution report whether the run succeeds or fails.

## When To Use

Use this skill when the user wants one or more of these behaviors:

- Create a `plan` markdown file before doing real work.
- Review that plan before showing it to the user.
- Wait for Korean approval before moving forward.
- Create a second `tech spec` markdown file from the approved plan.
- Review that tech spec before showing it to the user.
- Wait for Korean approval again before implementation or execution.
- Let the user choose the runtime command pattern, especially `local`, `srun`, or `sbatch`.
- Let the user specify runtime limits such as time or days and cluster options such as GPU, CPU, memory, partition, and job name.
- Stop on failure and always produce a final report with file and command tracking.

## Core Rules

- Do not start substantial execution before the plan is approved.
- Do not start implementation or execution before the tech spec is approved.
- Lightweight inspection is allowed before approval. Examples: reading files, checking the repo state, checking environment availability, estimating runtime, or identifying likely touched files.
- After self-reviewing a plan or tech spec, explain it to the user in detail before asking for approval.
- If the user asks questions or requests changes after the explanation, update the document, explain the revised version again, and then ask for approval again.
- If the approved plan changes in a meaningful way, update the plan file and ask for plan approval again.
- If the approved tech spec changes in a meaningful way, update the tech spec file and ask for tech spec approval again.
- Create `plans/`, `reports/`, and `logs/` if they do not already exist.
- Store both the plan and the tech spec under `plans/`.
- Store final reports under `reports/`.
- Store execution logs under `logs/`.
- Use absolute dates in filenames with the `YYYY-MM-DD` prefix.
- Always produce a final report, even when execution succeeds.
- Treat `docs/brain/` as the current stable project knowledge hub, not a dump for dated notes, one-off debug logs, or scratch plans.
- Update `docs/brain/` only when the work changes stable project knowledge such as pipeline behavior, storage rules, entrypoints, runbooks, analysis interpretations, glossary terms, or open questions.
- After execution ends, explain the result to the user in detail based on the final report.
- Default retry policy is zero retries. Allow at most one automatic retry only for a clearly transient and safe failure.

## Approval Rules

Treat only explicit Korean approval as approval. Examples:

- `승인`
- `실행`
- `진행`
- `계속 진행`
- `오케이 실행`
- `플랜 승인`
- `기술명세 승인`

Do not treat comments, questions, or tentative feedback as approval. Examples:

- `검토해볼게`
- `수정하자`
- `설명해봐`
- `이 부분 바꿔`

## Workflow

1. Inspect the task only enough to produce a good plan.
2. Draft the plan at `plans/<YYYY-MM-DD>-<slug>-plan.md` using [assets/templates/plan-template.md](assets/templates/plan-template.md).
3. Self-review the plan against the checklist in this skill and revise it before presenting it.
4. Explain the plan in detail, including the objective, why the approach was chosen, what each step does, the main risks, the compute recommendation, expected outputs, and what still needs user approval. Include the plan path and state clearly that substantial execution has not started.
5. Wait for explicit Korean approval of the plan.
6. Draft the tech spec at `plans/<YYYY-MM-DD>-<slug>-tech-spec.md` using [assets/templates/tech-spec-template.md](assets/templates/tech-spec-template.md).
7. Self-review the tech spec against the checklist in this skill and revise it before presenting it.
8. Explain the tech spec in detail, including how it maps to the approved plan, the implementation design, the files likely to change, the validation strategy, the launcher recommendation, expected runtime needs, and what execution settings still need user approval. Include the tech spec path and state clearly that implementation has not started.
9. Wait for explicit Korean approval of the tech spec.
10. If runtime settings are missing, ask the user to specify them before execution.
11. Capture the pre-execution baseline of the workspace.
12. Execute only the approved tech spec with the user-approved runtime settings.
13. Track commands run, files executed, logs written, and files changed.
14. If an unrecoverable error occurs, stop immediately.
15. Evaluate whether the execution changed stable project knowledge that belongs in `docs/brain/`.
16. If a brain update is required, update the relevant `docs/brain/` files and keep `docs/brain/INDEX.md` accurate if links or new stable documents changed.
17. Write the final report at `reports/<YYYY-MM-DD>-<slug>-report.md` using [assets/templates/report-template.md](assets/templates/report-template.md).
18. Explain the execution result in detail, including what changed, what was run, what was validated, what the outputs mean, whether any brain documents were updated, and any remaining risks or next steps.

## Plan Phase

The plan is the high-level execution proposal. It must be actionable, but not yet the implementation blueprint.

Every plan must include:

- objective
- current context
- assumptions
- inputs and dependencies
- step-by-step approach
- risks and blockers
- recommended compute mode: `local` or `gpu`
- expected outputs
- success criteria
- brain impact: `none` or `update required`
- approval status: `pending`

Self-review checklist for the plan:

- The objective is specific.
- The approach is implementable.
- Risks are concrete rather than generic.
- Success criteria are measurable.
- The compute recommendation is justified.
- The brain impact classification is justified.
- The file path and filename are correct.

When explaining the plan to the user, cover at least:

- the objective of the work
- why this approach was chosen
- what each planned step does
- the main risks or blockers
- why `local` or `gpu` is recommended
- what files or outputs are expected
- whether this work is likely to require a stable `docs/brain/` update
- what still needs explicit user approval

## Tech Spec Phase

The tech spec is the implementation-ready document derived from the approved plan. It should contain enough detail to execute directly after user approval.

Every tech spec must include:

- source approved plan path
- scope
- out of scope
- implementation design
- files expected to change
- ordered implementation steps
- validation plan
- expected outputs
- recommended launcher: `local`, `srun`, or `sbatch`
- recommended compute mode: `local` or `gpu`
- brain docs to update if required
- required user execution settings before run
- approval status: `pending`

Self-review checklist for the tech spec:

- It clearly maps back to the approved plan.
- The implementation steps are detailed enough to execute.
- The likely touched files are named.
- Validation is concrete.
- The recommended launcher is justified.
- If a brain update is required, the target brain docs are named.
- It explicitly lists what the user still needs to specify before execution.
- The file path and filename are correct.

When explaining the tech spec to the user, cover at least:

- how it maps back to the approved plan
- the implementation design and execution order
- which files are expected to change
- how validation will be done
- why the recommended launcher and compute mode were chosen
- which `docs/brain/` files are expected to change, if any
- what runtime settings the user still needs to provide
- any material risks or constraints before execution

## Brain Update Rules

Use `docs/brain/` for stable, current project knowledge only.

Update `docs/brain/` when the work changes one or more of these:

- project or codebase maps
- pipeline entrypoints or pipeline step descriptions
- artifact roots, taxonomy, storage rules, or sync rules
- analysis interpretations that should become the current project reading
- operational runbooks or environment guidance
- glossary terms or open questions that should stay current

Do not update `docs/brain/` for:

- dated handoff notes
- one-off debug scratchpads
- run-specific logs
- temporary plans
- obsolete notes that belong in `docs/archive/`

Prefer updating an existing `docs/brain/` file over creating a new one. If a new stable brain document is required, update `docs/brain/INDEX.md` so the hub stays accurate.

## Runtime Selection

Use `local` by default only when the user has not requested `srun` or `sbatch` and the work is reasonably small.

Recommend `gpu` when one or more of these are true:

- the task requires loading or serving an LLM locally
- the task is large and long-running work is expected
- model inference volume is high
- memory requirements are likely to exceed a practical CPU-only run
- accelerator-specific code is required

After tech spec approval, collect the execution settings that matter for the actual run. The user should be able to specify these directly:

- launcher choice: `local`, `srun`, or `sbatch`
- time limit
- day-based duration if relevant
- GPU count or GPU-related options
- CPU count
- memory
- partition or queue
- job name
- stdout or stderr log path
- environment setup commands if needed
- any extra launcher flags

If the task appears to require GPU but the user chooses `local`, call that risk out clearly before executing.

If GPU execution is recommended but unavailable in the current environment, stop and report the block instead of silently falling back.

## Execution Rules

- Do not invent cluster settings when the user wants to control them.
- If the user approves the tech spec but has not supplied enough execution settings for `srun` or `sbatch`, ask a concise follow-up question before running.
- Execute only the approved steps.
- Record the exact commands that were run.
- Record the files that were executed directly. Examples: scripts, notebooks, entrypoints, training jobs, shell wrappers.
- Record created, modified, and deleted files.
- If the workspace is a git repo, compare `git status --short` before and after execution.
- If the workspace is not a git repo, track touched files directly.

## Failure And Retry Rules

- On unrecoverable error, stop immediately.
- Unrecoverable examples: test failure, logic bug, schema mismatch, missing dependency that changes the approved plan, or partial output that is not clearly safe to overwrite.
- At most one automatic retry is allowed only for a transient and safe failure.
- Retry-eligible examples: temporary resource acquisition failure, transient startup timeout, or recoverable environment initialization issue.
- Any retry must be noted in the log and final report with the reason.

## Final Report

Every run must end with a report, whether it succeeds or fails.

The report must include:

- status: `success` or `failure`
- short summary
- plan path
- tech spec path
- execution settings chosen by the user
- commands run
- files executed
- files changed, created, or deleted
- output artifacts and their paths
- log paths
- validation results if successful
- brain updates performed, or an explicit note that no brain update was required
- failure point and error summary if unsuccessful
- whether an automatic retry was attempted

When explaining the execution result to the user, cover at least:

- whether the run succeeded or failed
- what files were created, modified, or deleted
- what commands or entrypoints were run
- what outputs or artifacts were produced
- how validation was performed and whether it passed
- whether `docs/brain/` was updated and why
- what the result means in practical terms
- any remaining risks, limitations, or recommended next steps

## Template Files

Reuse these template files instead of inventing fresh document structures each time:

- [assets/templates/plan-template.md](assets/templates/plan-template.md)
- [assets/templates/tech-spec-template.md](assets/templates/tech-spec-template.md)
- [assets/templates/report-template.md](assets/templates/report-template.md)

## Example Invocation

Example user request that should trigger this skill:

`$plan-techspec-executor로 데이터 전처리와 모델 실행 작업을 진행해. 먼저 plan md를 만들고 검토 후 내 승인 받기. 그 다음 tech spec md를 만들고 다시 승인 받기. 실행은 내가 srun 또는 sbatch 설정을 줄게. 성공해도 실패해도 report를 남겨.`
