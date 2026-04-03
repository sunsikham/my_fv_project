---
name: high-trust-research
description: Use when the task needs high-confidence repo research, literature review, paper verification, or source-grounded synthesis with explicit separation between local evidence, external evidence, and inference.
---

# High-Trust Research

## Purpose

Use this skill when the goal is to produce a trustworthy research brief for the main agent.

This skill is for:

- surveying project-local material before making claims
- checking papers or official documentation on the web
- validating claims with primary sources
- connecting external literature back to the current repository
- producing a detailed report that the main agent can reuse directly

This skill is not for:

- speculative brainstorming without evidence
- writing code
- summarizing unverified web snippets as if they were authoritative

## Source Priority

Use this priority order whenever possible:

1. Repo-local primary evidence: `AGENTS.md`, `docs/brain/`, code, scripts, metadata files, reports
2. Official documentation and original project repositories
3. Paper primary sources: publisher page, arXiv page, or paper PDF
4. High-quality secondary sources only when primary sources are unavailable

If you fall below tier 3, explain why.

## Repo Workflow

1. Start from the most relevant local entrypoints.
2. Build a short map of which files are actually relevant.
3. Extract concrete facts with file paths.
4. Only after the local map is clear, use external sources to verify or extend understanding.
5. Tie every external claim back to the repository task.

For this repo, start with these anchors unless the task clearly points elsewhere:

- `AGENTS.md`
- `docs/brain/INDEX.md`
- `docs/brain/PROJECT_MAP.md`
- `docs/brain/PIPELINE_MAP.md`
- `docs/brain/ops/storage_and_sync.md`

## Web And Paper Workflow

1. Prefer official documentation when the question is about a tool, library, or product behavior.
2. Prefer the original paper page or PDF when the question is about research claims.
3. Record title, source, and what exactly was verified.
4. If a result matters for this repo, explain the practical implication for the codebase or workflow here.
5. If internet access is unavailable, state that external verification could not be performed.

## Report Contract

Return results under these headings:

1. Objective
2. Local evidence
3. External evidence
4. Integrated interpretation
5. Uncertainties and limits
6. Recommended next reads

Under `Local evidence`:

- cite exact file paths
- distinguish direct observations from interpretation

Under `External evidence`:

- cite exact URLs
- say whether the source is primary or secondary
- include exact paper title when applicable

Under `Integrated interpretation`:

- explain how the evidence changes the main agent's understanding
- keep inference separate from sourced facts

## Quality Bar

Before finishing, check:

- Did I start with local evidence where relevant?
- Did I prefer primary sources?
- Did I mark unverified points clearly?
- Did I separate facts from inference?
- Did I cite every material claim?
- Did I explain what the main agent should inspect next?
