---
name: ccn-extended-abstract-review
description: Use when reviewing a full CCN extended abstract for logical flow, section structure, reader accessibility, evidence-to-claim fit, and compliance with the local CCN template and submission constraints.
---

# CCN Extended Abstract Review

## Purpose

Use this skill when the task is to review or improve a full CCN extended abstract, not just the 300-word abstract block.

This skill is for:

- evaluating the logic of the full 2-page story
- checking whether sections support a single clear contribution
- finding weak transitions, overclaim, and missing reader guidance
- checking compliance against the local CCN extended abstract template
- proposing concrete rewrites at the sentence, paragraph, or section level

This skill is not for:

- validating whether the underlying experiments are scientifically correct
- doing broad literature review unless explicitly asked
- polishing prose without first diagnosing the document-level argument

## Local Anchors

Start from these files before making format claims:

- `ccn/ccn-template-main 2/README.md`
- `ccn/ccn-template-main 2/ccn_extended_abstract.tex`

Relevant constraints from the local template:

- the abstract must be identical to the text version submitted in the web form
- the abstract should not exceed 300 words
- the extended abstract text, tables, and figures can be no longer than 2 pages, excluding references
- the audience is interdisciplinary
- supplementary technical appendices are not permitted
- double-blind anonymity must be preserved

## Review Priorities

Evaluate in this order:

1. Blocking format or anonymity issues
2. Core argument clarity
3. Section-level logic and flow
4. Evidence-to-claim fit
5. Reader accessibility for an interdisciplinary CCN audience
6. Sentence-level clarity and compression

Do not let minor wording feedback overshadow structural problems.

## Rubric

Score each dimension on a 1-5 scale.

- `problem framing`: Is the question and its importance clear early?
- `core contribution`: Is the paper's main claim stable and easy to restate?
- `logic flow`: Do sections and paragraphs build a coherent argument?
- `evidence-claim fit`: Do the presented results justify the interpretation?
- `method sufficiency`: Is there enough method detail to trust the results?
- `result prioritization`: Are the most important results foregrounded?
- `reader accessibility`: Can a broad CCN audience follow the story?
- `sentence clarity`: Are sentences direct, compact, and easy to parse?
- `figure/table integration`: Do visuals earn their space and support the text?
- `format safety`: Does the draft appear consistent with CCN constraints?

Interpret the scale as:

- `5`: submission-ready
- `4`: minor revision
- `3`: moderate revision needed
- `2`: major restructuring needed
- `1`: not viable without rewrite

## Structure Guidance

For full extended abstracts, treat the ideal progression as:

1. Problem and motivation
2. Approach or method
3. Main result
4. Interpretation and implication

The exact section names can vary, but the reader should never wonder:

- what question the paper is asking
- what was actually done
- what the main result is
- why the result matters

For the abstract block specifically, check that the same progression is compressed into one short paragraph.

## Review Workflow

1. Read the full draft, not only the abstract.
2. Map each section or paragraph to its argumentative role.
3. Identify the paper's one-sentence takeaway.
4. Check whether each major paragraph advances that takeaway.
5. Flag sections that are off-topic, redundant, or too compressed to be credible.
6. Check the CCN template constraints listed above.
7. Only after structure is clear, give sentence-level rewrite advice.

If a draft is in LaTeX, pay attention to:

- anonymization in title, authors, acknowledgments, figures, and links
- whether the abstract block is under 300 words
- whether figures or tables are likely to crowd the 2-page limit

## Report Contract

Return results under these headings:

1. Overall verdict
2. Score summary
3. Blocking issues
4. Major revision priorities
5. Section-by-section review
6. Concrete rewrite suggestions
7. Optional revised text

Under `Blocking issues`:

- list format violations, anonymity risks, and severe logic failures first

Under `Major revision priorities`:

- give the top 3-5 changes that most improve submission quality

Under `Section-by-section review`:

- explain each section's role
- say where the flow breaks
- note whether the section is necessary, underdeveloped, or overloaded

Under `Concrete rewrite suggestions`:

- prefer actionable edits over vague stylistic advice
- suggest reordering paragraphs when needed
- point out sentences that should be cut, merged, or split

## Style Rules For The Reviewer

- Findings first, praise second
- Be direct and specific
- Distinguish structural problems from polish
- Do not call something unclear without saying why
- Do not accuse the text of overclaim without naming the mismatch
- When possible, provide better wording, not just criticism

## Optional Tight Checks

Use lightweight checks when helpful:

- estimate abstract word count
- estimate whether the draft is likely to overflow 2 pages
- scan for explicit self-identifying text that breaks double-blind review

If page count cannot be verified from the available material, say `not verified`.
