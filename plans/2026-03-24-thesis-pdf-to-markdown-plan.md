# Plan Template

## Title

`Thesis PDF to Markdown conversion`

## Metadata

- Date: `2026-03-24`
- Slug: `thesis-pdf-to-markdown`
- Approval Status: `approved`

## Objective

`Convert /home/sunsik/my_fv_project/docs/Thesis_Selin Samra.pdf into a readable Markdown document that preserves the thesis structure and core text content while removing obvious PDF extraction artifacts.`

## Current Context

`The source thesis PDF already exists at /home/sunsik/my_fv_project/docs/Thesis_Selin Samra.pdf. Lightweight inspection shows it is a 70-page tagged PDF created from Microsoft Word, so OCR is not required. Local tools already available in the environment include pdftotext and pandoc. Sample extraction indicates the main body text is recoverable, while the title pages and table-of-contents area contain some full-width characters, page-number artifacts, and line-wrap noise that will need normalization during Markdown structuring.`

## Assumptions

- `The user wants a single Markdown deliverable under /home/sunsik/my_fv_project/docs/ unless they later request a chapter-split format.`
- `Text fidelity and readable structure are the goal; exact page-layout parity with the PDF is not required.`
- `Figure and table captions should be preserved as text, but embedded figure images do not need to be recreated unless the user asks for that separately.`
- `Intermediate extraction files can remain temporary unless keeping a repo-local text artifact becomes useful during implementation.`

## Inputs And Dependencies

- `Source PDF at /home/sunsik/my_fv_project/docs/Thesis_Selin Samra.pdf`
- `Local text extraction tool pdftotext`
- `Local document conversion tool pandoc`
- `Standard shell text-processing utilities for cleanup and restructuring`
- `User approval for the conversion plan and follow-on tech spec`

## Proposed Steps

1. `Map the thesis structure from the PDF by checking front matter, chapter boundaries, appendix, references, and the Korean summary so the Markdown output can follow the original document order.`
2. `Extract the full thesis text into a working representation that preserves enough layout information to distinguish headings, captions, references, and ordinary paragraphs.`
3. `Normalize extraction artifacts such as form-feed page breaks, standalone page numbers, full-width digits or letters in front matter, excessive line breaks, and noisy table-of-contents leader lines.`
4. `Rebuild the document as Markdown with a clean title/front-matter section, chapter and subsection headings, readable paragraphs, figure or table captions as text blocks, references, appendix content, and the final Korean summary section.`
5. `Spot-check representative sections against the source PDF, including Abstract, one methods/results region, References, and the Korean summary, and fix the most visible structural or formatting errors.`
6. `Write the final Markdown artifact under /home/sunsik/my_fv_project/docs/ and capture the conversion decisions, validations, and any remaining limitations in the required final report after execution.`

## Risks And Blockers

- `The title pages and contents pages already show full-width characters and numbering artifacts, so front-matter cleanup may require targeted manual normalization rather than only mechanical conversion.`
- `Paragraphs, captions, and references may be split across multiple short lines in the extracted text, which can create awkward Markdown unless reflow is handled carefully.`
- `If the user later wants near-visual parity with the PDF, plain Markdown will not preserve pagination, exact spacing, or original figure placement.`
- `The Korean summary section may expose encoding or spacing issues that differ from the English body text and need separate validation.`

## Recommended Compute Mode

- Mode: `local`
- Why: `This is a lightweight document-conversion workflow that relies on existing local text-processing tools and does not require GPU resources or long-running compute.`

## Expected Outputs

- `Approved plan artifact at /home/sunsik/my_fv_project/plans/2026-03-24-thesis-pdf-to-markdown-plan.md`
- `Follow-on tech spec at /home/sunsik/my_fv_project/plans/2026-03-24-thesis-pdf-to-markdown-tech-spec.md after plan approval`
- `If later executed: final Markdown thesis file under /home/sunsik/my_fv_project/docs/`
- `Final execution report under /home/sunsik/my_fv_project/reports/`

## Success Criteria

- `The final Markdown contains the thesis content in the correct order with recognizable front matter, chapter headings, references, appendix, and Korean summary sections.`
- `The output is readable as Markdown and no longer contains obvious page-break control characters or standalone PDF page numbers in the main flow.`
- `Representative spot checks confirm that the extracted text matches the source PDF content without major omissions in the validated sections.`
- `Remaining limitations, if any, are explicitly documented so the user can judge whether additional cleanup is needed.`

## Brain Impact

- Brain impact: `none`
- Why: `This work converts a single thesis document into Markdown but does not change stable project knowledge, pipeline behavior, storage policy, or operational guidance that belongs in /home/sunsik/my_fv_project/docs/brain/.`

## Approval Note

Substantial execution has not started. Waiting for Korean approval.
