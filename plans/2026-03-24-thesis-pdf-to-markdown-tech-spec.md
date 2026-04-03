# Tech Spec Template

## Title

`Thesis PDF to Markdown conversion implementation`

## Metadata

- Date: `2026-03-24`
- Slug: `thesis-pdf-to-markdown`
- Source Plan: `plans/2026-03-24-thesis-pdf-to-markdown-plan.md`
- Approval Status: `approved`

## Scope

- `Convert /home/sunsik/my_fv_project/docs/Thesis_Selin Samra.pdf into one cleaned Markdown document using a reproducible local conversion flow.`
- `Preserve the thesis reading order and major structure: front matter, table/list sections where useful, Abstract, chapters, references, appendix, and Korean summary.`
- `Normalize the most visible PDF extraction artifacts, including form-feed page breaks, standalone page numbers, full-width front-matter characters, and awkward line-level wrapping in ordinary paragraphs.`
- `Keep figure and table captions as text in the Markdown output when they appear in extracted text.`
- `Run targeted validation against the source PDF for representative sections before finalizing the Markdown output.`

## Out Of Scope

- `No attempt to recreate embedded figure images, original page layout, pagination, or Word-like formatting fidelity.`
- `No semantic rewriting of thesis content beyond minimal cleanup needed to restore readable text flow.`
- `No citation-style normalization beyond keeping the extracted references readable and correctly separated.`
- `No chapter-split Markdown export in this pass unless the user changes scope after approval.`
- `No docs/brain update, because this does not change stable project knowledge.`

## Implementation Design

`The implementation will use pdftotext as the extraction source of truth rather than trying to convert PDF directly with a general document converter. Lightweight inspection already showed that the PDF contains a usable text layer and that the main issues are not OCR failures but layout-derived artifacts: repeated title pages, page-number noise, full-width characters in the front matter, and aggressive line wrapping. That makes a structured text-cleaning pass the right level of intervention.`  

`A small dedicated local helper script will drive the conversion, tentatively /home/sunsik/my_fv_project/scripts/convert_thesis_pdf_to_markdown.py. The script will call pdftotext with layout-preserving output, split the result into page or block units, normalize Unicode using NFKC where that improves full-width characters, and then apply section-aware cleanup rules. Section-aware handling matters because body paragraphs, references, captions, title pages, and the table of contents do not reflow the same way.`  

`The front matter will be handled conservatively. Rather than preserving every decorative cover-page line exactly as extracted, the script will collapse the repeated title-page content into a readable Markdown title block and keep the essential metadata text. The table of contents, list of tables, and list of figures will be retained only if they remain readable after cleanup; otherwise they will be simplified into plain Markdown headings plus cleaned entry lines rather than dot-leader replicas.`  

`For the body text, the script will detect headings using patterns such as Abstract, Chapter N., numbered subsections like 1.1 and 3.5.1, References, Appendix, Appendix 1., and 논문요약. Consecutive ordinary text lines will be reflowed into paragraphs, while headings, captions, and explicit list-like blocks are emitted as separate Markdown blocks. Caption lines beginning with Figure N. or Table N. will be preserved as standalone text paragraphs so the thesis remains readable even without embedded images.`  

`The references section will use a stricter parser. Entries beginning with bracketed numbers such as [1] will open new reference blocks, and continuation lines will be joined into the same paragraph until the next numbered entry begins. This avoids the most common PDF-to-text problem in bibliography sections, where one citation is broken across several short lines.`  

`The appendix and Korean summary sections will be validated separately because they often expose different spacing and encoding behavior than the English body. After the scripted conversion completes, one targeted cleanup pass on the generated Markdown is expected for any remaining obvious issues that are easier to fix directly than to encode as more generic logic. The final result is intended to be a readable working Markdown document, not a fully lossless format conversion.`  

`The default output path will be /home/sunsik/my_fv_project/docs/thesis_selin_samra.md. This avoids spaces in the generated Markdown filename while keeping the source PDF untouched. If the user prefers a different output filename, that can be changed before execution without affecting the rest of the implementation design.`

## Expected File Changes

- Modify: `/home/sunsik/my_fv_project/plans/2026-03-24-thesis-pdf-to-markdown-plan.md`
- Create: `/home/sunsik/my_fv_project/plans/2026-03-24-thesis-pdf-to-markdown-tech-spec.md`
- Create: `/home/sunsik/my_fv_project/scripts/convert_thesis_pdf_to_markdown.py`
- Create: `/home/sunsik/my_fv_project/docs/thesis_selin_samra.md`
- Create after execution: `/home/sunsik/my_fv_project/reports/2026-03-24-thesis-pdf-to-markdown-report.md`
- Create if command logging is enabled during execution: `/home/sunsik/my_fv_project/logs/2026-03-24-thesis-pdf-to-markdown.log`

## Ordered Implementation Steps

1. `Create the helper conversion script that extracts text from the source PDF and normalizes Unicode, whitespace, page separators, and standalone page-number noise.`
2. `Implement structural detection rules for front matter, major headings, subsection headings, figure/table captions, references, appendix entries, and the Korean summary section.`
3. `Implement paragraph reflow rules so ordinary body text is joined into readable paragraphs while headings, captions, and bibliography entries remain separated.`
4. `Generate the initial Markdown output at /home/sunsik/my_fv_project/docs/thesis_selin_samra.md from the cleaned intermediate representation.`
5. `Perform one targeted manual cleanup pass on the generated Markdown for front-matter formatting and any obvious residual artifacts that are cheaper to fix directly than by adding more parser complexity.`
6. `Validate the final Markdown against the source PDF on representative sections: Abstract, one methods or results region, references, appendix, and the Korean summary.`
7. `Record the commands run, files changed, validation outcomes, and any remaining limitations in the final report.`

## Validation Plan

- `Extraction validation: confirm that pdftotext can extract the full thesis without OCR and that the generated Markdown is non-empty.`
- `Structure validation: confirm the Markdown contains recognizable sections for Abstract, Chapters 1-5, References, Appendix, and 논문요약 in the correct order.`
- `Artifact validation: confirm the main text flow no longer includes raw form-feed control characters or standalone numeric page-number lines in ordinary body sections.`
- `Spot-check validation: compare representative passages from Abstract, one body section, one figure-caption region, the references section, and the Korean summary against the PDF for content preservation.`
- `Reference validation: confirm numbered references remain distinct entries rather than collapsing into one paragraph or fragmenting into unreadable short lines.`
- `Output validation: confirm the final Markdown file opens cleanly as UTF-8 text and remains readable without depending on the source PDF.`

## Expected Outputs

- `/home/sunsik/my_fv_project/plans/2026-03-24-thesis-pdf-to-markdown-tech-spec.md`
- `/home/sunsik/my_fv_project/scripts/convert_thesis_pdf_to_markdown.py`
- `/home/sunsik/my_fv_project/docs/thesis_selin_samra.md`
- `/home/sunsik/my_fv_project/reports/2026-03-24-thesis-pdf-to-markdown-report.md`
- `/home/sunsik/my_fv_project/logs/2026-03-24-thesis-pdf-to-markdown.log` `if command logging is enabled during execution`

## Brain Docs To Update

- `none`

## Recommended Execution Strategy

- Launcher: `local`
- Compute Mode: `local`
- Reason: `The task is a short local document-conversion workflow using existing text-processing tools. It does not need GPU, cluster scheduling, or long runtime budgeting, and local execution keeps iteration fast for spot-checking and cleanup.`

## User Execution Settings Required Before Run

- Launcher choice: `local recommended`
- Output markdown path: `default /home/sunsik/my_fv_project/docs/thesis_selin_samra.md`
- Time limit: `not critical; expected to finish well under 10 minutes`
- Day-based duration if relevant: `not applicable`
- GPU options: `none`
- CPU count: `default shell is sufficient`
- Memory: `default shell is sufficient`
- Partition or queue: `not applicable`
- Job name: `not applicable for local execution`
- Log path: `optional; recommend /home/sunsik/my_fv_project/logs/2026-03-24-thesis-pdf-to-markdown.log if the user wants a saved execution log`
- Environment setup: `none beyond the current shell environment with pdftotext available`
- Extra launcher flags: `none`

## Approval Note

Implementation and execution have not started. Waiting for Korean approval.
