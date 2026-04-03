#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

INLINE_SECTION_HEADINGS = [
    "3.2. Stimuli",
    "3.3. Task Design",
    "3.4. Control Experiment",
    "3.5. Large Language Models",
    "3.6. Data Analysis",
    "Future Directions",
]
ABSTRACT_TITLE = "Human Analogical Reasoning Violates Geometric Assumptions of Vector-Based Models"


@dataclass
class Block:
    page: int
    lines: list[str]

    @property
    def text(self) -> str:
        return join_wrapped_lines(self.lines)


def run_pdftotext(pdf_path: Path) -> str:
    cmd = ["pdftotext", "-layout", str(pdf_path), "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "pdftotext failed")
    return proc.stdout


def normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line.rstrip())
    line = line.replace("\u00A0", " ")
    return line


def is_page_number_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and stripped.isdigit() and len(stripped) <= 3


def page_to_blocks(page_no: int, page_text: str) -> list[Block]:
    blocks: list[Block] = []
    current: list[str] = []
    blank_run = 0
    for raw_line in page_text.splitlines():
        line = normalize_line(raw_line)
        if is_page_number_line(line):
            continue
        stripped = line.strip()
        if not stripped:
            blank_run += 1
            continue
        if current and blank_run >= 2:
            blocks.append(Block(page=page_no, lines=current))
            current = []
        current.append(stripped)
        blank_run = 0
    if current:
        blocks.append(Block(page=page_no, lines=current))
    return blocks


def join_wrapped_lines(lines: Iterable[str]) -> str:
    parts: list[str] = []
    for raw_line in lines:
        line = cleanup_inline_text(raw_line)
        if not line:
            continue
        if not parts:
            parts.append(line)
            continue
        if parts[-1].endswith("-") and re.match(r"^[A-Za-z0-9]", line):
            parts[-1] = parts[-1][:-1] + line
        else:
            parts[-1] = f"{parts[-1]} {line}"
    return cleanup_inline_text(" ".join(parts))


def cleanup_inline_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r"(\d(?:\.\d+)+)(?=[A-Za-z])", r"\1 ", text)
    return text


def flatten_blocks(pages: list[list[Block]]) -> list[Block]:
    flat: list[Block] = []
    for page_blocks in pages:
        flat.extend(page_blocks)
    return flat


def is_heading_text(text: str) -> bool:
    text = cleanup_inline_text(text)
    return bool(
        text in {"Abstract", "References", "Appendix", "논문요약"}
        or text in INLINE_SECTION_HEADINGS
        or re.match(r"^Chapter \d+\.", text)
        or re.match(r"^Appendix \d+\.", text)
        or re.match(r"^\d+\.\d+(?:\.\d+)?\.?\s+", text)
        or re.match(r"^\d+\.\s+[A-Z].{0,100}$", text)
    )


def is_caption_text(text: str) -> bool:
    text = cleanup_inline_text(text)
    return bool(re.match(r"^(Figure|Table) [A-Z]?\d+\.", text))


def is_reference_start(text: str) -> bool:
    return bool(re.match(r"^\[\d+\]", cleanup_inline_text(text)))


def render_markdown(pdf_path: Path) -> str:
    raw = run_pdftotext(pdf_path)
    pages_raw = [page for page in raw.split("\f") if page.strip()]
    page_blocks = [page_to_blocks(idx + 1, page) for idx, page in enumerate(pages_raw)]
    blocks = split_semantic_blocks(flatten_blocks(page_blocks))

    abstract_idx = find_block_index(blocks, lambda b: b.text == "Abstract")
    if abstract_idx == -1:
        raise RuntimeError("Could not find Abstract section in extracted PDF text")

    lines: list[str] = []
    lines.extend(build_front_matter(blocks[:abstract_idx]))
    lines.append("")
    lines.extend(render_body(blocks[abstract_idx:]))
    markdown = "\n".join(trim_blank_lines(lines)).strip() + "\n"
    return postprocess_markdown(markdown)


def find_block_index(blocks: list[Block], predicate) -> int:
    for idx, block in enumerate(blocks):
        if predicate(block):
            return idx
    return -1


def split_semantic_blocks(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for block in blocks:
        current_lines: list[str] = []
        current_kind = "normal"
        for raw_line in block.lines:
            line = cleanup_inline_text(raw_line)
            if not line:
                continue
            split_heading, split_rest = split_inline_heading(line)
            if split_heading:
                if current_lines:
                    out.append(Block(page=block.page, lines=current_lines))
                    current_lines = []
                out.append(Block(page=block.page, lines=[split_heading]))
                current_kind = "normal"
                if split_rest:
                    line = split_rest
                else:
                    continue
            kind = classify_line_kind(line)
            if kind == "heading":
                if current_lines:
                    out.append(Block(page=block.page, lines=current_lines))
                    current_lines = []
                out.append(Block(page=block.page, lines=[line]))
                current_kind = "normal"
                continue
            if kind in {"caption", "reference"}:
                if current_lines:
                    out.append(Block(page=block.page, lines=current_lines))
                current_lines = [line]
                current_kind = kind
                continue
            if current_lines and current_kind in {"caption", "reference"}:
                current_lines.append(line)
            else:
                if not current_lines:
                    current_lines = [line]
                    current_kind = "normal"
                else:
                    current_lines.append(line)
        if current_lines:
            out.append(Block(page=block.page, lines=current_lines))
    return out


def classify_line_kind(text: str) -> str:
    if is_reference_start(text):
        return "reference"
    if is_heading_text(text):
        return "heading"
    if is_caption_text(text):
        return "caption"
    return "normal"


def split_inline_heading(text: str) -> tuple[str | None, str | None]:
    for heading in INLINE_SECTION_HEADINGS:
        if text == heading:
            return heading, None
        prefix = f"{heading} "
        if text.startswith(prefix):
            return heading, text[len(prefix) :].strip()
    return None, None


def build_front_matter(front_blocks: list[Block]) -> list[str]:
    lines: list[str] = []
    title_block = next(
        (b for b in front_blocks if "Human Analogical Reasoning Violates" in b.text and "Vector-Based Models" in b.text),
        None,
    )
    author_block = next((b for b in front_blocks if cleanup_inline_text(b.text) == "Selin Samra"), None)
    school_block = next(
        (
            b
            for b in front_blocks
            if "The Graduate School" in b.text and "Sungkyunkwan University" in b.text
        ),
        None,
    )
    submission_block = next((b for b in front_blocks if "A Master’s Thesis Submitted" in b.text), None)
    supervised_block = next((b for b in front_blocks if "Supervised by" in b.text), None)
    approval_block = next((b for b in front_blocks if "This certifies that the Master’s Thesis of" in b.text), None)

    title = join_wrapped_lines(title_block.lines) if title_block else pdf_guess_title(front_blocks)
    lines.append(f"# {title}")
    lines.append("")

    if author_block:
        lines.append(author_block.text)
        lines.append("")

    if school_block:
        for item in school_block.lines:
            lines.append(f"{cleanup_inline_text(item)}  ")
        if lines[-1].endswith("  "):
            lines[-1] = lines[-1][:-2]
        lines.append("")

    lines.append("## Front Matter")
    lines.append("")
    lines.append("- Degree: Master’s Thesis")
    if submission_block:
        lines.append(f"- Submission: {join_wrapped_lines(submission_block.lines)}")
    if supervised_block:
        supervised = [cleanup_inline_text(x) for x in supervised_block.lines if cleanup_inline_text(x) != "Supervised by"]
        if supervised:
            lines.append(f"- Supervisors: {', '.join(supervised)}")
    if approval_block:
        lines.append(f"- Approval: {join_wrapped_lines(approval_block.lines)}")
    lines.append("")
    return trim_blank_lines(lines)


def pdf_guess_title(front_blocks: list[Block]) -> str:
    for block in front_blocks:
        if "Violates" in block.text and "Vector-Based Models" in block.text:
            return block.text
    return "Thesis"


def render_body(body_blocks: list[Block]) -> list[str]:
    lines: list[str] = []
    skip_next_title = False
    in_references = False
    pending_paragraph: str | None = None
    pending_page: int | None = None
    pending_reference: str | None = None

    def flush_paragraph() -> None:
        nonlocal pending_paragraph, pending_page
        if pending_paragraph:
            lines.append(pending_paragraph)
            lines.append("")
        pending_paragraph = None
        pending_page = None

    def flush_reference() -> None:
        nonlocal pending_reference
        if pending_reference:
            lines.append(pending_reference)
            lines.append("")
        pending_reference = None

    for block in body_blocks:
        text = cleanup_inline_text(block.text)
        if not text:
            continue

        if text == "Abstract":
            flush_paragraph()
            flush_reference()
            lines.append("## Abstract")
            lines.append("")
            skip_next_title = True
            in_references = False
            continue

        if skip_next_title:
            stripped = strip_abstract_title(text)
            skip_next_title = False
            if not stripped:
                continue
            text = stripped
            block = Block(page=block.page, lines=[stripped])

        if text == "References":
            flush_paragraph()
            flush_reference()
            lines.append("## References")
            lines.append("")
            in_references = True
            continue

        if text == "Appendix":
            flush_paragraph()
            flush_reference()
            lines.append("## Appendix")
            lines.append("")
            in_references = False
            continue

        if text == "논문요약":
            flush_paragraph()
            flush_reference()
            lines.append("## 논문요약")
            lines.append("")
            in_references = False
            continue

        if in_references:
            flush_paragraph()
            rendered = render_reference_block(block)
            if not rendered:
                continue
            if pending_reference and not is_reference_start(rendered):
                pending_reference = join_paragraph_texts(pending_reference, rendered)
            else:
                flush_reference()
                pending_reference = rendered
            continue

        flush_reference()
        rendered_lines = render_general_block(block)
        if not rendered_lines:
            continue
        if len(rendered_lines) == 1 and is_plain_paragraph(block, text):
            paragraph = rendered_lines[0]
            if pending_paragraph and should_join_paragraphs(pending_paragraph, paragraph, pending_page, block.page):
                pending_paragraph = join_paragraph_texts(pending_paragraph, paragraph)
            else:
                flush_paragraph()
                pending_paragraph = paragraph
            pending_page = block.page
            continue
        flush_paragraph()
        lines.extend(rendered_lines)
        lines.append("")

    flush_paragraph()
    flush_reference()
    return trim_blank_lines(lines)


def render_reference_block(block: Block) -> str:
    text = cleanup_inline_text(block.text)
    if is_reference_start(text):
        return text
    return text


def is_plain_paragraph(block: Block, text: str) -> bool:
    if not text:
        return False
    if is_heading_text(text) or is_caption_text(text) or is_reference_start(text):
        return False
    if looks_like_equation_block(block):
        return False
    return True


def should_join_paragraphs(prev_text: str, curr_text: str, prev_page: int | None, curr_page: int) -> bool:
    if prev_page is None or prev_page == curr_page:
        return False
    if re.search(r"[.!?:;)\]”']$", prev_text):
        return False
    return True


def join_paragraph_texts(prev_text: str, curr_text: str) -> str:
    if prev_text.endswith("-") and re.match(r"^[A-Za-z0-9]", curr_text):
        return prev_text[:-1] + curr_text
    return cleanup_inline_text(f"{prev_text} {curr_text}")


def render_general_block(block: Block) -> list[str]:
    text = cleanup_inline_text(block.text)
    first_line = cleanup_inline_text(block.lines[0])

    if first_line.startswith("Figure ") or first_line.startswith("Table "):
        return render_caption_or_table_block(block)

    if text == "Future Directions":
        return ["### Future Directions"]
    if re.match(r"^Chapter \d+\.\d+\s+", text):
        return [f"### {text}"]
    if re.match(r"^Chapter \d+\.\s+", text):
        return [f"## {text}"]
    if re.match(r"^Appendix \d+\.\s+", text):
        return [f"### {text}"]
    if re.match(r"^\d+\.\d+\.\d+\.?\s+", text):
        return [f"#### {text}"]
    if re.match(r"^\d+\.\d+\.?\s+", text):
        return [f"### {text}"]
    if is_numbered_point_heading(text):
        return [f"#### {text.rstrip(':')}"]
    if looks_like_equation_block(block):
        return ["```text", *block.lines, "```"]

    return [text]


def is_numbered_point_heading(text: str) -> bool:
    if not re.match(r"^\d+\.\s+", text):
        return False
    if len(text) > 100:
        return False
    return text.endswith(":") or text.istitle()


def looks_like_equation_block(block: Block) -> bool:
    if len(block.lines) < 1 or len(block.lines) > 4:
        return False
    joined = " ".join(block.lines)
    avg_len = sum(len(line) for line in block.lines) / len(block.lines)
    if avg_len > 45:
        return False
    if re.search(r"[=≈∩−αβθ⃗]", joined):
        return True
    return False


def strip_abstract_title(text: str) -> str:
    if text.startswith(ABSTRACT_TITLE):
        return text[len(ABSTRACT_TITLE) :].strip()
    return text


def render_caption_or_table_block(block: Block) -> list[str]:
    lines = [cleanup_inline_text(line) for line in block.lines if cleanup_inline_text(line)]
    caption_idx = next((idx for idx, line in enumerate(lines) if re.match(r"^(Figure|Table) [A-Z]?\d+\.", line)), None)
    if caption_idx is None:
        return [cleanup_inline_text(block.text)]

    rendered: list[str] = []
    if caption_idx > 0:
        rendered.append("```text")
        rendered.extend(lines[:caption_idx])
        rendered.append("```")
    rendered.append(f"*{lines[caption_idx]}*")
    if caption_idx + 1 < len(lines):
        rendered.append(join_wrapped_lines(lines[caption_idx + 1 :]))
    return rendered


def trim_blank_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        out.append(line.rstrip())
        prev_blank = is_blank
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def postprocess_markdown(markdown: str) -> str:
    markdown = markdown.replace(
        "feature compression. Keywords:",
        "feature compression.\n\nKeywords:",
    )
    markdown = markdown.replace(
        "#### 1.3.2 Formalization of Analogical Reasoning in Computational\n\nModels While ",
        "#### 1.3.2 Formalization of Analogical Reasoning in Computational Models\n\nWhile ",
    )
    return markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert the thesis PDF into a cleaned Markdown document.")
    parser.add_argument("pdf_path", type=Path, help="Source PDF path")
    parser.add_argument("output_path", type=Path, help="Output Markdown path")
    args = parser.parse_args()

    markdown = render_markdown(args.pdf_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(markdown, encoding="utf-8")

    print(f"input_pdf={args.pdf_path}")
    print(f"output_markdown={args.output_path}")
    print(f"output_bytes={args.output_path.stat().st_size}")
    print(f"output_lines={markdown.count(chr(10))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
