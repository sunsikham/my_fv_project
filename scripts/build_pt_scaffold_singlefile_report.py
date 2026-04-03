#!/usr/bin/env python3
"""Build a single-file HTML viewer for PT valid-answer scaffolds."""

from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single-file HTML report from a PT valid-answer scaffold JSON."
    )
    parser.add_argument("--scaffold_json", required=True, help="Input scaffold JSON path")
    parser.add_argument("--out_html", required=True, help="Output HTML path")
    parser.add_argument(
        "--bundle_manifest",
        default=None,
        help="Optional bundle manifest JSON path for extra context",
    )
    parser.add_argument(
        "--title",
        default="PT Candidate Review",
        help="HTML page title",
    )
    return parser.parse_args()


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _q_sort_key(qid: str) -> int:
    digits = "".join(ch for ch in str(qid) if ch.isdigit())
    return int(digits) if digits else 1_000_000


def _unit_id(q_id: str, query_source: str, query_input: str, gold_target: str) -> str:
    return f"{q_id}::{query_source}::{query_input}->{gold_target}"


def _render_q_nav(qids: Sequence[str]) -> str:
    chips = [
        f'<a class="q-chip" href="#section-{html.escape(qid)}">{html.escape(qid)}</a>'
        for qid in qids
    ]
    options = [
        f'<option value="section-{html.escape(qid)}">{html.escape(qid)}</option>'
        for qid in qids
    ]
    return (
        '<div class="nav-controls">'
        '<select id="q-nav-select" class="nav-select">'
        '<option value="">Jump to question</option>'
        + "".join(options)
        + "</select>"
        '<label class="toggle"><input id="pending-only-toggle" type="checkbox"> Pending C only</label>'
        "</div>"
        f'<div class="chip-row">{"".join(chips)}</div>'
    )


def _fmt_list(values: Sequence[object]) -> str:
    return ", ".join(str(v) for v in values) if values else "-"


def _fmt_int(value: object) -> str:
    if value is None or value == "":
        return ""
    try:
        return str(int(value))
    except Exception:
        return str(value)


def _fmt_float(value: object, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _candidate_sort_key(row: Dict[str, object]) -> Tuple[float, float, float, float, str]:
    return (
        -float(row["top1_count"]),
        -float(row["row_coverage_count"]),
        -float(row["mean_logprob"]),
        float(row["best_rank"]),
        str(row["display_candidate"]),
    )


def _build_regime_candidate_lookup(
    topk_path: str,
    *,
    top_n: int = 20,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    units: Dict[Tuple[str, str, str, str], Dict[str, Dict[str, object]]] = {}
    for row in _load_jsonl(topk_path):
        q_id = str(row["q_id"])
        query_source = str(row.get("query_source", ""))
        query_input = str(row.get("query_input", ""))
        gold_target = str(row.get("target_str", ""))
        regime = str(row.get("edge", ""))
        trial_index = int(row.get("trial_index", 0))
        shot = int(row.get("shot", 0))
        unit_key = (q_id, query_source, query_input, gold_target)
        regime_bucket = units.setdefault(unit_key, {}).setdefault(
            regime,
            {
                "row_keys": set(),
                "candidate_stats": {},
                "shots_seen": set(),
            },
        )
        row_key = (trial_index, shot, regime)
        regime_bucket["row_keys"].add(row_key)
        regime_bucket["shots_seen"].add(shot)

        candidates = list(row.get("lexical_candidates", []))
        canonicals = list(row.get("lexical_candidate_canonical_forms", []))
        token_ids = list(row.get("lexical_candidate_token_ids", []))
        logits = list(row.get("lexical_candidate_logits", []))
        logprobs = list(row.get("lexical_candidate_logprobs", []))
        probs = list(row.get("lexical_candidate_probs", []))
        vocab_ranks = list(row.get("lexical_candidate_ranks", []))
        if len(token_ids) < len(candidates):
            token_ids = token_ids + [None] * (len(candidates) - len(token_ids))
        if len(logits) < len(candidates):
            logits = logits + [None] * (len(candidates) - len(logits))
        if len(vocab_ranks) < len(candidates):
            vocab_ranks = vocab_ranks + [None] * (len(candidates) - len(vocab_ranks))

        for rank_idx, (cand, canonical, token_id, logit, logprob, prob, vocab_rank) in enumerate(
            zip(candidates, canonicals, token_ids, logits, logprobs, probs, vocab_ranks),
            start=1,
        ):
            stats = regime_bucket["candidate_stats"].get(canonical)
            if stats is None:
                stats = {
                    "canonical": canonical,
                    "surface_counts": Counter(),
                    "occurrence_count": 0,
                    "row_keys": set(),
                    "top1_count": 0,
                    "sum_logprob": 0.0,
                    "sum_prob": 0.0,
                    "best_rank": rank_idx,
                    "best_vocab_rank": (int(vocab_rank) if vocab_rank is not None else None),
                }
                regime_bucket["candidate_stats"][canonical] = stats
            stats["surface_counts"][str(cand)] += 1
            stats["occurrence_count"] += 1
            stats["row_keys"].add(row_key)
            stats["sum_logprob"] += float(logprob)
            stats["sum_prob"] += float(prob)
            stats["best_rank"] = min(int(stats["best_rank"]), rank_idx)
            if rank_idx == 1:
                stats["top1_count"] += 1
            if vocab_rank is not None:
                if stats["best_vocab_rank"] is None:
                    stats["best_vocab_rank"] = int(vocab_rank)
                else:
                    stats["best_vocab_rank"] = min(int(stats["best_vocab_rank"]), int(vocab_rank))

    lookup: Dict[str, Dict[str, Dict[str, object]]] = {}
    for unit_key, regime_map in units.items():
        unit_id = _unit_id(*unit_key)
        lookup[unit_id] = {}
        for regime, bucket in regime_map.items():
            total_rows = len(bucket["row_keys"])
            candidates: List[Dict[str, object]] = []
            for stats in bucket["candidate_stats"].values():
                total_occ = int(stats["occurrence_count"])
                display_candidate = sorted(
                    stats["surface_counts"].items(),
                    key=lambda item: (-item[1], item[0]),
                )[0][0]
                row_coverage_count = len(stats["row_keys"])
                candidates.append(
                    {
                        "display_candidate": display_candidate,
                        "row_coverage_count": row_coverage_count,
                        "row_coverage_frac": (row_coverage_count / total_rows) if total_rows else 0.0,
                        "top1_count": int(stats["top1_count"]),
                        "best_rank": int(stats["best_rank"]),
                        "best_vocab_rank": (
                            int(stats["best_vocab_rank"]) if stats["best_vocab_rank"] is not None else None
                        ),
                        "mean_logprob": float(stats["sum_logprob"]) / total_occ,
                        "mean_prob": float(stats["sum_prob"]) / total_occ,
                        "shots_seen": sorted(bucket["shots_seen"]),
                        "regimes_seen": [regime],
                    }
                )
            candidates.sort(key=_candidate_sort_key)
            lookup[unit_id][regime] = {
                "source_row_count": total_rows,
                "candidate_suggestions": candidates[:top_n],
            }
    return lookup


def _render_candidate_rows(candidates: Sequence[Dict[str, object]]) -> str:
    rows: List[str] = []
    for idx, cand in enumerate(candidates, start=1):
        coverage_count = int(cand.get("row_coverage_count", 0))
        coverage_frac = float(cand.get("row_coverage_frac", 0.0))
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td><code>{html.escape(str(cand.get('display_candidate', '')))}</code></td>"
            f"<td>{coverage_count} ({coverage_frac:.2f})</td>"
            f"<td>{_fmt_int(cand.get('top1_count'))}</td>"
            f"<td>{_fmt_int(cand.get('best_rank'))}</td>"
            f"<td>{_fmt_int(cand.get('best_vocab_rank'))}</td>"
            f"<td>{_fmt_float(cand.get('mean_logprob'))}</td>"
            f"<td>{_fmt_float(cand.get('mean_prob'))}</td>"
            f"<td>{html.escape(_fmt_list(cand.get('shots_seen', [])))}</td>"
            f"<td>{html.escape(_fmt_list(cand.get('regimes_seen', [])))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_regime_panel(regime: str, payload: Dict[str, object]) -> str:
    candidates_html = _render_candidate_rows(payload.get("candidate_suggestions", []))
    return (
        '<div class="regime-card">'
        f'<div class="regime-head"><span class="regime-name">{html.escape(regime)}</span>'
        f'<span class="regime-meta">rows {html.escape(str(payload.get("source_row_count", "")))}</span></div>'
        '<div class="table-wrap"><table class="cand-table">'
        "<thead><tr>"
        "<th>#</th><th>Candidate</th><th>Coverage</th><th>Top1</th><th>Best Rank</th>"
        "<th>Best Vocab Rank</th><th>Mean LogP</th><th>Mean Prob</th><th>Shots</th><th>Regimes</th>"
        "</tr></thead>"
        f"<tbody>{candidates_html}</tbody></table></div>"
        "</div>"
    )


def _render_unit_card(unit: Dict[str, object], regime_lookup: Dict[str, Dict[str, Dict[str, object]]]) -> str:
    status = str(unit.get("review_status", "pending"))
    query_source = str(unit.get("query_source", ""))
    is_pending_c = status != "approved" and query_source == "C"
    card_classes = ["unit-card", f"status-{status}"]
    if is_pending_c:
        card_classes.append("pending-c")
    selected_target = str(unit.get("selected_target", ""))
    gold_target = str(unit.get("gold_target", ""))
    notes = str(unit.get("notes", "") or "").strip()
    notes_html = (
        f'<div class="notes"><strong>Notes</strong>: {html.escape(notes)}</div>' if notes else ""
    )
    unit_regimes = regime_lookup.get(str(unit.get("unit_id", "")), {})
    sorted_regimes = sorted(unit_regimes.keys())
    if sorted_regimes:
        regime_html = "".join(_render_regime_panel(regime, unit_regimes[regime]) for regime in sorted_regimes)
    else:
        candidates_html = _render_candidate_rows(unit.get("candidate_suggestions", []))
        regime_html = (
            '<div class="table-wrap"><table class="cand-table">'
            "<thead><tr>"
            "<th>#</th><th>Candidate</th><th>Coverage</th><th>Top1</th><th>Best Rank</th>"
            "<th>Best Vocab Rank</th><th>Mean LogP</th><th>Mean Prob</th><th>Shots</th><th>Regimes</th>"
            "</tr></thead>"
            f"<tbody>{candidates_html}</tbody></table></div>"
        )
    return (
        f'<article class="{" ".join(card_classes)}" data-status="{html.escape(status)}" '
        f'data-query-source="{html.escape(query_source)}">'
        '<div class="unit-head">'
        f'<div class="unit-title">{html.escape(query_source)} query: '
        f'<code>{html.escape(str(unit.get("query_input", "")))}</code> '
        f'-> gold <code>{html.escape(gold_target)}</code></div>'
        f'<span class="status-pill">{html.escape(status)}</span>'
        "</div>"
        '<div class="meta-grid">'
        f'<div><strong>Selected</strong>: <code>{html.escape(selected_target)}</code></div>'
        f'<div><strong>Regimes</strong>: {html.escape(_fmt_list(unit.get("regimes_seen", [])))}</div>'
        f'<div><strong>Shots</strong>: {html.escape(_fmt_list(unit.get("shots_seen", [])))}</div>'
        f'<div><strong>Rows</strong>: {html.escape(str(unit.get("source_row_count", "")))}</div>'
        "</div>"
        f"{notes_html}"
        f'<div class="regime-grid">{regime_html}</div>'
        "</article>"
    )


def _render_question_section(
    question: Dict[str, object],
    regime_lookup: Dict[str, Dict[str, Dict[str, object]]],
) -> str:
    qid = str(question.get("q_id", ""))
    units = list(question.get("units", []))
    pending_count = sum(1 for unit in units if str(unit.get("review_status")) != "approved")
    pending_c_count = sum(
        1
        for unit in units
        if str(unit.get("review_status")) != "approved" and str(unit.get("query_source")) == "C"
    )
    unit_cards = "".join(_render_unit_card(unit, regime_lookup) for unit in units)
    return (
        f'<section id="section-{html.escape(qid)}" class="q-section">'
        '<div class="q-header">'
        f"<h2>{html.escape(qid)}</h2>"
        '<div class="q-pills">'
        f'<span class="meta-pill">units {len(units)}</span>'
        f'<span class="meta-pill">pending {pending_count}</span>'
        f'<span class="meta-pill highlight">pending C {pending_c_count}</span>'
        "</div></div>"
        f'<div class="unit-grid">{unit_cards}</div>'
        "</section>"
    )


def _build_html(
    *,
    page_title: str,
    scaffold: Dict[str, object],
    manifest: Optional[Dict[str, object]],
    regime_lookup: Dict[str, Dict[str, Dict[str, object]]],
) -> str:
    questions = sorted(scaffold.get("questions", []), key=lambda q: _q_sort_key(str(q.get("q_id", ""))))
    qids = [str(question.get("q_id", "")) for question in questions]
    q_sections = "".join(_render_question_section(question, regime_lookup) for question in questions)
    q_nav = _render_q_nav(qids)
    notes_html = ""
    if manifest:
        notes = manifest.get("notes", [])
        if notes:
            notes_html = (
                '<div class="notes-panel"><div class="notes-title">Bundle Notes</div><ul>'
                + "".join(f"<li>{html.escape(str(note))}</li>" for note in notes)
                + "</ul></div>"
            )
    title = html.escape(page_title)
    canonical_root = html.escape(str(scaffold.get("canonical_root", "")))
    source_topk = html.escape(str(scaffold.get("source_topk_jsonl", "")))
    shots = html.escape(_fmt_list(scaffold.get("shots_included", [])))
    seed_applied = html.escape(str(scaffold.get("seed_selected_target_units_applied", "0")))
    question_count = len(questions)
    unit_count = sum(len(question.get("units", [])) for question in questions)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffdf8;
      --panel-2: #f8f2e8;
      --ink: #1d1c18;
      --muted: #5d5a52;
      --line: #d8ccba;
      --accent: #9f452f;
      --accent-2: #c97c56;
      --ok: #1c6a50;
      --warn: #a45a1f;
      --shadow: 0 10px 28px rgba(0,0,0,0.06);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(159,69,47,0.10), transparent 26%),
        linear-gradient(180deg, #fbf7f1 0%, var(--bg) 100%);
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      background: #f2ece2;
      border-radius: 6px;
      padding: 2px 6px;
      word-break: break-word;
    }}
    .wrap {{
      max-width: 1540px;
      margin: 0 auto;
      padding: 24px 20px 60px;
    }}
    .hero, .sticky-inner, .q-section, .notes-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .hero {{
      border-radius: 20px;
      padding: 24px;
      margin-bottom: 18px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 38px;
      letter-spacing: -0.03em;
      line-height: 1.06;
    }}
    .hero p {{
      margin: 8px 0 0;
      color: var(--muted);
      line-height: 1.6;
      font-size: 15px;
    }}
    .hero-meta {{
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .meta-pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #efe6d8;
      color: #5a3122;
      font-size: 12px;
      font-weight: 700;
    }}
    .meta-pill.highlight {{
      background: #f8e8d4;
      color: #8d4613;
    }}
    .sticky-nav {{
      position: sticky;
      top: 0;
      z-index: 20;
      padding-bottom: 14px;
      background: linear-gradient(180deg, rgba(244,239,231,0.97), rgba(244,239,231,0.76), transparent);
      backdrop-filter: blur(6px);
    }}
    .sticky-inner {{
      border-radius: 16px;
      padding: 14px 16px;
    }}
    .nav-title {{
      font-size: 13px;
      font-weight: 800;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 10px;
    }}
    .nav-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    .nav-select {{
      padding: 10px 12px;
      min-width: 240px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fffefb;
      color: var(--ink);
    }}
    .toggle {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
      font-weight: 600;
      color: var(--muted);
    }}
    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .q-chip {{
      text-decoration: none;
      color: var(--ink);
      background: #f2eadf;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 13px;
      font-weight: 700;
    }}
    .notes-panel {{
      border-radius: 18px;
      padding: 16px 18px;
      margin-bottom: 18px;
    }}
    .notes-title {{
      font-size: 14px;
      font-weight: 800;
      color: var(--muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .notes-panel ul {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.6;
    }}
    .q-section {{
      border-radius: 18px;
      padding: 18px;
      margin-top: 16px;
    }}
    .q-header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 14px;
    }}
    .q-header h2 {{
      margin: 0;
      font-size: 28px;
      letter-spacing: -0.03em;
    }}
    .q-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .unit-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 14px;
    }}
    .unit-card {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
    }}
    .unit-card.status-approved {{
      border-left: 5px solid var(--ok);
    }}
    .unit-card.status-pending {{
      border-left: 5px solid var(--warn);
    }}
    .unit-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: start;
      margin-bottom: 10px;
    }}
    .unit-title {{
      font-size: 16px;
      font-weight: 800;
      line-height: 1.45;
    }}
    .status-pill {{
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      background: #f0e3d1;
      color: #724029;
      white-space: nowrap;
    }}
    .status-approved .status-pill {{
      background: #d9f0e6;
      color: #1c6a50;
    }}
    .status-pending .status-pill {{
      background: #f8ead6;
      color: #9a5217;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 14px;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 12px;
    }}
    .notes {{
      margin-bottom: 12px;
      color: var(--muted);
      font-size: 13px;
      white-space: pre-wrap;
    }}
    .regime-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }}
    .regime-card {{
      background: #fffefb;
      border: 1px solid #eadfce;
      border-radius: 14px;
      padding: 12px;
    }}
    .regime-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }}
    .regime-name {{
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--accent);
    }}
    .regime-meta {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .cand-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      background: #fffefb;
      border-radius: 12px;
      overflow: hidden;
    }}
    .cand-table th, .cand-table td {{
      border-bottom: 1px solid #eadfce;
      padding: 8px 9px;
      text-align: left;
      vertical-align: top;
    }}
    .cand-table th {{
      position: sticky;
      top: 0;
      background: #f7efe3;
      z-index: 1;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
    }}
    .hidden-by-filter {{
      display: none !important;
    }}
    @media (max-width: 900px) {{
      .unit-grid {{
        grid-template-columns: 1fr;
      }}
      .hero h1 {{
        font-size: 32px;
      }}
      .q-header h2 {{
        font-size: 24px;
      }}
      .meta-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>{title}</h1>
      <p>One-stop viewer for PT candidate review. This HTML is for manual inspection convenience, not a final unified PT run.</p>
      <p><strong>Canonical Root</strong>: <code>{canonical_root}</code></p>
      <p><strong>Source Top-k</strong>: <code>{source_topk}</code></p>
      <div class="hero-meta">
        <span class="meta-pill">questions {question_count}</span>
        <span class="meta-pill">units {unit_count}</span>
        <span class="meta-pill">shots {shots}</span>
        <span class="meta-pill">seed applied {seed_applied}</span>
      </div>
    </section>
    <div class="sticky-nav">
      <div class="sticky-inner">
        <div class="nav-title">Question Navigation</div>
        {q_nav}
      </div>
    </div>
    {notes_html}
    {q_sections}
  </div>
  <script>
    const select = document.getElementById('q-nav-select');
    if (select) {{
      select.addEventListener('change', (event) => {{
        const id = event.target.value;
        if (!id) return;
        const el = document.getElementById(id);
        if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
      }});
    }}
    const toggle = document.getElementById('pending-only-toggle');
    if (toggle) {{
      toggle.addEventListener('change', () => {{
        const onlyPendingC = toggle.checked;
        document.querySelectorAll('.unit-card').forEach((card) => {{
          const keep = !onlyPendingC || card.classList.contains('pending-c');
          card.classList.toggle('hidden-by-filter', !keep);
        }});
        document.querySelectorAll('.q-section').forEach((section) => {{
          const visibleCards = section.querySelectorAll('.unit-card:not(.hidden-by-filter)');
          section.classList.toggle('hidden-by-filter', visibleCards.length === 0);
        }});
      }});
    }}
  </script>
</body>
</html>
"""


def main() -> int:
    args = _parse_args()
    scaffold = _load_json(args.scaffold_json)
    manifest = _load_json(args.bundle_manifest) if args.bundle_manifest else None
    regime_lookup = _build_regime_candidate_lookup(str(scaffold.get("source_topk_jsonl", "")))
    html_text = _build_html(
        page_title=args.title,
        scaffold=scaffold,
        manifest=manifest,
        regime_lookup=regime_lookup,
    )
    out_path = Path(args.out_html).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    print(f"saved_html={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
