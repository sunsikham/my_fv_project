#!/usr/bin/env python3
"""Build a single self-contained multi-Q HTML report for BD shuffle results."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import mimetypes
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REGIME_ORDER = {
    "D": ["BDBDBD_D", "BD_SHUF_D1_D", "BD_SHUF_D2_D", "BD_SHUF_D3_D", "BD_SHUF_D4_D", "BD_SHUF_D5_D"],
    "B": ["DBDBDB_B", "BD_SHUF_B1_B", "BD_SHUF_B2_B", "BD_SHUF_B3_B", "BD_SHUF_B4_B", "BD_SHUF_B5_B"],
}

BBB_REFERENCE_LABEL = "BBBBBB_D"
BBB_REFERENCE_PATTERN = "BBBBBBBBB"

BASELINE_BY_SIDE = {
    "D": "BDBDBD_D",
    "B": "DBDBDB_B",
}

SIDE_DISPLAY = {
    "D": "BDBDBD_D Family",
    "B": "DBDBDB_B Family",
}

QUERY_SOURCE_LABEL = {
    "D": "D query",
    "B": "B query",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single-file BD shuffle HTML report.")
    parser.add_argument("--run_dir", required=True, help="Canonical scratch run directory")
    parser.add_argument("--out_html", required=True, help="Output single-file HTML path")
    parser.add_argument(
        "--role_map_md",
        default="/home/sunsik/my_fv_project/docs/TRIANGLE_RELATION_ROLE_MAP.md",
        help="Markdown file that maps q_id to B/D relation role names",
    )
    parser.add_argument(
        "--page_title",
        default="BD Shuffle Multi-Q Target Probability Report",
        help="HTML page title",
    )
    parser.add_argument(
        "--bbb_reference_run_dir",
        default="",
        help="Optional pt_unified run directory that provides a BBB baseline reference for D-side panels",
    )
    parser.add_argument(
        "--bbb_reference_shot",
        default="9",
        help="Shot selector to use when aggregating the BBB baseline reference",
    )
    parser.add_argument(
        "--bbb_reference_regime_id",
        default="BASE_BD",
        help="Regime id to aggregate from the reference pt_unified shot sweep",
    )
    parser.add_argument(
        "--hero_image",
        default="",
        help="Optional image path to embed at the top of the report",
    )
    return parser.parse_args()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row in {path}")
        return list(reader)


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _image_data_uri(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _fmt_float(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "NA"


def _fmt_signed(value: object, digits: int = 4) -> str:
    try:
        num = float(value)
    except Exception:
        return "NA"
    return f"{num:+.{digits}f}"


def _space_pattern(pattern: str) -> str:
    return " ".join(list(pattern))


def _count_sources(pattern: str) -> str:
    return f"B={pattern.count('B')} D={pattern.count('D')}"


def _qid_sort_key(q_id: str) -> Tuple[int, str]:
    match = re.fullmatch(r"Q(\d+)", q_id)
    if match:
        return (int(match.group(1)), q_id)
    return (10**9, q_id)


def _meta_qid_order(meta: Dict[str, object]) -> List[str]:
    raw = str(meta.get("qid", "") or "").strip()
    if not raw:
        return []
    out: List[str] = []
    seen = set()
    for token in raw.split(","):
        q_id = token.strip()
        if not q_id or q_id in seen:
            continue
        seen.add(q_id)
        out.append(q_id)
    return out


def _case_delta_lookup(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    lookup: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in rows:
        key = (str(row["q_id"]), str(row["query_side"]), str(row["regime_id"]))
        lookup[key] = row
    return lookup


def _ordered_qids(regime_rows: Sequence[Dict[str, str]], meta_order: Sequence[str]) -> List[str]:
    found = {str(row["q_id"]) for row in regime_rows}
    out: List[str] = []
    for q_id in meta_order:
        if q_id in found and q_id not in out:
            out.append(q_id)
    for q_id in sorted(found, key=_qid_sort_key):
        if q_id not in out:
            out.append(q_id)
    return out


def _group_regime_rows(regime_rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    for row in regime_rows:
        q_id = str(row["q_id"])
        side = str(row["query_side"])
        grouped.setdefault(q_id, {}).setdefault(side, []).append(row)
    for q_id, side_map in grouped.items():
        for side, rows in side_map.items():
            side_map[side] = sorted(rows, key=lambda row: REGIME_ORDER[side].index(str(row["regime_id"])))
    return grouped


def _load_bbb_reference_lookup(
    *,
    run_dir: Path,
    shot: str,
    regime_id: str,
) -> Dict[str, Dict[str, object]]:
    sweep_path = run_dir / "pt_unified_shot_sweep.csv"
    if not sweep_path.exists():
        raise FileNotFoundError(f"missing BBB reference sweep csv: {sweep_path}")

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in _read_csv_rows(sweep_path):
        if str(row.get("regime_id", "")) != regime_id:
            continue
        if str(row.get("shot", "")) != str(shot):
            continue
        if str(row.get("edge_group", "")) != "BD":
            continue
        if str(row.get("query_source", "")) != "D":
            continue
        q_id = str(row.get("q_id", "")).strip()
        if not q_id:
            continue
        grouped.setdefault(q_id, []).append(row)

    lookup: Dict[str, Dict[str, object]] = {}
    for q_id, rows in grouped.items():
        query_inputs = {str(row["query_input"]) for row in rows}
        target_strs = {str(row["target_str"]) for row in rows}
        patterns = {str(row.get("demo_pattern", "")).strip() for row in rows if str(row.get("demo_pattern", "")).strip()}
        if len(query_inputs) != 1 or len(target_strs) != 1:
            raise ValueError(f"inconsistent BBB reference query/target rows for {q_id}")
        if len(patterns) > 1:
            raise ValueError(f"inconsistent BBB reference demo patterns for {q_id}: {sorted(patterns)}")
        pattern = next(iter(patterns), BBB_REFERENCE_PATTERN)
        if set(pattern) != {"B"}:
            raise ValueError(f"expected all-B demo pattern for {q_id}, got {pattern}")
        lookup[q_id] = {
            "q_id": q_id,
            "query_input": next(iter(query_inputs)),
            "target_str": next(iter(target_strs)),
            "layout_pattern": pattern,
            "mean_target_prob": sum(float(row["target_prob_raw"]) for row in rows) / len(rows),
            "n_trials": len(rows),
            "source_run_dir": str(run_dir),
            "shot": str(shot),
            "regime_id": regime_id,
        }
    return lookup


def _parse_role_map(markdown_text: str) -> Dict[str, Dict[str, str]]:
    role_map: Dict[str, Dict[str, str]] = {}
    current_qid: str | None = None
    q_header = re.compile(r"^##\s+(Q\d+)\s*$")
    role_line = re.compile(r"^-\s+`([ABD]):\s+(.+?)`\s*$")

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        match_q = q_header.match(line)
        if match_q:
            current_qid = match_q.group(1)
            role_map.setdefault(current_qid, {})
            continue
        if current_qid is None:
            continue
        match_role = role_line.match(line)
        if not match_role:
            continue
        side = match_role.group(1)
        role_text = match_role.group(2)
        role_map[current_qid][side] = role_text
    return role_map


def _family_payload(
    *,
    q_id: str,
    side: str,
    rows: Sequence[Dict[str, str]],
    delta_lookup: Dict[Tuple[str, str, str], Dict[str, str]],
    bbb_reference_lookup: Dict[str, Dict[str, object]] | None = None,
) -> Dict[str, object]:
    if not rows:
        raise ValueError(f"Missing rows for q_id={q_id} side={side}")

    baseline_id = BASELINE_BY_SIDE[side]
    baseline_row = next((row for row in rows if str(row["regime_id"]) == baseline_id), None)
    if baseline_row is None:
        raise ValueError(f"Missing baseline row for q_id={q_id} side={side}")

    baseline_mean_target_prob = float(baseline_row["mean_target_prob"])
    bars: List[Dict[str, object]] = []
    if side == "D" and bbb_reference_lookup:
        reference_row = bbb_reference_lookup.get(q_id)
        if reference_row is not None:
            if str(reference_row["query_input"]) != str(baseline_row["query_input"]):
                raise ValueError(f"BBB reference query mismatch for {q_id}")
            if str(reference_row["target_str"]) != str(baseline_row["target_str"]):
                raise ValueError(f"BBB reference target mismatch for {q_id}")
            bars.append(
                {
                    "regime_id": BBB_REFERENCE_LABEL,
                    "case_kind": "reference",
                    "layout_pattern": str(reference_row["layout_pattern"]),
                    "layout_spaced": _space_pattern(str(reference_row["layout_pattern"])),
                    "mean_target_prob": float(reference_row["mean_target_prob"]),
                    "delta_mean_target_prob": float(reference_row["mean_target_prob"]) - baseline_mean_target_prob,
                }
            )
    for row in rows:
        regime_id = str(row["regime_id"])
        case_kind = str(row["case_kind"])
        layout = str(row["layout_pattern"])
        delta_row = delta_lookup.get((q_id, side, regime_id))
        bars.append(
            {
                "regime_id": regime_id,
                "case_kind": case_kind,
                "layout_pattern": layout,
                "layout_spaced": _space_pattern(layout),
                "mean_target_prob": float(row["mean_target_prob"]),
                "delta_mean_target_prob": (
                    float(delta_row["delta_mean_target_prob"])
                    if delta_row is not None and case_kind == "shuffled"
                    else None
                ),
            }
        )

    return {
        "q_id": q_id,
        "side": side,
        "family_title": SIDE_DISPLAY[side],
        "baseline_id": baseline_id,
        "query_input": str(baseline_row["query_input"]),
        "target_str": str(baseline_row["target_str"]),
        "query_source_label": QUERY_SOURCE_LABEL[side],
        "baseline_layout": str(baseline_row["layout_pattern"]),
        "demo_count_text": _count_sources(str(baseline_row["layout_pattern"])),
        "bars": bars,
    }


def _render_bar_rows(bars: Sequence[Dict[str, object]]) -> str:
    parts: List[str] = []
    for bar in bars:
        prob = float(bar["mean_target_prob"])
        width_pct = max(0.0, min(100.0, prob * 100.0))
        case_kind = str(bar["case_kind"])
        if case_kind == "regular":
            fill_class = "fill baseline"
        elif case_kind == "reference":
            fill_class = "fill reference"
        else:
            fill_class = "fill shuffled"
        delta = bar["delta_mean_target_prob"]
        delta_html = ""
        if case_kind == "reference":
            delta_class = "delta-pos" if float(delta) >= 0 else "delta-neg"
            delta_html = (
                "<div class='delta delta-ref'>BBBBBB_D reference</div>"
                f"<div class='delta {delta_class}'>Δ vs base {_fmt_signed(delta)}</div>"
            )
        elif delta is not None:
            delta_class = "delta-pos" if float(delta) >= 0 else "delta-neg"
            delta_html = f"<div class='delta {delta_class}'>Δ vs base {_fmt_signed(delta)}</div>"
        else:
            delta_html = "<div class='delta delta-base'>baseline</div>"
        parts.append(
            "<div class='bar-row'>"
            "<div class='bar-label'>"
            f"<div class='regime-id'>{html.escape(str(bar['regime_id']))}</div>"
            f"<div class='layout'>{html.escape(str(bar['layout_spaced']))}</div>"
            "</div>"
            "<div class='bar-track'>"
            f"<div class='{fill_class}' style='width:{width_pct:.2f}%'></div>"
            "</div>"
            "<div class='bar-stats'>"
            f"<div class='value'>{_fmt_float(prob)}</div>"
            f"{delta_html}"
            "</div>"
            "</div>"
        )
    return "".join(parts)


def _render_family_panel(family: Dict[str, object]) -> str:
    title = (
        f"{family['q_id']} · {family['family_title']} · "
        f"query={family['query_input']} · target={family['target_str']}"
    )
    return (
        "<section class='family-panel'>"
        f"<h3>{html.escape(title)}</h3>"
        "<div class='family-meta'>"
        f"<span class='meta-pill'>baseline={html.escape(str(family['baseline_id']))}</span>"
        f"<span class='meta-pill'>{html.escape(str(family['query_source_label']))}</span>"
        f"<span class='meta-pill'>demo counts {html.escape(str(family['demo_count_text']))}</span>"
        "</div>"
        f"<div class='baseline-line'><strong>Baseline Layout</strong>: "
        f"<code>{html.escape(_space_pattern(str(family['baseline_layout'])))}</code></div>"
        "<div class='axis-label'>Mean Target Probability</div>"
        f"{_render_bar_rows(family['bars'])}"
        "</section>"
    )


def _render_q_nav(qids: Sequence[str]) -> str:
    chips = "".join(
        f"<a class='q-chip' href='#section-{html.escape(q_id)}'>{html.escape(q_id)}</a>" for q_id in qids
    )
    options = "".join(
        f"<option value='section-{html.escape(q_id)}'>{html.escape(q_id)}</option>" for q_id in qids
    )
    return (
        "<div class='sticky-nav'>"
        "<div class='sticky-inner'>"
        "<div class='nav-title'>Jump To Q</div>"
        "<label class='nav-select-wrap'>"
        "<span class='sr-only'>Select q_id</span>"
        f"<select class='nav-select' onchange=\"if(this.value){{location.hash=this.value;}}\">{options}</select>"
        "</label>"
        f"<div class='chip-row'>{chips}</div>"
        "</div>"
        "</div>"
    )


def _render_q_section(
    *,
    q_id: str,
    family_d: Dict[str, object],
    family_b: Dict[str, object],
    relation_roles: Dict[str, str],
) -> str:
    summary_text = (
        f"{q_id} · D: {family_d['query_input']} -> {family_d['target_str']} · "
        f"B: {family_b['query_input']} -> {family_b['target_str']}"
    )
    role_b = relation_roles.get("B", "unknown")
    role_d = relation_roles.get("D", "unknown")
    return (
        f"<details class='q-section' id='section-{html.escape(q_id)}' open>"
        f"<summary>{html.escape(summary_text)}</summary>"
        "<div class='q-body'>"
        "<div class='context-grid'>"
        "<div class='context-card'>"
        f"<h4>{html.escape(q_id)} Context</h4>"
        "<p class='context-line'><strong>Shuffle Rule</strong>: same q_id, same query row, same demo multiset, "
        "same B/D counts, only order changes.</p>"
        "<p class='context-line'><strong>Relation Roles</strong>: "
        f"B: {html.escape(role_b)} · D: {html.escape(role_d)}</p>"
        "</div>"
        "<div class='context-card'>"
        f"<h4>{html.escape(q_id)} Targets</h4>"
        f"<p class='context-line'><strong>D-side</strong>: query={html.escape(str(family_d['query_input']))} "
        f"target={html.escape(str(family_d['target_str']))}</p>"
        f"<p class='context-line'><strong>B-side</strong>: query={html.escape(str(family_b['query_input']))} "
        f"target={html.escape(str(family_b['target_str']))}</p>"
        "</div>"
        "</div>"
        "<div class='family-grid'>"
        f"{_render_family_panel(family_d)}"
        f"{_render_family_panel(family_b)}"
        "</div>"
        "</div>"
        "</details>"
    )


def _build_html(
    *,
    page_title: str,
    canonical_root: str,
    run_meta: Dict[str, object],
    q_sections: str,
    q_nav: str,
    bbb_reference_root: str,
    hero_image_uri: str,
) -> str:
    run_id = str(run_meta.get("run_id", "unknown"))
    shot_list = str(run_meta.get("shot_list", "unknown"))
    n_trials = str(run_meta.get("n_trials", "unknown"))
    model = str(run_meta.get("model", "unknown"))
    has_bbb_reference = bool(bbb_reference_root)
    bbb_reference_pill = '<span class="pill">BBBBBB_D Reference</span>' if has_bbb_reference else ""
    bbb_reference_text = ""
    reading_rule = (
        "for each q_id, compare the baseline target probability against five shuffled layouts separately for the "
        "<code>BDBDBD_D</code> family and the <code>DBDBDB_B</code> family. Full layout strings are shown on every bar."
    )
    if has_bbb_reference:
        bbb_reference_text = (
            f'<p><strong>BBBBBB_D Reference Run</strong>: <code>{html.escape(bbb_reference_root)}</code></p>'
        )
        reading_rule = (
            "for each q_id, compare the D-side <code>BBBBBB_D</code> reference against the alternating "
            "<code>BDBDBD_D</code> baseline and five shuffled layouts. The B-side still compares "
            "<code>DBDBDB_B</code> against its five shuffled layouts. Full layout strings are shown on every bar."
        )
    hero_image_html = ""
    if hero_image_uri:
        hero_image_html = (
            '<div class="hero-image-wrap">'
            f'<img class="hero-image" src="{hero_image_uri}" alt="Report hero image">'
            "</div>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page_title)}</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: #fffdf7;
      --panel-2: #f8f2e6;
      --ink: #1f1f1c;
      --muted: #5b5850;
      --line: #d9cfbf;
      --base: #9a442f;
      --base-2: #cb7a55;
      --shuf: #1f5b67;
      --shuf-2: #53919b;
      --good: #1a6a4f;
      --bad: #a03434;
      --shadow: 0 10px 30px rgba(0,0,0,0.06);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(154,68,47,0.12), transparent 24%),
        linear-gradient(180deg, #fbf8f2 0%, var(--bg) 100%);
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }}
    .wrap {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px 22px 60px;
    }}
    .hero, .q-section, .sticky-inner {{
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .hero {{
      border-radius: 20px;
      padding: 24px;
      margin-bottom: 18px;
    }}
    .hero-image-wrap {{
      margin-bottom: 18px;
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #fffefb, #f3eadc);
    }}
    .hero-image {{
      display: block;
      width: 100%;
      max-height: 460px;
      object-fit: contain;
      background: #f7f0e4;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 38px;
      letter-spacing: -0.03em;
      line-height: 1.08;
    }}
    .hero p {{
      margin: 10px 0 0;
      font-size: 15px;
      line-height: 1.6;
      color: var(--muted);
    }}
    .hero-meta {{
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .pill, .meta-pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #efe6d8;
      color: #5b2f21;
      font-size: 12px;
      font-weight: 700;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      background: #f2ece3;
      border-radius: 6px;
      padding: 2px 6px;
      word-break: break-all;
    }}
    .sticky-nav {{
      position: sticky;
      top: 0;
      z-index: 30;
      padding-bottom: 14px;
      background: linear-gradient(180deg, rgba(246,241,232,0.96), rgba(246,241,232,0.72), transparent);
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
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .nav-select {{
      width: 100%;
      max-width: 260px;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fffefb;
      color: var(--ink);
    }}
    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .q-chip {{
      text-decoration: none;
      border-radius: 999px;
      padding: 7px 11px;
      background: #f2ebdf;
      color: var(--shuf);
      font-weight: 700;
      font-size: 13px;
      border: 1px solid #ddd1c0;
    }}
    .q-section {{
      border-radius: 18px;
      margin-top: 14px;
      overflow: hidden;
    }}
    .q-section summary {{
      cursor: pointer;
      list-style: none;
      padding: 18px 22px;
      font-size: 19px;
      font-weight: 800;
      background: linear-gradient(180deg, #fffefb, #f7f0e4);
      border-bottom: 1px solid var(--line);
    }}
    .q-section summary::-webkit-details-marker {{
      display: none;
    }}
    .q-body {{
      padding: 18px 22px 22px;
    }}
    .context-grid, .family-grid {{
      display: grid;
      gap: 16px;
    }}
    .context-grid {{
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      margin-bottom: 16px;
    }}
    .family-grid {{
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
    }}
    .context-card, .family-panel {{
      background: linear-gradient(180deg, #fffefb, var(--panel-2));
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
    }}
    .context-card h4, .family-panel h3 {{
      margin: 0 0 10px;
      line-height: 1.2;
    }}
    .family-panel h3 {{
      font-size: 20px;
      letter-spacing: -0.02em;
    }}
    .context-line {{
      margin: 8px 0;
      line-height: 1.55;
      color: var(--muted);
      font-size: 14px;
    }}
    .family-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .baseline-line {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    .axis-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin-bottom: 8px;
      font-weight: 800;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(220px, 290px) minmax(160px, 1fr) 96px;
      gap: 12px;
      align-items: center;
      margin: 12px 0;
    }}
    .bar-label {{
      min-width: 0;
    }}
    .regime-id {{
      font-size: 13px;
      font-weight: 800;
      color: var(--ink);
      margin-bottom: 3px;
    }}
    .layout {{
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
      word-break: break-word;
    }}
    .bar-track {{
      width: 100%;
      height: 18px;
      border-radius: 999px;
      background: #e9dfd2;
      overflow: hidden;
      position: relative;
    }}
    .fill {{
      height: 100%;
      border-radius: 999px;
    }}
    .fill.baseline {{
      background: linear-gradient(90deg, var(--base), var(--base-2));
    }}
    .fill.reference {{
      background: linear-gradient(90deg, #706428, #b3a04a);
    }}
    .fill.shuffled {{
      background: linear-gradient(90deg, var(--shuf), var(--shuf-2));
    }}
    .bar-stats {{
      text-align: right;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
    }}
    .value {{
      font-size: 13px;
      font-weight: 800;
      color: var(--ink);
    }}
    .delta {{
      margin-top: 4px;
    }}
    .delta-base {{
      color: var(--muted);
    }}
    .delta-ref {{
      color: #6b5f26;
    }}
    .delta-pos {{
      color: var(--good);
    }}
    .delta-neg {{
      color: var(--bad);
    }}
    .sr-only {{
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }}
    @media (max-width: 960px) {{
      .family-grid {{
        grid-template-columns: 1fr;
      }}
      .bar-row {{
        grid-template-columns: 1fr;
      }}
      .bar-stats {{
        text-align: left;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      {hero_image_html}
      <div class="hero-meta">
        <span class="pill">Single File</span>
        <span class="pill">Multi-Q</span>
        <span class="pill">Target Probability Focus</span>
        {bbb_reference_pill}
      </div>
      <h1>{html.escape(page_title)}</h1>
      <p>
        This HTML is a self-contained human-view derivative built from the canonical scratch BD shuffle run.
        It is designed for direct opening from disk without a companion folder.
      </p>
      <p><strong>Canonical Scratch Run</strong>: <code>{html.escape(canonical_root)}</code></p>
      {bbb_reference_text}
      <p><strong>Run Metadata</strong>: run_id=<code>{html.escape(run_id)}</code> shot_list=<code>{html.escape(shot_list)}</code> n_trials=<code>{html.escape(n_trials)}</code> model=<code>{html.escape(model)}</code></p>
      <p><strong>Reading Rule</strong>: {reading_rule}</p>
    </section>
    {q_nav}
    {q_sections}
  </div>
</body>
</html>
"""


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    out_html = Path(args.out_html)

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    regime_metrics_path = run_dir / "bd_shuffle_regime_metrics.csv"
    case_deltas_path = run_dir / "bd_shuffle_case_deltas.csv"
    run_meta_path = run_dir / "run_meta.json"
    role_map_path = Path(args.role_map_md)

    for path in [regime_metrics_path, case_deltas_path, run_meta_path, role_map_path]:
        if not path.exists():
            raise FileNotFoundError(f"missing required artifact: {path}")

    regime_rows = _read_csv_rows(regime_metrics_path)
    case_rows = _read_csv_rows(case_deltas_path)
    run_meta = _read_json(run_meta_path)
    role_map = _parse_role_map(_read_text(role_map_path))
    bbb_reference_lookup: Dict[str, Dict[str, object]] = {}
    bbb_reference_root = ""
    hero_image_uri = ""
    if args.bbb_reference_run_dir.strip():
        bbb_reference_dir = Path(args.bbb_reference_run_dir).resolve()
        bbb_reference_root = str(bbb_reference_dir)
        bbb_reference_lookup = _load_bbb_reference_lookup(
            run_dir=bbb_reference_dir,
            shot=str(args.bbb_reference_shot),
            regime_id=str(args.bbb_reference_regime_id),
        )
    if args.hero_image.strip():
        hero_image_path = Path(args.hero_image).resolve()
        if not hero_image_path.exists():
            raise FileNotFoundError(f"hero_image not found: {hero_image_path}")
        hero_image_uri = _image_data_uri(hero_image_path)

    meta_order = _meta_qid_order(run_meta)
    delta_lookup = _case_delta_lookup(case_rows)
    grouped = _group_regime_rows(regime_rows)
    qids = _ordered_qids(regime_rows, meta_order)
    if bbb_reference_lookup:
        missing_reference_qids = [q_id for q_id in qids if q_id not in bbb_reference_lookup]
        if missing_reference_qids:
            raise ValueError(f"missing BBB reference rows for q_ids: {', '.join(missing_reference_qids)}")

    q_sections_parts: List[str] = []
    for q_id in qids:
        side_map = grouped.get(q_id, {})
        family_d = _family_payload(
            q_id=q_id,
            side="D",
            rows=side_map.get("D", []),
            delta_lookup=delta_lookup,
            bbb_reference_lookup=bbb_reference_lookup,
        )
        family_b = _family_payload(q_id=q_id, side="B", rows=side_map.get("B", []), delta_lookup=delta_lookup)
        q_sections_parts.append(
            _render_q_section(
                q_id=q_id,
                family_d=family_d,
                family_b=family_b,
                relation_roles=role_map.get(q_id, {}),
            )
        )

    html_text = _build_html(
        page_title=args.page_title,
        canonical_root=str(run_meta.get("canonical_root", run_dir)),
        run_meta=run_meta,
        q_sections="".join(q_sections_parts),
        q_nav=_render_q_nav(qids),
        bbb_reference_root=bbb_reference_root,
        hero_image_uri=hero_image_uri,
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
