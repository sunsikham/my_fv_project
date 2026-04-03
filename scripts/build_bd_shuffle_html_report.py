#!/usr/bin/env python3
"""Build a home-local HTML report for BD shuffle comparison results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REGIME_ORDER = {
    "D": ["BDBDBD_D", "BD_SHUF_D1_D", "BD_SHUF_D2_D", "BD_SHUF_D3_D", "BD_SHUF_D4_D", "BD_SHUF_D5_D"],
    "B": ["DBDBDB_B", "BD_SHUF_B1_B", "BD_SHUF_B2_B", "BD_SHUF_B3_B", "BD_SHUF_B4_B", "BD_SHUF_B5_B"],
}

BASELINE_BY_SIDE = {
    "D": "BDBDBD_D",
    "B": "DBDBDB_B",
}

SIDE_LABEL = {
    "D": "BDBDBD_D Group",
    "B": "DBDBDB_B Group",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BD shuffle HTML report.")
    parser.add_argument("--run_dir", required=True, help="Canonical scratch run directory")
    parser.add_argument("--out_html", required=True, help="Output HTML path")
    parser.add_argument("--asset_dir", required=True, help="Output asset directory for mirrored CSVs")
    parser.add_argument("--topk_limit", type=int, default=5, help="Top-k candidate rows to show per regime")
    return parser.parse_args()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row in {path}")
        return list(reader)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _fmt_float(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "NA"


def _fmt_pattern(pattern: object) -> str:
    text = str(pattern)
    if not text:
        return ""
    return " ".join(list(text))


def _display_layout(pattern: object, case_kind: object | None = None) -> str:
    base = _fmt_pattern(pattern)
    kind = str(case_kind or "").strip().lower()
    if kind == "regular":
        return f"{base}  regular"
    if kind == "shuffled":
        return f"{base}  shuffled"
    return base


def _html_table(rows: Sequence[Dict[str, object]], columns: Sequence[str], header_map: Dict[str, str] | None = None) -> str:
    header_map = header_map or {}
    parts = ["<table>", "<thead><tr>"]
    for col in columns:
        parts.append(f"<th>{html.escape(header_map.get(col, col))}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for col in columns:
            val = row.get(col, "")
            parts.append(f"<td>{html.escape(str(val))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _build_summary_cards(
    side_rows: List[Dict[str, str]],
    *,
    case_detail_lookup: Dict[tuple[str, str], Dict[str, str]],
    regime_detail_lookup: Dict[tuple[str, str], Dict[str, str]],
) -> str:
    cards: List[str] = []
    for row in side_rows:
        query_side = str(row["query_side"])
        baseline = str(row["baseline_regime_id"])
        best_case = str(row["best_case_regime_id"])
        worst_case = str(row["worst_case_regime_id"])
        best_key = (query_side, best_case)
        worst_key = (query_side, worst_case)
        base_key = (query_side, baseline)
        best_detail = case_detail_lookup.get(best_key, {})
        worst_detail = case_detail_lookup.get(worst_key, {})
        baseline_detail = regime_detail_lookup.get(base_key, {})
        cards.append(
            "<div class='card'>"
            f"<h3>Q1 {html.escape(query_side)} Side</h3>"
            f"<div class='metric'><span>Baseline Layout</span><strong>{html.escape(_fmt_pattern(baseline_detail.get('layout_pattern', '')))}</strong></div>"
            f"<div class='metric'><span>Baseline Acc</span><strong>{_fmt_float(row['baseline_top1_accuracy'])}</strong></div>"
            f"<div class='metric'><span>Mean Shuffled Acc</span><strong>{_fmt_float(row['mean_shuffled_top1_accuracy'])}</strong></div>"
            f"<div class='metric'><span>Mean Delta Acc</span><strong>{_fmt_float(row['mean_delta_top1_accuracy'])}</strong></div>"
            f"<div class='metric'><span>Best Layout</span><strong>{html.escape(_fmt_pattern(best_detail.get('layout_pattern', '')))}</strong></div>"
            f"<div class='metric'><span>Best Acc</span><strong>{_fmt_float(best_detail.get('shuffled_top1_accuracy'))}</strong></div>"
            f"<div class='metric'><span>Best Delta Acc</span><strong>{_fmt_float(best_detail.get('delta_top1_accuracy'))}</strong></div>"
            f"<div class='metric'><span>Worst Layout</span><strong>{html.escape(_fmt_pattern(worst_detail.get('layout_pattern', '')))}</strong></div>"
            f"<div class='metric'><span>Worst Acc</span><strong>{_fmt_float(worst_detail.get('shuffled_top1_accuracy'))}</strong></div>"
            f"<div class='metric'><span>Worst Delta Acc</span><strong>{_fmt_float(worst_detail.get('delta_top1_accuracy'))}</strong></div>"
            "</div>"
        )
    return "<div class='cards'>" + "".join(cards) + "</div>"


def _build_bar_chart(rows: List[Dict[str, str]], *, metric_key: str, title: str, scale_max: float) -> str:
    parts = ["<div class='chart-block'>", f"<h4>{html.escape(title)}</h4>"]
    for row in rows:
        display_label = _display_layout(row["layout_pattern"], row.get("case_kind"))
        value = float(row[metric_key])
        width_pct = max(0.0, min(100.0, (value / scale_max) * 100.0 if scale_max > 0 else 0.0))
        is_baseline = str(row["case_kind"]) == "regular"
        fill_class = "bar-fill baseline" if is_baseline else "bar-fill shuffled"
        parts.append(
            "<div class='bar-row'>"
            f"<div class='bar-label'>{html.escape(display_label)}</div>"
            "<div class='bar-track'>"
            f"<div class='{fill_class}' style='width:{width_pct:.2f}%'></div>"
            "</div>"
            f"<div class='bar-value'>{_fmt_float(value)}</div>"
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _build_side_chart_sections(regime_rows: List[Dict[str, str]]) -> str:
    grouped: Dict[str, List[Dict[str, str]]] = {"D": [], "B": []}
    for row in regime_rows:
        grouped[str(row["query_side"])].append(row)

    parts: List[str] = []
    for query_side in ("D", "B"):
        rows = grouped[query_side]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: REGIME_ORDER[query_side].index(str(row["regime_id"])))
        query_input = rows[0]["query_input"]
        target_str = rows[0]["target_str"]
        parts.append(
            "<section class='subsection'>"
            f"<h3>{html.escape(SIDE_LABEL[query_side])}</h3>"
            f"<p class='small'>query={html.escape(query_input)} target={html.escape(target_str)}</p>"
            "<div class='chart-grid'>"
            + _build_bar_chart(rows, metric_key="top1_accuracy", title="Top1 Accuracy", scale_max=1.0)
            + _build_bar_chart(rows, metric_key="mean_target_prob", title="Mean Target Probability", scale_max=1.0)
            + "</div></section>"
        )
    return "".join(parts)


def _group_topk(
    rows: List[Dict[str, str]],
    topk_limit: int,
    *,
    regime_detail_lookup: Dict[tuple[str, str], Dict[str, str]],
) -> str:
    grouped: Dict[str, Dict[str, List[Dict[str, str]]]] = {"D": {}, "B": {}}
    for row in rows:
        query_side = str(row["query_side"])
        regime = str(row["regime_id"])
        grouped.setdefault(query_side, {}).setdefault(regime, []).append(row)
    parts: List[str] = []
    for query_side in ("D", "B"):
        side_group = grouped.get(query_side, {})
        if not side_group:
            continue
        side_any_regime = next(iter(side_group.values()))
        query_input = side_any_regime[0]["query_input"]
        target_str = side_any_regime[0]["target_str"]
        parts.append(
            "<section class='subsection'>"
            f"<h3>{html.escape(SIDE_LABEL[query_side])}</h3>"
            f"<p class='small'>query={html.escape(query_input)} target={html.escape(target_str)}</p>"
        )
        for regime in REGIME_ORDER[query_side]:
            entries = [row for row in side_group.get(regime, []) if int(row["summary_rank"]) <= topk_limit]
            if not entries:
                continue
            regime_detail = regime_detail_lookup.get((query_side, regime), {})
            display_label = _display_layout(
                regime_detail.get("layout_pattern", regime),
                regime_detail.get("case_kind", "regular" if regime == BASELINE_BY_SIDE[query_side] else "shuffled"),
            )
            display_rows: List[Dict[str, object]] = []
            for row in entries:
                display_rows.append(
                    {
                        "summary_rank": row["summary_rank"],
                        "display_candidate": row["display_candidate"],
                        "mean_logprob": _fmt_float(row["mean_logprob"]),
                        "mean_prob": _fmt_float(row["mean_prob"]),
                        "mean_rank_within_row": _fmt_float(row["mean_rank_within_row"]),
                        "top1_count": f"{row['top1_count']}/{row['n_trials']}",
                        "trial_coverage_frac": _fmt_float(row["trial_coverage_frac"]),
                    }
                )
            parts.append(f"<h4>{html.escape(display_label)}</h4>")
            parts.append(
                _html_table(
                    display_rows,
                    [
                        "summary_rank",
                        "display_candidate",
                        "mean_logprob",
                        "mean_prob",
                        "mean_rank_within_row",
                        "top1_count",
                        "trial_coverage_frac",
                    ],
                    {
                        "summary_rank": "Rank",
                        "display_candidate": "Candidate",
                        "mean_logprob": "Mean Logprob",
                        "mean_prob": "Mean Prob",
                        "mean_rank_within_row": "Mean Rank",
                        "top1_count": "Top1 Count",
                        "trial_coverage_frac": "Coverage",
                    },
                )
            )
        parts.append("</section>")
    return "".join(parts)


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    out_html = Path(args.out_html)
    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    regime_metrics_src = run_dir / "bd_shuffle_regime_metrics.csv"
    case_deltas_src = run_dir / "bd_shuffle_case_deltas.csv"
    side_agg_src = run_dir / "bd_shuffle_side_aggregate.csv"
    topk_agg_src = run_dir / "bd_shuffle_edge_topk_trial_agg.csv"
    summary_src = run_dir / "bd_shuffle_summary.md"
    progress_src = run_dir / "progress_status.json"
    run_meta_src = run_dir / "run_meta.json"
    run_status_src = run_dir / "run_status.json"

    for src in [
        regime_metrics_src,
        case_deltas_src,
        side_agg_src,
        topk_agg_src,
        summary_src,
        progress_src,
        run_meta_src,
        run_status_src,
    ]:
        if not src.exists():
            raise FileNotFoundError(f"missing required artifact: {src}")

    mirror_paths = {
        "bd_shuffle_regime_metrics.csv": asset_dir / "bd_shuffle_regime_metrics.csv",
        "bd_shuffle_case_deltas.csv": asset_dir / "bd_shuffle_case_deltas.csv",
        "bd_shuffle_side_aggregate.csv": asset_dir / "bd_shuffle_side_aggregate.csv",
        "bd_shuffle_edge_topk_trial_agg.csv": asset_dir / "bd_shuffle_edge_topk_trial_agg.csv",
        "bd_shuffle_summary.md": asset_dir / "bd_shuffle_summary.md",
        "progress_status.json": asset_dir / "progress_status.json",
        "run_meta.json": asset_dir / "run_meta.json",
        "run_status.json": asset_dir / "run_status.json",
    }
    source_by_name = {
        "bd_shuffle_regime_metrics.csv": regime_metrics_src,
        "bd_shuffle_case_deltas.csv": case_deltas_src,
        "bd_shuffle_side_aggregate.csv": side_agg_src,
        "bd_shuffle_edge_topk_trial_agg.csv": topk_agg_src,
        "bd_shuffle_summary.md": summary_src,
        "progress_status.json": progress_src,
        "run_meta.json": run_meta_src,
        "run_status.json": run_status_src,
    }
    for name, dst in mirror_paths.items():
        _copy(source_by_name[name], dst)

    mirror_meta = {
        "canonical_root": str(run_dir),
        "sync_root": str(asset_dir),
        "sync_mode": "copy",
        "artifact_profile": "selected_summary_mirror",
        "files": {name: str(path) for name, path in mirror_paths.items()},
    }
    (asset_dir / "mirror_meta.json").write_text(
        json.dumps(mirror_meta, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    regime_rows = _read_csv(regime_metrics_src)
    case_delta_rows = _read_csv(case_deltas_src)
    side_rows = _read_csv(side_agg_src)
    topk_rows = _read_csv(topk_agg_src)

    case_table_rows: List[Dict[str, object]] = []
    case_detail_lookup: Dict[tuple[str, str], Dict[str, str]] = {}
    for row in case_delta_rows:
        case_detail_lookup[(str(row["query_side"]), str(row["regime_id"]))] = row
        case_table_rows.append(
            {
                "query_side": row["query_side"],
                "regime_id": row["case_index"],
                "layout_pattern": _fmt_pattern(row["layout_pattern"]),
                "baseline_top1_accuracy": _fmt_float(row["baseline_top1_accuracy"]),
                "shuffled_top1_accuracy": _fmt_float(row["shuffled_top1_accuracy"]),
                "delta_top1_accuracy": _fmt_float(row["delta_top1_accuracy"]),
                "delta_mean_target_logit": _fmt_float(row["delta_mean_target_logit"]),
                "delta_mean_target_logprob": _fmt_float(row["delta_mean_target_logprob"]),
                "delta_mean_target_rank": _fmt_float(row["delta_mean_target_rank"]),
            }
        )

    regime_table_rows: List[Dict[str, object]] = []
    regime_detail_lookup: Dict[tuple[str, str], Dict[str, str]] = {}
    for row in regime_rows:
        regime_detail_lookup[(str(row["query_side"]), str(row["regime_id"]))] = row
        regime_table_rows.append(
            {
                "query_side": row["query_side"],
                "regime_id": _display_layout(row["layout_pattern"], row["case_kind"]),
                "case_kind": row["case_kind"],
                "layout_pattern": _fmt_pattern(row["layout_pattern"]),
                "query_input": row["query_input"],
                "target_str": row["target_str"],
                "top1_accuracy": _fmt_float(row["top1_accuracy"]),
                "mean_target_logprob": _fmt_float(row["mean_target_logprob"]),
                "mean_target_prob": _fmt_float(row["mean_target_prob"]),
                "mean_target_logit": _fmt_float(row["mean_target_logit"]),
                "mean_target_rank": _fmt_float(row["mean_target_rank"]),
            }
        )

    side_table_rows: List[Dict[str, object]] = []
    for row in side_rows:
        best_case_detail = case_detail_lookup.get((str(row["query_side"]), str(row["best_case_regime_id"])), {})
        worst_case_detail = case_detail_lookup.get((str(row["query_side"]), str(row["worst_case_regime_id"])), {})
        baseline_detail = regime_detail_lookup.get((str(row["query_side"]), str(row["baseline_regime_id"])), {})
        side_table_rows.append(
            {
                "query_side": row["query_side"],
                "baseline_regime_id": _fmt_pattern(baseline_detail.get("layout_pattern", "")),
                "baseline_top1_accuracy": _fmt_float(row["baseline_top1_accuracy"]),
                "mean_shuffled_top1_accuracy": _fmt_float(row["mean_shuffled_top1_accuracy"]),
                "mean_delta_top1_accuracy": _fmt_float(row["mean_delta_top1_accuracy"]),
                "min_delta_top1_accuracy": _fmt_float(row["min_delta_top1_accuracy"]),
                "max_delta_top1_accuracy": _fmt_float(row["max_delta_top1_accuracy"]),
                "mean_delta_target_logit": _fmt_float(row["mean_delta_target_logit"]),
                "mean_delta_target_logprob": _fmt_float(row["mean_delta_target_logprob"]),
                "mean_delta_target_rank": _fmt_float(row["mean_delta_target_rank"]),
                "best_case_regime_id": _fmt_pattern(best_case_detail.get("layout_pattern", "")),
                "worst_case_regime_id": _fmt_pattern(worst_case_detail.get("layout_pattern", "")),
            }
        )

    summary_cards_html = _build_summary_cards(
        side_rows,
        case_detail_lookup=case_detail_lookup,
        regime_detail_lookup=regime_detail_lookup,
    )
    side_table_html = _html_table(
        side_table_rows,
        [
            "query_side",
            "baseline_regime_id",
            "baseline_top1_accuracy",
            "mean_shuffled_top1_accuracy",
            "mean_delta_top1_accuracy",
            "min_delta_top1_accuracy",
            "max_delta_top1_accuracy",
            "mean_delta_target_logit",
            "mean_delta_target_logprob",
            "mean_delta_target_rank",
            "best_case_regime_id",
            "worst_case_regime_id",
        ],
        {
            "query_side": "Side",
            "baseline_regime_id": "Baseline Layout",
            "baseline_top1_accuracy": "Baseline Acc",
            "mean_shuffled_top1_accuracy": "Mean Shuffled Acc",
            "mean_delta_top1_accuracy": "Mean Delta Acc",
            "min_delta_top1_accuracy": "Min Delta Acc",
            "max_delta_top1_accuracy": "Max Delta Acc",
            "mean_delta_target_logit": "Mean Delta Logit",
            "mean_delta_target_logprob": "Mean Delta Logprob",
            "mean_delta_target_rank": "Mean Delta Rank",
            "best_case_regime_id": "Best Layout",
            "worst_case_regime_id": "Worst Layout",
        },
    )
    case_table_html = _html_table(
        case_table_rows,
        [
            "query_side",
            "regime_id",
            "layout_pattern",
            "baseline_top1_accuracy",
            "shuffled_top1_accuracy",
            "delta_top1_accuracy",
            "delta_mean_target_logit",
            "delta_mean_target_logprob",
            "delta_mean_target_rank",
        ],
        {
            "query_side": "Side",
            "regime_id": "Case",
            "layout_pattern": "Layout",
            "baseline_top1_accuracy": "Baseline Acc",
            "shuffled_top1_accuracy": "Shuffled Acc",
            "delta_top1_accuracy": "Delta Acc",
            "delta_mean_target_logit": "Delta Logit",
            "delta_mean_target_logprob": "Delta Logprob",
            "delta_mean_target_rank": "Delta Rank",
        },
    )
    regime_table_html = _html_table(
        regime_table_rows,
        [
            "query_side",
            "regime_id",
            "case_kind",
            "layout_pattern",
            "query_input",
            "target_str",
            "top1_accuracy",
            "mean_target_logprob",
            "mean_target_prob",
            "mean_target_logit",
            "mean_target_rank",
        ],
        {
            "query_side": "Side",
            "regime_id": "Displayed Layout",
            "case_kind": "Kind",
            "layout_pattern": "Layout",
            "query_input": "Query",
            "target_str": "Target",
            "top1_accuracy": "Acc",
            "mean_target_logprob": "Mean Logprob",
            "mean_target_prob": "Mean Prob",
            "mean_target_logit": "Mean Logit",
            "mean_target_rank": "Mean Rank",
        },
    )
    charts_html = _build_side_chart_sections(regime_rows)
    topk_html = _group_topk(
        topk_rows,
        args.topk_limit,
        regime_detail_lookup=regime_detail_lookup,
    )

    canonical_link = html.escape(str(run_dir))
    asset_rel = html.escape(os.path.relpath(asset_dir, out_html.parent))
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BD Shuffle Q1 Report</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --ink: #1d1d1b;
      --muted: #5a5851;
      --line: #d8d0c4;
      --accent: #8c3b2a;
      --accent-2: #204e5a;
      --good: #1f6b4f;
      --bad: #9c2f2f;
      --shadow: 0 10px 30px rgba(0,0,0,0.06);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(140,59,42,0.10), transparent 28%),
        linear-gradient(180deg, #f8f4ee 0%, var(--bg) 100%);
    }}
    .wrap {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 32px 28px 56px;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px;
      line-height: 1.15;
    }}
    h1 {{
      font-size: 40px;
      letter-spacing: -0.03em;
    }}
    h2 {{
      font-size: 26px;
      margin-top: 28px;
    }}
    h3 {{
      font-size: 20px;
      margin-top: 18px;
    }}
    p, li {{
      font-size: 16px;
      line-height: 1.55;
    }}
    .hero, .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 24px;
      margin-bottom: 20px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 14px;
      margin-top: 8px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }}
    .card {{
      background: linear-gradient(180deg, #fffefb, #f8f3eb);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
    }}
    .metric {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 6px 0;
      border-bottom: 1px dashed rgba(0,0,0,0.08);
    }}
    .metric:last-child {{
      border-bottom: 0;
    }}
    .metric span {{
      color: var(--muted);
    }}
    .metric strong {{
      color: var(--accent-2);
      font-weight: 700;
    }}
    .small {{
      color: var(--muted);
      font-size: 14px;
    }}
    .pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #efe7dc;
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      margin-right: 8px;
      margin-bottom: 8px;
    }}
    .table-wrap {{
      overflow-x: auto;
      margin-top: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 900px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f7f0e7;
      color: #322d27;
      z-index: 1;
    }}
    tr:nth-child(even) td {{
      background: rgba(0,0,0,0.015);
    }}
    .subsection {{
      margin-top: 18px;
      border-top: 1px solid var(--line);
      padding-top: 18px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 18px;
      margin-top: 12px;
    }}
    .chart-block {{
      background: linear-gradient(180deg, #fffefb, #f5eee4);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }}
    .chart-block h4 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 160px 1fr 70px;
      gap: 10px;
      align-items: center;
      margin: 10px 0;
    }}
    .bar-label {{
      font-size: 13px;
      color: #3a352f;
      word-break: break-word;
    }}
    .bar-track {{
      height: 14px;
      background: #e8dfd2;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
    }}
    .bar-fill.baseline {{
      background: linear-gradient(90deg, #8c3b2a, #c06b49);
    }}
    .bar-fill.shuffled {{
      background: linear-gradient(90deg, #204e5a, #4f7b88);
    }}
    .bar-value {{
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      color: #3b3b39;
      text-align: right;
    }}
    .links a {{
      color: var(--accent-2);
      text-decoration: none;
    }}
    .links a:hover {{
      text-decoration: underline;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      background: #f2ece3;
      padding: 1px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="pill">Home Mirror</div>
      <div class="pill">Q1</div>
      <div class="pill">Shot 9 Only</div>
      <h1>BD Shuffle Behavior Report</h1>
      <p>Regular BD alternating baselines versus five shuffled cases per query side, mirrored from scratch canonical artifacts into the home project root for easier review.</p>
      <div class="meta">Canonical scratch run: <code>{canonical_link}</code></div>
      <div class="meta">Home mirror asset dir: <code>{asset_rel}</code></div>
      <div class="links small" style="margin-top:10px;">
        Mirrored files:
        <a href="{asset_rel}/bd_shuffle_side_aggregate.csv">side aggregate CSV</a>,
        <a href="{asset_rel}/bd_shuffle_case_deltas.csv">case deltas CSV</a>,
        <a href="{asset_rel}/bd_shuffle_regime_metrics.csv">regime metrics CSV</a>,
        <a href="{asset_rel}/bd_shuffle_edge_topk_trial_agg.csv">top candidate aggregate CSV</a>,
        <a href="{asset_rel}/bd_shuffle_summary.md">markdown summary</a>
      </div>
    </section>

    <section class="section">
      <h2>Quick Read</h2>
      <p class="small">These cards summarize the side-level picture before drilling into full tables.</p>
      {summary_cards_html}
    </section>

    <section class="section">
      <h2>Side Aggregate Table</h2>
      <div class="table-wrap">
        {side_table_html}
      </div>
    </section>

    <section class="section">
      <h2>Per-Case Delta Table</h2>
      <div class="table-wrap">
        {case_table_html}
      </div>
    </section>

    <section class="section">
      <h2>Per-Regime Metrics</h2>
      <div class="table-wrap">
        {regime_table_html}
      </div>
    </section>

    <section class="section">
      <h2>Bar Charts</h2>
      <p class="small">Separated by baseline family so the `BDBDBD_D` group and `DBDBDB_B` group are easier to compare at a glance.</p>
      {charts_html}
    </section>

    <section class="section">
      <h2>Top Candidates By Regime</h2>
      <p class="small">Top {args.topk_limit} aggregated lexical candidates for each regime, split by the two baseline groups for easier reading.</p>
      {topk_html}
    </section>
  </div>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
