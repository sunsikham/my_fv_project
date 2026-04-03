#!/usr/bin/env python3
"""
Build a human-readable HTML report for baseline PT outputs.

Outputs:
  - index.html
  - one page per q_id
  - per-q plots for target probability and PT summary
"""

from __future__ import annotations

import argparse
import html
import os
import re
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SHOT_ORDER = [1, 3, 5, 7, 10]
EDGE_ORDER = ["AB", "AD", "BD"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build human-readable PT HTML report")
    p.add_argument("--run_dir", required=True, help="PT run directory")
    p.add_argument("--out_dir", default=None, help="Output report directory")
    p.add_argument("--sweep_csv", default=None, help="Override pt_5edge_shot_sweep.csv")
    p.add_argument("--bootstrap_csv", default=None, help="Override pt_bootstrap_summary.csv")
    p.add_argument("--topk_summary_csv", default=None, help="Override top-k summary CSV")
    p.add_argument("--dpi", type=int, default=180, help="Plot dpi")
    return p.parse_args()


def _qid_sort_key(qid: str) -> int:
    m = re.search(r"(\d+)", str(qid))
    return int(m.group(1)) if m else 1_000_000


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _plot_target_prob(df: pd.DataFrame, qid: str, out_path: str, dpi: int) -> None:
    qdf = (
        df[(df["q_id"] == qid) & (df["edge"].isin(EDGE_ORDER))]
        .groupby(["edge", "shot"], as_index=False)["target_prob_raw"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    styles = {
        "AB": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "AD": {"color": "#d62728", "linestyle": "--", "marker": "s"},
        "BD": {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
    }
    local_y = []
    for edge in EDGE_ORDER:
        sdf = qdf[qdf["edge"] == edge].sort_values("shot")
        if sdf.empty:
            continue
        xs = sdf["shot"].tolist()
        ys = sdf["target_prob_raw"].tolist()
        local_y.extend(ys)
        ax.plot(xs, ys, linewidth=2.1, markersize=5.0, label=edge, **styles[edge])
    if local_y:
        y_min = min(local_y)
        y_max = max(local_y)
        pad = max((y_max - y_min) * 0.15, 0.01) if y_max != y_min else max(abs(y_min) * 0.08, 0.01)
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel("Shot")
    ax.set_ylabel("Mean target probability")
    ax.set_title(f"{qid} target probability")
    ax.grid(alpha=0.28, linewidth=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_pt(df: pd.DataFrame, qid: str, out_path: str, dpi: int) -> None:
    qdf = df[df["q_id"] == qid].sort_values("shot")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    if not qdf.empty:
        ax.plot(qdf["shot"], qdf["pt_abc_mean"], marker="o", linewidth=2.0, color="#1f77b4", label="ABC")
        ax.plot(qdf["shot"], qdf["pt_abd_mean"], marker="s", linewidth=2.0, color="#d62728", label="ABD")
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Shot")
    ax.set_ylabel("PT")
    ax.set_title(f"{qid} PT summary")
    ax.grid(alpha=0.28, linewidth=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _build_edge_meta(topk_df: pd.DataFrame, qid: str) -> Dict[str, Tuple[str, str]]:
    meta = {}
    qdf = topk_df[topk_df["q_id"] == qid]
    for edge in EDGE_ORDER:
        sdf = qdf[qdf["edge"] == edge]
        if sdf.empty:
            continue
        row = sdf.iloc[0]
        meta[edge] = (str(row["query_input"]), str(row["target_str"]))
    return meta


def _build_candidate_table(topk_df: pd.DataFrame, qid: str, edge: str) -> str:
    qdf = topk_df[(topk_df["q_id"] == qid) & (topk_df["edge"] == edge)].copy()
    if qdf.empty:
        return "<p>No candidate summary available.</p>"

    pieces = [
        '<table class="cand-table">',
        "<thead><tr><th>Rank</th>" + "".join(f"<th>Shot {shot}</th>" for shot in SHOT_ORDER) + "</tr></thead>",
        "<tbody>",
    ]
    for rank in range(1, 21):
        pieces.append(f"<tr><td class='rank'>{rank}</td>")
        for shot in SHOT_ORDER:
            sdf = qdf[(qdf["shot"] == shot) & (qdf["summary_rank"] == rank)]
            if sdf.empty:
                pieces.append("<td></td>")
                continue
            row = sdf.iloc[0]
            candidate = html.escape(str(row["display_candidate"]))
            mean_lp = float(row["mean_logprob"])
            coverage = float(row["trial_coverage_frac"])
            top1_count = int(row["top1_count"])
            pieces.append(
                "<td>"
                f"<div class='cand'>{candidate}</div>"
                f"<div class='meta'>lp={mean_lp:.3f}</div>"
                f"<div class='meta'>cov={coverage:.2f} top1={top1_count}</div>"
                "</td>"
            )
        pieces.append("</tr>")
    pieces.append("</tbody></table>")
    return "".join(pieces)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _page_template(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: Georgia, "Times New Roman", serif;
      margin: 24px;
      color: #1d1d1d;
      background: #f7f4ed;
      line-height: 1.4;
    }}
    a {{ color: #0f4c5c; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .nav {{ margin-bottom: 18px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      margin: 18px 0 24px 0;
    }}
    .card {{
      background: #fffdfa;
      border: 1px solid #d9d2c3;
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    img {{
      max-width: 100%;
      height: auto;
      border: 1px solid #ddd2c4;
      border-radius: 8px;
      background: white;
    }}
    .cand-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      table-layout: fixed;
      background: #fff;
    }}
    .cand-table th, .cand-table td {{
      border: 1px solid #ddd2c4;
      padding: 7px 8px;
      vertical-align: top;
      font-size: 13px;
    }}
    .cand-table th {{
      background: #efe8d8;
    }}
    .rank {{
      width: 44px;
      text-align: center;
      font-weight: 700;
      background: #f5efe2;
    }}
    .cand {{
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .meta {{
      color: #5b5348;
      font-size: 11px;
      white-space: nowrap;
    }}
    .qid-list {{
      columns: 3 220px;
      column-gap: 24px;
    }}
    .qid-list li {{
      margin-bottom: 8px;
    }}
    h1, h2, h3 {{
      color: #222;
    }}
    .small {{
      color: #5b5348;
      font-size: 13px;
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> int:
    args = _parse_args()
    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(run_dir, "human_report"))
    plot_dir = os.path.join(out_dir, "plots")
    _ensure_dir(plot_dir)

    sweep_csv = args.sweep_csv or os.path.join(run_dir, "pt_5edge_shot_sweep.csv")
    bootstrap_csv = args.bootstrap_csv or os.path.join(run_dir, "pt_bootstrap_summary.csv")
    topk_summary_csv = args.topk_summary_csv or os.path.join(run_dir, "pt_edge_topk_trial_agg_top20_by_q_shot_edge.csv")

    sweep_df = pd.read_csv(sweep_csv)
    boot_df = pd.read_csv(bootstrap_csv)
    topk_df = pd.read_csv(topk_summary_csv)

    qids = sorted(topk_df["q_id"].dropna().unique().tolist(), key=_qid_sort_key)

    summary_rows = []
    for qid in qids:
        target_prob_plot = os.path.join(plot_dir, f"{qid}_target_prob.png")
        pt_plot = os.path.join(plot_dir, f"{qid}_pt.png")
        _plot_target_prob(sweep_df, qid, target_prob_plot, dpi=args.dpi)
        _plot_pt(boot_df, qid, pt_plot, dpi=args.dpi)

        edge_meta = _build_edge_meta(topk_df, qid)
        q_boot = boot_df[boot_df["q_id"] == qid].sort_values("shot")
        shot10 = q_boot[q_boot["shot"] == 10]
        pt_abd_10 = float(shot10["pt_abd_mean"].iloc[0]) if not shot10.empty else None
        summary_rows.append((qid, pt_abd_10))

        sections = [
            '<div class="nav"><a href="index.html">Back to index</a></div>',
            f"<h1>{html.escape(qid)}</h1>",
            '<div class="grid">',
            f'<div class="card"><h2>Target Probability</h2><img src="plots/{qid}_target_prob.png" alt="{qid} target probability"></div>',
            f'<div class="card"><h2>PT Summary</h2><img src="plots/{qid}_pt.png" alt="{qid} PT summary"></div>',
            "</div>",
        ]

        for edge in EDGE_ORDER:
            query_input, target_str = edge_meta.get(edge, ("", ""))
            sections.append(
                '<div class="card">'
                f"<h2>{edge}</h2>"
                f"<div class='small'>query={html.escape(query_input)} target={html.escape(target_str)}</div>"
                f"{_build_candidate_table(topk_df, qid, edge)}"
                "</div>"
            )

        page_html = _page_template(
            f"{qid} PT report",
            "".join(sections),
        )
        _write_text(os.path.join(out_dir, f"{qid}.html"), page_html)

    list_items = []
    for qid, pt_abd_10 in summary_rows:
        suffix = f" shot10 ABD={pt_abd_10:.3f}" if pt_abd_10 is not None else ""
        list_items.append(f'<li><a href="{qid}.html">{html.escape(qid)}</a><span class="small"> {html.escape(suffix)}</span></li>')

    index_body = "".join(
        [
            "<h1>PT Human Report</h1>",
            f"<p class='small'>run_dir={html.escape(run_dir)}</p>",
            '<div class="grid">',
            '<div class="card"><h2>Overview</h2><p class="small">Open each q page to compare AB/AD/BD shot curves and shot-wise top candidates.</p></div>',
            '<div class="card"><h2>Available q_ids</h2><ul class="qid-list">' + "".join(list_items) + "</ul></div>",
            "</div>",
        ]
    )
    index_html = _page_template("PT Human Report", index_body)
    _write_text(os.path.join(out_dir, "index.html"), index_html)
    print(f"report_dir={out_dir}")
    print(f"index_html={os.path.join(out_dir, 'index.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
