#!/usr/bin/env python3
"""Build a single self-contained unified PT dashboard HTML."""

from __future__ import annotations

import argparse
import base64
import html
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from scripts.build_pt_unified_human_report import (
    EDGE_GROUP_CONFIG,
    FAMILY_ORDER,
    REGIME_ORDER,
    _aggregate_topk,
    _build_a_only_table,
    _build_edge_group_compare_table,
    _edge_group_explainer,
    _build_meta,
    _default_baseline_abc_csv,
    _load_baseline_abc_df,
    _discover_pca_3d_html,
    _family_status_map,
    _humanize_pca_variant,
    _load_role_map,
    _plot_across_q_pt_median,
    _plot_control,
    _plot_edge_group_compare,
    _plot_pt,
    _qid_sort_key,
    _role_line_for_edge,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build one-file unified PT dashboard HTML")
    p.add_argument("--run_dir", required=True, help="Unified PT run directory")
    p.add_argument("--out_html", required=True, help="Output standalone HTML path")
    p.add_argument("--dpi", type=int, default=180, help="Plot dpi")
    return p.parse_args()


def _img_to_data_uri(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _load_pca_srcdoc(qid: str) -> str | None:
    items = _discover_pca_3d_html(qid)
    if not items:
        return None
    src = Path(items[0]["source_path"])
    return src.read_text(encoding="utf-8")


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
    h1, h2, h3 {{ color: #222; }}
    .small {{ color: #5b5348; font-size: 13px; }}
    .explain {{ line-height: 1.55; margin-bottom: 14px; font-size: 18px; font-weight: 500; }}
    .explain-role {{ font-size: 18px; font-weight: 500; margin-bottom: 6px; }}
    .explain-prompt {{ font-size: 18px; font-weight: 500; margin: 10px 0 6px 0; color: #1d1d1d; }}
    .explain-query, .explain-target {{ font-size: 17px; margin-bottom: 4px; font-weight: 500; }}
    .toc {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(247,244,237,0.96);
      padding: 10px 0 12px 0;
      border-bottom: 1px solid #ddd2c4;
      margin-bottom: 18px;
    }}
    .toc a {{
      display: inline-block;
      margin: 0 10px 8px 0;
      padding: 6px 10px;
      border: 1px solid #d9d2c3;
      border-radius: 999px;
      background: #fffdfa;
    }}
    .dashboard-q {{
      margin-bottom: 18px;
      border: 1px solid #d9d2c3;
      border-radius: 12px;
      background: #fffdfa;
      overflow: hidden;
    }}
    .dashboard-q summary {{
      cursor: pointer;
      list-style: none;
      padding: 14px 16px;
      font-weight: 700;
      background: #efe8d8;
    }}
    .dashboard-body {{ padding: 16px; }}
    .stack {{
      display: flex;
      flex-direction: column;
      gap: 18px;
      margin: 18px 0 24px 0;
    }}
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
      width: 100%;
      max-width: 980px;
      display: block;
      margin: 0 auto;
      border: 1px solid #ddd2c4;
      border-radius: 8px;
      background: white;
    }}
    .pca-frame {{
      width: 100%;
      height: 680px;
      border: 1px solid #ddd2c4;
      border-radius: 8px;
      background: white;
    }}
    .setup-box {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin: 0 0 14px 0;
    }}
    .setup-panel {{
      border: 1px solid #ddd2c4;
      border-radius: 10px;
      background: #faf7f0;
      padding: 12px 14px;
    }}
    .setup-panel h3 {{
      margin: 0 0 8px 0;
      font-size: 17px;
    }}
    .setup-line {{ font-size: 13px; margin: 4px 0; color: #2b2b2b; }}
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
    .cand-table th {{ background: #efe8d8; }}
    .rank {{
      width: 86px;
      text-align: center;
      font-weight: 700;
      background: #f5efe2;
    }}
    .muted-cell {{ background: #faf7f0; }}
    .cand {{ font-weight: 700; margin-bottom: 4px; }}
    .meta {{ color: #5b5348; font-size: 11px; white-space: nowrap; }}
    .pct {{ color: #0f4c5c; font-weight: 700; margin-left: 6px; }}
    .cand-compact {{
      margin-bottom: 8px;
      padding-bottom: 6px;
      border-bottom: 1px dotted #e6ddcf;
    }}
    .cand-compact:last-child {{ margin-bottom: 0; padding-bottom: 0; border-bottom: 0; }}
    .cand-line {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: baseline;
    }}
    .goals-box {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      margin: 18px 0 26px 0;
    }}
    .goal-panel {{
      border: 1px solid #ddd2c4;
      border-radius: 12px;
      background: #fffdfa;
      padding: 24px 26px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .goal-panel h2 {{
      margin: 0 0 14px 0;
      font-size: 28px;
    }}
    .goal-panel p {{
      margin: 0 0 12px 0;
      font-size: 19px;
      line-height: 1.65;
      color: #2b2b2b;
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
    run_dir = Path(args.run_dir)
    sweep_df = pd.read_csv(run_dir / "pt_unified_shot_sweep.csv")
    boot_df = pd.read_csv(run_dir / "pt_unified_bootstrap_summary.csv")
    abc_df = _load_baseline_abc_df(str(_default_baseline_abc_csv()) if _default_baseline_abc_csv() else None)
    eligibility_df = pd.read_csv(run_dir / "pt_unified_family_eligibility.csv")
    topk_df = pd.read_json(run_dir / "pt_unified_edge_topk.jsonl", lines=True)
    summary_csv = run_dir / "pt_unified_edge_topk_trial_agg_top20_by_q_shot_regime.csv"
    summary_df = _aggregate_topk(topk_df, str(summary_csv))
    role_map = _load_role_map()

    eligible_qids = sorted(
        eligibility_df[eligibility_df["eligible"] == 1]["q_id"].dropna().astype(str).unique().tolist(),
        key=_qid_sort_key,
    )

    toc = '<a href="#action-goals">Action Goals</a>' + "".join(
        f'<a href="#{html.escape(qid)}">{html.escape(qid)}</a>' for qid in eligible_qids
    )

    sections: List[str] = [
        "<h1>Unified PT All-Q Dashboard</h1>",
        f"<p class='small'>run_dir={html.escape(str(run_dir))}</p>",
        f'<div class="toc">{toc}</div>',
        '<section id="action-goals" class="goals-box">'
        '<div class="goal-panel">'
        '<h2>Action Goals</h2>'
        '<p>Goal 1 — Candidate-based evaluation.</p>'
        '<p>Inspect model behavior through generated candidates rather than exact-match accuracy.</p>'
        '<p>For each prompt, generate candidate answers and check whether they align with the intended relation.</p>'
        '</div>'
        '<div class="goal-panel">'
        '<h2>Action Goals</h2>'
        '<p>Goal 2 — Representation warping in a common PCA space.</p>'
        '<p>Test whether context warps the representation of A toward B or D in a shared PCA space.</p>'
        '<p>Build one trial-level embedding per condition and project all conditions into the same PCA space.</p>'
        '</div>'
        '</section>',
    ]

    with tempfile.TemporaryDirectory(prefix="unified_html_") as tmp:
        tmp_dir = Path(tmp)
        all_q_pt_median_path = tmp_dir / "all_q_pt_median.png"
        if _plot_across_q_pt_median(boot_df, abc_df, eligible_qids, str(all_q_pt_median_path), args.dpi):
            all_q_pt_median_uri = _img_to_data_uri(str(all_q_pt_median_path))
            sections.insert(
                2,
                '<div class="card"><h2>Across-Q PT Median</h2>'
                '<div class="small" style="margin-bottom: 10px;">'
                'Shot-wise median across q. Each PT line is aggregated independently.'
                '</div>'
                f'<img src="{all_q_pt_median_uri}" alt="Across-q PT median summary"></div>',
            )
        for qid in eligible_qids:
            q_elig = eligibility_df[eligibility_df["q_id"] == qid]
            meta = _build_meta(summary_df, sweep_df, qid)

            pt_path = tmp_dir / f"{qid}_pt.png"
            _plot_pt(boot_df, abc_df, qid, str(pt_path), args.dpi)
            edge_paths = _plot_edge_group_compare(sweep_df, qid, str(tmp_dir / f"{qid}_edge_compare"), args.dpi)
            control_path = tmp_dir / f"{qid}_control.png"
            _plot_control(sweep_df, qid, str(control_path), args.dpi)

            pt_uri = _img_to_data_uri(str(pt_path)) if pt_path.exists() else ""
            control_uri = _img_to_data_uri(str(control_path)) if control_path.exists() else ""
            edge_uris = {edge: _img_to_data_uri(path) for edge, path in edge_paths.items()}

            status_parts = []
            for family_id in FAMILY_ORDER:
                row = q_elig[q_elig["family_id"] == family_id]
                if row.empty:
                    continue
                rec = row.iloc[0]
                status_parts.append(f"{family_id}={'ok' if int(rec['eligible']) == 1 else 'skip'}")
            header_suffix = " | ".join(status_parts)

            pca_srcdoc = _load_pca_srcdoc(qid)
            pca_items = _discover_pca_3d_html(qid)
            pca_label = ""
            if pca_items:
                pca_label = _humanize_pca_variant(pca_items[0]["subdir"])
            pca_block = (
                f'<div class="card"><h2>PCA 3D</h2>'
                '<div class="small explain" style="font-size: 16px; font-weight: 400;">'
                'Using the top 20 heads, we build a single vector for each trial. '
                'We then fit one common PCA on trial vectors pooled across all conditions and project everything into the same PCA space. '
                'Each dot is one trial vector.'
                '</div>'
                f'<div class="small" style="margin-bottom: 10px;">{html.escape(pca_label)}</div>'
                f'<iframe class="pca-frame" srcdoc="{html.escape(pca_srcdoc, quote=True)}"></iframe></div>'
                if pca_srcdoc
                else '<div class="card"><h2>PCA 3D</h2><div class="small">No PCA 3D HTML found for this q.</div></div>'
            )

            sections.append(
                f'<details class="dashboard-q" id="{html.escape(qid)}" {"open" if qid == "Q1" else ""}>'
                f"<summary>{html.escape(qid)} <span class='small'>{html.escape(header_suffix)}</span></summary>"
                '<div class="dashboard-body">'
                '<div class="stack">'
                + (
                    f'<div class="card"><h2>Test Triangle Inequality</h2>'
                    '<div class="setup-box">'
                    '<div class="setup-panel"><h3>ABC</h3><div class="setup-line">AAAAA -> B</div><div class="setup-line">AAAAA -> C</div><div class="setup-line">BBBBB -> C</div></div>'
                    '<div class="setup-panel"><h3>ABD</h3><div class="setup-line">AAAAA -> B</div><div class="setup-line">AAAAA -> D</div><div class="setup-line">BBBBB -> D</div></div>'
                    '<div class="setup-panel"><h3>Mixed ABD</h3><div class="setup-line">ABABAB -> B</div><div class="setup-line">ADADAD -> D</div><div class="setup-line">BDBDBD -> D</div></div>'
                    '</div>'
                    f'<img src="{pt_uri}" alt="{qid} PT overlay"></div>'
                    if pt_uri else '<div class="card"><h2>Test Triangle Inequality</h2><div class="small">No PT plot available.</div></div>'
                )
                + (
                    "<div class=\"card\"><h2>AB / AD / BD Target Probability</h2>"
                    "<div class=\"small\" style=\"margin-bottom: 10px;\">"
                    "Mean target probability: average probability assigned to the target token across trials."
                    "</div><div class=\"grid\">"
                    + "".join(
                        f'<div>{_edge_group_explainer(role_map, qid, edge, meta)}<img src="{edge_uris[edge]}" alt="{qid} {edge} comparison"></div>'
                        for edge in ["AB", "AD", "BD"] if edge in edge_uris
                    )
                    + "</div></div>"
                    if edge_uris else '<div class="card"><h2>AB / AD / BD Target Probability</h2><div class="small">No comparison plot available.</div></div>'
                )
                + "</div>"
                + '<div class="stack">'
                + f'<div class="card"><h2>AB Candidates</h2>{_role_line_for_edge(role_map, qid, "AB")}<div class="small" style="font-size: 15px; line-height: 1.5; margin-bottom: 10px;">For each shot, candidates are taken from the next-token distribution after the demos and query, filtered to interpretable word-level items, and ranked by mean log probability across trials.</div>{_build_edge_group_compare_table(summary_df, eligibility_df, qid, "AB", meta)}</div>'
                + f'<div class="card"><h2>AD Candidates</h2>{_role_line_for_edge(role_map, qid, "AD")}{_build_edge_group_compare_table(summary_df, eligibility_df, qid, "AD", meta)}</div>'
                + f'<div class="card"><h2>BD Candidates</h2>{_role_line_for_edge(role_map, qid, "BD")}{_build_edge_group_compare_table(summary_df, eligibility_df, qid, "BD", meta)}</div>'
                + pca_block
                + "</div></div></details>"
            )

    out_path = Path(args.out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_page_template("Unified PT All-Q Dashboard", "".join(sections)), encoding="utf-8")
    print(f"out_html={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
