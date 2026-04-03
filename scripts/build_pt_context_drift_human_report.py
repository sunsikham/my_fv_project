#!/usr/bin/env python3
"""
Build a human-readable HTML report for mixed-context PT outputs.
"""

from __future__ import annotations

import argparse
import html
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SHOT_ORDER = [1, 3, 5, 7, 10]
REGIME_ORDER = ["ABABAB_B", "ADADAD_D", "BDBDBD_D"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build human-readable mixed-context PT HTML report")
    p.add_argument("--run_dir", required=True, help="Mixed-context PT run directory")
    p.add_argument("--out_dir", default=None, help="Output report directory")
    p.add_argument("--sweep_csv", default=None, help="Override pt_context_drift_shot_sweep.csv")
    p.add_argument("--bootstrap_csv", default=None, help="Override pt_context_drift_bootstrap_summary.csv")
    p.add_argument("--topk_jsonl", default=None, help="Override pt_context_drift_edge_topk.jsonl")
    p.add_argument("--summary_csv", default=None, help="Optional output path for aggregated top-k summary CSV")
    p.add_argument("--dpi", type=int, default=180, help="Plot dpi")
    return p.parse_args()


def _qid_sort_key(qid: str) -> int:
    m = re.search(r"(\d+)", str(qid))
    return int(m.group(1)) if m else 1_000_000


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _aggregate_topk(topk_df: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    grouped_rows: List[Dict[str, object]] = []
    trial_counts: Dict[Tuple[str, int, str], set] = defaultdict(set)
    by_key: Dict[Tuple[str, int, str, str], Dict[str, object]] = {}

    for row in topk_df.to_dict(orient="records"):
        q_id = str(row["q_id"])
        shot = int(row["shot"])
        regime = str(row["regime_id"])
        trial_index = int(row["trial_index"])
        trial_counts[(q_id, shot, regime)].add(trial_index)

        candidates = list(row.get("lexical_candidates", []))
        canonicals = list(row.get("lexical_candidate_canonical_forms", []))
        logprobs = list(row.get("lexical_candidate_logprobs", []))
        probs = list(row.get("lexical_candidate_probs", []))
        for rank_idx, (cand, canon, lp, prob) in enumerate(zip(candidates, canonicals, logprobs, probs), start=1):
            key = (q_id, shot, regime, canon)
            stat = by_key.get(key)
            if stat is None:
                stat = {
                    "q_id": q_id,
                    "shot": shot,
                    "regime_id": regime,
                    "query_input": str(row.get("query_input", "")),
                    "target_str": str(row.get("target_str", "")),
                    "candidate_canonical": canon,
                    "surface_counts": defaultdict(int),
                    "count": 0,
                    "sum_logprob": 0.0,
                    "sum_prob": 0.0,
                    "sum_rank": 0.0,
                    "top1_count": 0,
                    "trial_indices": set(),
                }
                by_key[key] = stat
            stat["surface_counts"][cand] += 1
            stat["count"] += 1
            stat["sum_logprob"] += float(lp)
            stat["sum_prob"] += float(prob)
            stat["sum_rank"] += float(rank_idx)
            stat["trial_indices"].add(trial_index)
            if rank_idx == 1:
                stat["top1_count"] += 1

    grouped: Dict[Tuple[str, int, str], List[Dict[str, object]]] = defaultdict(list)
    for stat in by_key.values():
        q_id = stat["q_id"]
        shot = stat["shot"]
        regime = stat["regime_id"]
        n_trials = len(trial_counts[(q_id, shot, regime)])
        display_candidate = sorted(stat["surface_counts"].items(), key=lambda x: (-x[1], x[0]))[0][0]
        count = int(stat["count"])
        grouped[(q_id, shot, regime)].append(
            {
                "q_id": q_id,
                "shot": shot,
                "regime_id": regime,
                "query_input": stat["query_input"],
                "target_str": stat["target_str"],
                "candidate_canonical": stat["candidate_canonical"],
                "display_candidate": display_candidate,
                "mean_logprob": float(stat["sum_logprob"]) / count,
                "mean_prob": float(stat["sum_prob"]) / count,
                "mean_rank_within_row": float(stat["sum_rank"]) / count,
                "occurrence_count": count,
                "trial_coverage_count": len(stat["trial_indices"]),
                "trial_coverage_frac": (len(stat["trial_indices"]) / n_trials) if n_trials else 0.0,
                "top1_count": int(stat["top1_count"]),
                "n_trials": n_trials,
            }
        )

    final_rows: List[Dict[str, object]] = []
    for _, entries in grouped.items():
        entries.sort(
            key=lambda r: (
                -float(r["mean_logprob"]),
                -int(r["occurrence_count"]),
                -float(r["mean_prob"]),
                str(r["display_candidate"]),
            )
        )
        for summary_rank, entry in enumerate(entries[:20], start=1):
            e = dict(entry)
            e["summary_rank"] = summary_rank
            final_rows.append(e)

    out_df = pd.DataFrame(final_rows).sort_values(["q_id", "shot", "regime_id", "summary_rank"]).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)
    return out_df


def _plot_target_prob(sweep_df: pd.DataFrame, qid: str, out_path: str, dpi: int) -> None:
    qdf = (
        sweep_df[sweep_df["q_id"] == qid]
        .groupby(["regime_id", "shot"], as_index=False)["target_prob_raw"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(5.3, 3.7))
    styles = {
        "ABABAB_B": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "ADADAD_D": {"color": "#d62728", "linestyle": "--", "marker": "s"},
        "BDBDBD_D": {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
    }
    local_y: List[float] = []
    for regime in REGIME_ORDER:
        sdf = qdf[qdf["regime_id"] == regime].sort_values("shot")
        if sdf.empty:
            continue
        xs = sdf["shot"].tolist()
        ys = sdf["target_prob_raw"].tolist()
        local_y.extend(ys)
        ax.plot(xs, ys, linewidth=2.1, markersize=5.0, label=regime, **styles[regime])
    if local_y:
        y_min = min(local_y)
        y_max = max(local_y)
        pad = max((y_max - y_min) * 0.15, 0.01) if y_max != y_min else max(abs(y_min) * 0.08, 0.01)
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel("Shot")
    ax.set_ylabel("Mean target probability")
    ax.set_title(f"{qid} target probability")
    ax.grid(alpha=0.28, linewidth=0.6)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_pt(boot_df: pd.DataFrame, qid: str, out_path: str, dpi: int) -> None:
    qdf = boot_df[boot_df["q_id"] == qid].sort_values("shot")
    fig, ax = plt.subplots(figsize=(5.3, 3.7))
    if not qdf.empty:
        ax.plot(qdf["shot"], qdf["pt_ctx_abd_mean"], marker="o", linewidth=2.0, color="#b05a00", label="CTX_ABD")
        if "baseline_pt_abd_mean" in qdf.columns and qdf["baseline_pt_abd_mean"].notna().any():
            base = qdf[qdf["baseline_pt_abd_mean"].notna()]
            ax.plot(base["shot"], base["baseline_pt_abd_mean"], marker="s", linewidth=2.0, linestyle="--", color="#4c78a8", label="BASE_ABD")
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Shot")
    ax.set_ylabel("PT")
    ax.set_title(f"{qid} context-drift PT")
    ax.grid(alpha=0.28, linewidth=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _build_regime_meta(summary_df: pd.DataFrame, qid: str) -> Dict[str, Tuple[str, str]]:
    meta = {}
    qdf = summary_df[summary_df["q_id"] == qid]
    for regime in REGIME_ORDER:
        sdf = qdf[qdf["regime_id"] == regime]
        if sdf.empty:
            continue
        row = sdf.iloc[0]
        meta[regime] = (str(row["query_input"]), str(row["target_str"]))
    return meta


def _build_candidate_table(summary_df: pd.DataFrame, qid: str, regime: str) -> str:
    qdf = summary_df[(summary_df["q_id"] == qid) & (summary_df["regime_id"] == regime)].copy()
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

    sweep_csv = args.sweep_csv or os.path.join(run_dir, "pt_context_drift_shot_sweep.csv")
    bootstrap_csv = args.bootstrap_csv or os.path.join(run_dir, "pt_context_drift_bootstrap_summary.csv")
    topk_jsonl = args.topk_jsonl or os.path.join(run_dir, "pt_context_drift_edge_topk.jsonl")
    summary_csv = args.summary_csv or os.path.join(run_dir, "pt_context_drift_edge_topk_trial_agg_top20_by_q_shot_regime.csv")

    sweep_df = pd.read_csv(sweep_csv)
    boot_df = pd.read_csv(bootstrap_csv)
    topk_df = pd.read_json(topk_jsonl, lines=True)
    summary_df = _aggregate_topk(topk_df, summary_csv)

    qids = sorted(summary_df["q_id"].dropna().unique().tolist(), key=_qid_sort_key)
    summary_rows = []
    for qid in qids:
        target_prob_plot = os.path.join(plot_dir, f"{qid}_target_prob.png")
        pt_plot = os.path.join(plot_dir, f"{qid}_pt.png")
        _plot_target_prob(sweep_df, qid, target_prob_plot, dpi=args.dpi)
        _plot_pt(boot_df, qid, pt_plot, dpi=args.dpi)

        q_boot = boot_df[boot_df["q_id"] == qid].sort_values("shot")
        shot10 = q_boot[q_boot["shot"] == 10]
        pt_ctx_10 = float(shot10["pt_ctx_abd_mean"].iloc[0]) if not shot10.empty else None
        delta_10 = float(shot10["delta_vs_baseline_mean"].iloc[0]) if ("delta_vs_baseline_mean" in shot10.columns and not shot10.empty) else None
        summary_rows.append((qid, pt_ctx_10, delta_10))

        regime_meta = _build_regime_meta(summary_df, qid)
        sections = [
            '<div class="nav"><a href="index.html">Back to index</a></div>',
            f"<h1>{html.escape(qid)}</h1>",
            '<div class="grid">',
            f'<div class="card"><h2>Target Probability</h2><img src="plots/{qid}_target_prob.png" alt="{qid} target probability"></div>',
            f'<div class="card"><h2>Context PT Summary</h2><img src="plots/{qid}_pt.png" alt="{qid} context PT"></div>',
            "</div>",
        ]

        for regime in REGIME_ORDER:
            query_input, target_str = regime_meta.get(regime, ("", ""))
            sections.append(
                '<div class="card">'
                f"<h2>{regime}</h2>"
                f"<div class='small'>query={html.escape(query_input)} target={html.escape(target_str)}</div>"
                f"{_build_candidate_table(summary_df, qid, regime)}"
                "</div>"
            )

        page_html = _page_template(f"{qid} mixed PT report", "".join(sections))
        _write_text(os.path.join(out_dir, f"{qid}.html"), page_html)

    list_items = []
    for qid, pt_ctx_10, delta_10 in summary_rows:
        suffix_parts = []
        if pt_ctx_10 is not None:
            suffix_parts.append(f"shot10 CTX_ABD={pt_ctx_10:.3f}")
        if delta_10 is not None:
            suffix_parts.append(f"delta={delta_10:+.3f}")
        suffix = " ".join(suffix_parts)
        list_items.append(f'<li><a href="{qid}.html">{html.escape(qid)}</a><span class="small"> {html.escape(suffix)}</span></li>')

    index_body = "".join(
        [
            "<h1>Mixed-Context PT Human Report</h1>",
            f"<p class='small'>run_dir={html.escape(run_dir)}</p>",
            '<div class="grid">',
            '<div class="card"><h2>Overview</h2><p class="small">Open each q page to compare regime-wise shot curves, mixed-context PT, and shot-wise top candidates.</p></div>',
            '<div class="card"><h2>Available q_ids</h2><ul class="qid-list">' + "".join(list_items) + "</ul></div>",
            "</div>",
        ]
    )
    index_html = _page_template("Mixed-Context PT Human Report", index_body)
    _write_text(os.path.join(out_dir, "index.html"), index_html)
    print(f"report_dir={out_dir}")
    print(f"index_html={os.path.join(out_dir, 'index.html')}")
    print(f"summary_csv={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
