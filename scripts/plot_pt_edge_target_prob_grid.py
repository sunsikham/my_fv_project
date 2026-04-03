#!/usr/bin/env python3
"""
Plot per-q small multiples of mean target probability vs shot for selected PT edges.

Example:
  /home/sunsik/.venvs/pt442/bin/python scripts/plot_pt_edge_target_prob_grid.py \
    --in_csv /scratch/sunsik/my_fv_project/pt_analysis/<run>/pt_5edge_shot_sweep.csv \
    --out_png /scratch/sunsik/my_fv_project/pt_analysis/<run>/pt_ab_ad_bd_target_prob_grid.png
"""

import argparse
import math
import re

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot mean target probability vs shot for selected PT edges by q_id."
    )
    p.add_argument("--in_csv", required=True, help="Input pt_5edge_shot_sweep.csv")
    p.add_argument("--out_png", required=True, help="Output PNG path")
    p.add_argument("--out_pdf", default=None, help="Optional output PDF path")
    p.add_argument("--edges", default="AB,AD,BD", help="Comma-separated edges to plot")
    p.add_argument("--metric", default="target_prob_raw", choices=["target_prob_raw", "target_logprob_raw", "target_s_norm"])
    p.add_argument("--ncols", type=int, default=4, help="Subplot columns")
    p.add_argument("--dpi", type=int, default=220, help="PNG dpi")
    return p.parse_args()


def _qid_sort_key(qid: str) -> int:
    m = re.search(r"(\d+)", str(qid))
    return int(m.group(1)) if m else 1_000_000


def main() -> int:
    args = _parse_args()
    df = pd.read_csv(args.in_csv)
    required = {"q_id", "shot", "edge", "trial_index", "query_input", "target_str", args.metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    edges = [x.strip() for x in args.edges.split(",") if x.strip()]
    if not edges:
        raise ValueError("No edges provided")

    df = df[df["edge"].isin(edges)].copy()
    if df.empty:
        raise ValueError(f"No rows found for edges={edges}")

    agg = (
        df.groupby(["q_id", "edge", "shot"], as_index=False)[args.metric]
        .mean()
        .rename(columns={args.metric: "mean_value"})
    )
    meta_cols = ["q_id", "query_input", "target_str"]
    if "gold_target_str" in df.columns:
        meta_cols.append("gold_target_str")
    meta = (
        df.sort_values(["q_id", "edge", "shot", "trial_index"])
        .groupby("q_id", as_index=False)
        .first()[meta_cols]
    )

    qids = sorted(agg["q_id"].unique().tolist(), key=_qid_sort_key)
    ncols = max(1, int(args.ncols))
    nrows = math.ceil(len(qids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 3.3 * nrows), sharex=True, sharey=False)
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    color_map = {"AB": "#1f77b4", "AD": "#d62728", "BD": "#2ca02c"}
    linestyle_map = {"AB": "-", "AD": "--", "BD": "-."}
    marker_map = {"AB": "o", "AD": "s", "BD": "^"}
    label_map = {"AB": "AB", "AD": "AD", "BD": "BD"}
    y_label = {
        "target_prob_raw": "Mean target probability",
        "target_logprob_raw": "Mean target logprob",
        "target_s_norm": "Mean normalized target score",
    }[args.metric]

    for idx, qid in enumerate(qids):
        ax = axes[idx]
        q_agg = agg[agg["q_id"] == qid]
        local_y = []
        for edge in edges:
            series = q_agg[q_agg["edge"] == edge].sort_values("shot")
            if series.empty:
                continue
            y_vals = series["mean_value"].tolist()
            local_y.extend(y_vals)
            ax.plot(
                series["shot"].tolist(),
                y_vals,
                marker=marker_map.get(edge, "o"),
                linestyle=linestyle_map.get(edge, "-"),
                linewidth=2.2,
                markersize=5.0,
                color=color_map.get(edge),
                label=label_map.get(edge, edge),
            )
        if local_y:
            y_min = min(local_y)
            y_max = max(local_y)
            if y_min == y_max:
                pad = max(abs(y_min) * 0.08, 0.01)
            else:
                pad = max((y_max - y_min) * 0.12, 0.01)
            ax.set_ylim(y_min - pad, y_max + pad)
        meta_row = meta[meta["q_id"] == qid].iloc[0]
        title = f"{qid}: {meta_row['query_input']} -> {meta_row['target_str']}"
        gold_target = str(meta_row["gold_target_str"]) if "gold_target_str" in meta_row.index else ""
        if gold_target and gold_target != str(meta_row["target_str"]):
            title += f" (gold: {gold_target})"
        ax.set_title(title, fontsize=10, pad=8)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(alpha=0.28, linewidth=0.6)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Shot", fontsize=11)
        if idx % ncols == 0:
            ax.set_ylabel(y_label, fontsize=11)

    for idx in range(len(qids), len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=len(edges),
            frameon=True,
            fontsize=11,
            fancybox=True,
        )
    fig.suptitle("PT Target Probability by Shot and Edge", fontsize=15, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(args.out_png, dpi=args.dpi)
    if args.out_pdf:
        fig.savefig(args.out_pdf)
    plt.close(fig)
    print(f"saved_png={args.out_png}")
    if args.out_pdf:
        print(f"saved_pdf={args.out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
