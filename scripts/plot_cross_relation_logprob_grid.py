#!/usr/bin/env python3
"""
Example:
  python scripts/plot_cross_relation_logprob_grid.py \
    --in_csv results/cross_relation_logits.csv \
    --out_pdf results/cross_relation_logprob_grid.pdf \
    --out_png results/cross_relation_logprob_grid.png \
    --ncols 6 \
    --sharey 1
"""

import argparse
import re
import warnings

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-q_id mean target_logprob vs n_demos grid."
    )
    parser.add_argument("--in_csv", required=True, help="Input CSV path")
    parser.add_argument("--out_pdf", required=True, help="Output PDF path")
    parser.add_argument("--out_png", required=False, default=None, help="Output PNG path")
    parser.add_argument("--ncols", type=int, default=6, help="Number of subplot columns")
    parser.add_argument("--sharey", type=int, default=1, help="Share y-axis (1/0)")
    parser.add_argument("--title_fontsize", type=int, default=7, help="Title font size")
    parser.add_argument("--label_fontsize", type=int, default=7, help="Label font size")
    parser.add_argument("--tick_fontsize", type=int, default=6, help="Tick font size")
    return parser.parse_args()


def _qid_sort_key(qid: str) -> int:
    match = re.search(r"(\d+)", str(qid))
    return int(match.group(1)) if match else 1_000_000


def main() -> int:
    args = _parse_args()

    df = pd.read_csv(args.in_csv)
    required = {
        "q_id",
        "n_demos",
        "trial_index",
        "query_input",
        "target_str",
        "target_token_str",
        "target_logprob",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    agg = (
        df.groupby(["q_id", "n_demos"], as_index=False)["target_logprob"]
        .mean()
        .rename(columns={"target_logprob": "mean_target_logprob"})
    )
    meta = (
        df.sort_values(["q_id", "n_demos", "trial_index"])
        .groupby("q_id", as_index=False)
        .first()
    )

    qids = sorted(meta["q_id"].unique(), key=_qid_sort_key)
    n_plots = len(qids)
    if n_plots == 0:
        raise ValueError("No q_id entries found in CSV")

    ncols = max(1, int(args.ncols))
    nrows = (n_plots + ncols - 1) // ncols
    figsize = (ncols * 2.0, nrows * 2.2)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharey=bool(args.sharey),
    )
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    missing_qids = []
    for idx, qid in enumerate(qids):
        ax = axes[idx]
        series = agg[agg["q_id"] == qid].sort_values("n_demos")
        if series.empty:
            missing_qids.append(qid)
            ax.set_visible(False)
            continue
        x = series["n_demos"].tolist()
        y = series["mean_target_logprob"].tolist()
        ax.plot(x, y, marker="o")

        meta_row = meta[meta["q_id"] == qid]
        query_input = meta_row["query_input"].iloc[0]
        target_str = meta_row["target_str"].iloc[0]
        target_token_str = meta_row["target_token_str"].iloc[0]
        title = f"{query_input}:{target_str},{target_token_str}"
        ax.set_title(title, fontsize=args.title_fontsize)

        ax.tick_params(axis="both", labelsize=args.tick_fontsize)

        row_idx = idx // ncols
        col_idx = idx % ncols
        if row_idx == nrows - 1:
            ax.set_xlabel("n_demos", fontsize=args.label_fontsize)
        if col_idx == 0:
            ax.set_ylabel("mean logprob", fontsize=args.label_fontsize)

    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    if missing_qids:
        warnings.warn(f"Missing q_id series for: {sorted(missing_qids)}")

    fig.tight_layout()
    fig.savefig(args.out_pdf)
    if args.out_png:
        fig.savefig(args.out_png, dpi=250)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
