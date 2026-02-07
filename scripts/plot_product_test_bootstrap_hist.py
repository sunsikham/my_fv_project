#!/usr/bin/env python3
"""Plot bootstrap histogram distributions for PT (ABC/ABD/Delta)."""

import argparse
import os
import csv
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PT bootstrap histograms.")
    parser.add_argument(
        "--in_npz",
        required=False,
        help="Path to pt_bootstrap_<q_id>.npz",
    )
    parser.add_argument(
        "--in_dir",
        required=False,
        help="Directory containing pt_bootstrap_<q_id>.npz files",
    )
    parser.add_argument(
        "--summary_csv",
        required=False,
        default=None,
        help="Optional summary CSV with perm_p_one for significance stars",
    )
    parser.add_argument("--out_png", required=True, help="Output PNG path")
    parser.add_argument(
        "--bins", type=int, default=50, help="Histogram bins (default: 50)"
    )
    parser.add_argument(
        "--max_cols",
        type=int,
        default=6,
        help="Max columns for grid when using --in_dir",
    )
    parser.add_argument(
        "--title", required=False, default=None, help="Optional plot title"
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure dpi")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    abd_gt1_map = {}
    if args.summary_csv:
        with open(args.summary_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_id = str(row.get("q_id", "")).strip()
                if not q_id:
                    continue
                try:
                    abd_gt1_map[q_id] = float(row.get("pt_abd_p_gt1", ""))
                except ValueError:
                    continue

    if args.in_dir:
        files = [
            os.path.join(args.in_dir, f)
            for f in sorted(os.listdir(args.in_dir))
            if f.startswith("pt_bootstrap_") and f.endswith(".npz")
        ]
        if not files:
            raise ValueError("No pt_bootstrap_*.npz files found in --in_dir")
        n = len(files)
        n_cols = min(args.max_cols, n)
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = axes.flatten()
        for ax, path in zip(axes, files):
            data = np.load(path)
            pt_abc = data["pt_abc"]
            pt_abd = data["pt_abd"]
            delta = data["delta"]
            q_id = data.get("q_id", None)
            ax.hist(pt_abc, bins=args.bins, color="#1f77b4", alpha=0.6, label="ABC")
            ax.hist(pt_abd, bins=args.bins, color="#ff7f0e", alpha=0.6, label="ABD")
            ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
            if q_id is None:
                q_label = os.path.basename(path).replace("pt_bootstrap_", "").replace(".npz", "")
            else:
                q_label = str(q_id)
            star = ""
            p_gt1 = abd_gt1_map.get(q_label)
            if p_gt1 is not None:
                if p_gt1 >= 0.999:
                    star = "***"
                elif p_gt1 >= 0.99:
                    star = "**"
                elif p_gt1 >= 0.95:
                    star = "*"
            title = f"Q={q_label} (mean ABC={pt_abc.mean():.3f}, ABD={pt_abd.mean():.3f}){star}"
            title_color = "red" if star else "black"
            ax.set_title(title, color=title_color)
            ax.legend(loc="best", fontsize=8)
        for ax in axes[len(files):]:
            ax.axis("off")
        if args.title:
            fig.suptitle(args.title)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
        fig.savefig(args.out_png, dpi=args.dpi)
        return 0

    if not args.in_npz:
        raise ValueError("Provide --in_npz or --in_dir")

    data = np.load(args.in_npz)
    pt_abc = data["pt_abc"]
    pt_abd = data["pt_abd"]
    delta = data["delta"]
    q_id = data.get("q_id", None)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axes[0].hist(pt_abc, bins=args.bins, color="#1f77b4", alpha=0.75)
    axes[0].set_title("PT_ABC")
    axes[0].axvline(1.0, color="gray", linestyle="--", linewidth=1)

    axes[1].hist(pt_abd, bins=args.bins, color="#ff7f0e", alpha=0.75)
    axes[1].set_title("PT_ABD")
    axes[1].axvline(1.0, color="gray", linestyle="--", linewidth=1)

    axes[2].hist(delta, bins=args.bins, color="#2ca02c", alpha=0.75)
    axes[2].set_title("DELTA")
    axes[2].axvline(0.0, color="gray", linestyle="--", linewidth=1)

    if args.title:
        fig.suptitle(args.title)
    elif q_id is not None:
        fig.suptitle(f"Bootstrap Distributions (q_id={q_id})")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.savefig(args.out_png, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
