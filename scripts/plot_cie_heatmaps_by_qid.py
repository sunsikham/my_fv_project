#!/usr/bin/env python3
"""Plot CIE heatmaps per q_id with the same style as AIE heatmap."""

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CIE heatmaps per q_id.")
    parser.add_argument("--in_csv", required=False, help="Path to cie_scores.csv")
    parser.add_argument(
        "--in_dir",
        required=False,
        default=None,
        help="Directory containing cie_scores.csv (default: results/attention_head)",
    )
    parser.add_argument(
        "--metric", default="mean_delta_p", help="Metric column (default: mean_delta_p)"
    )
    parser.add_argument(
        "--metric_abs",
        action="store_true",
        help="Plot absolute value of metric for easier comparison (default: off)",
    )
    parser.add_argument(
        "--out_dir",
        required=False,
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap name")
    parser.add_argument("--vmin", type=float, default=None, help="Color scale min")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale max")
    parser.add_argument(
        "--robust",
        type=int,
        default=1,
        help="Enable robust clipping via percentiles (default: 1)",
    )
    parser.add_argument("--q_low", type=float, default=1.0, help="Low percentile (default: 1)")
    parser.add_argument("--q_high", type=float, default=99.0, help="High percentile (default: 99)")
    parser.add_argument(
        "--nonzero_only",
        type=int,
        default=None,
        help="Use nonzero values for percentile stats (default: 1 for mean_delta_p, else 0)",
    )
    parser.add_argument(
        "--norm",
        choices=["linear", "twoslope", "symlog"],
        default="symlog",
        help="Normalization (default: symlog)",
    )
    parser.add_argument(
        "--linthresh",
        type=float,
        default=None,
        help="Symlog linthresh (default: auto)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Overlay top-k points by metric (default: 0 = off)",
    )
    parser.add_argument("--title", default=None, help="Optional plot title prefix")
    parser.add_argument("--dpi", type=int, default=200, help="Output dpi (default: 200)")
    return parser.parse_args()


def _apply_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _resolve_heatmap_scale(
    values: np.ndarray,
    vmin: Optional[float],
    vmax: Optional[float],
    robust: bool,
    q_low: float,
    q_high: float,
    nonzero_only: bool,
) -> Tuple[Optional[float], Optional[float], float, float, float]:
    if vmin is not None or vmax is not None:
        used_vmin = vmin
        used_vmax = vmax
        return used_vmin, used_vmax, float("nan"), float("nan"), 1.0

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None, float("nan"), float("nan"), 1.0

    use_vals = finite
    nonzero_ratio = float(np.count_nonzero(finite)) / float(finite.size)
    if nonzero_only:
        nonzero = finite[finite != 0]
        if nonzero.size > 0:
            use_vals = nonzero

    if robust:
        ql = float(np.percentile(use_vals, q_low))
        qh = float(np.percentile(use_vals, q_high))
        return ql, qh, ql, qh, nonzero_ratio

    return float(np.min(use_vals)), float(np.max(use_vals)), float("nan"), float("nan"), nonzero_ratio


def _auto_linthresh(values: np.ndarray, nonzero_only: bool) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1e-6
    if nonzero_only:
        finite = finite[finite != 0]
    if finite.size == 0:
        return 1e-6
    abs_vals = np.abs(finite)
    p = np.percentile(abs_vals, 20)
    return float(p if p > 0 else np.max(abs_vals) * 0.01 + 1e-6)


def _build_norm(norm_name: str, vmin: Optional[float], vmax: Optional[float], linthresh: Optional[float]):
    if norm_name == "linear":
        return None
    if norm_name == "twoslope":
        if vmin is None or vmax is None:
            return None
        return mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    if norm_name == "symlog":
        if vmin is None or vmax is None:
            return None
        if linthresh is None:
            linthresh = 1e-6
        return mcolors.SymLogNorm(
            linthresh=linthresh,
            linscale=1.0,
            vmin=vmin,
            vmax=vmax,
            base=10,
        )
    return None


def _resolve_grid(df: pd.DataFrame, metric: str) -> Tuple[List[int], List[int], np.ndarray]:
    n_layers = int(df["layer"].max()) + 1
    n_heads = int(df["head"].max()) + 1
    matrix = np.zeros((n_layers, n_heads), dtype=float)
    for _, row in df.iterrows():
        layer = int(row["layer"])
        head = int(row["head"])
        value = float(row[metric])
        if 0 <= layer < n_layers and 0 <= head < n_heads:
            matrix[layer, head] = value
    layers = list(range(n_layers))
    heads = list(range(n_heads))
    return layers, heads, matrix


def main() -> int:
    args = parse_args()
    _apply_plot_style()

    in_csv = args.in_csv
    if in_csv is None:
        base_dir = args.in_dir or os.path.join("results", "attention_head")
        in_csv = os.path.join(base_dir, "cie_scores.csv")
    if not os.path.exists(in_csv):
        print(f"Missing cie_scores.csv at: {in_csv}")
        return 1

    df = pd.read_csv(in_csv)
    if args.metric not in df.columns:
        print(f"Metric not found: {args.metric}")
        return 1

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(in_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    metric_label = f"abs({args.metric})" if args.metric_abs else args.metric
    nonzero_only = args.nonzero_only
    if nonzero_only is None:
        nonzero_only = 1 if args.metric in {"mean_delta_p", "mean_delta_logit", "mean_delta_logprob"} else 0

    for q_id in sorted(df["q_id"].astype(str).unique()):
        sub = df[df["q_id"].astype(str) == q_id].copy()
        if args.metric_abs:
            sub[metric_label] = sub[args.metric].abs()
            metric_key = metric_label
        else:
            metric_key = args.metric

        layers, heads, matrix = _resolve_grid(sub, metric_key)
        values = matrix if not args.metric_abs else np.abs(matrix)

        vmin, vmax, ql, qh, nonzero_ratio = _resolve_heatmap_scale(
            values,
            args.vmin,
            args.vmax,
            bool(args.robust),
            args.q_low,
            args.q_high,
            bool(nonzero_only),
        )
        linthresh = args.linthresh
        if linthresh is None and args.norm == "symlog":
            linthresh = _auto_linthresh(values, bool(nonzero_only))
        norm = _build_norm(args.norm, vmin, vmax, linthresh)

        fig, ax = plt.subplots(figsize=(7, 5))
        imshow_kwargs = {"aspect": "auto", "cmap": args.cmap}
        if norm is not None:
            imshow_kwargs["norm"] = norm
        if vmin is not None and vmax is not None and norm is None:
            imshow_kwargs["vmin"] = vmin
            imshow_kwargs["vmax"] = vmax
        im = ax.imshow(matrix, **imshow_kwargs)

        if args.topk and args.topk > 0:
            flat = matrix.flatten()
            topk = min(args.topk, flat.size)
            idxs = np.argsort(flat)[-topk:]
            for idx in idxs:
                i = idx // len(heads)
                j = idx % len(heads)
                ax.scatter(j, i, s=18, c="white", marker="o", edgecolors="black", linewidths=0.3)

        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels(heads)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)

        title = f"CIE heatmap Q={q_id} ({metric_label})"
        if args.title:
            title = f"{args.title} Q={q_id} ({metric_label})"
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"cie_heatmap_q{q_id}_{metric_label}.png")
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
