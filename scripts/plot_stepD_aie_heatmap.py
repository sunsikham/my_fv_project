#!/usr/bin/env python3
"""Plot StepD AIE heatmap.

Example:
  python scripts/plot_stepD_aie_heatmap.py --run_id stepD_llama31_8b_smoke --metric mean_delta_p --topk 20
"""

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def parse_args():
    parser = argparse.ArgumentParser(description="Plot StepD AIE heatmap from aie_scores.csv.")
    parser.add_argument("--run_id", required=True, help="StepD run id (required)")
    parser.add_argument(
        "--metric", default="mean_delta_p", help="Metric column (default: mean_delta_p)"
    )
    parser.add_argument(
        "--metric_abs",
        action="store_true",
        help="Plot absolute value of metric for easier comparison (default: off)",
    )
    parser.add_argument(
        "--out_name",
        default="aie_heatmap_{metric}.png",
        help="Output filename (default: aie_heatmap_{metric}.png)",
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name")
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
    parser.add_argument(
        "--topk_layers",
        type=int,
        default=50,
        help="Top-k heads for layer distribution plot (default: 50; 0 = off)",
    )
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--dpi", type=int, default=200, help="Output dpi (default: 200)")
    return parser.parse_args()


def _suggest_numeric_columns(df: pd.DataFrame) -> List[str]:
    candidates = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    return candidates[:5]


def _safe_int_series(series: pd.Series) -> pd.Series:
    try:
        return series.astype(int)
    except Exception:
        return series


def _tick_positions(values: List[int], max_ticks: int = 20) -> Tuple[List[int], List[str]]:
    if not values:
        return [], []
    if len(values) <= max_ticks:
        return list(range(len(values))), [str(v) for v in values]
    step = max(1, len(values) // max_ticks)
    idxs = list(range(0, len(values), step))
    return idxs, [str(values[i]) for i in idxs]


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

    csv_path = os.path.join("runs", args.run_id, "artifacts", "aie_scores.csv")
    if not os.path.exists(csv_path):
        print(f"Missing aie_scores.csv at: {csv_path}")
        return 1

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"Failed to read CSV at {csv_path}: {exc}")
        return 1

    required_cols = {"layer", "head"}
    if args.metric != "mean_abs_delta_p" or "mean_abs_delta_p" in df.columns:
        required_cols.add(args.metric)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        suggestions = _suggest_numeric_columns(df)
        suggestion_msg = ""
        if suggestions:
            suggestion_msg = f" Available numeric columns: {', '.join(suggestions)}"
        print(f"Missing columns in CSV: {', '.join(missing)}.{suggestion_msg}")
        return 1

    df = df.copy()
    df["layer"] = _safe_int_series(df["layer"])
    df["head"] = _safe_int_series(df["head"])
    df = df.sort_values(["layer", "head"])

    metric_name = args.metric
    metric_label = metric_name
    if args.metric == "mean_abs_delta_p" and "mean_abs_delta_p" not in df.columns:
        if "mean_delta_p" not in df.columns:
            print("Missing mean_delta_p needed to compute mean_abs_delta_p.")
            return 1
        df["mean_abs_delta_p"] = df["mean_delta_p"].abs()
    if args.metric_abs:
        metric_label = f"abs({metric_name})"
        df[metric_label] = df[metric_name].abs()
        metric_name = metric_label

    layers, heads, matrix = _resolve_grid(df, metric_name)

    if args.out_name == "aie_heatmap_{metric}.png":
        out_name = args.out_name.format(metric=args.metric)
    else:
        out_name = args.out_name
    out_path = os.path.join("runs", args.run_id, "artifacts", out_name)

    if args.nonzero_only is None:
        nonzero_only = args.metric == "mean_delta_p"
    else:
        nonzero_only = bool(args.nonzero_only)

    title = args.title
    if title is None:
        title = f"AIE heatmap ({metric_label})"

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=args.dpi)
    vmin, vmax, ql, qh, nonzero_ratio = _resolve_heatmap_scale(
        matrix,
        args.vmin,
        args.vmax,
        robust=bool(args.robust),
        q_low=args.q_low,
        q_high=args.q_high,
        nonzero_only=nonzero_only,
    )
    cmap = args.cmap
    if args.norm in {"twoslope", "symlog"} and args.cmap == "viridis":
        cmap = "coolwarm"
    linthresh = args.linthresh
    if args.norm == "symlog" and linthresh is None:
        linthresh = _auto_linthresh(matrix, nonzero_only=nonzero_only)
    norm = _build_norm(args.norm, vmin, vmax, linthresh)
    scale_bits = [f"vmin={vmin:.3g}", f"vmax={vmax:.3g}", f"norm={args.norm}"]
    if args.robust:
        scale_bits.append(f"q={args.q_low:g}..{args.q_high:g}")
    if args.norm == "symlog":
        scale_bits.append(f"linthresh={linthresh:.3g}")
    scale_note = ", ".join(scale_bits)
    imshow_kwargs = {
        "aspect": "auto",
        "origin": "lower",
        "cmap": cmap,
        "interpolation": "nearest",
        "norm": norm,
    }
    if norm is None:
        imshow_kwargs["vmin"] = vmin
        imshow_kwargs["vmax"] = vmax
    im = ax.imshow(matrix, **imshow_kwargs)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_label)

    x_ticks, x_labels = _tick_positions(heads)
    y_ticks, y_labels = _tick_positions(layers)
    ax.set_xticks(x_ticks, x_labels)
    ax.set_yticks(y_ticks, y_labels)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(f"{title}\n{scale_note}")

    if args.topk and args.topk > 0:
        topk = df.sort_values(metric_name, ascending=False).head(args.topk)
        head_to_idx = {head: i for i, head in enumerate(heads)}
        layer_to_idx = {layer: i for i, layer in enumerate(layers)}
        xs = [head_to_idx[row["head"]] for _, row in topk.iterrows() if row["head"] in head_to_idx]
        ys = [layer_to_idx[row["layer"]] for _, row in topk.iterrows() if row["layer"] in layer_to_idx]
        ax.scatter(xs, ys, s=28, c="white", marker="o", edgecolors="black", linewidths=0.6)

    fig.tight_layout()
    try:
        fig.savefig(out_path)
    except Exception as exc:
        print(f"Failed to save figure to {out_path}: {exc}")
        return 1
    finally:
        plt.close(fig)

    print(f"Saved: {out_path}")

    if args.topk_layers and args.topk_layers > 0:
        n_layers = len(layers)
        topk_n = args.topk_layers
        if topk_n > n_layers:
            topk_n = n_layers
            print(f"Warning: topk_layers clamped to {topk_n} (n_layers={n_layers})")
        if topk_n > len(df):
            topk_n = len(df)
        if topk_n > n_layers * len(heads):
            topk_n = n_layers * len(heads)
        if args.topk_layers > topk_n and args.topk_layers <= n_layers:
            print(f"Warning: topk_layers clamped to {topk_n} (requested {args.topk_layers})")
        topk_layers = df.sort_values(metric_name, ascending=False).head(topk_n)
        layer_counts = (
            topk_layers.groupby("layer").size().reindex(range(n_layers), fill_value=0)
        )
        if layer_counts.empty:
            print("No data for layer distribution plot.")
            return 0

        max_count = int(layer_counts.max())
        max_layers = layer_counts[layer_counts == max_count].index.tolist()
        max_layer_str = ", ".join(str(layer) for layer in max_layers)
        print(
            f"Top heads by layer (topk_layers={topk_n}): "
            f"max_count={max_count} at layer(s) {max_layer_str}"
        )

        layer_out_name = f"aie_top_heads_by_layer_{metric_label}.png"
        layer_out_path = os.path.join("runs", args.run_id, "artifacts", layer_out_name)

        fig, ax = plt.subplots(figsize=(11, 4.5), dpi=args.dpi)
        bars = ax.bar(layer_counts.index.astype(int), layer_counts.values, color="#2f5aa6")
        for bar, layer in zip(bars, layer_counts.index):
            if layer in max_layers:
                bar.set_color("#c44536")
        ax.set_title(f"Top-{topk_n} head distribution by layer ({metric_label})")
        ax.set_xlabel("layer")
        ax.set_ylabel("count")
        ax.set_ylim(0, max_count * 1.15)
        ax.set_xticks(*_tick_positions(layer_counts.index.astype(int).tolist(), max_ticks=20))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        try:
            fig.savefig(layer_out_path)
        except Exception as exc:
            print(f"Failed to save layer distribution to {layer_out_path}: {exc}")
            return 1
        finally:
            plt.close(fig)
        print(f"Saved: {layer_out_path}")

    print(
        "Summary: n_layers={}, n_heads={}, nonzero_ratio={:.3f}, vmin={}, vmax={}, top_layer={}".format(
            len(layers),
            len(heads),
            nonzero_ratio,
            f"{vmin:.3g}" if vmin is not None else "None",
            f"{vmax:.3g}" if vmax is not None else "None",
            max_layer_str if args.topk_layers and args.topk_layers > 0 else "n/a",
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
