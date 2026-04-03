#!/usr/bin/env python3
"""Plot mixed-context PT summary across shots, optionally against baseline ABD."""

import argparse
import csv
import math
import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLS = {
    "q_id",
    "shot",
    "pt_ctx_abd_mean",
    "pt_ctx_abd_ci2_low",
    "pt_ctx_abd_ci2_high",
    "pt_ctx_abd_p_gt1",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mixed-context PT summary across shots."
    )
    parser.add_argument("--in_csv", required=True, help="Input mixed-context summary CSV")
    parser.add_argument("--out_png", required=True, help="Output PNG path")
    parser.add_argument("--qid", required=False, default=None, help="Single q_id")
    parser.add_argument("--qids", required=False, default=None, help="Comma-separated q_id list")
    parser.add_argument("--baseline_csv", required=False, default=None, help="Optional baseline PT bootstrap summary CSV")
    parser.add_argument("--max_cols", type=int, default=6, help="Max columns for subplot grid")
    parser.add_argument("--show_p", choices=["none", "stars", "text"], default="stars", help="p-value display mode")
    parser.add_argument("--ymin", type=float, default=None, help="Optional y-axis lower bound")
    parser.add_argument("--ymax", type=float, default=None, help="Optional y-axis upper bound")
    parser.add_argument("--title", required=False, default=None, help="Figure title")
    parser.add_argument("--dpi", type=int, default=200, help="Figure dpi")
    return parser.parse_args()


def _load_rows(path: str, required_cols) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Missing header row in input CSV")
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")
        return list(reader)


def _parse_qids(args: argparse.Namespace) -> Optional[Sequence[str]]:
    if args.qid:
        return [args.qid]
    if args.qids:
        return [q.strip() for q in args.qids.split(",") if q.strip()]
    return None


def _p_from_gt1(p_gt1: float) -> float:
    return max(0.0, min(1.0, 1.0 - p_gt1))


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _group_rows(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["q_id"], []).append(row)
    return grouped


def _safe_float(row: Dict[str, str], key: str):
    val = row.get(key)
    if val is None or val == "":
        return None
    return float(val)


def _index_by_qid_shot(rows: List[Dict[str, str]]) -> Dict[tuple, Dict[str, str]]:
    return {(row["q_id"], int(row["shot"])): row for row in rows}


def _plot_panel(ax, rows: List[Dict[str, str]], baseline_index, show_p: str, ymin, ymax):
    rows_sorted = sorted(rows, key=lambda r: int(r["shot"]))
    shots = [int(r["shot"]) for r in rows_sorted]
    ctx_mean = [_safe_float(r, "pt_ctx_abd_mean") for r in rows_sorted]
    ctx_lo = [_safe_float(r, "pt_ctx_abd_ci2_low") for r in rows_sorted]
    ctx_hi = [_safe_float(r, "pt_ctx_abd_ci2_high") for r in rows_sorted]

    ctx_line = ax.plot(shots, ctx_mean, marker="o", label="CTX_ABD")[0]
    if all(v is not None for v in ctx_lo + ctx_hi):
        ctx_err = [
            [m - l for m, l in zip(ctx_mean, ctx_lo)],
            [h - m for m, h in zip(ctx_mean, ctx_hi)],
        ]
        ax.errorbar(shots, ctx_mean, yerr=ctx_err, fmt="none", capsize=3, color=ctx_line.get_color())

    if baseline_index:
        q_id = rows_sorted[0]["q_id"]
        base_rows = [baseline_index[(q_id, shot)] for shot in shots if (q_id, shot) in baseline_index]
        if base_rows:
            base_shots = [int(r["shot"]) for r in base_rows]
            base_mean = [_safe_float(r, "pt_abd_mean") for r in base_rows]
            base_lo = [_safe_float(r, "pt_abd_ci2_low") for r in base_rows]
            base_hi = [_safe_float(r, "pt_abd_ci2_high") for r in base_rows]
            base_line = ax.plot(base_shots, base_mean, marker="o", linestyle="--", label="BASE_ABD")[0]
            if all(v is not None for v in base_lo + base_hi):
                base_err = [
                    [m - l for m, l in zip(base_mean, base_lo)],
                    [h - m for m, h in zip(base_mean, base_hi)],
                ]
                ax.errorbar(base_shots, base_mean, yerr=base_err, fmt="none", capsize=3, color=base_line.get_color())

    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("shot")
    ax.set_ylabel("PT")
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    if show_p != "none":
        for i, row in enumerate(rows_sorted):
            p_ctx = _p_from_gt1(float(row["pt_ctx_abd_p_gt1"]))
            if show_p == "stars":
                mark = _stars(p_ctx)
                if mark:
                    ax.text(shots[i], ctx_mean[i], mark, ha="center", va="bottom", fontsize=10)
            else:
                ax.text(shots[i], ctx_mean[i], f"p={p_ctx:.3f}", ha="center", va="bottom", fontsize=8)


def main() -> int:
    args = _parse_args()
    rows = _load_rows(args.in_csv, REQUIRED_COLS)
    qid_filter = _parse_qids(args)
    if qid_filter is not None:
        rows = [r for r in rows if r["q_id"] in set(qid_filter)]
    if not rows:
        raise ValueError("No rows to plot after filtering")

    baseline_index = None
    if args.baseline_csv:
        baseline_rows = _load_rows(
            args.baseline_csv,
            {"q_id", "shot", "pt_abd_mean", "pt_abd_ci2_low", "pt_abd_ci2_high", "pt_abd_p_gt1"},
        )
        baseline_index = _index_by_qid_shot(baseline_rows)

    grouped = _group_rows(rows)
    qids = sorted(grouped.keys())
    n = len(qids)
    n_cols = min(max(1, args.max_cols), n)
    n_rows = int(math.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, q_id in zip(axes, qids):
        _plot_panel(ax, grouped[q_id], baseline_index, args.show_p, args.ymin, args.ymax)
        ax.set_title(q_id)
    for ax in axes[len(qids):]:
        ax.axis("off")

    axes[0].legend(loc="best")
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    out_dir = os.path.dirname(args.out_png) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out_png, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
