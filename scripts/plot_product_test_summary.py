#!/usr/bin/env python3
"""Plot product test summary (ABC vs ABD) across shots with CI bars."""

import argparse
import csv
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLS = {
    "q_id",
    "shot",
    "pt_abc_mean",
    "pt_abd_mean",
    "pt_abc_ci2_low",
    "pt_abc_ci2_high",
    "pt_abd_ci2_low",
    "pt_abd_ci2_high",
    "pt_abc_p_gt1",
    "pt_abd_p_gt1",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot product test summary (ABC vs ABD) across shots."
    )
    parser.add_argument("--in_csv", required=True, help="Input summary CSV")
    parser.add_argument("--out_png", required=True, help="Output PNG path")
    parser.add_argument("--qid", required=False, default=None, help="Single q_id")
    parser.add_argument(
        "--qids",
        required=False,
        default=None,
        help="Comma-separated q_id list",
    )
    parser.add_argument(
        "--max_cols",
        type=int,
        default=6,
        help="Max columns for subplot grid",
    )
    parser.add_argument(
        "--show_p",
        choices=["none", "stars", "text"],
        default="stars",
        help="p-value display mode",
    )
    parser.add_argument(
        "--p_mode",
        choices=["tail"],
        default="tail",
        help="p-value calculation mode",
    )
    parser.add_argument("--title", required=False, default=None, help="Figure title")
    parser.add_argument("--dpi", type=int, default=200, help="Figure dpi")
    return parser.parse_args()


def _load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Missing header row in input CSV")
        missing = REQUIRED_COLS - set(reader.fieldnames)
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


def _safe_float(row: Dict[str, str], key: str) -> Optional[float]:
    val = row.get(key)
    if val is None or val == "":
        return None
    return float(val)


def _plot_panel(
    ax,
    rows: List[Dict[str, str]],
    show_p: str,
):
    rows_sorted = sorted(rows, key=lambda r: int(r["shot"]))
    shots = [int(r["shot"]) for r in rows_sorted]
    abc_mean = [_safe_float(r, "pt_abc_mean") for r in rows_sorted]
    abd_mean = [_safe_float(r, "pt_abd_mean") for r in rows_sorted]

    abc_lo = [_safe_float(r, "pt_abc_ci2_low") for r in rows_sorted]
    abc_hi = [_safe_float(r, "pt_abc_ci2_high") for r in rows_sorted]
    abd_lo = [_safe_float(r, "pt_abd_ci2_low") for r in rows_sorted]
    abd_hi = [_safe_float(r, "pt_abd_ci2_high") for r in rows_sorted]

    abc_line = ax.plot(shots, abc_mean, marker="o", label="ABC")[0]
    abd_line = ax.plot(shots, abd_mean, marker="o", label="ABD")[0]

    # error bars if CI2 available
    if all(v is not None for v in abc_lo + abc_hi):
        abc_err = [
            [m - l for m, l in zip(abc_mean, abc_lo)],
            [h - m for m, h in zip(abc_mean, abc_hi)],
        ]
        ax.errorbar(
            shots,
            abc_mean,
            yerr=abc_err,
            fmt="none",
            capsize=3,
            color=abc_line.get_color(),
        )
    if all(v is not None for v in abd_lo + abd_hi):
        abd_err = [
            [m - l for m, l in zip(abd_mean, abd_lo)],
            [h - m for m, h in zip(abd_mean, abd_hi)],
        ]
        ax.errorbar(
            shots,
            abd_mean,
            yerr=abd_err,
            fmt="none",
            capsize=3,
            color=abd_line.get_color(),
        )

    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("shot")
    ax.set_ylabel("PT")

    if show_p != "none":
        for i, row in enumerate(rows_sorted):
            p_abd = _p_from_gt1(float(row["pt_abd_p_gt1"]))
            if show_p == "stars":
                mark = _stars(p_abd)
                if mark:
                    ax.text(
                        shots[i],
                        abd_mean[i],
                        mark,
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
            elif show_p == "text":
                ax.text(
                    shots[i],
                    abd_mean[i],
                    f"p={p_abd:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )


def main() -> int:
    args = _parse_args()
    rows = _load_rows(args.in_csv)

    qid_filter = _parse_qids(args)
    if qid_filter is not None:
        rows = [r for r in rows if r["q_id"] in set(qid_filter)]

    if not rows:
        raise ValueError("No rows to plot after filtering")

    grouped = _group_rows(rows)
    qids = sorted(grouped.keys())

    n = len(qids)
    max_cols = max(1, args.max_cols)
    n_cols = min(max_cols, n)
    n_rows = int(math.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, q_id in zip(axes, qids):
        _plot_panel(ax, grouped[q_id], args.show_p)
        ax.set_title(q_id)

    for ax in axes[len(qids) :]:
        ax.axis("off")

    # legend: first axis only
    axes[0].legend(loc="best")

    if args.title:
        fig.suptitle(args.title)

    fig.tight_layout()
    out_dir = os.path.dirname(args.out_png) or "."
    if out_dir == "results":
        out_dir = os.path.join(out_dir, "pt_analysis")
        args.out_png = os.path.join(out_dir, os.path.basename(args.out_png))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out_png, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
