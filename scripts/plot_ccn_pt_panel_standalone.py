#!/usr/bin/env python3
"""Render a standalone CCN product-test panel for LaTeX assembly."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker


CONSISTENT_COLOR = "#7f8c8d"
MIXED_COLOR = "#c0392b"
STAR_COLOR = "#c0392b"
SEPARATOR_COLOR = "#e0e0e0"
REFLINE_COLOR = "#555555"

HEADER_X = 0.055
HEADER_Y = 0.945
LEGEND_ANCHOR = (0.975, 0.955)
SUBPLOT_ADJUST = {
    "left": 0.13,
    "right": 0.985,
    "bottom": 0.19,
    "top": 0.82,
}

DEFAULT_QIDS = "Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18"
DEFAULT_STAR_QIDS = "Q1,Q4,Q8,Q10,Q11,Q16"

REQUIRED_COLS = {
    "q_id",
    "pt_abc_ci2_low",
    "pt_abc_ci2_high",
    "pt_abd_ci2_low",
    "pt_abd_ci2_high",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a standalone CCN PT panel (e.g. Figure 2A or 2B)."
    )
    parser.add_argument("--in_csv", required=True, help="Bootstrap summary CSV")
    parser.add_argument("--out_png", required=True, help="Output PNG path")
    parser.add_argument("--out_pdf", default=None, help="Optional output PDF path")
    parser.add_argument("--panel_letter", default="A", help="Panel letter")
    parser.add_argument("--panel_title", default="Llama", help="Panel title")
    parser.add_argument("--qids", default=DEFAULT_QIDS, help="Comma-separated q ids")
    parser.add_argument("--star_qids", default=DEFAULT_STAR_QIDS, help="Comma-separated q ids to star")
    parser.add_argument(
        "--summary_stat",
        choices=["mean", "median"],
        default="mean",
        help="Which summary statistic to plot from the bootstrap CSV",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=None,
        help="Optional shot filter for multi-shot LLM bootstrap summaries",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional y-axis cap. Values above the cap are annotated with arrows.",
    )
    parser.add_argument("--width_in", type=float, default=3.45, help="Figure width in inches")
    parser.add_argument("--height_in", type=float, default=2.80, help="Figure height in inches")
    parser.add_argument("--dpi", type=int, default=300, help="Raster DPI for PNG output")
    return parser.parse_args()


def _norm_qid(raw: str) -> str:
    match = re.search(r"(\d+)", str(raw))
    if not match:
        raise ValueError(f"Cannot parse q_id: {raw!r}")
    return f"Q{int(match.group(1))}"


def _parse_qids(raw: str) -> list[str]:
    return [_norm_qid(part.strip()) for part in raw.split(",") if part.strip()]


def _display_qid(qid: str) -> str:
    match = re.fullmatch(r"Q(\d+)", qid)
    if not match:
        return qid
    return f"Q {int(match.group(1))}"


def _load_rows(
    path: str,
    summary_stat: str,
    shot: int | None,
) -> dict[str, dict[str, float]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row: {csv_path}")
        value_cols = {f"pt_abc_{summary_stat}", f"pt_abd_{summary_stat}"}
        missing = (REQUIRED_COLS | value_cols) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        rows: dict[str, dict[str, float]] = {}
        for row in reader:
            if shot is not None:
                row_shot = row.get("shot")
                if row_shot is None or int(row_shot) != shot:
                    continue
            qid = _norm_qid(row["q_id"])
            rows[qid] = {
                "pt_abc_value": float(row[f"pt_abc_{summary_stat}"]),
                "pt_abd_value": float(row[f"pt_abd_{summary_stat}"]),
                "pt_abc_ci2_low": float(row["pt_abc_ci2_low"]),
                "pt_abc_ci2_high": float(row["pt_abc_ci2_high"]),
                "pt_abd_ci2_low": float(row["pt_abd_ci2_low"]),
                "pt_abd_ci2_high": float(row["pt_abd_ci2_high"]),
            }
    return rows


def _draw_capped_errorbar(
    ax: plt.Axes,
    x: float,
    value: float,
    low: float,
    high: float,
    color: str,
    ymax: float | None,
) -> float:
    plot_value = value
    plot_low = low
    plot_high = high
    clipped = False

    if ymax is not None:
        plot_value = min(value, ymax)
        plot_low = min(low, plot_value)
        plot_high = min(high, ymax)
        clipped = (value > ymax) or (high > ymax)

    ax.errorbar(
        x,
        plot_value,
        yerr=[[max(plot_value - plot_low, 0.0)], [max(plot_high - plot_value, 0.0)]],
        fmt="o",
        markersize=3.8,
        color=color,
        ecolor=color,
        elinewidth=1.0,
        capsize=2.2,
        zorder=3,
    )

    anchor_y = plot_high
    if clipped and ymax is not None:
        arrow_base = plot_value + max(0.05, ymax * 0.008)
        arrow_top = ymax + max(0.24, ymax * 0.04)
        ax.annotate(
            "",
            xy=(x, arrow_top),
            xytext=(x, arrow_base),
            arrowprops=dict(
                arrowstyle="-|>,head_width=0.18,head_length=0.28",
                color=color,
                lw=0.9,
                mutation_scale=8,
            ),
            annotation_clip=False,
            zorder=6,
        )
        if value > ymax:
            ax.text(
                x + 0.05,
                arrow_top,
                f"{value:.1f}",
                ha="left",
                va="center",
                fontsize=6.4,
                fontweight="bold",
                color=color,
                clip_on=False,
                zorder=5,
            )
        anchor_y = arrow_top
    return anchor_y


def _draw_panel(
    ax: plt.Axes,
    rows: dict[str, dict[str, float]],
    qids: list[str],
    star_qids: set[str],
    ymax: float | None,
) -> None:
    xs = list(range(len(qids)))
    dx = 0.16

    for idx, qid in enumerate(qids):
        row = rows[qid]
        x_consistent = xs[idx] - dx
        x_mixed = xs[idx] + dx

        if idx > 0:
            ax.axvline(idx - 0.5, color=SEPARATOR_COLOR, linewidth=0.6, zorder=0)

        _draw_capped_errorbar(
            ax,
            x_consistent,
            row["pt_abc_value"],
            row["pt_abc_ci2_low"],
            row["pt_abc_ci2_high"],
            CONSISTENT_COLOR,
            ymax,
        )
        mixed_anchor_y = _draw_capped_errorbar(
            ax,
            x_mixed,
            row["pt_abd_value"],
            row["pt_abd_ci2_low"],
            row["pt_abd_ci2_high"],
            MIXED_COLOR,
            ymax,
        )
        if qid in star_qids:
            star_y = mixed_anchor_y + (0.18 if ymax is None else max(0.2, ymax * 0.04))
            ax.text(
                x_mixed,
                star_y,
                "*",
                ha="center",
                va="bottom",
                fontsize=9.3,
                color=STAR_COLOR,
                fontweight="bold",
                zorder=4,
            )

    ax.axhline(1.0, color=REFLINE_COLOR, ls="--", lw=0.9, zorder=1)
    ax.set_xlim(-0.55, len(qids) - 0.45)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_display_qid(qid) for qid in qids],
        fontsize=6.6,
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("Item", fontsize=7.8, labelpad=2.0)
    ax.set_ylabel("Product Test Value", fontsize=7.8, labelpad=2.0)
    if ymax is not None:
        ax.set_ylim(0.0, ymax + max(0.6, ymax * 0.12))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(axis="both", labelsize=6.6, length=2.5, pad=1.0)
    ax.grid(axis="y", alpha=0.20, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_header(fig: plt.Figure, panel_letter: str, panel_title: str) -> None:
    fig.text(
        HEADER_X,
        HEADER_Y,
        f"{panel_letter}. {panel_title}",
        fontsize=8.6,
        fontweight="bold",
        ha="left",
        va="top",
    )
    legend_handles = [
        Line2D([], [], marker="o", linestyle="None", markersize=5.0, color=CONSISTENT_COLOR),
        Line2D([], [], marker="o", linestyle="None", markersize=5.0, color=MIXED_COLOR),
    ]
    fig.legend(
        legend_handles,
        ["Consistent (ABC)", "Mixed (ABD)"],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=LEGEND_ANCHOR,
        fontsize=6.0,
        handletextpad=0.3,
        borderaxespad=0.0,
        labelspacing=0.25,
    )


def main() -> int:
    args = _parse_args()
    qids = _parse_qids(args.qids)
    star_qids = set(_parse_qids(args.star_qids)) if args.star_qids.strip() else set()
    rows = _load_rows(args.in_csv, summary_stat=args.summary_stat, shot=args.shot)
    missing = [qid for qid in qids if qid not in rows]
    if missing:
        raise ValueError(f"Missing rows for qids: {missing}")

    fig, ax = plt.subplots(figsize=(args.width_in, args.height_in))
    fig.subplots_adjust(**SUBPLOT_ADJUST)
    _draw_panel(
        ax,
        rows,
        qids,
        star_qids,
        args.ymax,
    )
    _add_header(fig, args.panel_letter, args.panel_title)

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=args.dpi, facecolor="white")

    if args.out_pdf:
        out_pdf = Path(args.out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, facecolor="white")

    plt.close(fig)
    print(f"Saved -> {out_png}")
    if args.out_pdf:
        print(f"Saved -> {args.out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
