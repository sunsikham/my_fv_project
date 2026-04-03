#!/usr/bin/env python3
"""Plot q-wise human PT means for ABC vs ABD with confidence intervals."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLS = {
    "q_id",
    "pt_abc_mean",
    "pt_abd_mean",
    "pt_abc_ci2_low",
    "pt_abc_ci2_high",
    "pt_abd_ci2_low",
    "pt_abd_ci2_high",
}

ABC_COLOR = "#2b6cb0"
ABD_COLOR = "#c05621"
LINE_COLOR = "#9a9a9a"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot q-wise human PT paired points with 95% confidence intervals."
    )
    parser.add_argument("--in_csv", required=True, help="Input human bootstrap summary CSV")
    parser.add_argument("--out_png", required=True, help="Output PNG path")
    parser.add_argument("--out_pdf", default=None, help="Optional output PDF path")
    parser.add_argument(
        "--qids",
        required=True,
        help="Comma-separated q ids, e.g. Q1,Q3,Q4 or 1,3,4",
    )
    parser.add_argument(
        "--title",
        default="B1 Human: q-wise ABC vs ABD",
        help="Figure title",
    )
    parser.add_argument(
        "--layout",
        choices=["paired", "flat"],
        default="paired",
        help="Plot layout: paired points per q, or flat ABC/ABD sequence",
    )
    parser.add_argument(
        "--star_qids",
        default="",
        help="Comma-separated q ids to mark with a star on the ABD point",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Output dpi")
    return parser.parse_args()


def _normalize_qid(qid: str) -> str:
    match = re.search(r"(\d+)", str(qid))
    if not match:
        raise ValueError(f"Could not parse q_id from {qid!r}")
    return f"Q{int(match.group(1))}"


def _load_rows(path: str) -> dict[str, dict[str, float]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row: {csv_path}")
        missing = REQUIRED_COLS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

        rows: dict[str, dict[str, float]] = {}
        for row in reader:
            qid = _normalize_qid(row["q_id"])
            rows[qid] = {
                "pt_abc_mean": float(row["pt_abc_mean"]),
                "pt_abd_mean": float(row["pt_abd_mean"]),
                "pt_abc_ci2_low": float(row["pt_abc_ci2_low"]),
                "pt_abc_ci2_high": float(row["pt_abc_ci2_high"]),
                "pt_abd_ci2_low": float(row["pt_abd_ci2_low"]),
                "pt_abd_ci2_high": float(row["pt_abd_ci2_high"]),
            }
        return rows


def _parse_qids(raw_qids: str) -> list[str]:
    qids = [_normalize_qid(part.strip()) for part in raw_qids.split(",") if part.strip()]
    if not qids:
        raise ValueError("No q_ids provided")
    return qids


def _parse_optional_qids(raw_qids: str) -> set[str]:
    if not raw_qids.strip():
        return set()
    return set(_parse_qids(raw_qids))


def _plot_paired(ax, rows: dict[str, dict[str, float]], qids: list[str], star_qids: set[str]) -> None:
    x_centers = list(range(len(qids)))
    dx = 0.18

    for idx, qid in enumerate(qids):
        row = rows[qid]
        x_abc = x_centers[idx] - dx
        x_abd = x_centers[idx] + dx

        abc_mean = row["pt_abc_mean"]
        abd_mean = row["pt_abd_mean"]
        abc_low = row["pt_abc_ci2_low"]
        abc_high = row["pt_abc_ci2_high"]
        abd_low = row["pt_abd_ci2_low"]
        abd_high = row["pt_abd_ci2_high"]

        ax.plot([x_abc, x_abd], [abc_mean, abd_mean], color=LINE_COLOR, linewidth=1.3, zorder=1)
        ax.errorbar(
            x_abc,
            abc_mean,
            yerr=[[abc_mean - abc_low], [abc_high - abc_mean]],
            fmt="o",
            markersize=6.5,
            color=ABC_COLOR,
            ecolor=ABC_COLOR,
            elinewidth=1.6,
            capsize=3.5,
            zorder=3,
        )
        ax.errorbar(
            x_abd,
            abd_mean,
            yerr=[[abd_mean - abd_low], [abd_high - abd_mean]],
            fmt="o",
            markersize=6.5,
            color=ABD_COLOR,
            ecolor=ABD_COLOR,
            elinewidth=1.6,
            capsize=3.5,
            zorder=3,
        )
        if qid in star_qids:
            ax.text(
                x_abd,
                abd_high + 0.12,
                "*",
                ha="center",
                va="bottom",
                fontsize=15,
                color=ABD_COLOR,
                fontweight="bold",
                zorder=4,
            )

    ax.set_xlim(-0.6, len(qids) - 0.4)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(qids, rotation=0)


def _plot_flat(ax, rows: dict[str, dict[str, float]], qids: list[str], star_qids: set[str]) -> None:
    tick_positions = []
    tick_labels = []

    for idx, qid in enumerate(qids):
        base = 2 * idx
        row = rows[qid]
        x_abc = base
        x_abd = base + 1

        if idx % 2 == 0:
            ax.axvspan(base - 0.5, base + 1.5, color="#f7f7f7", zorder=0)
        if idx > 0:
            ax.axvline(base - 0.5, color="#dddddd", linewidth=0.8, zorder=0)

        abc_mean = row["pt_abc_mean"]
        abd_mean = row["pt_abd_mean"]
        abc_low = row["pt_abc_ci2_low"]
        abc_high = row["pt_abc_ci2_high"]
        abd_low = row["pt_abd_ci2_low"]
        abd_high = row["pt_abd_ci2_high"]

        ax.errorbar(
            x_abc,
            abc_mean,
            yerr=[[abc_mean - abc_low], [abc_high - abc_mean]],
            fmt="o",
            markersize=6.5,
            color=ABC_COLOR,
            ecolor=ABC_COLOR,
            elinewidth=1.6,
            capsize=3.5,
            zorder=3,
        )
        ax.errorbar(
            x_abd,
            abd_mean,
            yerr=[[abd_mean - abd_low], [abd_high - abd_mean]],
            fmt="o",
            markersize=6.5,
            color=ABD_COLOR,
            ecolor=ABD_COLOR,
            elinewidth=1.6,
            capsize=3.5,
            zorder=3,
        )
        if qid in star_qids:
            ax.text(
                x_abd,
                abd_high + 0.12,
                "*",
                ha="center",
                va="bottom",
                fontsize=15,
                color=ABD_COLOR,
                fontweight="bold",
                zorder=4,
            )

        tick_positions.extend([x_abc, x_abd])
        tick_labels.extend(["ABC", "ABD"])
        ax.text(
            base + 0.5,
            -0.12,
            qid,
            ha="center",
            va="top",
            fontsize=10,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xlim(-0.7, 2 * len(qids) - 0.3)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=10)


def _plot(
    rows: dict[str, dict[str, float]],
    qids: list[str],
    title: str,
    out_png: str,
    out_pdf: str | None,
    dpi: int,
    layout: str,
    star_qids: set[str],
) -> None:
    missing = [qid for qid in qids if qid not in rows]
    if missing:
        raise ValueError(f"Missing rows for q_ids: {missing}")

    fig_width = 12.0 if layout == "paired" else 14.0
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    if layout == "paired":
        _plot_paired(ax, rows, qids, star_qids)
    else:
        _plot_flat(ax, rows, qids, star_qids)

    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Product Test")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.scatter([], [], color=ABC_COLOR, label="ABC")
    ax.scatter([], [], color=ABD_COLOR, label="ABD")
    ax.legend(frameon=False, loc="upper left", ncol=2)

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if out_pdf:
        Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    rows = _load_rows(args.in_csv)
    qids = _parse_qids(args.qids)
    star_qids = _parse_optional_qids(args.star_qids)
    _plot(rows, qids, args.title, args.out_png, args.out_pdf, args.dpi, args.layout, star_qids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
