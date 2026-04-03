#!/usr/bin/env python3
"""Render a standalone CCN PCA panel for LaTeX assembly."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HEADER_X = 0.055
HEADER_Y = 0.945
SUBPLOT_ADJUST = {
    "left": 0.13,
    "right": 0.985,
    "bottom": 0.16,
    "top": 0.82,
}


COLORS = {
    "AAA": "#8e44ad",
    "BBB": "#2471a3",
    "BABA": "#e67e22",
    "DDD": "#1a8a5c",
    "DADA": "#d4ac0d",
}

MARKER_STYLES = {
    "AAA": {"marker": "o", "s_trial": 28, "s_centroid": 250},
    "BBB": {"marker": "s", "s_trial": 28, "s_centroid": 235},
    "BABA": {"marker": "D", "s_trial": 24, "s_centroid": 235},
    "DDD": {"marker": "^", "s_trial": 30, "s_centroid": 250},
    "DADA": {"marker": "d", "s_trial": 24, "s_centroid": 235},
}

CONDITION_ORDER = ["AAA", "BBB", "BABA", "DDD", "DADA"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a standalone CCN PCA panel.")
    parser.add_argument("--points_csv", required=True)
    parser.add_argument("--centroids_csv", required=True)
    parser.add_argument("--meta_json", required=True)
    parser.add_argument("--out_png", required=True)
    parser.add_argument("--out_pdf", default=None)
    parser.add_argument("--panel_letter", default="B")
    parser.add_argument("--panel_title", default="PCA projection")
    parser.add_argument("--width_in", type=float, default=3.45)
    parser.add_argument("--height_in", type=float, default=2.80)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _load_points(path: str) -> dict[str, np.ndarray]:
    groups: dict[str, list[list[float]]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cond = row["condition"]
            groups.setdefault(cond, []).append([float(row["pc1"]), float(row["pc2"])])
    return {key: np.array(vals) for key, vals in groups.items()}


def _load_centroids(path: str) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            centroids[row["condition"]] = np.array([float(row["pc1"]), float(row["pc2"])])
    return centroids


def _load_meta(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _draw_panel(
    ax: plt.Axes,
    points: dict[str, np.ndarray],
    centroids: dict[str, np.ndarray],
    var_ratio: list[float],
) -> None:
    for cond in CONDITION_ORDER:
        if cond not in points:
            continue
        pts = points[cond]
        marker_style = MARKER_STYLES[cond]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=COLORS[cond],
            marker=marker_style["marker"],
            s=marker_style["s_trial"],
            alpha=0.45,
            edgecolors="white",
            linewidths=0.3,
            zorder=2,
        )

    for cond in CONDITION_ORDER:
        if cond not in centroids:
            continue
        centroid = centroids[cond]
        marker_style = MARKER_STYLES[cond]
        ax.scatter(
            [centroid[0]],
            [centroid[1]],
            c=COLORS[cond],
            marker=marker_style["marker"],
            s=marker_style["s_centroid"],
            edgecolors="white",
            linewidths=1.8,
            zorder=8,
        )

    aaa = centroids["AAA"]
    arrow_kw = dict(arrowstyle="->,head_width=0.3,head_length=0.2", lw=2.0, mutation_scale=14)
    if "BABA" in centroids:
        direction = centroids["BABA"] - aaa
        start = aaa + direction * 0.18
        end = aaa + direction * 0.82
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(**arrow_kw, color=COLORS["BABA"], alpha=0.75),
            zorder=5,
        )
    if "DADA" in centroids:
        direction = centroids["DADA"] - aaa
        start = aaa + direction * 0.18
        end = aaa + direction * 0.82
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(**arrow_kw, color=COLORS["DADA"], alpha=0.75),
            zorder=5,
        )

    if "BABA" in centroids and "BBB" in centroids:
        ax.plot(
            [centroids["BABA"][0], centroids["BBB"][0]],
            [centroids["BABA"][1], centroids["BBB"][1]],
            color=COLORS["BABA"],
            lw=1.0,
            alpha=0.35,
            ls="--",
            zorder=3,
        )
    if "DADA" in centroids and "DDD" in centroids:
        ax.plot(
            [centroids["DADA"][0], centroids["DDD"][0]],
            [centroids["DADA"][1], centroids["DDD"][1]],
            color=COLORS["DADA"],
            lw=1.0,
            alpha=0.35,
            ls="--",
            zorder=3,
        )

    label_offsets = {
        "AAA": np.array([0.30, 0.70]),
        "BBB": np.array([-0.68, -0.46]),
        "BABA": np.array([-0.68, 0.56]),
        "DDD": np.array([0.78, -0.48]),
        "DADA": np.array([0.78, 0.48]),
    }
    for cond in CONDITION_ORDER:
        if cond not in centroids:
            continue
        centroid = centroids[cond]
        offset = label_offsets.get(cond, np.zeros(2))
        ax.text(
            centroid[0] + offset[0],
            centroid[1] + offset[1],
            cond,
            fontsize=8.8,
            fontweight="bold",
            color=COLORS[cond],
            ha="center",
            va="bottom" if cond == "AAA" else "center",
            zorder=10,
        )

    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=7.2, labelpad=4.0)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=7.2, labelpad=5.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", length=2.5, pad=1.0)


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


def main() -> int:
    args = _parse_args()
    points = _load_points(args.points_csv)
    centroids = _load_centroids(args.centroids_csv)
    meta = _load_meta(args.meta_json)
    var_ratio = meta["explained_variance_ratio"]

    fig, ax = plt.subplots(figsize=(args.width_in, args.height_in))
    fig.subplots_adjust(**SUBPLOT_ADJUST)
    _draw_panel(ax, points, centroids, var_ratio)
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
