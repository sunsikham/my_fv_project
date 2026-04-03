#!/usr/bin/env python3
"""CCN Figure Panel C: 3D PCA scatter of condition-specific function vectors.

Shows that interleaved conditions (BABA, DADA) shift from AAA toward
BBB / DDD respectively — evidence of context-dependent re-representation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d  # noqa: F401
from mpl_toolkits.mplot3d.proj3d import proj_transform
import numpy as np


# ── colour palette ────────────────────────────────────────────────────
COLORS = {
    "AAA":  "#8e44ad",   # purple — ambiguous pair
    "BBB":  "#2471a3",   # dark blue — pure B
    "BABA": "#e67e22",   # orange — A in B context
    "DDD":  "#1a8a5c",   # dark green — pure D
    "DADA": "#d4ac0d",   # gold/yellow — A in D context
}

MARKER_STYLES = {
    "AAA":  {"marker": "o", "s_trial": 28, "s_centroid": 260},
    "BBB":  {"marker": "s", "s_trial": 28, "s_centroid": 240},
    "BABA": {"marker": "D", "s_trial": 24, "s_centroid": 240},
    "DDD":  {"marker": "^", "s_trial": 30, "s_centroid": 260},
    "DADA": {"marker": "d", "s_trial": 24, "s_centroid": 240},
}

CONDITION_ORDER = ["AAA", "BBB", "BABA", "DDD", "DADA"]

# Display names for legend
DISPLAY_NAMES = {
    "AAA":  "AAA (pure A)",
    "BBB":  "BBB (pure B)",
    "BABA": "BABA (A in B context)",
    "DDD":  "DDD (pure D)",
    "DADA": "DADA (A in D context)",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCN Panel C: 3D PCA scatter")
    p.add_argument("--points_csv", required=True)
    p.add_argument("--centroids_csv", required=True)
    p.add_argument("--meta_json", required=True)
    p.add_argument("--out_png", required=True)
    p.add_argument("--out_pdf", default=None)
    p.add_argument("--elev", type=float, default=28)
    p.add_argument("--azim", type=float, default=-60)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _load_points(path: str) -> dict[str, np.ndarray]:
    groups: dict[str, list[list[float]]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            groups.setdefault(cond, []).append(
                [float(row["pc1"]), float(row["pc2"]), float(row["pc3"])]
            )
    return {k: np.array(v) for k, v in groups.items()}


def _load_centroids(path: str) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            centroids[row["condition"]] = np.array(
                [float(row["pc1"]), float(row["pc2"]), float(row["pc3"])]
            )
    return centroids


def _load_meta(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _draw_shift_arrow(ax, start: np.ndarray, end: np.ndarray, color: str,
                      lw: float = 2.5, alpha: float = 0.75) -> None:
    """Draw a connecting line from AAA centroid toward interleaved centroid."""
    direction = end - start
    end_short = start + direction * 0.82
    start_short = start + direction * 0.18
    ax.plot(
        [start_short[0], end_short[0]],
        [start_short[1], end_short[1]],
        [start_short[2], end_short[2]],
        color=color, linewidth=lw, alpha=alpha, zorder=5,
        linestyle="-", solid_capstyle="round",
    )


def _draw_dashed_link(ax, p1: np.ndarray, p2: np.ndarray, color: str,
                      lw: float = 1.0, alpha: float = 0.35) -> None:
    """Draw a faint dashed line between interleaved and pure target centroids."""
    ax.plot(
        [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
        color=color, linewidth=lw, alpha=alpha, zorder=3,
        linestyle="--",
    )


def main() -> int:
    args = _parse_args()

    points = _load_points(args.points_csv)
    centroids = _load_centroids(args.centroids_csv)
    meta = _load_meta(args.meta_json)
    var_ratio = meta["explained_variance_ratio"]

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    # ── trial-level scatter (small, transparent) ──────────────────────
    for cond in CONDITION_ORDER:
        if cond not in points:
            continue
        pts = points[cond]
        ms = MARKER_STYLES[cond]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=COLORS[cond], marker=ms["marker"], s=ms["s_trial"],
            alpha=0.55, edgecolors="white", linewidths=0.3, zorder=2,
        )

    # ── centroid markers (large, bold, white edge) ────────────────────
    for cond in CONDITION_ORDER:
        if cond not in centroids:
            continue
        c = centroids[cond]
        ms = MARKER_STYLES[cond]
        ax.scatter(
            [c[0]], [c[1]], [c[2]],
            c=COLORS[cond], marker=ms["marker"], s=ms["s_centroid"],
            edgecolors="white", linewidths=2.0, zorder=8,
            label=DISPLAY_NAMES.get(cond, cond),
        )

    # ── arrows: AAA → BABA, AAA → DADA ───────────────────────────────
    aaa = centroids["AAA"]
    if "BABA" in centroids:
        _draw_shift_arrow(ax, aaa, centroids["BABA"], color=COLORS["BABA"])
    if "DADA" in centroids:
        _draw_shift_arrow(ax, aaa, centroids["DADA"], color=COLORS["DADA"])

    # ── dashed links: BABA···BBB, DADA···DDD ─────────────────────────
    if "BABA" in centroids and "BBB" in centroids:
        _draw_dashed_link(ax, centroids["BABA"], centroids["BBB"], COLORS["BABA"])
    if "DADA" in centroids and "DDD" in centroids:
        _draw_dashed_link(ax, centroids["DADA"], centroids["DDD"], COLORS["DADA"])

    # ── centroid text labels (offset to avoid overlap) ────────────────
    label_offsets = {
        "AAA":  np.array([0.20,  0.65,  0.50]),
        "BBB":  np.array([-0.55, -0.55,  0.40]),
        "BABA": np.array([-0.50,  0.65,  0.30]),
        "DDD":  np.array([0.55,  -0.80,  0.0]),
        "DADA": np.array([0.70,   0.65,  0.0]),
    }
    for cond in CONDITION_ORDER:
        if cond not in centroids:
            continue
        c = centroids[cond]
        off = label_offsets.get(cond, np.zeros(3))
        ax.text(
            c[0] + off[0], c[1] + off[1], c[2] + off[2],
            cond, fontsize=9.5, fontweight="bold", color=COLORS[cond],
            ha="center", va="center", zorder=10,
        )

    # ── axes labels with variance explained ───────────────────────────
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=9, labelpad=8)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=9, labelpad=8)
    ax.set_zlabel(f"PC3 ({var_ratio[2]*100:.1f}%)", fontsize=9, labelpad=8)

    # ── camera angle ──────────────────────────────────────────────────
    ax.view_init(elev=args.elev, azim=args.azim)

    # ── clean up: minimal — no panes, no grid, no tick labels ───────
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # legend — above the plot, single row
    legend = ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 0.92),
        fontsize=6.5, frameon=True, framealpha=0.92,
        edgecolor="#cccccc", ncol=5, handletextpad=0.3,
        borderpad=0.4, labelspacing=0.3, columnspacing=0.8,
        markerscale=0.9,
    )
    legend.get_frame().set_linewidth(0.5)

    fig.tight_layout(pad=1.5)

    # ── save ──────────────────────────────────────────────────────────
    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight", facecolor="white")
    if args.out_pdf:
        Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out}")
    if args.out_pdf:
        print(f"Saved → {args.out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
