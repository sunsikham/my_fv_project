#!/usr/bin/env python3
"""CCN Figure Panel C (2D version): PC1 vs PC2 scatter of condition-specific function vectors.

Same data/colours as the 3D version but projected onto the PC1–PC2 plane.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── colour palette (same as 3D) ──────────────────────────────────────
COLORS = {
    "AAA":  "#8e44ad",
    "BBB":  "#2471a3",
    "BABA": "#e67e22",
    "DDD":  "#1a8a5c",
    "DADA": "#d4ac0d",
}

MARKER_STYLES = {
    "AAA":  {"marker": "o", "s_trial": 32, "s_centroid": 280},
    "BBB":  {"marker": "s", "s_trial": 32, "s_centroid": 260},
    "BABA": {"marker": "D", "s_trial": 28, "s_centroid": 260},
    "DDD":  {"marker": "^", "s_trial": 34, "s_centroid": 280},
    "DADA": {"marker": "d", "s_trial": 28, "s_centroid": 260},
}

CONDITION_ORDER = ["AAA", "BBB", "BABA", "DDD", "DADA"]

DISPLAY_NAMES = {
    "AAA":  "AAA (pure A)",
    "BBB":  "BBB (pure B)",
    "BABA": "BABA (A in B context)",
    "DDD":  "DDD (pure D)",
    "DADA": "DADA (A in D context)",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCN Panel C: 2D PCA scatter")
    p.add_argument("--points_csv", required=True)
    p.add_argument("--centroids_csv", required=True)
    p.add_argument("--meta_json", required=True)
    p.add_argument("--out_png", required=True)
    p.add_argument("--out_pdf", default=None)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _load_points(path: str) -> dict[str, np.ndarray]:
    groups: dict[str, list[list[float]]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            groups.setdefault(cond, []).append(
                [float(row["pc1"]), float(row["pc2"])]
            )
    return {k: np.array(v) for k, v in groups.items()}


def _load_centroids(path: str) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            centroids[row["condition"]] = np.array(
                [float(row["pc1"]), float(row["pc2"])]
            )
    return centroids


def _load_meta(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = _parse_args()

    points = _load_points(args.points_csv)
    centroids = _load_centroids(args.centroids_csv)
    meta = _load_meta(args.meta_json)
    var_ratio = meta["explained_variance_ratio"]

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    # ── trial-level scatter ───────────────────────────────────────────
    for cond in CONDITION_ORDER:
        if cond not in points:
            continue
        pts = points[cond]
        ms = MARKER_STYLES[cond]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=COLORS[cond], marker=ms["marker"], s=ms["s_trial"],
            alpha=0.45, edgecolors="white", linewidths=0.3, zorder=2,
        )

    # ── centroid markers ──────────────────────────────────────────────
    for cond in CONDITION_ORDER:
        if cond not in centroids:
            continue
        c = centroids[cond]
        ms = MARKER_STYLES[cond]
        ax.scatter(
            [c[0]], [c[1]],
            c=COLORS[cond], marker=ms["marker"], s=ms["s_centroid"],
            edgecolors="white", linewidths=2.0, zorder=8,
        )

    # ── arrows: AAA → BABA, AAA → DADA ───────────────────────────────
    aaa = centroids["AAA"]
    arrow_kw = dict(
        arrowstyle="->,head_width=0.3,head_length=0.2",
        lw=2.2, mutation_scale=15,
    )
    if "BABA" in centroids:
        direction = centroids["BABA"] - aaa
        start = aaa + direction * 0.18
        end = aaa + direction * 0.82
        ax.annotate("", xy=end, xytext=start,
                     arrowprops=dict(**arrow_kw, color=COLORS["BABA"], alpha=0.75),
                     zorder=5)
    if "DADA" in centroids:
        direction = centroids["DADA"] - aaa
        start = aaa + direction * 0.18
        end = aaa + direction * 0.82
        ax.annotate("", xy=end, xytext=start,
                     arrowprops=dict(**arrow_kw, color=COLORS["DADA"], alpha=0.75),
                     zorder=5)

    # ── dashed links: BABA···BBB, DADA···DDD ─────────────────────────
    if "BABA" in centroids and "BBB" in centroids:
        ax.plot(
            [centroids["BABA"][0], centroids["BBB"][0]],
            [centroids["BABA"][1], centroids["BBB"][1]],
            color=COLORS["BABA"], lw=1.0, alpha=0.35, ls="--", zorder=3,
        )
    if "DADA" in centroids and "DDD" in centroids:
        ax.plot(
            [centroids["DADA"][0], centroids["DDD"][0]],
            [centroids["DADA"][1], centroids["DDD"][1]],
            color=COLORS["DADA"], lw=1.0, alpha=0.35, ls="--", zorder=3,
        )

    # ── centroid text labels ──────────────────────────────────────────
    label_offsets = {
        "AAA":  np.array([0.25,  0.40]),
        "BBB":  np.array([-0.45, -0.40]),
        "BABA": np.array([-0.40,  0.45]),
        "DDD":  np.array([0.45,  -0.45]),
        "DADA": np.array([0.50,   0.40]),
    }
    for cond in CONDITION_ORDER:
        if cond not in centroids:
            continue
        c = centroids[cond]
        off = label_offsets.get(cond, np.zeros(2))
        ax.text(
            c[0] + off[0], c[1] + off[1],
            cond, fontsize=10, fontweight="bold", color=COLORS[cond],
            ha="center", va="center", zorder=10,
        )

    # ── axes ──────────────────────────────────────────────────────────
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # no equal aspect — let the figure breathe horizontally

    # no legend — centroid labels are sufficient

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
