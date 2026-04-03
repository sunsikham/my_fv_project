#!/usr/bin/env python3
"""CCN Composite Figure: Panel A (triangle) + Panel B (PT) + Panel C (PCA 2D).

Layout (2×2):
  Top-left:  A  — stimulus design (consistent vs mixed triangles)
  Top-right: C  — 2D PCA scatter (Q8)
  Bottom:    B1 — Human PT  |  B2 — LLM PT (placeholder)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# SHARED
# ═══════════════════════════════════════════════════════════════════════
TEXT_COLOR = "#2c3e50"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCN composite figure")
    # Panel B data
    p.add_argument("--human_csv", required=True)
    p.add_argument("--llm_csv", default=None)
    p.add_argument("--qids", default="Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18")
    p.add_argument("--star_qids_human", default="Q1,Q4,Q8,Q10,Q11,Q16")
    p.add_argument("--star_qids_llm", default="")
    p.add_argument("--llm_label", default="Llama-3.1-70B")
    # Panel C data
    p.add_argument("--pca_points_csv", required=True)
    p.add_argument("--pca_centroids_csv", required=True)
    p.add_argument("--pca_meta_json", required=True)
    # output
    p.add_argument("--out_png", required=True)
    p.add_argument("--out_pdf", default=None)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# PANEL A — Triangle stimulus design
# ═══════════════════════════════════════════════════════════════════════
PA_CONSISTENT_FILL = "#d5d8dc"
PA_MIXED_FILL      = "#f5b7b1"
PA_CONSISTENT_EDGE = "#7f8c8d"
PA_MIXED_EDGE      = "#c0392b"
PA_AMBIG_COLOR     = "#6c3483"
PA_DASHED_EDGE     = "#999999"
PA_RELATION_COLOR  = "#555555"


def _draw_triangle(
    ax, vertices, fill_color, edge_color, edge_styles,
    pair_labels, pair_ids, edge_labels, relation_labels,
    title, ambig_idx=0,
):
    top, bl, br = vertices
    tri_patch = plt.Polygon(
        vertices, closed=True,
        facecolor=fill_color, edgecolor="none", alpha=0.55, zorder=1,
    )
    ax.add_patch(tri_patch)

    edges = [(top, bl), (top, br), (bl, br)]
    for i, (p1, p2) in enumerate(edges):
        ls = edge_styles[i]
        lw = 2.0 if ls == "-" else 1.5
        ec = edge_color if ls == "-" else PA_DASHED_EDGE
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=ec, linewidth=lw, linestyle=ls, zorder=2)

    edge_midpoints = [(top + bl) / 2, (top + br) / 2, (bl + br) / 2]
    centre = vertices.mean(axis=0)
    for i, mid in enumerate(edge_midpoints):
        direction = mid - centre
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        scale = 0.30 if i == 2 else 0.42
        label_pos = mid + direction * scale
        ax.text(label_pos[0], label_pos[1] + 0.06, edge_labels[i],
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=TEXT_COLOR, zorder=5)
        ax.text(label_pos[0], label_pos[1] - 0.08, relation_labels[i],
                ha="center", va="top", fontsize=7, color=TEXT_COLOR, zorder=5)

    vert_offsets = [
        np.array([0, 0.18]),
        np.array([-0.15, -0.18]),
        np.array([0.15, -0.18]),
    ]
    for i, (vx, vy) in enumerate(vertices):
        ox, oy = vert_offsets[i]
        is_ambig = (i == ambig_idx)
        id_color = PA_AMBIG_COLOR if is_ambig else TEXT_COLOR
        cx, cy = vx + ox, vy + oy
        circle = plt.Circle((cx, cy), 0.11,
                             facecolor="white", edgecolor=id_color,
                             linewidth=1.8, zorder=4)
        ax.add_patch(circle)
        ax.text(cx, cy - 0.012, pair_ids[i],
                ha="center", va="center", fontsize=8, fontweight="bold",
                color=id_color, zorder=5)
        pair_offset_y = 0.18 if i == 0 else -0.18
        ax.text(vx + ox, vy + oy + pair_offset_y, pair_labels[i],
                ha="center", va="center" if pair_offset_y > 0 else "top",
                fontsize=7, fontweight="bold" if is_ambig else "normal",
                color=id_color, zorder=5)

    ax.text(centre[0], vertices[:, 1].max() + 0.65, title,
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=TEXT_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_panel_a(ax_l, ax_r):
    h = np.sqrt(3)
    tri = np.array([[0.0, h], [-1.0, 0.0], [1.0, 0.0]])

    _draw_triangle(
        ax_l, tri, PA_CONSISTENT_FILL, PA_CONSISTENT_EDGE,
        ["-", "-", "-"],
        ["Snail : Shell", "Owl : Tree", "Human : House"],
        ["A", "B", "C"], ["AB", "AC", "BC"],
        ["lives in", "lives in", "lives in"],
        "Consistent (ABC)", ambig_idx=0,
    )
    ax_l.set_xlim(-2.0, 2.0)
    ax_l.set_ylim(-0.8, h + 1.0)

    _draw_triangle(
        ax_r, tri, PA_MIXED_FILL, PA_MIXED_EDGE,
        ["-", "-", "--"],
        ["Snail : Shell", "Owl : Tree", "Student : Backpack"],
        ["A", "B", "D"], ["AB", "AD", "BD"],
        ["lives in", "carries", "?"],
        "Mixed (ABD)", ambig_idx=0,
    )
    ax_r.set_xlim(-2.0, 2.0)
    ax_r.set_ylim(-0.8, h + 1.0)


# ═══════════════════════════════════════════════════════════════════════
# PANEL B — Product Test plots
# ═══════════════════════════════════════════════════════════════════════
PB_CONSISTENT_COLOR = "#7f8c8d"
PB_MIXED_COLOR      = "#c0392b"
PB_STAR_COLOR       = "#c0392b"
PB_SEPARATOR_COLOR  = "#e0e0e0"
PB_REFLINE_COLOR    = "#555555"

PB_REQUIRED_COLS = {
    "q_id", "pt_abc_mean", "pt_abd_mean",
    "pt_abc_ci2_low", "pt_abc_ci2_high",
    "pt_abd_ci2_low", "pt_abd_ci2_high",
}


def _norm_qid(raw):
    m = re.search(r"(\d+)", str(raw))
    if not m:
        raise ValueError(f"Cannot parse q_id: {raw!r}")
    return f"Q{int(m.group(1))}"


def _load_pt(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = {}
        for r in reader:
            qid = _norm_qid(r["q_id"])
            rows[qid] = {k: float(r[k]) for k in PB_REQUIRED_COLS if k != "q_id"}
        return rows


def _parse_qids(raw):
    return [_norm_qid(s.strip()) for s in raw.split(",") if s.strip()]


def _parse_star_qids(raw):
    if not raw.strip():
        return set()
    return set(_parse_qids(raw))


def draw_panel_b(ax, rows, qids, star_qids, panel_label, show_xlabel=True):
    n = len(qids)
    xs = list(range(n))
    dx = 0.16

    for i, qid in enumerate(qids):
        r = rows[qid]
        x_c, x_m = xs[i] - dx, xs[i] + dx
        if i > 0:
            ax.axvline(i - 0.5, color=PB_SEPARATOR_COLOR, linewidth=0.6, zorder=0)
        ax.errorbar(
            x_c, r["pt_abc_mean"],
            yerr=[[r["pt_abc_mean"] - r["pt_abc_ci2_low"]],
                  [r["pt_abc_ci2_high"] - r["pt_abc_mean"]]],
            fmt="o", markersize=3.5, color=PB_CONSISTENT_COLOR,
            ecolor=PB_CONSISTENT_COLOR, elinewidth=1.0, capsize=2, zorder=3,
        )
        ax.errorbar(
            x_m, r["pt_abd_mean"],
            yerr=[[r["pt_abd_mean"] - r["pt_abd_ci2_low"]],
                  [r["pt_abd_ci2_high"] - r["pt_abd_mean"]]],
            fmt="o", markersize=3.5, color=PB_MIXED_COLOR,
            ecolor=PB_MIXED_COLOR, elinewidth=1.0, capsize=2, zorder=3,
        )
        if qid in star_qids:
            star_y = r["pt_abd_ci2_high"] + 0.12
            ax.text(x_m, star_y, "*", ha="center", va="bottom", fontsize=10,
                    color=PB_STAR_COLOR, fontweight="bold", zorder=4)

    ax.axhline(1.0, color=PB_REFLINE_COLOR, ls="--", lw=0.8, zorder=1)
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_xticks(xs)
    ax.set_xticklabels(qids, fontsize=6)
    if show_xlabel:
        ax.set_xlabel("Item", fontsize=7)
    ax.set_ylabel("Product Test", fontsize=7)
    ax.tick_params(axis="both", labelsize=6)
    ax.grid(axis="y", alpha=0.20, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.08, 1.05, panel_label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def draw_panel_b_placeholder(ax, label, panel_label, qids):
    n = len(qids)
    xs = list(range(n))
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_ylim(0, 5)
    ax.text(0.5, 0.5, f"{label}\n(data in progress)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color="#aaaaaa", style="italic")
    ax.axhline(1.0, color=PB_REFLINE_COLOR, ls="--", lw=0.8, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(qids, fontsize=6)
    ax.set_ylabel("Product Test", fontsize=7)
    ax.set_xlabel("Item", fontsize=7)
    ax.tick_params(axis="both", labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.08, 1.05, panel_label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


# ═══════════════════════════════════════════════════════════════════════
# PANEL C — 2D PCA scatter
# ═══════════════════════════════════════════════════════════════════════
PC_COLORS = {
    "AAA": "#8e44ad", "BBB": "#2471a3", "BABA": "#e67e22",
    "DDD": "#1a8a5c", "DADA": "#d4ac0d",
}
PC_MARKERS = {
    "AAA": {"m": "o", "st": 24, "sc": 220},
    "BBB": {"m": "s", "st": 24, "sc": 200},
    "BABA": {"m": "D", "st": 20, "sc": 200},
    "DDD": {"m": "^", "st": 26, "sc": 220},
    "DADA": {"m": "d", "st": 20, "sc": 200},
}
PC_ORDER = ["AAA", "BBB", "BABA", "DDD", "DADA"]


def _load_pca_points(path):
    groups = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            groups.setdefault(cond, []).append(
                [float(row["pc1"]), float(row["pc2"])])
    return {k: np.array(v) for k, v in groups.items()}


def _load_pca_centroids(path):
    centroids = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            centroids[row["condition"]] = np.array(
                [float(row["pc1"]), float(row["pc2"])])
    return centroids


def draw_panel_c(ax, points, centroids, var_ratio):
    for cond in PC_ORDER:
        if cond not in points:
            continue
        pts = points[cond]
        ms = PC_MARKERS[cond]
        ax.scatter(pts[:, 0], pts[:, 1], c=PC_COLORS[cond],
                   marker=ms["m"], s=ms["st"], alpha=0.45,
                   edgecolors="white", linewidths=0.3, zorder=2)

    for cond in PC_ORDER:
        if cond not in centroids:
            continue
        c = centroids[cond]
        ms = PC_MARKERS[cond]
        ax.scatter([c[0]], [c[1]], c=PC_COLORS[cond],
                   marker=ms["m"], s=ms["sc"],
                   edgecolors="white", linewidths=2.0, zorder=8)

    aaa = centroids["AAA"]
    arrow_kw = dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                    lw=2.0, mutation_scale=14)
    for target, color in [("BABA", PC_COLORS["BABA"]), ("DADA", PC_COLORS["DADA"])]:
        if target in centroids:
            d = centroids[target] - aaa
            ax.annotate("", xy=aaa + d * 0.82, xytext=aaa + d * 0.18,
                        arrowprops=dict(**arrow_kw, color=color, alpha=0.75),
                        zorder=5)

    for inter, pure in [("BABA", "BBB"), ("DADA", "DDD")]:
        if inter in centroids and pure in centroids:
            ax.plot([centroids[inter][0], centroids[pure][0]],
                    [centroids[inter][1], centroids[pure][1]],
                    color=PC_COLORS[inter], lw=1.0, alpha=0.35,
                    ls="--", zorder=3)

    label_offsets = {
        "AAA": np.array([0.25, 0.40]),
        "BBB": np.array([-0.45, -0.40]),
        "BABA": np.array([-0.40, 0.45]),
        "DDD": np.array([0.45, -0.45]),
        "DADA": np.array([0.50, 0.40]),
    }
    for cond in PC_ORDER:
        if cond not in centroids:
            continue
        c = centroids[cond]
        off = label_offsets.get(cond, np.zeros(2))
        ax.text(c[0] + off[0], c[1] + off[1], cond,
                fontsize=8, fontweight="bold", color=PC_COLORS[cond],
                ha="center", va="center", zorder=10)

    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


# ═══════════════════════════════════════════════════════════════════════
# MAIN — composite
# ═══════════════════════════════════════════════════════════════════════
def main() -> int:
    args = _parse_args()

    # load data
    qids = _parse_qids(args.qids)
    star_human = _parse_star_qids(args.star_qids_human)
    star_llm = _parse_star_qids(args.star_qids_llm)
    human_rows = _load_pt(args.human_csv)

    pca_points = _load_pca_points(args.pca_points_csv)
    pca_centroids = _load_pca_centroids(args.pca_centroids_csv)
    with Path(args.pca_meta_json).open("r") as f:
        pca_meta = json.load(f)
    var_ratio = pca_meta["explained_variance_ratio"]

    # figure layout
    fig = plt.figure(figsize=(7.5, 7.5))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[1.0, 0.8],
        width_ratios=[1.0, 1.0],
        hspace=0.35, wspace=0.30,
    )

    # top-left: Panel A (two sub-axes for the two triangles)
    gs_a = gs[0, 0].subgridspec(1, 2, wspace=0.05)
    ax_a_l = fig.add_subplot(gs_a[0, 0])
    ax_a_r = fig.add_subplot(gs_a[0, 1])
    draw_panel_a(ax_a_l, ax_a_r)
    # Panel A label
    ax_a_l.text(-0.05, 1.08, "A", transform=ax_a_l.transAxes,
                fontsize=13, fontweight="bold", va="bottom", ha="left")

    # top-right: Panel C (PCA 2D)
    ax_c = fig.add_subplot(gs[0, 1])
    draw_panel_c(ax_c, pca_points, pca_centroids, var_ratio)
    ax_c.text(-0.05, 1.08, "C", transform=ax_c.transAxes,
              fontsize=13, fontweight="bold", va="bottom", ha="left")

    # bottom-left: Panel B1 (Human PT)
    ax_b1 = fig.add_subplot(gs[1, 0])
    draw_panel_b(ax_b1, human_rows, qids, star_human,
                 panel_label="B1  Human", show_xlabel=True)
    # shared legend for B panels
    ax_b1.scatter([], [], color=PB_CONSISTENT_COLOR, s=20, label="Consistent (ABC)")
    ax_b1.scatter([], [], color=PB_MIXED_COLOR, s=20, label="Mixed (ABD)")
    ax_b1.legend(frameon=False, loc="upper left", fontsize=6, ncol=2,
                 handletextpad=0.3, columnspacing=0.8)

    # bottom-right: Panel B2 (LLM)
    ax_b2 = fig.add_subplot(gs[1, 1])
    if args.llm_csv:
        llm_rows = _load_pt(args.llm_csv)
        draw_panel_b(ax_b2, llm_rows, qids, star_llm,
                     panel_label=f"B2  {args.llm_label}", show_xlabel=True)
    else:
        draw_panel_b_placeholder(ax_b2, args.llm_label,
                                 panel_label="B2  " + args.llm_label, qids=qids)

    # save
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
