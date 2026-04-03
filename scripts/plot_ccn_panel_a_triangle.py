#!/usr/bin/env python3
"""CCN Figure Panel A: Consistent vs Mixed triangle stimulus design.

Generates a clean, publication-ready diagram showing:
  - Left:  Consistent triangle (ABC) — all edges share "lives in"
  - Right: Mixed triangle (ABD) — A is ambiguous across relations
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── colours (matching PT plot palette) ──────────────────────────────
CONSISTENT_FILL = "#d5d8dc"   # light grey
MIXED_FILL      = "#f5b7b1"   # light red
CONSISTENT_EDGE = "#7f8c8d"   # grey
MIXED_EDGE      = "#c0392b"   # red
AMBIG_COLOR     = "#6c3483"   # purple for the ambiguous pair A
TEXT_COLOR       = "#2c3e50"
RELATION_COLOR  = "#555555"
DASHED_EDGE     = "#999999"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCN Panel A: triangle design diagram")
    p.add_argument("--out_png", required=True)
    p.add_argument("--out_pdf", default=None)
    p.add_argument("--out_svg", default=None)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _draw_triangle(
    ax: plt.Axes,
    vertices: np.ndarray,        # shape (3, 2) — top, bottom-left, bottom-right
    fill_color: str,
    edge_color: str,
    edge_styles: list[str],      # 3 styles for edges: top-left, top-right, bottom
    pair_labels: list[str],      # 3 vertex labels
    pair_ids: list[str],         # 3 vertex IDs (A, B, C / A, B, D)
    edge_labels: list[str],      # 3 edge names (AB, AC, BC)
    relation_labels: list[str],  # 3 relation names per edge
    title: str,
    ambig_idx: int = 0,          # which vertex is the ambiguous pair
) -> None:
    top, bl, br = vertices

    # filled triangle
    triangle = plt.Polygon(
        vertices, closed=True,
        facecolor=fill_color, edgecolor="none", alpha=0.55, zorder=1,
    )
    ax.add_patch(triangle)

    # edges
    edges = [(top, bl), (top, br), (bl, br)]
    for i, (p1, p2) in enumerate(edges):
        ls = edge_styles[i]
        lw = 2.0 if ls == "-" else 1.5
        ec = edge_color if ls == "-" else DASHED_EDGE
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color=ec, linewidth=lw, linestyle=ls, zorder=2,
        )

    # edge labels (name + relation)
    edge_midpoints = [(top + bl) / 2, (top + br) / 2, (bl + br) / 2]
    # offsets to push labels away from triangle centre
    centre = vertices.mean(axis=0)
    for i, mid in enumerate(edge_midpoints):
        direction = mid - centre
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        # bottom edge (i==2) closer to triangle; side edges further out
        scale = 0.50 if i == 2 else 0.58
        offset = direction * scale

        label_pos = mid + offset

        if i == 2:
            # Put BC/BD just above the bottom edge inside the triangle,
            # and move the relation label into the former BC/BD slot below.
            ax.text(
                mid[0], mid[1] + 0.10, edge_labels[i],
                ha="center", va="bottom", fontsize=14, fontweight="bold",
                color=TEXT_COLOR, zorder=5,
            )
            ax.text(
                label_pos[0], label_pos[1] + 0.11, relation_labels[i],
                ha="center", va="bottom", fontsize=15, fontweight="bold",
                color="#1a1a1a", zorder=5,
            )
        else:
            # edge name (bold)
            ax.text(
                label_pos[0], label_pos[1] + 0.07, edge_labels[i],
                ha="center", va="bottom", fontsize=14, fontweight="bold",
                color=TEXT_COLOR, zorder=5,
            )
            # relation label (bold, dark)
            ax.text(
                label_pos[0], label_pos[1] - 0.10, relation_labels[i],
                ha="center", va="top", fontsize=15, fontweight="bold",
                color="#1a1a1a", zorder=5,
            )

    # vertex labels
    vert_offsets = [
        np.array([0, 0.18]),    # top → push up
        np.array([-0.15, -0.18]),  # bottom-left → push down-left
        np.array([0.15, -0.18]),   # bottom-right → push down-right
    ]
    for i, (vx, vy) in enumerate(vertices):
        ox, oy = vert_offsets[i]
        is_ambig = (i == ambig_idx)
        id_color = AMBIG_COLOR if is_ambig else TEXT_COLOR
        lbl_weight = "bold" if is_ambig else "normal"

        # ID circle — nudge text down slightly so the letter sits
        # at the optical centre of the circle
        cx, cy = vx + ox, vy + oy
        circle = plt.Circle(
            (cx, cy), 0.13,
            facecolor="white", edgecolor=id_color, linewidth=2.0, zorder=4,
        )
        ax.add_patch(circle)
        ax.text(
            cx, cy - 0.012, pair_ids[i],
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=id_color, zorder=5,
        )

        # pair name below/above the circle
        pair_offset_y = -0.32 if oy > 0 else 0.32
        if i == 0:  # top vertex: label above circle
            pair_offset_y = 0.32
        else:  # bottom vertices: label below circle
            pair_offset_y = -0.32

        ax.text(
            vx + ox, vy + oy + pair_offset_y, pair_labels[i],
            ha="center", va="center" if pair_offset_y > 0 else "top",
            fontsize=14, fontweight="bold", color="#1a1a1a", zorder=5,
        )

    # title
    ax.text(
        centre[0], vertices[:, 1].max() + 0.72, title,
        ha="center", va="bottom", fontsize=15, fontweight="bold",
        color=TEXT_COLOR,
    )

    ax.set_aspect("equal")
    ax.axis("off")


def main() -> int:
    args = _parse_args()

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.0, 4.8))

    # equilateral triangle: side = 2, height = sqrt(3)
    h = np.sqrt(3)
    tri = np.array([
        [0.0, h],        # top
        [-1.0, 0.0],     # bottom-left
        [1.0, 0.0],      # bottom-right
    ])

    # ── Left: Consistent (ABC) ──────────────────────────────────────
    _draw_triangle(
        ax_l, tri,
        fill_color=CONSISTENT_FILL,
        edge_color=CONSISTENT_EDGE,
        edge_styles=["-", "-", "-"],          # all solid
        pair_labels=["Snail : Shell", "Owl : Tree", "Human : House"],
        pair_ids=["A", "B", "C"],
        edge_labels=["AB", "AC", "BC"],
        relation_labels=["lives in", "lives in", "lives in"],
        title="Consistent (ABC)",
        ambig_idx=0,
    )
    ax_l.set_xlim(-2.0, 2.0)
    ax_l.set_ylim(-0.8, h + 1.0)

    # ── Right: Mixed (ABD) ──────────────────────────────────────────
    _draw_triangle(
        ax_r, tri,
        fill_color=MIXED_FILL,
        edge_color=MIXED_EDGE,
        edge_styles=["-", "-", "--"],         # BD is dashed
        pair_labels=["Snail : Shell", "Owl : Tree", "Student : Backpack"],
        pair_ids=["A", "B", "D"],
        edge_labels=["AB", "AD", "BD"],
        relation_labels=["lives in", "carries", "?"],
        title="Mixed (ABD)",
        ambig_idx=0,
    )
    ax_r.set_xlim(-2.0, 2.0)
    ax_r.set_ylim(-0.8, h + 1.0)

    fig.tight_layout(pad=1.0)

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight", facecolor="white")
    if args.out_pdf:
        Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_pdf, bbox_inches="tight", facecolor="white")
    if args.out_svg:
        Path(args.out_svg).parent.mkdir(parents=True, exist_ok=True)
        # Keep text as SVG text instead of converting glyphs to paths so
        # PowerPoint can edit labels more reliably after import.
        with matplotlib.rc_context({"svg.fonttype": "none"}):
            fig.savefig(args.out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out}")
    if args.out_pdf:
        print(f"Saved → {args.out_pdf}")
    if args.out_svg:
        print(f"Saved → {args.out_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
