#!/usr/bin/env python3
"""CCN Figure 2: Three-panel full-width figure.

Panel A  – Human PT (q-wise product-test ratios)
Panel B  – LLM PT (placeholder or real data)
Panel C  – PCA 2D scatter of condition-specific function vectors
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
import matplotlib.ticker as mticker
import numpy as np


# ── PT colour palette ────────────────────────────────────────────────
CONSISTENT_COLOR = "#7f8c8d"
MIXED_COLOR      = "#c0392b"
STAR_COLOR       = "#c0392b"
SEPARATOR_COLOR  = "#e0e0e0"
REFLINE_COLOR    = "#555555"

# ── PCA colour palette ──────────────────────────────────────────────
PCA_COLORS = {
    "AAA":  "#8e44ad",
    "BBB":  "#2471a3",
    "BABA": "#e67e22",
    "DDD":  "#1a8a5c",
    "DADA": "#d4ac0d",
}
PCA_MARKERS = {
    "AAA":  {"marker": "o", "s_trial": 32, "s_centroid": 280},
    "BBB":  {"marker": "s", "s_trial": 32, "s_centroid": 260},
    "BABA": {"marker": "D", "s_trial": 28, "s_centroid": 260},
    "DDD":  {"marker": "^", "s_trial": 34, "s_centroid": 280},
    "DADA": {"marker": "d", "s_trial": 28, "s_centroid": 260},
}
CONDITION_ORDER = ["AAA", "BBB", "BABA", "DDD", "DADA"]

# ── PT defaults ──────────────────────────────────────────────────────
DEFAULT_QIDS = "Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18"
DEFAULT_STAR_QIDS = "Q1,Q4,Q8,Q10,Q11,Q16"

BASE_REQUIRED_COLS = {
    "q_id",
    "pt_abc_ci2_low", "pt_abc_ci2_high",
    "pt_abd_ci2_low", "pt_abd_ci2_high",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCN Figure 2: 3-panel (Human PT | LLM PT | PCA)")
    p.add_argument("--human_csv", required=True)
    p.add_argument("--llm_csv", default=None)
    p.add_argument("--pca_points_csv", required=True)
    p.add_argument("--pca_centroids_csv", required=True)
    p.add_argument("--pca_meta_json", required=True)
    p.add_argument("--out_png", required=True)
    p.add_argument("--out_pdf", default=None)
    p.add_argument("--out_svg", default=None)
    p.add_argument("--qids", default=DEFAULT_QIDS)
    p.add_argument("--star_qids_human", default=DEFAULT_STAR_QIDS)
    p.add_argument("--star_qids_llm", default="")
    p.add_argument("--summary_stat", choices=["mean", "median"], default="mean")
    p.add_argument("--llm_shot", type=int, default=None)
    p.add_argument("--llm_ymax", type=float, default=None)
    p.add_argument("--llm_label", default="Llama-3.1-70B")
    p.add_argument("--mode", choices=["composite", "llm_only", "pca_only"], default="composite")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


# ── PT helpers ───────────────────────────────────────────────────────
def _norm_qid(raw: str) -> str:
    m = re.search(r"(\d+)", str(raw))
    if not m:
        raise ValueError(f"Cannot parse q_id: {raw!r}")
    return f"Q{int(m.group(1))}"


def _load_pt(
    path: str,
    summary_stat: str,
    shot: int | None = None,
) -> dict[str, dict[str, float]]:
    value_cols = {f"pt_abc_{summary_stat}", f"pt_abd_{summary_stat}"}
    required_cols = BASE_REQUIRED_COLS | value_cols
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row in {path}")
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        rows: dict[str, dict[str, float]] = {}
        for r in reader:
            if shot is not None:
                row_shot = r.get("shot")
                if row_shot is None or int(row_shot) != shot:
                    continue
            qid = _norm_qid(r["q_id"])
            rows[qid] = {
                "pt_abc_value": float(r[f"pt_abc_{summary_stat}"]),
                "pt_abd_value": float(r[f"pt_abd_{summary_stat}"]),
                "pt_abc_ci2_low": float(r["pt_abc_ci2_low"]),
                "pt_abc_ci2_high": float(r["pt_abc_ci2_high"]),
                "pt_abd_ci2_low": float(r["pt_abd_ci2_low"]),
                "pt_abd_ci2_high": float(r["pt_abd_ci2_high"]),
            }
        return rows


def _parse_qids(raw: str) -> list[str]:
    return [_norm_qid(s.strip()) for s in raw.split(",") if s.strip()]


def _parse_star_qids(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return set(_parse_qids(raw))


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
        markersize=5,
        color=color,
        ecolor=color,
        elinewidth=1.2,
        capsize=3,
        zorder=3,
    )

    anchor_y = plot_high
    was_clipped = False
    if clipped and ymax is not None:
        was_clipped = True
        # 선 + 세모: 점에서 세모까지 연결
        arrow_top = plot_value + ymax * 0.12
        # 선: 점 위에서 세모 아래까지
        ax.plot([x, x], [plot_value, arrow_top], color=color, lw=1.2,
                solid_capstyle="butt", zorder=4, clip_on=False)
        # 세모(▲) 마커
        ax.plot(x, arrow_top, marker="^", markersize=6, color=color,
                zorder=5, clip_on=False)
        # 숫자: 점 옆(오른쪽)에 표시
        if value > ymax:
            ax.text(
                x + 0.25, plot_value,
                f"{value:.0f}",
                ha="left", va="center",
                fontsize=7, fontweight="bold", color=color,
                zorder=5,
            )
        anchor_y = arrow_top
    return anchor_y, was_clipped


# ── PCA helpers ──────────────────────────────────────────────────────
def _load_pca_points(path: str) -> dict[str, np.ndarray]:
    groups: dict[str, list[list[float]]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cond = row["condition"]
            groups.setdefault(cond, []).append(
                [float(row["pc1"]), float(row["pc2"])]
            )
    return {k: np.array(v) for k, v in groups.items()}


def _load_pca_centroids(path: str) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            centroids[row["condition"]] = np.array(
                [float(row["pc1"]), float(row["pc2"])]
            )
    return centroids


# ── PT panel drawer ──────────────────────────────────────────────────
def draw_pt_panel(
    ax: plt.Axes,
    rows: dict[str, dict[str, float]],
    qids: list[str],
    star_qids: set[str],
    panel_label: str,
    ymax: float | None = None,
) -> None:
    n = len(qids)
    xs = list(range(n))
    dx = 0.16

    for i, qid in enumerate(qids):
        r = rows[qid]
        x_c, x_m = xs[i] - dx, xs[i] + dx

        if i > 0:
            ax.axvline(i - 0.5, color=SEPARATOR_COLOR, linewidth=0.6, zorder=0)

        _, _ = _draw_capped_errorbar(
            ax,
            x_c,
            r["pt_abc_value"],
            r["pt_abc_ci2_low"],
            r["pt_abc_ci2_high"],
            CONSISTENT_COLOR,
            ymax,
        )
        mixed_anchor_y, mixed_clipped = _draw_capped_errorbar(
            ax,
            x_m,
            r["pt_abd_value"],
            r["pt_abd_ci2_low"],
            r["pt_abd_ci2_high"],
            MIXED_COLOR,
            ymax,
        )
        if qid in star_qids:
            if mixed_clipped:
                # star above the value number (number is at x_m+0.25, ymax)
                ax.text(
                    x_m + 0.25, ymax + 0.55, "*",
                    ha="center", va="bottom", fontsize=13,
                    color=STAR_COLOR, fontweight="bold", zorder=4,
                )
            else:
                star_y = mixed_anchor_y + (0.18 if ymax is None else max(0.2, ymax * 0.04))
                ax.text(
                    x_m, star_y, "*",
                    ha="center", va="bottom", fontsize=13,
                    color=STAR_COLOR, fontweight="bold", zorder=4,
                )

    ax.axhline(1.0, color=REFLINE_COLOR, ls="--", lw=0.9, zorder=1)

    ax.set_xlim(-0.55, n - 0.45)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i + 1) for i in range(n)], fontsize=8)
    ax.set_xlabel("Item", fontsize=10)
    ax.set_ylabel("Product Test Ratio", fontsize=10)
    if ymax is not None:
        ax.set_ylim(0.0, ymax + ymax * 0.25)
        ax.set_yticks(np.linspace(0.0, ymax, 6))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="y", alpha=0.20, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        -0.02, 1.06, panel_label,
        transform=ax.transAxes, fontsize=11, fontweight="bold",
        va="bottom", ha="left",
    )


def draw_pt_placeholder(
    ax: plt.Axes, label: str, panel_label: str, qids: list[str],
) -> None:
    n = len(qids)
    xs = list(range(n))
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_ylim(0, 5)
    ax.text(
        0.5, 0.5, f"{label}\n(data in progress)",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=9, color="#aaaaaa", style="italic",
    )
    ax.axhline(1.0, color=REFLINE_COLOR, ls="--", lw=0.9, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(qids, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Product Test Ratio", fontsize=8)
    ax.set_xlabel("Item", fontsize=8)
    ax.tick_params(axis="both", labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        -0.02, 1.06, panel_label,
        transform=ax.transAxes, fontsize=11, fontweight="bold",
        va="bottom", ha="left",
    )


# ── PCA panel drawer ────────────────────────────────────────────────
def draw_pca_panel(
    ax: plt.Axes,
    points: dict[str, np.ndarray],
    centroids: dict[str, np.ndarray],
    var_ratio: list[float],
) -> None:
    # trial-level scatter
    for cond in CONDITION_ORDER:
        if cond not in points:
            continue
        pts = points[cond]
        ms = PCA_MARKERS[cond]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=PCA_COLORS[cond], marker=ms["marker"], s=ms["s_trial"],
            alpha=0.45, edgecolors="white", linewidths=0.3, zorder=2,
        )

    # centroid markers
    for cond in CONDITION_ORDER:
        if cond not in centroids:
            continue
        c = centroids[cond]
        ms = PCA_MARKERS[cond]
        ax.scatter(
            [c[0]], [c[1]],
            c=PCA_COLORS[cond], marker=ms["marker"], s=ms["s_centroid"],
            edgecolors="white", linewidths=2.0, zorder=8,
        )

    # arrows: AAA → BABA, AAA → DADA
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
                     arrowprops=dict(**arrow_kw, color=PCA_COLORS["BABA"], alpha=0.75),
                     zorder=5)
    if "DADA" in centroids:
        direction = centroids["DADA"] - aaa
        start = aaa + direction * 0.18
        end = aaa + direction * 0.82
        ax.annotate("", xy=end, xytext=start,
                     arrowprops=dict(**arrow_kw, color=PCA_COLORS["DADA"], alpha=0.75),
                     zorder=5)

    # dashed links: BABA···BBB, DADA···DDD
    if "BABA" in centroids and "BBB" in centroids:
        ax.plot(
            [centroids["BABA"][0], centroids["BBB"][0]],
            [centroids["BABA"][1], centroids["BBB"][1]],
            color=PCA_COLORS["BABA"], lw=1.0, alpha=0.35, ls="--", zorder=3,
        )
    if "DADA" in centroids and "DDD" in centroids:
        ax.plot(
            [centroids["DADA"][0], centroids["DDD"][0]],
            [centroids["DADA"][1], centroids["DDD"][1]],
            color=PCA_COLORS["DADA"], lw=1.0, alpha=0.35, ls="--", zorder=3,
        )

    # centroid text labels
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
            cond, fontsize=9, fontweight="bold", color=PCA_COLORS[cond],
            ha="center", va="center", zorder=10,
        )

    # axes
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(
        -0.02, 1.06, "C",
        transform=ax.transAxes, fontsize=11, fontweight="bold",
        va="bottom", ha="left",
    )


# ── main ─────────────────────────────────────────────────────────────
def main() -> int:
    args = _parse_args()
    qids = _parse_qids(args.qids)
    star_human = _parse_star_qids(args.star_qids_human)
    star_llm = _parse_star_qids(args.star_qids_llm)
    human_rows = _load_pt(args.human_csv, summary_stat=args.summary_stat)

    # PCA data
    pca_points = _load_pca_points(args.pca_points_csv)
    pca_centroids = _load_pca_centroids(args.pca_centroids_csv)
    with Path(args.pca_meta_json).open("r", encoding="utf-8") as f:
        pca_meta = json.load(f)
    var_ratio = pca_meta["explained_variance_ratio"]

    if args.mode == "llm_only":
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        if args.llm_csv:
            llm_rows = _load_pt(args.llm_csv, summary_stat=args.summary_stat, shot=args.llm_shot)
            draw_pt_panel(ax, llm_rows, qids, star_llm,
                          panel_label=f"A. Llama (9 demonstrations)",
                          ymax=args.llm_ymax)
        else:
            draw_pt_placeholder(ax, args.llm_label, panel_label="A", qids=qids)
        ax.scatter([], [], color=CONSISTENT_COLOR, s=30, label="Consistent (ABC)")
        ax.scatter([], [], color=MIXED_COLOR, s=30, label="Mixed (ABD)")
        ax.legend(frameon=False, loc="upper right", fontsize=8, ncol=1,
                  bbox_to_anchor=(1.0, 1.18),
                  handletextpad=0.4, borderpad=0.0, labelspacing=0.1)
        fig.subplots_adjust(top=0.85, bottom=0.15, left=0.15, right=0.95)

    elif args.mode == "pca_only":
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        draw_pca_panel(ax, pca_points, pca_centroids, var_ratio)
        ax.texts[-1].set_text("B. PCA projection")  # override panel label
        fig.subplots_adjust(top=0.85, bottom=0.12, left=0.08, right=0.95)

    else:  # composite
        fig, (ax_a, ax_b, ax_c) = plt.subplots(
            1, 3, figsize=(7.5, 2.8),
            gridspec_kw={"wspace": 0.45, "width_ratios": [1.0, 1.0, 0.85]},
        )
        draw_pt_panel(ax_a, human_rows, qids, star_human, panel_label="A  Human")
        ax_a.scatter([], [], color=CONSISTENT_COLOR, s=20, label="Consistent (ABC)")
        ax_a.scatter([], [], color=MIXED_COLOR, s=20, label="Mixed (ABD)")
        ax_a.legend(frameon=False, loc="upper left", fontsize=6, ncol=1,
                    handletextpad=0.3, borderpad=0.3)
        if args.llm_csv:
            llm_rows = _load_pt(args.llm_csv, summary_stat=args.summary_stat, shot=args.llm_shot)
            draw_pt_panel(ax_b, llm_rows, qids, star_llm,
                          panel_label=f"B  {args.llm_label}", ymax=args.llm_ymax)
        else:
            draw_pt_placeholder(ax_b, args.llm_label, panel_label=f"B  {args.llm_label}", qids=qids)
        draw_pca_panel(ax_c, pca_points, pca_centroids, var_ratio)
        fig.tight_layout(pad=0.8)

    # save
    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight", facecolor="white")
    if args.out_pdf:
        Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_pdf, bbox_inches="tight", facecolor="white")
    if args.out_svg:
        Path(args.out_svg).parent.mkdir(parents=True, exist_ok=True)
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
