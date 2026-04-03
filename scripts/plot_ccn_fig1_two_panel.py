#!/usr/bin/env python3
"""CCN Figure 1: Two-panel q-wise PT comparison (Human + LLM placeholder).

Panel A  – Human behavioural data
Panel B  – LLM (placeholder, to be filled later)

Improvements over the diagnostic flat_ci_starred plot:
  - No figure-level title (use LaTeX caption)
  - Panel labels (A, B) in bold
  - Cleaner legend: Consistent (ABC) / Mixed (ABD)
  - Background bands removed; subtle vertical separators
  - y = 1 reference with zone annotation
  - Star offset above CI bar for readability
  - Publication colour palette (grey vs red-orange)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── colour palette ──────────────────────────────────────────────────
CONSISTENT_COLOR = "#7f8c8d"   # muted grey  → "nothing special"
MIXED_COLOR      = "#c0392b"   # red-orange  → "violation alert"
STAR_COLOR       = "#c0392b"
SEPARATOR_COLOR  = "#e0e0e0"
REFLINE_COLOR    = "#555555"

# ── default q-id list (matches the original starred figure) ─────────
DEFAULT_QIDS = "Q1,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q16,Q18"
DEFAULT_STAR_QIDS = "Q1,Q4,Q8,Q10,Q11,Q16"

REQUIRED_COLS = {
    "q_id",
    "pt_abc_mean", "pt_abd_mean",
    "pt_abc_ci2_low", "pt_abc_ci2_high",
    "pt_abd_ci2_low", "pt_abd_ci2_high",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCN Figure 1: two-panel PT plot")
    p.add_argument("--human_csv", required=True, help="Human bootstrap summary CSV")
    p.add_argument("--llm_csv", default=None, help="LLM bootstrap summary CSV (omit for placeholder)")
    p.add_argument("--out_png", required=True)
    p.add_argument("--out_pdf", default=None)
    p.add_argument("--qids", default=DEFAULT_QIDS)
    p.add_argument("--star_qids_human", default=DEFAULT_STAR_QIDS)
    p.add_argument("--star_qids_llm", default="")
    p.add_argument("--llm_label", default="Llama-3.1-70B")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


# ── helpers ─────────────────────────────────────────────────────────
def _norm_qid(raw: str) -> str:
    m = re.search(r"(\d+)", str(raw))
    if not m:
        raise ValueError(f"Cannot parse q_id: {raw!r}")
    return f"Q{int(m.group(1))}"


def _load(path: str) -> dict[str, dict[str, float]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        missing = REQUIRED_COLS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        rows: dict[str, dict[str, float]] = {}
        for r in reader:
            qid = _norm_qid(r["q_id"])
            rows[qid] = {k: float(r[k]) for k in REQUIRED_COLS if k != "q_id"}
        return rows


def _parse_qids(raw: str) -> list[str]:
    return [_norm_qid(s.strip()) for s in raw.split(",") if s.strip()]


def _parse_star_qids(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return set(_parse_qids(raw))


# ── single-panel plotter ───────────────────────────────────────────
def _draw_panel(
    ax: plt.Axes,
    rows: dict[str, dict[str, float]],
    qids: list[str],
    star_qids: set[str],
    panel_label: str,
    show_xlabel: bool = True,
) -> None:
    n = len(qids)
    xs = list(range(n))
    dx = 0.16  # dodge

    for i, qid in enumerate(qids):
        r = rows[qid]
        x_c, x_m = xs[i] - dx, xs[i] + dx

        # subtle separator between items
        if i > 0:
            ax.axvline(i - 0.5, color=SEPARATOR_COLOR, linewidth=0.6, zorder=0)

        # consistent (ABC)
        ax.errorbar(
            x_c, r["pt_abc_mean"],
            yerr=[[r["pt_abc_mean"] - r["pt_abc_ci2_low"]],
                  [r["pt_abc_ci2_high"] - r["pt_abc_mean"]]],
            fmt="o", markersize=5, color=CONSISTENT_COLOR,
            ecolor=CONSISTENT_COLOR, elinewidth=1.4, capsize=3, zorder=3,
        )
        # mixed (ABD)
        ax.errorbar(
            x_m, r["pt_abd_mean"],
            yerr=[[r["pt_abd_mean"] - r["pt_abd_ci2_low"]],
                  [r["pt_abd_ci2_high"] - r["pt_abd_mean"]]],
            fmt="o", markersize=5, color=MIXED_COLOR,
            ecolor=MIXED_COLOR, elinewidth=1.4, capsize=3, zorder=3,
        )
        # significance star
        if qid in star_qids:
            star_y = r["pt_abd_ci2_high"] + 0.15
            ax.text(
                x_m, star_y, "*",
                ha="center", va="bottom", fontsize=13,
                color=STAR_COLOR, fontweight="bold", zorder=4,
            )

    # reference line + zone labels (placed via axes transform so they
    # sit at a fixed x-fraction, avoiding overlap with data points)
    ax.axhline(1.0, color=REFLINE_COLOR, ls="--", lw=0.9, zorder=1)
    ax.annotate(
        "TI violated  \u2191", xy=(1.01, 1.0), xycoords=("axes fraction", "data"),
        fontsize=6.5, color="#999999", ha="left", va="bottom",
        annotation_clip=False,
    )
    ax.annotate(
        "TI preserved \u2193", xy=(1.01, 1.0), xycoords=("axes fraction", "data"),
        fontsize=6.5, color="#999999", ha="left", va="top",
        annotation_clip=False,
    )

    # axes
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_xticks(xs)
    if show_xlabel:
        ax.set_xticklabels(qids, fontsize=8)
        ax.set_xlabel("Item", fontsize=9)
    else:
        ax.set_xticklabels(qids, fontsize=8)

    ax.set_ylabel("Product Test Ratio", fontsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="y", alpha=0.20, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # panel label (bold A / B)
    ax.text(
        -0.08, 1.05, panel_label,
        transform=ax.transAxes, fontsize=13, fontweight="bold",
        va="bottom", ha="left",
    )


def _draw_placeholder(
    ax: plt.Axes, label: str, panel_label: str, qids: list[str],
) -> None:
    """Empty panel with centred text, x-axis matching Panel A."""
    n = len(qids)
    xs = list(range(n))
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_ylim(0, 5)
    ax.text(
        0.5, 0.5, f"{label}\n(data in progress)",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=11, color="#aaaaaa", style="italic",
    )
    ax.axhline(1.0, color=REFLINE_COLOR, ls="--", lw=0.9, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(qids, fontsize=8)
    ax.set_ylabel("Product Test Ratio", fontsize=9)
    ax.set_xlabel("Item", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        -0.08, 1.05, panel_label,
        transform=ax.transAxes, fontsize=13, fontweight="bold",
        va="bottom", ha="left",
    )


# ── main ────────────────────────────────────────────────────────────
def main() -> int:
    args = _parse_args()
    qids = _parse_qids(args.qids)
    star_human = _parse_star_qids(args.star_qids_human)
    star_llm = _parse_star_qids(args.star_qids_llm)
    human_rows = _load(args.human_csv)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.0, 5.8), sharex=False,
        gridspec_kw={"hspace": 0.38},
    )

    # Panel A — Human
    _draw_panel(ax_top, human_rows, qids, star_human, panel_label="A  Human", show_xlabel=False)

    # Panel B — LLM (real data or placeholder)
    if args.llm_csv:
        llm_rows = _load(args.llm_csv)
        _draw_panel(ax_bot, llm_rows, qids, star_llm, panel_label=f"B  {args.llm_label}", show_xlabel=True)
    else:
        _draw_placeholder(ax_bot, args.llm_label, panel_label="B  " + args.llm_label, qids=qids)

    # shared legend (only once, at top)
    ax_top.scatter([], [], color=CONSISTENT_COLOR, s=30, label="Consistent (ABC)")
    ax_top.scatter([], [], color=MIXED_COLOR, s=30, label="Mixed (ABD)")
    ax_top.legend(
        frameon=False, loc="upper left", fontsize=8, ncol=2,
        handletextpad=0.4, columnspacing=1.0,
    )

    # save
    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
    if args.out_pdf:
        pdf = Path(args.out_pdf)
        pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(pdf), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
    if args.out_pdf:
        print(f"Saved → {args.out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
