#!/usr/bin/env python3
"""Plot endpoint movement presentation figures from q-wise movement artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc  # noqa: E402


BLUE = "#4e79a7"
BLUE_LIGHT = "#9ecae9"
ORANGE = "#f28e2b"
ORANGE_LIGHT = "#fdd0a2"
GRAY = "#4d4d4d"
GRID = "#d0d7de"
PANEL_BG = "#fcfcfc"

REQUIRED_COLS = {
    "q_id",
    "p_b_x_bab",
    "p_d_x_bab",
    "p_d_x_dad",
    "p_b_x_dad",
    "r_b_x_bab",
    "r_d_x_dad",
    "alpha_bab",
    "beta_bab",
    "alpha_dad",
    "beta_dad",
    "joint_selectivity_bab",
    "joint_selectivity_dad",
    "joint_resid_bab",
    "joint_resid_dad",
    "axis_angle_deg",
}


@dataclass(frozen=True)
class MovementRow:
    q_id: str
    p_b_x_bab: float
    p_d_x_bab: float
    p_d_x_dad: float
    p_b_x_dad: float
    r_b_x_bab: float
    r_d_x_dad: float
    alpha_bab: float
    beta_bab: float
    alpha_dad: float
    beta_dad: float
    joint_selectivity_bab: float
    joint_selectivity_dad: float
    joint_resid_bab: float
    joint_resid_dad: float
    axis_angle_deg: float


@dataclass(frozen=True)
class GeometryData:
    coords: Mapping[str, np.ndarray]
    proj_bab: np.ndarray
    proj_dad: np.ndarray
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot endpoint movement presentation figures."
    )
    parser.add_argument("--movement_csv", required=True, help="movement_qwise.csv path")
    parser.add_argument(
        "--means_npz",
        required=True,
        help="movement_condition_means.npz path",
    )
    parser.add_argument(
        "--focus_q",
        default="Q1",
        help="Focus q_id for the geometry panel (default: Q1)",
    )
    parser.add_argument("--out_png", default=None, help="Composite PNG output path")
    parser.add_argument("--out_pdf", default=None, help="Composite PDF output path")
    parser.add_argument(
        "--panel_out_dir",
        default=None,
        help="Optional directory for standalone geometry/summary panel exports",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Figure dpi")
    return parser.parse_args()


def qid_sort_key(qid: str) -> Tuple[int, str]:
    text = (qid or "").strip()
    if text.upper().startswith("Q") and text[1:].isdigit():
        return (int(text[1:]), text)
    return (10**9, text)


def _float_field(row: Mapping[str, str], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"missing numeric field {key!r} in q_id={row.get('q_id')!r}")
    return float(value)


def load_rows(path: Path) -> List[MovementRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing header row: {path}")
        missing = REQUIRED_COLS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"movement CSV missing required columns: {sorted(missing)}")
        rows = [
            MovementRow(
                q_id=row["q_id"],
                p_b_x_bab=_float_field(row, "p_b_x_bab"),
                p_d_x_bab=_float_field(row, "p_d_x_bab"),
                p_d_x_dad=_float_field(row, "p_d_x_dad"),
                p_b_x_dad=_float_field(row, "p_b_x_dad"),
                r_b_x_bab=_float_field(row, "r_b_x_bab"),
                r_d_x_dad=_float_field(row, "r_d_x_dad"),
                alpha_bab=_float_field(row, "alpha_bab"),
                beta_bab=_float_field(row, "beta_bab"),
                alpha_dad=_float_field(row, "alpha_dad"),
                beta_dad=_float_field(row, "beta_dad"),
                joint_selectivity_bab=_float_field(row, "joint_selectivity_bab"),
                joint_selectivity_dad=_float_field(row, "joint_selectivity_dad"),
                joint_resid_bab=_float_field(row, "joint_resid_bab"),
                joint_resid_dad=_float_field(row, "joint_resid_dad"),
                axis_angle_deg=_float_field(row, "axis_angle_deg"),
            )
            for row in reader
        ]
    rows.sort(key=lambda item: qid_sort_key(item.q_id))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def row_by_qid(rows: Sequence[MovementRow]) -> Dict[str, MovementRow]:
    return {row.q_id: row for row in rows}


def load_focus_means(path: Path, qid: str) -> Dict[str, np.ndarray]:
    keys = {
        "a": f"{qid}__a",
        "b": f"{qid}__b",
        "d": f"{qid}__d",
        "x_bab": f"{qid}__x_bab",
        "x_dad": f"{qid}__x_dad",
    }
    payload = np.load(path)
    missing = [name for name, key in keys.items() if key not in payload]
    if missing:
        raise KeyError(f"means npz missing {missing} for focus_q={qid}")
    return {name: np.asarray(payload[keys[name]], dtype=np.float64) for name in keys}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def compute_geometry(row: MovementRow, points: Mapping[str, np.ndarray]) -> GeometryData:
    a = points["a"]
    b = points["b"]
    d = points["d"]
    x_bab = points["x_bab"]
    x_dad = points["x_dad"]

    u_b = b - a
    norm_ab = float(np.linalg.norm(u_b))
    if norm_ab == 0.0:
        raise ValueError("focus_q has zero-length A->B axis")
    e1 = u_b / norm_ab

    u_d = d - a
    u_d_ortho = u_d - (float(np.dot(u_d, e1)) * e1)
    norm_u_d_ortho = float(np.linalg.norm(u_d_ortho))
    if norm_u_d_ortho == 0.0:
        raise ValueError("focus_q has degenerate B/D span")
    e2 = u_d_ortho / norm_u_d_ortho

    coords: Dict[str, np.ndarray] = {}
    for name, vec in points.items():
        delta = vec - a
        coords[name] = np.array(
            [float(np.dot(delta, e1)), float(np.dot(delta, e2))],
            dtype=np.float64,
        )

    proj_bab = row.p_b_x_bab * coords["b"]
    proj_dad = row.p_d_x_dad * coords["d"]

    extent_points = [
        coords["a"],
        coords["b"],
        coords["d"],
        coords["x_bab"],
        coords["x_dad"],
        proj_bab,
        proj_dad,
    ]
    x_vals = [float(point[0]) for point in extent_points]
    y_vals = [float(point[1]) for point in extent_points]
    x_min = min(x_vals)
    x_max = max(x_vals)
    y_min = min(y_vals)
    y_max = max(y_vals)
    x_pad = max(0.35, 0.18 * max(1e-6, x_max - x_min))
    y_pad = max(0.35, 0.22 * max(1e-6, y_max - y_min))
    return GeometryData(
        coords=coords,
        proj_bab=np.asarray(proj_bab, dtype=np.float64),
        proj_dad=np.asarray(proj_dad, dtype=np.float64),
        xlim=(x_min - x_pad, x_max + x_pad),
        ylim=(y_min - y_pad, y_max + y_pad),
    )


def add_text_panel(ax, title: str, lines: Sequence[str]) -> None:
    ax.set_axis_off()
    ax.set_facecolor(PANEL_BG)
    ax.text(
        0.0,
        1.0,
        title,
        va="top",
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=GRAY,
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.88,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10.8,
        linespacing=1.45,
        color=GRAY,
        transform=ax.transAxes,
    )


def _callout_text(label: str, progress: float, cross: float, residual: float) -> str:
    return (
        f"{label}\n"
        f"progress = {progress:.3f}\n"
        f"cross = {cross:.3f}\n"
        f"residual = {residual:.3f}"
    )


def plot_geometry_panel(
    ax,
    row: MovementRow,
    geometry: GeometryData,
) -> None:
    coords = geometry.coords
    origin = coords["a"]
    b = coords["b"]
    d = coords["d"]
    x_bab = coords["x_bab"]
    x_dad = coords["x_dad"]

    ax.set_title(
        f"B. {row.q_id} Endpoint Geometry",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=GRAY,
    )
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(*geometry.xlim)
    ax.set_ylim(*geometry.ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.35, color=GRID)
    ax.axhline(0.0, color=GRID, linewidth=0.8, zorder=0)
    ax.axvline(0.0, color=GRID, linewidth=0.8, zorder=0)

    ax.annotate(
        "",
        xy=b,
        xytext=origin,
        arrowprops=dict(arrowstyle="->", lw=2.0, color=BLUE_LIGHT),
        zorder=1,
    )
    ax.annotate(
        "",
        xy=d,
        xytext=origin,
        arrowprops=dict(arrowstyle="->", lw=2.0, color=ORANGE_LIGHT),
        zorder=1,
    )
    ax.annotate(
        "",
        xy=x_bab,
        xytext=origin,
        arrowprops=dict(arrowstyle="->", lw=2.8, color=BLUE),
        zorder=2,
    )
    ax.annotate(
        "",
        xy=x_dad,
        xytext=origin,
        arrowprops=dict(arrowstyle="->", lw=2.8, color=ORANGE),
        zorder=2,
    )

    ax.plot(
        [origin[0], geometry.proj_bab[0]],
        [origin[1], geometry.proj_bab[1]],
        linestyle=(0, (2, 2)),
        linewidth=1.7,
        color=BLUE,
        alpha=0.85,
        zorder=2,
    )
    ax.plot(
        [geometry.proj_bab[0], x_bab[0]],
        [geometry.proj_bab[1], x_bab[1]],
        linestyle=(0, (4, 2)),
        linewidth=1.7,
        color=BLUE,
        alpha=0.85,
        zorder=2,
    )
    ax.plot(
        [origin[0], geometry.proj_dad[0]],
        [origin[1], geometry.proj_dad[1]],
        linestyle=(0, (2, 2)),
        linewidth=1.7,
        color=ORANGE,
        alpha=0.85,
        zorder=2,
    )
    ax.plot(
        [geometry.proj_dad[0], x_dad[0]],
        [geometry.proj_dad[1], x_dad[1]],
        linestyle=(0, (4, 2)),
        linewidth=1.7,
        color=ORANGE,
        alpha=0.85,
        zorder=2,
    )

    ax.scatter(
        [origin[0]],
        [origin[1]],
        s=90,
        color=GRAY,
        edgecolors="white",
        linewidths=0.9,
        zorder=4,
    )
    ax.scatter(
        [b[0]],
        [b[1]],
        s=95,
        facecolors="white",
        edgecolors=BLUE,
        linewidths=2.0,
        zorder=4,
    )
    ax.scatter(
        [d[0]],
        [d[1]],
        s=95,
        facecolors="white",
        edgecolors=ORANGE,
        linewidths=2.0,
        zorder=4,
    )
    ax.scatter(
        [x_bab[0]],
        [x_bab[1]],
        s=98,
        color=BLUE,
        edgecolors="white",
        linewidths=0.9,
        zorder=5,
    )
    ax.scatter(
        [x_dad[0]],
        [x_dad[1]],
        s=98,
        color=ORANGE,
        edgecolors="white",
        linewidths=0.9,
        zorder=5,
    )
    ax.scatter(
        [geometry.proj_bab[0], geometry.proj_dad[0]],
        [geometry.proj_bab[1], geometry.proj_dad[1]],
        s=28,
        color=[BLUE, ORANGE],
        alpha=0.95,
        zorder=4,
    )

    label_offsets = {
        "a": (-0.20, 0.18),
        "b": (0.10, -0.12),
        "d": (0.10, 0.10),
        "x_bab": (0.10, 0.16),
        "x_dad": (0.08, -0.16),
    }
    for key, label, color in [
        ("a", "A", GRAY),
        ("b", "B", BLUE),
        ("d", "D", ORANGE),
        ("x_bab", "BAB", BLUE),
        ("x_dad", "DAD", ORANGE),
    ]:
        point = coords[key]
        dx, dy = label_offsets[key]
        ax.text(
            point[0] + dx,
            point[1] + dy,
            label,
            fontsize=11,
            fontweight="bold",
            color=color,
            zorder=6,
        )

    x_range = geometry.xlim[1] - geometry.xlim[0]
    y_range = geometry.ylim[1] - geometry.ylim[0]
    bab_box_xy = (x_bab[0] + 0.16 * x_range, x_bab[1] + 0.12 * y_range)
    dad_box_xy = (x_dad[0] + 0.10 * x_range, x_dad[1] - 0.18 * y_range)

    ax.annotate(
        _callout_text("BABABA", row.p_b_x_bab, row.p_d_x_bab, row.r_b_x_bab),
        xy=x_bab,
        xytext=bab_box_xy,
        textcoords="data",
        fontsize=9.6,
        color=GRAY,
        bbox=dict(boxstyle="round,pad=0.35", fc="#eff6ff", ec=BLUE, lw=1.0),
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.1),
        zorder=6,
    )
    ax.annotate(
        _callout_text("DADADA", row.p_d_x_dad, row.p_b_x_dad, row.r_d_x_dad),
        xy=x_dad,
        xytext=dad_box_xy,
        textcoords="data",
        fontsize=9.6,
        color=GRAY,
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff7ed", ec=ORANGE, lw=1.0),
        arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.1),
        zorder=6,
    )

    theta = float(np.degrees(np.arctan2(d[1], d[0])))
    arc_radius = 0.30 * min(np.linalg.norm(b - origin), np.linalg.norm(d - origin))
    ax.add_patch(
        Arc(
            xy=origin,
            width=2.0 * arc_radius,
            height=2.0 * arc_radius,
            theta1=0.0,
            theta2=theta,
            color=GRAY,
            linewidth=1.2,
            zorder=3,
        )
    )
    ax.text(
        origin[0] + 0.58 * arc_radius,
        origin[1] + 0.46 * arc_radius,
        f"{row.axis_angle_deg:.1f} deg",
        fontsize=9.4,
        color=GRAY,
        zorder=6,
    )

    joint_text = (
        "Joint decomposition\n"
        f"alpha_bab = {row.alpha_bab:.3f}\n"
        f"beta_bab = {row.beta_bab:.3f}\n"
        f"alpha_dad = {row.alpha_dad:.3f}\n"
        f"beta_dad = {row.beta_dad:.3f}"
    )
    ax.text(
        0.98,
        0.97,
        joint_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9.8,
        color=GRAY,
        bbox=dict(boxstyle="round,pad=0.40", fc="white", ec=GRID, lw=0.9),
        zorder=7,
    )
    ax.text(
        0.02,
        0.03,
        "Not PCA: coordinates lie in span{B-A, D-A}",
        transform=ax.transAxes,
        fontsize=9.2,
        color=GRAY,
        ha="left",
        va="bottom",
    )
    ax.set_xlabel("Coordinate along A->B", fontsize=10.5, color=GRAY)
    ax.set_ylabel("Orthogonalized coordinate in span{A->B, A->D}", fontsize=10.5, color=GRAY)
    ax.tick_params(labelsize=8.5, colors=GRAY)


def _metric_xlim(values_a: Sequence[float], values_b: Sequence[float], include_zero: bool) -> Tuple[float, float]:
    vals = [float(v) for v in values_a] + [float(v) for v in values_b]
    if include_zero:
        vals.append(0.0)
    x_min = min(vals)
    x_max = max(vals)
    span = max(1e-6, x_max - x_min)
    pad = 0.14 * span
    return (x_min - pad, x_max + pad)


def plot_pair_metric(
    ax,
    rows: Sequence[MovementRow],
    focus_q: str,
    value_a: str,
    value_b: str,
    xlabel: str,
    title: str,
    include_zero: bool,
    note: Optional[str] = None,
) -> None:
    rows_sorted = sorted(rows, key=lambda item: qid_sort_key(item.q_id))
    qids = [row.q_id for row in rows_sorted]
    y = np.arange(len(rows_sorted), dtype=np.float64)
    bab = [float(getattr(row, value_a)) for row in rows_sorted]
    dad = [float(getattr(row, value_b)) for row in rows_sorted]

    ax.set_facecolor(PANEL_BG)
    ax.grid(axis="x", linestyle=":", alpha=0.35, color=GRID)
    if include_zero:
        ax.axvline(0.0, color=GRAY, linestyle="--", linewidth=1.0, alpha=0.7)

    for yi, bab_val, dad_val in zip(y, bab, dad):
        ax.plot([bab_val, dad_val], [yi, yi], color=GRID, linewidth=0.9, zorder=1)

    sizes_bab = [90 if qid == focus_q else 58 for qid in qids]
    sizes_dad = [90 if qid == focus_q else 58 for qid in qids]
    edge_bab = ["black" if qid == focus_q else "white" for qid in qids]
    edge_dad = ["black" if qid == focus_q else "white" for qid in qids]

    ax.scatter(
        bab,
        y - 0.12,
        s=sizes_bab,
        color=BLUE,
        edgecolors=edge_bab,
        linewidths=0.9,
        label="BABABA",
        zorder=3,
    )
    ax.scatter(
        dad,
        y + 0.12,
        s=sizes_dad,
        color=ORANGE,
        edgecolors=edge_dad,
        linewidths=0.9,
        label="DADADA",
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(qids)
    ax.invert_yaxis()
    ax.set_xlim(*_metric_xlim(bab, dad, include_zero=include_zero))
    ax.set_xlabel(xlabel, fontsize=10, color=GRAY)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=GRAY)
    ax.tick_params(labelsize=8.8, colors=GRAY)

    if note:
        ax.text(
            0.0,
            0.985,
            note,
            transform=ax.transAxes,
            fontsize=8.8,
            color=GRAY,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.90),
        )


def build_composite_figure(
    rows: Sequence[MovementRow],
    focus_row: MovementRow,
    geometry: GeometryData,
    dpi: int,
) -> plt.Figure:
    fig = plt.figure(figsize=(16.0, 9.0), dpi=dpi)
    outer = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[0.74, 1.55],
        width_ratios=[1.25, 1.10, 1.05],
        hspace=0.30,
        wspace=0.30,
    )

    ax_info = fig.add_subplot(outer[0, 0:2])
    ax_setup = fig.add_subplot(outer[0, 2])
    ax_geom = fig.add_subplot(outer[1, 0:2])
    summary_grid = outer[1, 2].subgridspec(2, 1, hspace=0.35)
    ax_sel = fig.add_subplot(summary_grid[0, 0])
    ax_resid = fig.add_subplot(summary_grid[1, 0])

    info_lines = [
        "PCA suggested drift, but 2D/3D plots cannot quantify movement in the original residual space.",
        "Endpoint movement turns the final-query state into three presentation-friendly questions:",
        "1. Direction: did the mixed condition move toward the intended endpoint?",
        "2. Selectivity: did it favor the intended branch over the cross branch?",
        "3. Cleanliness: is the shift mostly on-axis, or does structured off-axis change remain?",
    ]
    setup_lines = [
        "Five mean points",
        "a = mean(AAAAAA)",
        "b = mean(BBBBBB)",
        "d = mean(DDDDDD)",
        "x_bab = mean(BABABA)",
        "x_dad = mean(DADADA)",
        "",
        "Main metrics",
        "progress = intended-axis position",
        "selectivity = intended - cross",
        "residual = off-axis remainder",
    ]
    add_text_panel(ax_info, "A. Why Endpoint Movement", info_lines)
    add_text_panel(ax_setup, "Five-Point Setup", setup_lines)
    plot_geometry_panel(ax_geom, focus_row, geometry)
    plot_pair_metric(
        ax_sel,
        rows,
        focus_row.q_id,
        value_a="joint_selectivity_bab",
        value_b="joint_selectivity_dad",
        xlabel="Joint selectivity",
        title="C. Across-q summary: selectivity",
        include_zero=True,
        note="Joint coefficients are used because B/D axes are not orthogonal.",
    )
    ax_sel.legend(loc="lower right", frameon=False, fontsize=8.8)
    plot_pair_metric(
        ax_resid,
        rows,
        focus_row.q_id,
        value_a="joint_resid_bab",
        value_b="joint_resid_dad",
        xlabel="Joint residual",
        title="Across-q summary: residual structure",
        include_zero=True,
    )

    fig.text(
        0.5,
        0.02,
        "Endpoint shift is real and selective, but not fully one-axis clean, which motivates Stage 2: query-only reweighting.",
        ha="center",
        va="bottom",
        fontsize=11.2,
        color=GRAY,
    )
    fig.subplots_adjust(left=0.055, right=0.975, top=0.95, bottom=0.09)
    return fig


def build_geometry_figure(focus_row: MovementRow, geometry: GeometryData, dpi: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.2, 6.5), dpi=dpi)
    plot_geometry_panel(ax, focus_row, geometry)
    fig.tight_layout()
    return fig


def build_summary_figure(rows: Sequence[MovementRow], focus_q: str, dpi: int) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.4), dpi=dpi)
    plot_pair_metric(
        axes[0],
        rows,
        focus_q,
        value_a="joint_selectivity_bab",
        value_b="joint_selectivity_dad",
        xlabel="Joint selectivity",
        title="Across-q summary: selectivity",
        include_zero=True,
        note="Joint coefficients are used because B/D axes are not orthogonal.",
    )
    axes[0].legend(loc="lower right", frameon=False, fontsize=8.6)
    plot_pair_metric(
        axes[1],
        rows,
        focus_q,
        value_a="joint_resid_bab",
        value_b="joint_resid_dad",
        xlabel="Joint residual",
        title="Across-q summary: residual structure",
        include_zero=True,
    )
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, out_path: Optional[Path], dpi: int) -> Optional[Path]:
    if out_path is None:
        return None
    ensure_parent(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path


def save_panel_exports(
    panel_out_dir: Path,
    rows: Sequence[MovementRow],
    focus_row: MovementRow,
    geometry: GeometryData,
    dpi: int,
) -> List[Path]:
    panel_out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    fig_geom = build_geometry_figure(focus_row, geometry, dpi=dpi)
    geom_path = panel_out_dir / f"{focus_row.q_id.lower()}_endpoint_geometry.png"
    fig_geom.savefig(geom_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_geom)
    saved.append(geom_path)

    fig_summary = build_summary_figure(rows, focus_row.q_id, dpi=dpi)
    summary_path = panel_out_dir / "endpoint_allq_summary.png"
    fig_summary.savefig(summary_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_summary)
    saved.append(summary_path)
    return saved


def main() -> int:
    args = parse_args()
    if args.out_png is None and args.out_pdf is None:
        raise ValueError("At least one of --out_png or --out_pdf must be provided")

    rows = load_rows(Path(args.movement_csv))
    rows_by_qid = row_by_qid(rows)
    if args.focus_q not in rows_by_qid:
        raise KeyError(f"focus_q not found in movement CSV: {args.focus_q}")
    focus_row = rows_by_qid[args.focus_q]

    focus_points = load_focus_means(Path(args.means_npz), args.focus_q)
    geometry = compute_geometry(focus_row, focus_points)

    fig = build_composite_figure(rows, focus_row, geometry, dpi=args.dpi)
    saved_paths: List[Path] = []
    out_png = Path(args.out_png) if args.out_png else None
    out_pdf = Path(args.out_pdf) if args.out_pdf else None
    maybe_png = save_figure(fig, out_png, dpi=args.dpi)
    maybe_pdf = save_figure(fig, out_pdf, dpi=args.dpi)
    plt.close(fig)

    if maybe_png is not None:
        saved_paths.append(maybe_png)
    if maybe_pdf is not None:
        saved_paths.append(maybe_pdf)

    panel_paths: List[Path] = []
    if args.panel_out_dir:
        panel_paths = save_panel_exports(
            Path(args.panel_out_dir),
            rows=rows,
            focus_row=focus_row,
            geometry=geometry,
            dpi=args.dpi,
        )

    for path in saved_paths:
        print(f"saved={path}")
    for path in panel_paths:
        print(f"panel_saved={path}")
    print(f"focus_q={focus_row.q_id}")
    print(f"q_count={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
