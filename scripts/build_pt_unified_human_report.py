#!/usr/bin/env python3
"""Build a human-readable HTML report for unified PT outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.run_condition_common_pca import maybe_plot_scatter_3d_html


SHOT_ORDER = [0, 1, 3, 5, 7, 9]
FAMILY_ORDER = ["BASE_ABD", "CTX_ABD", "ZERO_CTRL", "A_ONLY"]
REGIME_ORDER = {
    "BASE_ABD": ["BASE_AB", "BASE_AD", "BASE_BD"],
    "CTX_ABD": ["CTX_ABABAB_B", "CTX_ADADAD_D", "CTX_BDBDBD_D"],
    "ZERO_CTRL": ["ZERO_A", "ZERO_B", "ZERO_D"],
    "A_ONLY": ["AAAA_A"],
}
REGIME_LABELS = {
    "BASE_AB": "BASE_AB",
    "BASE_AD": "BASE_AD",
    "BASE_BD": "BASE_BD",
    "CTX_ABABAB_B": "CTX_ABABAB_B",
    "CTX_ADADAD_D": "CTX_ADADAD_D",
    "CTX_BDBDBD_D": "CTX_BDBDBD_D",
    "ZERO_A": "ZERO_A",
    "ZERO_B": "ZERO_B",
    "ZERO_D": "ZERO_D",
    "AAAA_A": "AAAA_A",
}
EDGE_GROUP_CONFIG = {
    "AB": {
        "zero_regime": "ZERO_B",
        "base_regime": "BASE_AB",
        "ctx_regime": "CTX_ABABAB_B",
        "color": "#1f77b4",
    },
    "AD": {
        "zero_regime": "ZERO_D",
        "base_regime": "BASE_AD",
        "ctx_regime": "CTX_ADADAD_D",
        "color": "#d62728",
    },
    "BD": {
        "zero_regime": "ZERO_D",
        "base_regime": "BASE_BD",
        "ctx_regime": "CTX_BDBDBD_D",
        "color": "#2ca02c",
    },
}
ROLE_MAP_MD = Path("/home/sunsik/my_fv_project/docs/TRIANGLE_RELATION_ROLE_MAP.md")
ENDPOINT_MOVEMENT_CSV = Path(
    "/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_multi_root/movement_qwise.csv"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build human-readable unified PT HTML report")
    p.add_argument("--run_dir", required=True, help="Unified PT run directory")
    p.add_argument("--out_dir", default=None, help="Output report directory")
    p.add_argument("--sweep_csv", default=None, help="Override pt_unified_shot_sweep.csv")
    p.add_argument("--bootstrap_csv", default=None, help="Override pt_unified_bootstrap_summary.csv")
    p.add_argument("--baseline_abc_csv", default=None, help="Optional baseline pt_bootstrap_summary.csv for ABC overlay")
    p.add_argument("--eligibility_csv", default=None, help="Override pt_unified_family_eligibility.csv")
    p.add_argument("--topk_jsonl", default=None, help="Override pt_unified_edge_topk.jsonl")
    p.add_argument("--summary_csv", default=None, help="Optional output path for aggregated top-k summary CSV")
    p.add_argument("--dpi", type=int, default=180, help="Plot dpi")
    return p.parse_args()


def _qid_sort_key(qid: str) -> int:
    m = re.search(r"(\d+)", str(qid))
    return int(m.group(1)) if m else 1_000_000


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _humanize_pca_variant(name: str) -> str:
    mapping = {
        "AAA_ref_with_D": "5-condition PCA (AAA, BBB, BABA, DDD, DADA)",
        "AAA_ref": "3-condition PCA (AAA, BBB, BABA)",
        "union_ref": "Union-reference PCA",
    }
    return mapping.get(name, name)


def _default_baseline_abc_csv() -> Path | None:
    candidates = [
        Path("/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_20260310_104803/pt_bootstrap_summary.csv"),
        Path("/home/sunsik/my_fv_project/pt_analysis/llama31_70b_20260224_031506/pt_bootstrap_summary.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_baseline_abc_df(path: str | None) -> pd.DataFrame:
    if path:
        csv_path = Path(path)
    else:
        csv_path = _default_baseline_abc_csv()
    if csv_path is None or not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _load_endpoint_movement_df(path: str | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path else ENDPOINT_MOVEMENT_CSV
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _fmt_num(value) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "NA"


def _format_target_label(target_str: str, gold_target_str: str) -> str:
    target = str(target_str or "")
    gold = str(gold_target_str or "")
    if gold and gold != target:
        return f"{target} (gold: {gold})"
    return target


def _build_endpoint_movement_diagram_svg(progress: float, residual: float) -> str:
    x0 = 70.0
    x1 = 360.0
    y0 = 170.0
    x_mid = x0 + min(1.0, max(0.0, progress)) * (x1 - x0)
    y_mid = y0 - min(90.0, max(0.0, residual) * 145.0)
    total = float(np.sqrt(progress**2 + residual**2))
    return (
        '<svg viewBox="0 0 430 220" role="img" aria-label="Endpoint movement diagram" '
        'style="width:100%; max-width:520px; display:block; margin:0 auto;">'
        '<defs><marker id="endpoint-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#0f4c5c"/></marker></defs>'
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#2b2b2b" stroke-width="2"/>'
        f'<circle cx="{x0}" cy="{y0}" r="4" fill="#2b2b2b"/>'
        f'<circle cx="{x1}" cy="{y0}" r="4" fill="#2b2b2b"/>'
        f'<line x1="{x0}" y1="{y0}" x2="{x_mid}" y2="{y0}" stroke="#0f4c5c" stroke-width="3" marker-end="url(#endpoint-arrow)"/>'
        f'<line x1="{x_mid}" y1="{y0}" x2="{x_mid}" y2="{y_mid}" stroke="#7b6d5d" stroke-width="3" stroke-dasharray="6,4"/>'
        f'<line x1="{x0}" y1="{y0}" x2="{x_mid}" y2="{y_mid}" stroke="#c7522a" stroke-width="3"/>'
        f'<circle cx="{x_mid}" cy="{y_mid}" r="5" fill="#c7522a"/>'
        f'<text x="{x0-10}" y="{y0+22}" font-size="15" fill="#2b2b2b">A</text>'
        f'<text x="{x1-5}" y="{y0+22}" font-size="15" fill="#2b2b2b">B</text>'
        f'<text x="{x_mid+8}" y="{y_mid-8}" font-size="15" fill="#c7522a">x_bab</text>'
        '<text x="154" y="161" font-size="13" fill="#0f4c5c">progress</text>'
        f'<text x="{x_mid+10}" y="{(y0+y_mid)/2:.1f}" font-size="13" fill="#7b6d5d">residual</text>'
        f'<text x="230" y="126" font-size="13" fill="#c7522a">total shift ≈ {_fmt_num(total)}</text>'
        '</svg>'
    )


def _build_endpoint_movement_big_svg(row: pd.Series) -> str:
    bab_p = float(row["p_b_x_bab"])
    bab_r = float(row["r_b_x_bab"])
    dad_p = float(row["p_d_x_dad"])
    dad_r = float(row["r_d_x_dad"])
    bab_total = float(np.sqrt(bab_p**2 + bab_r**2))
    dad_total = float(np.sqrt(dad_p**2 + dad_r**2))

    def panel(x0: float, title: str, end_label: str, p: float, r: float, total: float, main_color: str, point_label: str) -> str:
        axis_len = 270.0
        y0 = 220.0
        x1 = x0 + axis_len
        x_mid = x0 + min(1.0, max(0.0, p)) * axis_len
        y_mid = y0 - min(95.0, max(0.0, r) * 155.0)
        return (
            f'<text x="{x0}" y="54" font-size="22" font-weight="700" fill="#222">{html.escape(title)}</text>'
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#2b2b2b" stroke-width="2.5"/>'
            f'<circle cx="{x0}" cy="{y0}" r="4.5" fill="#2b2b2b"/>'
            f'<circle cx="{x1}" cy="{y0}" r="4.5" fill="#2b2b2b"/>'
            f'<line x1="{x0}" y1="{y0}" x2="{x_mid}" y2="{y0}" stroke="{main_color}" stroke-width="4" marker-end="url(#metric-arrow)"/>'
            f'<line x1="{x_mid}" y1="{y0}" x2="{x_mid}" y2="{y_mid}" stroke="#7b6d5d" stroke-width="3" stroke-dasharray="7,5"/>'
            f'<line x1="{x0}" y1="{y0}" x2="{x_mid}" y2="{y_mid}" stroke="#c7522a" stroke-width="4"/>'
            f'<circle cx="{x_mid}" cy="{y_mid}" r="6" fill="#c7522a"/>'
            f'<text x="{x0-8}" y="{y0+26}" font-size="16" fill="#2b2b2b">A</text>'
            f'<text x="{x1-2}" y="{y0+26}" font-size="16" fill="#2b2b2b">{html.escape(end_label)}</text>'
            f'<text x="{x_mid+10}" y="{y_mid-10}" font-size="16" fill="#c7522a">{html.escape(point_label)}</text>'
            f'<text x="{x0+85}" y="{y0-10}" font-size="14" fill="{main_color}">progress = {_fmt_num(p)}</text>'
            f'<text x="{x_mid+12}" y="{(y0+y_mid)/2:.1f}" font-size="14" fill="#7b6d5d">residual = {_fmt_num(r)}</text>'
            f'<text x="{x0+108}" y="95" font-size="14" fill="#c7522a">total shift ≈ {_fmt_num(total)}</text>'
        )

    return "".join(
        [
            '<svg viewBox="0 0 860 300" role="img" aria-label="Endpoint movement metric schematic" '
            'style="width:100%; max-width:1040px; display:block; margin:0 auto;">',
            '<defs><marker id="metric-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#0f4c5c"/></marker></defs>',
            '<rect x="10" y="10" width="840" height="280" rx="14" ry="14" fill="#fffdfa" stroke="#ddd2c4"/>',
            '<text x="28" y="30" font-size="15" fill="#5b5348">Metric-based schematic, not PCA: the geometry is drawn from the measured progress and residual values.</text>',
            panel(70.0, "AAA → BABABA vs BBBBBB", "B", bab_p, bab_r, bab_total, "#0f4c5c", "x_bab"),
            panel(470.0, "AAA → DADADA vs DDDDDD", "D", dad_p, dad_r, dad_total, "#6a4c93", "x_dad"),
            '</svg>',
        ]
    )


def _build_endpoint_movement_card(qid: str, endpoint_df: pd.DataFrame) -> str:
    if endpoint_df.empty:
        return (
            '<div class="card"><h2>Joint Main</h2>'
            "<div class='small'>No endpoint movement summary available.</div></div>"
        )
    qdf = endpoint_df[endpoint_df["q_id"] == qid]
    if qdf.empty:
        return (
            '<div class="card"><h2>Joint Main</h2>'
            f"<div class='small'>No endpoint movement row found for {html.escape(qid)}.</div></div>"
        )
    row = qdf.iloc[0]
    return (
        '<div class="card">'
        '<h3>Joint Main</h3>'
        '<div class="small">We treat the mixed shift as a joint combination of the clean A→B and A→D directions.</div>'
        '<div class="small" style="margin-top:8px;"><code>x-a ≈ α(b-a) + β(d-a) + ε</code></div>'
        '<div class="small" style="margin-top:8px;">α: clean intended component</div>'
        '<div class="small">β: competing component</div>'
        '<div class="small">ε: off-axis remainder</div>'
        '</div>'
    )


def _build_endpoint_movement_section(
    endpoint_df: pd.DataFrame,
    qid: str = "Q1",
    figure_src: str | None = None,
) -> str:
    if endpoint_df.empty:
        return (
            '<section id="stage-1-endpoint" class="goals-box">'
            '<div class="goal-panel"><h2>Stage 1. Endpoint Movement</h2>'
            "<p>No endpoint movement summary available.</p></div></section>"
        )
    qdf = endpoint_df[endpoint_df["q_id"] == qid]
    if qdf.empty:
        return (
            '<section id="stage-1-endpoint" class="goals-box">'
            f'<div class="goal-panel"><h2>Stage 1. Endpoint Movement</h2><p>No endpoint movement row found for {html.escape(qid)}.</p></div></section>'
        )
    row = qdf.iloc[0]
    raw_progress = float(row["p_b_x_bab"])
    raw_cross = float(row["p_d_x_bab"])
    raw_resid = float(row["r_b_x_bab"])
    raw_angle_deg = float(np.degrees(np.arctan2(raw_resid, raw_progress))) if raw_progress > 0 else float("nan")
    return (
        '<section id="stage-1-endpoint" class="goals-box">'
        '<div class="goal-panel" style="grid-column: 1 / -1;">'
        '<h2>Stage 1. Endpoint Movement</h2>'
        '<p>Question: does the mixed condition move the final query state toward the intended endpoint, and is that movement one-dimensional?</p>'
        '<p>We use joint decomposition as the main readout and treat raw progress / cross / residual as supporting intuition only.</p>'
        '</div>'
        + (
            '<div class="card" style="grid-column: 1 / -1;"><h2>Q1 Metric Schematic</h2>'
            + _build_endpoint_movement_big_svg(row)
            + '<div class="small" style="margin-top:8px;">First show the measured endpoint movement itself: how far the mixed state moves toward the clean endpoint, and how much off-axis change remains.</div></div>'
        )
        + _build_endpoint_movement_card(qid, endpoint_df)
        + (
            '<div class="card">'
            '<h2>How We Measured It</h2>'
            '<div class="small">Raw progress projects the mixed state onto the clean A→B axis. Residual measures the remaining off-axis component.</div>'
            '<div class="small" style="margin-top:8px;"><code>p_B(x) = ((x-a)·(b-a)) / ||b-a||²</code></div>'
            '<div class="small"><code>r_B(x) = ||(x-a)-p_B(x)(b-a)|| / ||b-a||</code></div>'
            f'<div class="small">BABABA → B progress: {_fmt_num(raw_progress)}</div>'
            f'<div class="small">BABABA → D raw cross: {_fmt_num(raw_cross)}</div>'
            f'<div class="small">Raw residual: {_fmt_num(raw_resid)}</div>'
            f'<div class="small">Off-axis angle: {_fmt_num(raw_angle_deg)}°</div>'
            '</div>'
        )
        + (
            '<div class="card">'
            '<h2>Q1 Result</h2>'
            f'<div class="small"><strong>BABABA</strong>: α={_fmt_num(row["alpha_bab"])}, β={_fmt_num(row["beta_bab"])}, joint residual={_fmt_num(row["joint_resid_bab"])}, selectivity={_fmt_num(row["joint_selectivity_bab"])}</div>'
            f'<div class="small" style="margin-top:8px;"><strong>DADADA</strong>: α={_fmt_num(row["alpha_dad"])}, β={_fmt_num(row["beta_dad"])}, joint residual={_fmt_num(row["joint_resid_dad"])}, selectivity={_fmt_num(row["joint_selectivity_dad"])}</div>'
            '<div class="small" style="margin-top:10px;">Takeaway: endpoint directionality is real, but a one-axis account is insufficient.</div>'
            '</div>'
        )
        + (
            '<div class="card">'
            '<h2>Raw Geometry</h2>'
            + _build_endpoint_movement_diagram_svg(raw_progress, raw_resid)
            + '<div class="small" style="margin-top:8px;">The mixed state moves toward B, but not along a single straight axis.</div>'
            '</div>'
        )
        + '</section>'
    )


def _load_role_map(md_path: Path = ROLE_MAP_MD) -> Dict[str, Dict[str, str]]:
    if not md_path.exists():
        return {}
    text = md_path.read_text(encoding="utf-8")
    out: Dict[str, Dict[str, str]] = {}
    current_qid: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            current_qid = line[3:].strip()
            out.setdefault(current_qid, {})
            continue
        if current_qid is None or not line.startswith("- `"):
            continue
        match = re.match(r"- `([ABD]):\s*(.+?)`?$", line)
        if not match:
            continue
        role_key = match.group(1)
        role_text = match.group(2).rstrip("`").strip()
        out[current_qid][role_key] = role_text
    return out


def _role_line_for_edge(role_map: Dict[str, Dict[str, str]], qid: str, edge_group: str) -> str:
    roles = role_map.get(qid, {})
    if edge_group == "A_ONLY":
        a_role = roles.get("A", "")
        return f"<div class='small'>A: {html.escape(a_role)}</div>" if a_role else ""
    if edge_group == "AB":
        a_role = roles.get("A", "")
        b_role = roles.get("B", "")
        parts = []
        if a_role:
            parts.append(f"A: {html.escape(a_role)}")
        if b_role:
            parts.append(f"B: {html.escape(b_role)}")
        return f"<div class='small'>{' | '.join(parts)}</div>" if parts else ""
    if edge_group == "AD":
        a_role = roles.get("A", "")
        d_role = roles.get("D", "")
        parts = []
        if a_role:
            parts.append(f"A: {html.escape(a_role)}")
        if d_role:
            parts.append(f"D: {html.escape(d_role)}")
        return f"<div class='small'>{' | '.join(parts)}</div>" if parts else ""
    if edge_group == "BD":
        b_role = roles.get("B", "")
        d_role = roles.get("D", "")
        parts = []
        if b_role:
            parts.append(f"B: {html.escape(b_role)}")
        if d_role:
            parts.append(f"D: {html.escape(d_role)}")
        return f"<div class='small'>{' | '.join(parts)}</div>" if parts else ""
    return ""


def _default_pca_roots() -> List[Path]:
    roots = [
        Path("/home/sunsik/my_fv_project/results_fv/relation_condition_qwise"),
        Path("/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise"),
    ]
    return [root for root in roots if root.exists()]


def _ensure_pca_3d_html_from_points(pca_dir: Path) -> bool:
    html_path = pca_dir / "scatter_3d_interactive.html"
    if html_path.exists():
        return True
    points_path = pca_dir / "pca_points.csv"
    if not points_path.exists():
        return False

    pts: List[List[float]] = []
    labels: List[str] = []
    trial_ids: List[str] = []
    with points_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"condition", "trial_id", "pc1", "pc2", "pc3"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            return False
        for row in reader:
            pts.append([float(row["pc1"]), float(row["pc2"]), float(row["pc3"])])
            labels.append(str(row["condition"]))
            trial_ids.append(str(row["trial_id"]))
    if not pts:
        return False

    model_meta_path = pca_dir / "pca_model_meta.json"
    plot_title = _humanize_pca_variant(pca_dir.name)
    if model_meta_path.exists():
        try:
            payload = json.loads(model_meta_path.read_text(encoding="utf-8"))
            out_subdir = str(payload.get("out_subdir", pca_dir.name))
            plot_title = _humanize_pca_variant(out_subdir)
        except Exception:
            pass

    return bool(
        maybe_plot_scatter_3d_html(
            out_path=html_path,
            points=np.array(pts, dtype=np.float32),
            labels=labels,
            trial_ids=trial_ids,
            title=plot_title,
        )
    )


def _discover_pca_3d_html(qid: str) -> List[Dict[str, str]]:
    found: Dict[Tuple[str, str], Dict[str, str]] = {}
    for root in _default_pca_roots():
        pattern = f"*/{qid}/_pca_common/*"
        for pca_dir in sorted(root.glob(pattern)):
            if not pca_dir.is_dir():
                continue
            html_path = pca_dir / "scatter_3d_interactive.html"
            if not html_path.exists():
                _ensure_pca_3d_html_from_points(pca_dir)
            if not html_path.exists():
                continue
            relation_name = pca_dir.parents[2].name
            subdir = pca_dir.name
            key = (relation_name, subdir)
            current = {
                "relation_name": relation_name,
                "subdir": subdir,
                "source_path": str(html_path),
                "source_root": str(root),
            }
            # Prefer home copy over scratch copy when both exist.
            if key not in found or str(root).startswith("/home/"):
                found[key] = current
    def sort_key(key: Tuple[str, str]) -> Tuple[int, str, str]:
        relation_name, subdir = key
        priority = {"AAA_ref_with_D": 0, "AAA_ref": 1, "union_ref": 2}.get(subdir, 99)
        return (priority, relation_name, subdir)

    return [found[key] for key in sorted(found.keys(), key=sort_key)]


def _discover_pca_centroids_csv(qid: str) -> Path | None:
    preferred = ["AAA_ref_with_D", "AAA_ref", "union_ref"]
    found: Dict[str, Path] = {}
    for root in _default_pca_roots():
        pattern = f"*/{qid}/_pca_common/*"
        for pca_dir in sorted(root.glob(pattern)):
            if not pca_dir.is_dir():
                continue
            centroids = pca_dir / "pca_centroids.csv"
            points = pca_dir / "pca_points.csv"
            if centroids.exists() and points.exists():
                key = pca_dir.name
                if key not in found or str(root).startswith("/home/"):
                    found[key] = centroids
    for key in preferred:
        if key in found:
            return found[key]
    return None


def _plot_endpoint_stage1(
    qid: str,
    out_path: str,
    dpi: int,
    *,
    endpoint_row: pd.Series | None = None,
) -> bool:
    centroids_csv = _discover_pca_centroids_csv(qid)
    if centroids_csv is None or not centroids_csv.exists():
        return False
    centroids: Dict[str, Tuple[float, float, float]] = {}
    with centroids_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"condition", "pc1", "pc2", "pc3"}.issubset(set(reader.fieldnames)):
            return False
        for row in reader:
            centroids[str(row["condition"])] = (
                float(row["pc1"]),
                float(row["pc2"]),
                float(row["pc3"]),
            )
    needed = {"AAA", "BBB", "BABA", "DDD", "DADA"}
    if not needed.issubset(set(centroids.keys())):
        return False

    fig = plt.figure(figsize=(9.6, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    colors = {
        "AAA": "#2f3e46",
        "BBB": "#0f4c5c",
        "BABA": "#c7522a",
        "DDD": "#6a4c93",
        "DADA": "#8a5a44",
    }

    for cond, (x, y, z) in centroids.items():
        ax.scatter(
            x,
            y,
            z,
            s=130 if cond == "AAA" else 115,
            color=colors.get(cond, "#555"),
            depthshade=False,
        )
        ax.text(x + 0.06, y + 0.06, z + 0.06, cond, fontsize=10, color=colors.get(cond, "#555"))

    a = np.array(centroids["AAA"], dtype=np.float32)
    b = np.array(centroids["BBB"], dtype=np.float32)
    bab = np.array(centroids["BABA"], dtype=np.float32)
    d = np.array(centroids["DDD"], dtype=np.float32)
    dad = np.array(centroids["DADA"], dtype=np.float32)

    def arrow(src, dst, color, style="-", width=2.2):
        delta = dst - src
        ax.quiver(
            float(src[0]),
            float(src[1]),
            float(src[2]),
            float(delta[0]),
            float(delta[1]),
            float(delta[2]),
            color=color,
            linewidth=width,
            linestyle=style,
            arrow_length_ratio=0.08,
        )

    arrow(a, b, colors["BBB"], style="--", width=2.0)
    arrow(a, bab, colors["BABA"], style="-", width=2.8)
    arrow(a, d, colors["DDD"], style="--", width=2.0)
    arrow(a, dad, colors["DADA"], style="-", width=2.8)

    ax.set_title("Q1 Common PCA: clean endpoints and mixed shifts", fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.grid(True, alpha=0.18)
    try:
        ax.view_init(elev=20, azim=-58)
    except Exception:
        pass

    if endpoint_row is not None:
        bab_progress = _fmt_num(endpoint_row.get("p_b_x_bab"))
        bab_resid = _fmt_num(endpoint_row.get("r_b_x_bab"))
        bab_total = _fmt_num(
            float(np.sqrt(float(endpoint_row["p_b_x_bab"]) ** 2 + float(endpoint_row["r_b_x_bab"]) ** 2))
        )
        dad_progress = _fmt_num(endpoint_row.get("p_d_x_dad"))
        dad_resid = _fmt_num(endpoint_row.get("r_d_x_dad"))
        dad_total = _fmt_num(
            float(np.sqrt(float(endpoint_row["p_d_x_dad"]) ** 2 + float(endpoint_row["r_d_x_dad"]) ** 2))
        )
        fig.text(
            0.02,
            0.95,
            "Q1 joint intuition\n"
            f"BABABA: progress={bab_progress}, residual={bab_resid}, total≈{bab_total}\n"
            f"DADADA: progress={dad_progress}, residual={dad_resid}, total≈{dad_total}",
            ha="left",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffdfa", edgecolor="#d9d2c3"),
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _copy_pca_links_for_q(qid: str, report_dir: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    pca_dir = Path(report_dir) / "pca_html" / qid
    pca_dir.mkdir(parents=True, exist_ok=True)
    for item in _discover_pca_3d_html(qid):
        src = Path(item["source_path"])
        dst_name = f"{item['relation_name']}__{item['subdir']}__scatter_3d_interactive.html"
        dst = pca_dir / dst_name
        shutil.copy2(src, dst)
        copied = dict(item)
        copied["report_rel_path"] = str(Path("pca_html") / qid / dst_name)
        out.append(copied)
    return out


def _build_pca_card(qid: str, report_dir: str) -> str:
    pca_items = _copy_pca_links_for_q(qid, report_dir)
    pieces = ['<div class="card"><h2>PCA 3D HTML</h2>']
    if not pca_items:
        pieces.append("<div class='small'>No PCA 3D HTML found for this q.</div></div>")
        return "".join(pieces)
    pieces.append("<ul>")
    for item in pca_items:
        label = _humanize_pca_variant(item["subdir"])
        href = html.escape(item["report_rel_path"])
        source_path = html.escape(item["source_path"])
        pieces.append(
            "<li>"
            f"<a href=\"{href}\">{html.escape(label)}</a>"
            f"<div class='small'>{source_path}</div>"
            "</li>"
        )
    pieces.append("</ul></div>")
    return "".join(pieces)


def _build_pca_iframe_card(qid: str, report_dir: str) -> str:
    pca_items = _copy_pca_links_for_q(qid, report_dir)
    pieces = ['<div class="card"><h2>PCA 3D</h2>']
    if not pca_items:
        pieces.append("<div class='small'>No PCA 3D HTML found for this q.</div></div>")
        return "".join(pieces)
    default_src = pca_items[0]["report_rel_path"]
    label = _humanize_pca_variant(pca_items[0]["subdir"])
    pieces.append(
        "<div class='small explain' style='font-size: 16px; font-weight: 400;'>"
        "Using the top 20 heads, we build a single vector for each trial. "
        "We then fit one common PCA on trial vectors pooled across all conditions and project everything into the same PCA space. "
        "Each dot is one trial vector."
        "</div>"
    )
    pieces.append(f"<div class='small' style='margin-bottom: 10px;'>{html.escape(label)}</div>")
    pieces.append(
        f'<iframe class="pca-frame" src="{html.escape(default_src)}" loading="lazy"></iframe>'
    )
    pieces.append("</div>")
    return "".join(pieces)


def _aggregate_topk(topk_df: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    if topk_df.empty:
        out_df = pd.DataFrame(
            columns=[
                "q_id",
                "shot",
                "family_id",
                "regime_id",
                "query_input",
                "target_str",
                "candidate_canonical",
                "display_candidate",
                "mean_logprob",
                "mean_prob",
                "mean_rank_within_row",
                "occurrence_count",
                "trial_coverage_count",
                "trial_coverage_frac",
                "top1_count",
                "n_trials",
                "summary_rank",
            ]
        )
        out_df.to_csv(out_csv, index=False)
        return out_df

    trial_counts: Dict[Tuple[str, int, str, str], set] = defaultdict(set)
    by_key: Dict[Tuple[str, int, str, str, str], Dict[str, object]] = {}
    for row in topk_df.to_dict(orient="records"):
        q_id = str(row["q_id"])
        shot = int(row["shot"])
        family_id = str(row["family_id"])
        regime_id = str(row["regime_id"])
        trial_index = int(row["trial_index"])
        trial_counts[(q_id, shot, family_id, regime_id)].add(trial_index)

        candidates = list(row.get("lexical_candidates", []))
        canonicals = list(row.get("lexical_candidate_canonical_forms", []))
        logprobs = list(row.get("lexical_candidate_logprobs", []))
        probs = list(row.get("lexical_candidate_probs", []))
        for rank_idx, (cand, canon, lp, prob) in enumerate(zip(candidates, canonicals, logprobs, probs), start=1):
            key = (q_id, shot, family_id, regime_id, canon)
            stat = by_key.get(key)
            if stat is None:
                stat = {
                    "q_id": q_id,
                    "shot": shot,
                    "family_id": family_id,
                    "regime_id": regime_id,
                    "query_input": str(row.get("query_input", "")),
                    "target_str": str(row.get("target_str", "")),
                    "gold_target_str": str(row.get("gold_target_str", "")),
                    "candidate_canonical": canon,
                    "surface_counts": defaultdict(int),
                    "count": 0,
                    "sum_logprob": 0.0,
                    "sum_prob": 0.0,
                    "sum_rank": 0.0,
                    "top1_count": 0,
                    "trial_indices": set(),
                }
                by_key[key] = stat
            stat["surface_counts"][cand] += 1
            stat["count"] += 1
            stat["sum_logprob"] += float(lp)
            stat["sum_prob"] += float(prob)
            stat["sum_rank"] += float(rank_idx)
            stat["trial_indices"].add(trial_index)
            if rank_idx == 1:
                stat["top1_count"] += 1

    grouped: Dict[Tuple[str, int, str, str], List[Dict[str, object]]] = defaultdict(list)
    for stat in by_key.values():
        q_id = stat["q_id"]
        shot = stat["shot"]
        family_id = stat["family_id"]
        regime_id = stat["regime_id"]
        n_trials = len(trial_counts[(q_id, shot, family_id, regime_id)])
        count = int(stat["count"])
        display_candidate = sorted(stat["surface_counts"].items(), key=lambda x: (-x[1], x[0]))[0][0]
        grouped[(q_id, shot, family_id, regime_id)].append(
            {
                "q_id": q_id,
                "shot": shot,
                "family_id": family_id,
                "regime_id": regime_id,
                "query_input": stat["query_input"],
                "target_str": stat["target_str"],
                "gold_target_str": stat["gold_target_str"],
                "candidate_canonical": stat["candidate_canonical"],
                "display_candidate": display_candidate,
                "mean_logprob": float(stat["sum_logprob"]) / count,
                "mean_prob": float(stat["sum_prob"]) / count,
                "mean_rank_within_row": float(stat["sum_rank"]) / count,
                "occurrence_count": count,
                "trial_coverage_count": len(stat["trial_indices"]),
                "trial_coverage_frac": (len(stat["trial_indices"]) / n_trials) if n_trials else 0.0,
                "top1_count": int(stat["top1_count"]),
                "n_trials": n_trials,
            }
        )

    final_rows: List[Dict[str, object]] = []
    for _, entries in grouped.items():
        entries.sort(
            key=lambda r: (
                -float(r["mean_logprob"]),
                -int(r["occurrence_count"]),
                -float(r["mean_prob"]),
                str(r["display_candidate"]),
            )
        )
        for summary_rank, entry in enumerate(entries[:20], start=1):
            row = dict(entry)
            row["summary_rank"] = summary_rank
            final_rows.append(row)

    out_df = pd.DataFrame(final_rows).sort_values(["q_id", "family_id", "regime_id", "shot", "summary_rank"]).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)
    return out_df


def _plot_family_prob(sweep_df: pd.DataFrame, qid: str, family_id: str, out_path: str, dpi: int) -> bool:
    family_df = (
        sweep_df[(sweep_df["q_id"] == qid) & (sweep_df["family_id"] == family_id)]
        .groupby(["regime_id", "shot"], as_index=False)["target_prob_raw"]
        .mean()
    )
    if family_df.empty:
        return False
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    style_cycle = {
        "BASE_AB": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "BASE_AD": {"color": "#d62728", "linestyle": "--", "marker": "s"},
        "BASE_BD": {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
        "CTX_ABABAB_B": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "CTX_ADADAD_D": {"color": "#d62728", "linestyle": "--", "marker": "s"},
        "CTX_BDBDBD_D": {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
    }
    local_y: List[float] = []
    for regime_id in REGIME_ORDER[family_id]:
        sdf = family_df[family_df["regime_id"] == regime_id].sort_values("shot")
        if sdf.empty:
            continue
        xs = sdf["shot"].tolist()
        ys = sdf["target_prob_raw"].tolist()
        local_y.extend(ys)
        ax.plot(xs, ys, linewidth=2.1, markersize=5.0, label=REGIME_LABELS[regime_id], **style_cycle[regime_id])
    if local_y:
        y_min = min(local_y)
        y_max = max(local_y)
        pad = max((y_max - y_min) * 0.15, 0.01) if y_max != y_min else max(abs(y_min) * 0.08, 0.01)
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel("Shot")
    ax.set_ylabel("Mean target probability")
    ax.set_title(f"{qid} {family_id}")
    ax.grid(alpha=0.28, linewidth=0.6)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_control(sweep_df: pd.DataFrame, qid: str, out_path: str, dpi: int) -> bool:
    qdf = sweep_df[sweep_df["q_id"] == qid]
    zero_df = qdf[qdf["family_id"] == "ZERO_CTRL"].copy()
    a_df = qdf[qdf["family_id"] == "A_ONLY"].copy()
    if zero_df.empty and a_df.empty:
        return False
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    if not a_df.empty:
        a_line = (
            a_df.groupby("shot", as_index=False)["target_prob_raw"]
            .mean()
            .sort_values("shot")
        )
        ax.plot(
            a_line["shot"],
            a_line["target_prob_raw"],
            color="#7a3fb0",
            linestyle="-",
            marker="o",
            linewidth=2.1,
            markersize=5.0,
            label="AAAA_A",
        )
    zero_styles = {
        "ZERO_A": {"color": "#333333", "marker": "o"},
        "ZERO_B": {"color": "#8c564b", "marker": "s"},
        "ZERO_D": {"color": "#bcbd22", "marker": "^"},
    }
    for regime_id, style in zero_styles.items():
        sdf = zero_df[zero_df["regime_id"] == regime_id]
        if sdf.empty:
            continue
        y_val = float(sdf["target_prob_raw"].mean())
        ax.scatter([0], [y_val], s=55, label=regime_id, **style)
        ax.annotate(regime_id, (0, y_val), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("Shot")
    ax.set_ylabel("Mean target probability")
    ax.set_title(f"{qid} controls")
    ax.grid(alpha=0.28, linewidth=0.6)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_pt(boot_df: pd.DataFrame, abc_df: pd.DataFrame, qid: str, out_path: str, dpi: int) -> bool:
    qdf = boot_df[boot_df["q_id"] == qid].sort_values("shot")
    abc_qdf = abc_df[abc_df["q_id"] == qid].sort_values("shot") if not abc_df.empty else pd.DataFrame()
    if qdf.empty and abc_qdf.empty:
        return False
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    if not abc_qdf.empty and abc_qdf["pt_abc_mean"].notna().any():
        abc = abc_qdf[abc_qdf["pt_abc_mean"].notna()]
        ax.plot(abc["shot"], abc["pt_abc_mean"], color="#2b8a3e", marker="^", linewidth=2.0, label="ABC")
    if qdf["base_abd_mean"].notna().any():
        base = qdf[qdf["base_abd_mean"].notna()]
        ax.plot(base["shot"], base["base_abd_mean"], color="#4c78a8", marker="o", linewidth=2.0, label="ABD")
    if qdf["ctx_abd_mean"].notna().any():
        ctx = qdf[qdf["ctx_abd_mean"].notna()]
        ax.plot(ctx["shot"], ctx["ctx_abd_mean"], color="#f58518", marker="s", linewidth=2.0, linestyle="--", label="Mixed ABD")
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Shot")
    ax.set_ylabel("PT")
    ax.set_title(f"{qid} PT summary")
    ax.grid(alpha=0.28, linewidth=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def _plot_across_q_pt_median(
    boot_df: pd.DataFrame,
    abc_df: pd.DataFrame,
    qids: List[str],
    out_path: str,
    dpi: int,
) -> bool:
    qid_set = {str(qid) for qid in qids}
    boot_use = boot_df[boot_df["q_id"].astype(str).isin(qid_set)].copy()
    abc_use = abc_df[abc_df["q_id"].astype(str).isin(qid_set)].copy() if not abc_df.empty else pd.DataFrame()
    if boot_use.empty and abc_use.empty:
        return False

    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    line_count = 0

    if not abc_use.empty and "pt_abc_mean" in abc_use.columns:
        abc = (
            abc_use[abc_use["pt_abc_mean"].notna()]
            .groupby("shot", as_index=False)["pt_abc_mean"]
            .median()
            .sort_values("shot")
        )
        if not abc.empty:
            ax.plot(abc["shot"], abc["pt_abc_mean"], color="#2b8a3e", marker="^", linewidth=2.0, label="ABC median")
            line_count += 1

    if "base_abd_mean" in boot_use.columns:
        base = (
            boot_use[boot_use["base_abd_mean"].notna()]
            .groupby("shot", as_index=False)["base_abd_mean"]
            .median()
            .sort_values("shot")
        )
        if not base.empty:
            ax.plot(base["shot"], base["base_abd_mean"], color="#4c78a8", marker="o", linewidth=2.0, label="ABD median")
            line_count += 1

    if "ctx_abd_mean" in boot_use.columns:
        ctx = (
            boot_use[boot_use["ctx_abd_mean"].notna()]
            .groupby("shot", as_index=False)["ctx_abd_mean"]
            .median()
            .sort_values("shot")
        )
        if not ctx.empty:
            ax.plot(ctx["shot"], ctx["ctx_abd_mean"], color="#f58518", marker="s", linewidth=2.0, linestyle="--", label="Mixed ABD median")
            line_count += 1

    if line_count == 0:
        plt.close(fig)
        return False

    shot_values = []
    if "shot" in boot_use.columns:
        shot_values.extend(boot_use["shot"].dropna().tolist())
    if not abc_use.empty and "shot" in abc_use.columns:
        shot_values.extend(abc_use["shot"].dropna().tolist())
    if shot_values:
        ax.set_xticks(sorted({int(shot) for shot in shot_values}))
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Shot")
    ax.set_ylabel("PT")
    ax.set_title("Across-Q PT median summary")
    ax.grid(alpha=0.28, linewidth=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_edge_group_compare(sweep_df: pd.DataFrame, qid: str, out_prefix: str, dpi: int) -> Dict[str, str]:
    qdf = sweep_df[sweep_df["q_id"] == qid].copy()
    if qdf.empty:
        return {}
    all_y: List[float] = []
    prepared: Dict[str, Dict[str, object]] = {}
    for edge_group in ["AB", "AD", "BD"]:
        cfg = EDGE_GROUP_CONFIG[edge_group]
        color = cfg["color"]
        base_df = (
            qdf[qdf["regime_id"] == cfg["base_regime"]]
            .groupby("shot", as_index=False)["target_prob_raw"]
            .mean()
            .sort_values("shot")
        )
        ctx_df = (
            qdf[qdf["regime_id"] == cfg["ctx_regime"]]
            .groupby("shot", as_index=False)["target_prob_raw"]
            .mean()
            .sort_values("shot")
        )
        zero_df = (
            qdf[qdf["regime_id"] == cfg["zero_regime"]]
            .groupby("shot", as_index=False)["target_prob_raw"]
            .mean()
            .sort_values("shot")
        )
        local_y: List[float] = []
        payload = {
            "base_df": base_df,
            "ctx_df": ctx_df,
            "zero_df": zero_df,
            "color": color,
        }
        if not base_df.empty:
            xs = [int(x) for x in base_df["shot"].tolist()]
            ys = [float(y) for y in base_df["target_prob_raw"].tolist()]
            local_y.extend(ys)
            payload["base_xs"] = xs
            payload["base_ys"] = ys
        if not ctx_df.empty:
            xs = [int(x) for x in ctx_df["shot"].tolist()]
            ys = [float(y) for y in ctx_df["target_prob_raw"].tolist()]
            local_y.extend(ys)
            payload["ctx_xs"] = xs
            payload["ctx_ys"] = ys
        if not zero_df.empty:
            zero_row = zero_df[zero_df["shot"] == 0]
            if not zero_row.empty:
                y_val = float(zero_row["target_prob_raw"].iloc[0])
                local_y.append(y_val)
                payload["zero_y"] = y_val
        all_y.extend(local_y)
        prepared[edge_group] = payload
    if not all_y:
        return {}
    y_min = min(all_y)
    y_max = max(all_y)
    pad = max((y_max - y_min) * 0.15, 0.01) if y_max != y_min else max(abs(y_min) * 0.08, 0.01)
    outputs: Dict[str, str] = {}
    for edge_group in ["AB", "AD", "BD"]:
        payload = prepared.get(edge_group)
        if not payload:
            continue
        fig, ax = plt.subplots(figsize=(6.1, 4.6))
        color = payload["color"]
        if "base_xs" in payload:
            ax.plot(payload["base_xs"], payload["base_ys"], color=color, linestyle="-", marker="o", linewidth=2.2, markersize=5.0, label="baseline")
        if "ctx_xs" in payload:
            ax.plot(payload["ctx_xs"], payload["ctx_ys"], color=color, linestyle="--", marker="s", linewidth=2.2, markersize=5.0, label="mixed")
        if "zero_y" in payload:
            ax.scatter([0], [payload["zero_y"]], color=color, marker="X", s=68, label="zero-shot")
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_title(f"{qid} {edge_group}", fontsize=12)
        ax.set_xlabel("Shot")
        ax.set_ylabel("Mean target probability")
        ax.set_xticks(SHOT_ORDER)
        ax.grid(alpha=0.28, linewidth=0.6)
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()
        out_path = f"{out_prefix}_{edge_group}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        outputs[edge_group] = out_path
    return outputs


def _build_meta(summary_df: pd.DataFrame, sweep_df: pd.DataFrame, qid: str) -> Dict[str, Tuple[str, str]]:
    meta: Dict[str, Tuple[str, str]] = {}
    q_sweep = sweep_df[sweep_df["q_id"] == qid]
    for regime_id in sorted(q_sweep["regime_id"].dropna().unique().tolist()):
        sdf = q_sweep[q_sweep["regime_id"] == regime_id]
        row = sdf.iloc[0]
        meta[regime_id] = (
            str(row["query_input"]),
            _format_target_label(str(row["target_str"]), str(row.get("gold_target_str", ""))),
        )
    q_summary = summary_df[summary_df["q_id"] == qid]
    if not q_summary.empty:
        for regime_id in sorted(q_summary["regime_id"].dropna().unique().tolist()):
            if regime_id in meta:
                continue
            sdf = q_summary[q_summary["regime_id"] == regime_id]
            row = sdf.iloc[0]
            meta[regime_id] = (
                str(row["query_input"]),
                _format_target_label(str(row["target_str"]), str(row.get("gold_target_str", ""))),
            )
    return meta


def _build_candidate_table(summary_df: pd.DataFrame, qid: str, regime_id: str) -> str:
    qdf = summary_df[(summary_df["q_id"] == qid) & (summary_df["regime_id"] == regime_id)].copy()
    if qdf.empty:
        return "<p>No candidate summary available.</p>"
    pieces = [
        '<table class="cand-table">',
        "<thead><tr><th>Rank</th>" + "".join(f"<th>Shot {shot}</th>" for shot in SHOT_ORDER) + "</tr></thead>",
        "<tbody>",
    ]
    for rank in range(1, 21):
        pieces.append(f"<tr><td class='rank'>{rank}</td>")
        for shot in SHOT_ORDER:
            sdf = qdf[(qdf["shot"] == shot) & (qdf["summary_rank"] == rank)]
            if sdf.empty:
                pieces.append("<td></td>")
                continue
            row = sdf.iloc[0]
            candidate = html.escape(str(row["display_candidate"]))
            mean_lp = float(row["mean_logprob"])
            coverage = float(row["trial_coverage_frac"])
            top1_count = int(row["top1_count"])
            pieces.append(
                "<td>"
                f"<div class='cand'>{candidate}</div>"
                f"<div class='meta'>lp={mean_lp:.3f}</div>"
                f"<div class='meta'>cov={coverage:.2f} top1={top1_count}</div>"
                "</td>"
            )
        pieces.append("</tr>")
    pieces.append("</tbody></table>")
    return "".join(pieces)


def _format_candidate_cell(summary_df: pd.DataFrame, qid: str, regime_id: str, shot: int, top_n: int = 5) -> str:
    sdf = summary_df[
        (summary_df["q_id"] == qid)
        & (summary_df["regime_id"] == regime_id)
        & (summary_df["shot"] == shot)
    ].sort_values("summary_rank")
    if sdf.empty:
        return "<div class='small'>-</div>"
    pieces: List[str] = []
    for _, row in sdf.head(top_n).iterrows():
        cand = html.escape(str(row["display_candidate"]))
        pct = float(row["mean_prob"]) * 100.0
        mean_lp = float(row["mean_logprob"])
        cov = float(row["trial_coverage_frac"])
        pieces.append(
            "<div class='cand-compact'>"
            f"<div class='cand-line'><span class='cand'>{cand}</span> <span class='pct'>{pct:.1f}%</span></div>"
            f"<div class='meta'>lp={mean_lp:.3f} cov={cov:.2f}</div>"
            "</div>"
        )
    return "".join(pieces)


def _family_status_map(eligibility_df: pd.DataFrame, qid: str) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for _, row in eligibility_df[eligibility_df["q_id"] == qid].iterrows():
        out[str(row["family_id"])] = {
            "eligible": int(row["eligible"]) == 1,
            "reason": str(row["reason"]),
        }
    return out


def _meta_line(meta: Dict[str, Tuple[str, str]], regime_id: str) -> str:
    query_input, target_str = meta.get(regime_id, ("", ""))
    if not query_input and not target_str:
        return "<div class='small'>query= target=</div>"
    return f"<div class='small'>query={html.escape(query_input)} target={html.escape(target_str)}</div>"


def _edge_group_meta_line(meta: Dict[str, Tuple[str, str]], edge_group: str) -> str:
    cfg = EDGE_GROUP_CONFIG[edge_group]
    base_q, base_t = meta.get(cfg["base_regime"], ("", ""))
    return (
        "<div class='small'>"
        f"query: {html.escape(base_q)} | target: {html.escape(base_t)}"
        "</div>"
    )


def _edge_group_explainer(
    role_map: Dict[str, Dict[str, str]],
    qid: str,
    edge_group: str,
    meta: Dict[str, Tuple[str, str]],
) -> str:
    roles = role_map.get(qid, {})
    cfg = EDGE_GROUP_CONFIG[edge_group]
    query_input, target_str = meta.get(cfg["base_regime"], ("", ""))
    if edge_group == "AB":
        role_lines = [
            f"A: {html.escape(roles.get('A', ''))}",
            f"B: {html.escape(roles.get('B', ''))}",
        ]
        prompt_line = "AAAAA / ABABAB"
    elif edge_group == "AD":
        role_lines = [
            f"A: {html.escape(roles.get('A', ''))}",
            f"D: {html.escape(roles.get('D', ''))}",
        ]
        prompt_line = "AAAAA / ADADAD"
    else:
        role_lines = [
            f"B: {html.escape(roles.get('B', ''))}",
            f"D: {html.escape(roles.get('D', ''))}",
        ]
        prompt_line = "BBBBB / BDBDBD"
    return (
        '<div class="small explain">'
        + "".join(f'<div class="explain-role">{line}</div>' for line in role_lines)
        + f'<div class="explain-prompt">{html.escape(prompt_line)}</div>'
        + f'<div class="explain-query">query: {html.escape(query_input)}</div>'
        + f'<div class="explain-target">target: {html.escape(target_str)}</div>'
        + "</div>"
    )


def _build_edge_group_compare_table(summary_df: pd.DataFrame, eligibility_df: pd.DataFrame, qid: str, edge_group: str, meta: Dict[str, Tuple[str, str]]) -> str:
    cfg = EDGE_GROUP_CONFIG[edge_group]
    status = _family_status_map(eligibility_df, qid)
    row_specs = [
        ("baseline", cfg["base_regime"], status.get("BASE_ABD", {}).get("eligible", False), status.get("BASE_ABD", {}).get("reason", "")),
        ("mixed", cfg["ctx_regime"], status.get("CTX_ABD", {}).get("eligible", False), status.get("CTX_ABD", {}).get("reason", "")),
    ]
    pieces = [
        _edge_group_meta_line(meta, edge_group),
        '<table class="cand-table compare-table">',
        "<thead><tr><th>Mode</th>" + "".join(f"<th>Shot {shot}</th>" for shot in SHOT_ORDER) + "</tr></thead>",
        "<tbody>",
    ]
    for label, regime_id, enabled, reason_text in row_specs:
        pieces.append(f"<tr><td class='rank rowlabel'>{html.escape(label)}</td>")
        for shot in SHOT_ORDER:
            if not enabled:
                reason = html.escape(reason_text or "not eligible")
                pieces.append(f"<td><div class='small'>N/A: {reason}</div></td>")
                continue
            regime_for_cell = cfg["zero_regime"] if shot == 0 else regime_id
            pieces.append(f"<td>{_format_candidate_cell(summary_df, qid, regime_for_cell, shot)}</td>")
        pieces.append("</tr>")
    pieces.append("</tbody></table>")
    return "".join(pieces)


def _build_a_only_table(summary_df: pd.DataFrame, eligibility_df: pd.DataFrame, qid: str, meta: Dict[str, Tuple[str, str]]) -> str:
    status = _family_status_map(eligibility_df, qid)
    zero_ok = status.get("ZERO_CTRL", {}).get("eligible", False)
    a_only_ok = status.get("A_ONLY", {}).get("eligible", False)
    pieces = [
        "<div class='small'>"
        f"query: {html.escape(meta.get('AAAA_A', ('', ''))[0])} | target: {html.escape(meta.get('AAAA_A', ('', ''))[1])}"
        "</div>",
        '<table class="cand-table compare-table">',
        "<thead><tr><th>Mode</th>" + "".join(f"<th>Shot {shot}</th>" for shot in SHOT_ORDER) + "</tr></thead>",
        "<tbody>",
    ]
    for label, regime_id, valid_shots, enabled, reason in [
        ("A-only", "AAAA_A", [0, 1, 3, 5, 7, 9], a_only_ok, status.get("A_ONLY", {}).get("reason", "")),
    ]:
        pieces.append(f"<tr><td class='rank rowlabel'>{html.escape(label)}</td>")
        for shot in SHOT_ORDER:
            if shot not in valid_shots:
                pieces.append("<td class='muted-cell'></td>")
                continue
            if shot == 0 and not zero_ok:
                pieces.append(f"<td><div class='small'>N/A: {html.escape(status.get('ZERO_CTRL', {}).get('reason', 'not eligible'))}</div></td>")
                continue
            if shot > 0 and not enabled:
                pieces.append(f"<td><div class='small'>N/A: {html.escape(reason)}</div></td>")
                continue
            regime_for_cell = "ZERO_A" if shot == 0 else regime_id
            pieces.append(f"<td>{_format_candidate_cell(summary_df, qid, regime_for_cell, shot)}</td>")
        pieces.append("</tr>")
    pieces.append("</tbody></table>")
    return "".join(pieces)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _page_template(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: Georgia, "Times New Roman", serif;
      margin: 24px;
      color: #1d1d1d;
      background: #f7f4ed;
      line-height: 1.4;
    }}
    a {{ color: #0f4c5c; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .nav {{ margin-bottom: 18px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      margin: 18px 0 24px 0;
    }}
    .card {{
      background: #fffdfa;
      border: 1px solid #d9d2c3;
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    img {{
      max-width: 100%;
      height: auto;
      border: 1px solid #ddd2c4;
      border-radius: 8px;
      background: white;
    }}
    .cand-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      table-layout: fixed;
      background: #fff;
    }}
    .cand-table th, .cand-table td {{
      border: 1px solid #ddd2c4;
      padding: 7px 8px;
      vertical-align: top;
      font-size: 13px;
    }}
    .cand-table th {{
      background: #efe8d8;
    }}
    .rank {{
      width: 44px;
      text-align: center;
      font-weight: 700;
      background: #f5efe2;
    }}
    .cand {{
      font-weight: 700;
      margin-bottom: 4px;
    }}
    .meta {{
      color: #5b5348;
      font-size: 11px;
      white-space: nowrap;
    }}
    .pct {{
      color: #0f4c5c;
      font-weight: 700;
      margin-left: 6px;
    }}
    .cand-compact {{
      margin-bottom: 8px;
      padding-bottom: 6px;
      border-bottom: 1px dotted #e6ddcf;
    }}
    .cand-compact:last-child {{
      margin-bottom: 0;
      padding-bottom: 0;
      border-bottom: 0;
    }}
    .cand-line {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: baseline;
    }}
    .qid-list {{
      columns: 3 220px;
      column-gap: 24px;
    }}
    .qid-list li {{
      margin-bottom: 8px;
    }}
    h1, h2, h3 {{
      color: #222;
    }}
    .small {{
      color: #5b5348;
      font-size: 13px;
    }}
    .explain {{
      line-height: 1.55;
      margin-bottom: 14px;
      font-size: 18px;
      font-weight: 500;
    }}
    .explain-role {{
      font-size: 18px;
      font-weight: 500;
      margin-bottom: 6px;
    }}
    .explain-prompt {{
      font-size: 18px;
      font-weight: 500;
      margin: 10px 0 6px 0;
      color: #1d1d1d;
    }}
    .explain-query, .explain-target {{
      font-size: 17px;
      margin-bottom: 4px;
      font-weight: 500;
    }}
    .status-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .compare-table td {{
      min-width: 150px;
    }}
    .rowlabel {{
      width: 86px;
    }}
    .muted-cell {{
      background: #faf7f0;
    }}
    .dashboard-q {{
      margin-bottom: 18px;
      border: 1px solid #d9d2c3;
      border-radius: 12px;
      background: #fffdfa;
      overflow: hidden;
    }}
    .dashboard-q summary {{
      cursor: pointer;
      list-style: none;
      padding: 14px 16px;
      font-weight: 700;
      background: #efe8d8;
    }}
    .dashboard-q summary::-webkit-details-marker {{
      display: none;
    }}
    .dashboard-body {{
      padding: 16px;
    }}
    .stack {{
      display: flex;
      flex-direction: column;
      gap: 18px;
      margin: 18px 0 24px 0;
    }}
    .toc {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(247,244,237,0.96);
      padding: 10px 0 12px 0;
      border-bottom: 1px solid #ddd2c4;
      margin-bottom: 18px;
    }}
    .toc a {{
      display: inline-block;
      margin: 0 10px 8px 0;
      padding: 6px 10px;
      border: 1px solid #d9d2c3;
      border-radius: 999px;
      background: #fffdfa;
    }}
    .goals-box {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      margin: 18px 0 26px 0;
    }}
    .goal-panel {{
      border: 1px solid #ddd2c4;
      border-radius: 12px;
      background: #fffdfa;
      padding: 24px 26px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .goal-panel h2 {{
      margin: 0 0 14px 0;
      font-size: 28px;
    }}
    .goal-panel p {{
      margin: 0 0 12px 0;
      font-size: 19px;
      line-height: 1.65;
      color: #2b2b2b;
    }}
    .pca-tabbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .pca-tab {{
      border: 1px solid #d9d2c3;
      border-radius: 999px;
      background: #fffdfa;
      padding: 6px 10px;
      cursor: pointer;
      font: inherit;
    }}
    .pca-tab.active {{
      background: #0f4c5c;
      color: #fffdfa;
      border-color: #0f4c5c;
    }}
    .pca-frame {{
      width: 100%;
      height: 680px;
      border: 1px solid #ddd2c4;
      border-radius: 8px;
      background: white;
    }}
    .dashboard-q .card img {{
      width: 100%;
      max-width: 980px;
      display: block;
      margin: 0 auto;
    }}
    .setup-box {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin: 0 0 14px 0;
    }}
    .setup-panel {{
      border: 1px solid #ddd2c4;
      border-radius: 10px;
      background: #faf7f0;
      padding: 12px 14px;
    }}
    .setup-panel h3 {{
      margin: 0 0 8px 0;
      font-size: 17px;
    }}
    .setup-line {{
      font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 13px;
      margin: 4px 0;
      color: #2b2b2b;
    }}
  </style>
</head>
<body>
{body}
<script>
function setPcaFrame(frameId, src, btn) {{
  const frame = document.getElementById(frameId);
  if (frame) frame.src = src;
  const parent = btn.parentElement;
  if (parent) {{
    parent.querySelectorAll('.pca-tab').forEach((node) => node.classList.remove('active'));
  }}
  btn.classList.add('active');
}}
</script>
</body>
</html>
"""


def _build_dashboard_section(
    *,
    qid: str,
    eligibility_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    role_map: Dict[str, Dict[str, str]],
    report_dir: str,
    pt_plot_rel: str,
    edge_plot_rels: Dict[str, str],
    control_plot_rel: str,
) -> str:
    q_elig = eligibility_df[eligibility_df["q_id"] == qid]
    meta = _build_meta(summary_df, sweep_df, qid)
    status_parts = []
    for family_id in FAMILY_ORDER:
        row = q_elig[q_elig["family_id"] == family_id]
        if row.empty:
            continue
        rec = row.iloc[0]
        if int(rec["eligible"]) == 1:
            status_parts.append(f"{family_id}=ok")
        else:
            status_parts.append(f"{family_id}=skip")
    header_suffix = " | ".join(status_parts)
    pieces = [
        f'<details class="dashboard-q" id="{qid}" {"open" if qid == "Q1" else ""}>',
        f"<summary>{html.escape(qid)} <span class='small'>{html.escape(header_suffix)}</span></summary>",
        '<div class="dashboard-body">',
        '<div class="stack">',
        (
            f'<div class="card"><h2>Test Triangle Inequality</h2>'
            '<div class="setup-box">'
            '<div class="setup-panel">'
            '<h3>ABC</h3>'
            '<div class="setup-line">AAAAA -> B</div>'
            '<div class="setup-line">AAAAA -> C</div>'
            '<div class="setup-line">BBBBB -> C</div>'
            '</div>'
            '<div class="setup-panel">'
            '<h3>ABD</h3>'
            '<div class="setup-line">AAAAA -> B</div>'
            '<div class="setup-line">AAAAA -> D</div>'
            '<div class="setup-line">BBBBB -> D</div>'
            '</div>'
            '<div class="setup-panel">'
            '<h3>Mixed ABD</h3>'
            '<div class="setup-line">ABABAB -> B</div>'
            '<div class="setup-line">ADADAD -> D</div>'
            '<div class="setup-line">BDBDBD -> D</div>'
            '</div>'
            '</div>'
            f'<img src="{html.escape(pt_plot_rel)}" alt="{qid} PT overlay"></div>'
            if pt_plot_rel
            else '<div class="card"><h2>Test Triangle Inequality</h2><div class="small">No PT plot available.</div></div>'
        ),
        (
            "<div class=\"card\"><h2>AB / AD / BD Target Probability</h2>"
            "<div class=\"small\" style=\"margin-bottom: 10px; font-size: 16px; line-height: 1.5;\">"
            "Mean target probability: average probability assigned to the target token across trials."
            "</div>"
            "<div class=\"grid\">"
            + "".join(
                (
                    f'<div>{_edge_group_explainer(role_map, qid, edge_group, meta)}'
                    f'<img src="{html.escape(edge_plot_rels[edge_group])}" alt="{qid} {edge_group} comparison"></div>'
                )
                for edge_group in ["AB", "AD", "BD"]
                if edge_group in edge_plot_rels
            )
            + "</div></div>"
            if edge_plot_rels
            else '<div class="card"><h2>AB / AD / BD Target Probability</h2><div class="small">No comparison plot available.</div></div>'
        ),
        "</div>",
        '<div class="stack">',
        f'<div class="card"><h2>AB Candidates</h2>{_role_line_for_edge(role_map, qid, "AB")}<div class="small" style="font-size: 15px; line-height: 1.5; margin-bottom: 10px;">For each shot, candidates are taken from the next-token distribution after the demos and query, filtered to interpretable word-level items, and ranked by mean log probability across trials.</div>{_build_edge_group_compare_table(summary_df, eligibility_df, qid, "AB", meta)}</div>',
        f'<div class="card"><h2>AD Candidates</h2>{_role_line_for_edge(role_map, qid, "AD")}{_build_edge_group_compare_table(summary_df, eligibility_df, qid, "AD", meta)}</div>',
        f'<div class="card"><h2>BD Candidates</h2>{_role_line_for_edge(role_map, qid, "BD")}{_build_edge_group_compare_table(summary_df, eligibility_df, qid, "BD", meta)}</div>',
        _build_pca_iframe_card(qid, report_dir),
        "</div>",
        "</div>",
        "</details>",
    ]
    return "".join(pieces)


def main() -> int:
    args = _parse_args()
    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(run_dir, "human_report"))
    plot_dir = os.path.join(out_dir, "plots")
    _ensure_dir(plot_dir)

    sweep_csv = args.sweep_csv or os.path.join(run_dir, "pt_unified_shot_sweep.csv")
    bootstrap_csv = args.bootstrap_csv or os.path.join(run_dir, "pt_unified_bootstrap_summary.csv")
    baseline_abc_csv = args.baseline_abc_csv
    eligibility_csv = args.eligibility_csv or os.path.join(run_dir, "pt_unified_family_eligibility.csv")
    topk_jsonl = args.topk_jsonl or os.path.join(run_dir, "pt_unified_edge_topk.jsonl")
    summary_csv = args.summary_csv or os.path.join(run_dir, "pt_unified_edge_topk_trial_agg_top20_by_q_shot_regime.csv")

    sweep_df = pd.read_csv(sweep_csv)
    boot_df = pd.read_csv(bootstrap_csv) if os.path.exists(bootstrap_csv) else pd.DataFrame()
    abc_df = _load_baseline_abc_df(baseline_abc_csv)
    eligibility_df = pd.read_csv(eligibility_csv)
    topk_df = pd.read_json(topk_jsonl, lines=True) if os.path.exists(topk_jsonl) else pd.DataFrame()
    summary_df = _aggregate_topk(topk_df, summary_csv)
    role_map = _load_role_map()

    eligible_qids = sorted(
        eligibility_df[eligibility_df["eligible"] == 1]["q_id"].dropna().astype(str).unique().tolist(),
        key=_qid_sort_key,
    )
    if not eligible_qids:
        eligible_qids = sorted(sweep_df["q_id"].dropna().astype(str).unique().tolist(), key=_qid_sort_key)

    all_q_pt_median_path = os.path.join(plot_dir, "all_q_pt_median.png")
    has_all_q_pt_median = _plot_across_q_pt_median(boot_df, abc_df, eligible_qids, all_q_pt_median_path, args.dpi)

    summary_rows = []
    dashboard_rows = []
    for qid in eligible_qids:
        base_plot = os.path.join(plot_dir, f"{qid}_base.png")
        ctx_plot = os.path.join(plot_dir, f"{qid}_ctx.png")
        control_plot = os.path.join(plot_dir, f"{qid}_control.png")
        pt_plot = os.path.join(plot_dir, f"{qid}_pt.png")
        edge_compare_prefix = os.path.join(plot_dir, f"{qid}_edge_compare")
        has_base_plot = _plot_family_prob(sweep_df, qid, "BASE_ABD", base_plot, args.dpi)
        has_ctx_plot = _plot_family_prob(sweep_df, qid, "CTX_ABD", ctx_plot, args.dpi)
        has_control_plot = _plot_control(sweep_df, qid, control_plot, args.dpi)
        has_pt_plot = _plot_pt(boot_df, abc_df, qid, pt_plot, args.dpi)
        edge_plot_paths = _plot_edge_group_compare(sweep_df, qid, edge_compare_prefix, args.dpi)

        q_boot = boot_df[boot_df["q_id"] == qid].sort_values("shot") if not boot_df.empty else pd.DataFrame()
        shot9 = q_boot[q_boot["shot"] == 9] if not q_boot.empty else pd.DataFrame()
        base9 = float(shot9["base_abd_mean"].iloc[0]) if not shot9.empty and shot9["base_abd_mean"].notna().any() else None
        ctx9 = float(shot9["ctx_abd_mean"].iloc[0]) if not shot9.empty and shot9["ctx_abd_mean"].notna().any() else None
        delta9 = float(shot9["delta_ctx_minus_base"].iloc[0]) if not shot9.empty and shot9["delta_ctx_minus_base"].notna().any() else None
        summary_rows.append((qid, base9, ctx9, delta9))
        dashboard_rows.append(
            {
                "qid": qid,
                "pt_plot_rel": f"plots/{qid}_pt.png" if has_pt_plot else "",
                "edge_plot_rels": {
                    edge_group: f"plots/{qid}_edge_compare_{edge_group}.png"
                    for edge_group in edge_plot_paths.keys()
                },
                "control_plot_rel": f"plots/{qid}_control.png" if has_control_plot else "",
            }
        )

        meta = _build_meta(summary_df, sweep_df, qid)
        q_elig = eligibility_df[eligibility_df["q_id"] == qid]
        sections = [
            '<div class="nav"><a href="index.html">Back to index</a></div>',
            f"<h1>{html.escape(qid)}</h1>",
            '<div class="status-grid">',
        ]
        for family_id in FAMILY_ORDER:
            row = q_elig[q_elig["family_id"] == family_id]
            if row.empty:
                sections.append(f'<div class="card"><h3>{family_id}</h3><div class="small">No manifest row</div></div>')
                continue
            rec = row.iloc[0]
            status = "eligible" if int(rec["eligible"]) == 1 else f"not eligible: {rec['reason']}"
            sections.append(
                '<div class="card">'
                f"<h3>{family_id}</h3>"
                f"<div class='small'>{html.escape(str(status))}</div>"
                f"<div class='small'>A_pool={int(rec['A_pool'])} B_pool={int(rec['B_pool'])} D_pool={int(rec['D_pool'])}</div>"
                "</div>"
            )
        sections.append("</div>")
        sections.append('<div class="grid">')
        sections.append(
            '<div class="card"><h2>BASE_ABD</h2>'
            + (f'<img src="plots/{qid}_base.png" alt="{qid} base plot">' if has_base_plot else '<div class="small">No BASE_ABD plot</div>')
            + "</div>"
        )
        sections.append(
            '<div class="card"><h2>CTX_ABD</h2>'
            + (f'<img src="plots/{qid}_ctx.png" alt="{qid} ctx plot">' if has_ctx_plot else '<div class="small">No CTX_ABD plot</div>')
            + "</div>"
        )
        sections.append(
            '<div class="card"><h2>Controls</h2>'
            + (f'<img src="plots/{qid}_control.png" alt="{qid} control plot">' if has_control_plot else '<div class="small">No control plot</div>')
            + "</div>"
        )
        sections.append(
            '<div class="card"><h2>PT Summary</h2>'
            + (f'<img src="plots/{qid}_pt.png" alt="{qid} PT plot">' if has_pt_plot else '<div class="small">No PT plot</div>')
            + "</div>"
        )
        sections.append(_build_pca_card(qid, out_dir))
        sections.append("</div>")

        for family_id in FAMILY_ORDER:
            row = q_elig[q_elig["family_id"] == family_id]
            rec = row.iloc[0] if not row.empty else None
            if rec is None or int(rec["eligible"]) != 1:
                reason = rec["reason"] if rec is not None else "unknown"
                sections.append(
                    '<div class="card">'
                    f"<h2>{family_id}</h2>"
                    f"<div class='small'>not eligible: {html.escape(str(reason))}</div>"
                    "</div>"
                )
                continue
            family_body = ['<div class="card">', f"<h2>{family_id}</h2>"]
            for regime_id in REGIME_ORDER[family_id]:
                query_input, target_str = meta.get(regime_id, ("", ""))
                edge_group = (
                    "A_ONLY" if regime_id == "AAAA_A"
                    else "AB" if regime_id in {"BASE_AB", "CTX_ABABAB_B"}
                    else "AD" if regime_id in {"BASE_AD", "CTX_ADADAD_D"}
                    else "BD"
                )
                family_body.append(
                    f"<h3>{REGIME_LABELS[regime_id]}</h3>"
                    f"{_role_line_for_edge(role_map, qid, edge_group)}"
                    f"<div class='small'>query={html.escape(query_input)} target={html.escape(target_str)}</div>"
                    f"{_build_candidate_table(summary_df, qid, regime_id)}"
                )
            family_body.append("</div>")
            sections.append("".join(family_body))

        page_html = _page_template(f"{qid} unified PT report", "".join(sections))
        _write_text(os.path.join(out_dir, f"{qid}.html"), page_html)

    list_items = []
    for qid, base9, ctx9, delta9 in summary_rows:
        parts = []
        if base9 is not None:
            parts.append(f"shot9 BASE={base9:.3f}")
        if ctx9 is not None:
            parts.append(f"CTX={ctx9:.3f}")
        if delta9 is not None:
            parts.append(f"delta={delta9:+.3f}")
        suffix = " ".join(parts)
        list_items.append(f'<li><a href="{qid}.html">{html.escape(qid)}</a><span class="small"> {html.escape(suffix)}</span></li>')

    index_body = "".join(
        [
            "<h1>Unified PT Human Report</h1>",
            f"<p class='small'>run_dir={html.escape(run_dir)}</p>",
            '<div class="grid">',
            '<div class="card"><h2>Overview</h2><p class="small">Open the single-page dashboard for side-by-side BASE/CTX/zero-shot/A-only comparison, or use per-q pages below.</p><p><a href="all_q_dashboard.html">Open all-q dashboard</a></p></div>',
            '<div class="card"><h2>Available q_ids</h2><ul class="qid-list">' + "".join(list_items) + "</ul></div>",
            "</div>",
        ]
    )
    index_html = _page_template("Unified PT Human Report", index_body)
    _write_text(os.path.join(out_dir, "index.html"), index_html)
    toc = '<a href="#action-goals">Action Goals</a>' + "".join(
        f'<a href="#{html.escape(row["qid"])}">{html.escape(row["qid"])}</a>' for row in dashboard_rows
    )
    action_goals = (
        '<section id="action-goals" class="goals-box">'
        '<div class="goal-panel">'
        '<h2>Action Goals</h2>'
        '<p>Goal 1 — Candidate-based evaluation.</p>'
        '<p>Inspect model behavior through generated candidates rather than exact-match accuracy.</p>'
        '<p>For each prompt, generate candidate answers and check whether they align with the intended relation.</p>'
        '</div>'
        '<div class="goal-panel">'
        '<h2>Action Goals</h2>'
        '<p>Goal 2 — Representation warping in a common PCA space.</p>'
        '<p>Test whether context warps the representation of A toward B or D in a shared PCA space.</p>'
        '<p>Build one trial-level embedding per condition and project all conditions into the same PCA space.</p>'
        '</div>'
        '</section>'
    )
    across_q_pt_card = (
        '<div class="card">'
        '<h2>Across-Q PT Median</h2>'
        '<div class="small" style="margin-bottom: 10px;">'
        'Shot-wise median across q. Each PT line is aggregated independently.'
        '</div>'
        '<img src="plots/all_q_pt_median.png" alt="Across-q PT median summary">'
        '</div>'
        if has_all_q_pt_median else ""
    )
    dashboard_sections = []
    for row in dashboard_rows:
        dashboard_sections.append(
            _build_dashboard_section(
                qid=row["qid"],
                eligibility_df=eligibility_df,
                summary_df=summary_df,
                sweep_df=sweep_df,
                role_map=role_map,
                report_dir=out_dir,
                pt_plot_rel=row["pt_plot_rel"],
                edge_plot_rels=row["edge_plot_rels"],
                control_plot_rel=row["control_plot_rel"],
            )
        )
    dashboard_body = "".join(
        [
            "<h1>Unified PT All-Q Dashboard</h1>",
            f"<p class='small'>run_dir={html.escape(run_dir)}</p>",
            across_q_pt_card,
            f'<div class="toc">{toc}</div>',
            action_goals,
            "".join(dashboard_sections),
        ]
    )
    _write_text(os.path.join(out_dir, "all_q_dashboard.html"), _page_template("Unified PT All-Q Dashboard", dashboard_body))
    print(f"report_dir={out_dir}")
    print(f"index_html={os.path.join(out_dir, 'index.html')}")
    print(f"dashboard_html={os.path.join(out_dir, 'all_q_dashboard.html')}")
    print(f"summary_csv={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
