#!/usr/bin/env python3
"""Run common PCA for per-q condition vectors."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("CSV rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_conditions(text: str) -> List[str]:
    raw = [part.strip().upper() for part in (text or "").split(",") if part.strip()]
    if not raw:
        raise ValueError("conditions must not be empty")
    return raw


def pairwise_distances(centroids: Dict[str, np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    keys = sorted(centroids.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            out[f"{a}__{b}"] = float(np.linalg.norm(centroids[a] - centroids[b]))
    return out


def maybe_plot_scatter(
    *,
    out_path: Path,
    points: np.ndarray,
    labels: Sequence[str],
    title: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    unique = sorted(set(labels))
    color_map = {name: idx for idx, name in enumerate(unique)}
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in unique:
        idxs = [i for i, value in enumerate(labels) if value == name]
        arr = points[idxs]
        x = arr[:, 0]
        y = arr[:, 1] if arr.shape[1] > 1 else np.zeros_like(x)
        ax.scatter(x, y, label=name, s=14, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    ax.grid(alpha=0.2, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def maybe_plot_scatter_3d_html(
    *,
    out_path: Path,
    points: np.ndarray,
    labels: Sequence[str],
    trial_ids: Sequence[str],
    title: str,
) -> bool:
    if points.shape[1] < 3:
        return False

    unique = sorted(set(labels))
    color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    color_map = {name: color_palette[idx % len(color_palette)] for idx, name in enumerate(unique)}

    traces = []
    for name in unique:
        idxs = [i for i, value in enumerate(labels) if value == name]
        arr = points[idxs]
        text = [trial_ids[i] for i in idxs]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": name,
                "x": [float(v) for v in arr[:, 0]],
                "y": [float(v) for v in arr[:, 1]],
                "z": [float(v) for v in arr[:, 2]],
                "text": text,
                "hovertemplate": "cond=%{fullData.name}<br>trial=%{text}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}<br>PC3=%{z:.3f}<extra></extra>",
                "marker": {
                    "size": 5,
                    "opacity": 0.85,
                    "color": color_map[name],
                },
            }
        )

    layout = {
        "title": f"{title} - Interactive",
        "paper_bgcolor": "#f5f5f5",
        "plot_bgcolor": "#f5f5f5",
        "legend": {"x": 0.85, "y": 0.95},
        "margin": {"l": 0, "r": 0, "b": 0, "t": 55},
        "scene": {
            "xaxis": {"title": "PC1", "gridcolor": "#d9d9d9", "zerolinecolor": "#b0b0b0"},
            "yaxis": {"title": "PC2", "gridcolor": "#d9d9d9", "zerolinecolor": "#b0b0b0"},
            "zaxis": {"title": "PC3", "gridcolor": "#d9d9d9", "zerolinecolor": "#b0b0b0"},
            "camera": {"eye": {"x": 1.55, "y": 1.35, "z": 0.95}},
        },
    }

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    html, body {{ margin: 0; padding: 0; background: #f5f5f5; font-family: Helvetica, Arial, sans-serif; }}
    #plot {{ width: 100vw; height: 100vh; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const data = {json.dumps(traces, ensure_ascii=True)};
    const layout = {json.dumps(layout, ensure_ascii=True)};
    const config = {{responsive: true, displaylogo: false}};
    Plotly.newPlot('plot', data, layout, config);
  </script>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    return True


def maybe_plot_scatter_3d_png(
    *,
    out_path: Path,
    points: np.ndarray,
    labels: Sequence[str],
    title: str,
) -> bool:
    if points.shape[1] < 3:
        return False
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        return False

    unique = sorted(set(labels))
    fig = plt.figure(figsize=(7.5, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    for name in unique:
        idxs = [i for i, value in enumerate(labels) if value == name]
        arr = points[idxs]
        ax.scatter(
            arr[:, 0],
            arr[:, 1],
            arr[:, 2],
            label=name,
            s=18,
            alpha=0.82,
            depthshade=True,
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(elev=23, azim=38)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run condition common PCA.")
    parser.add_argument("--q_dir", required=True, help="Per-q output directory root")
    parser.add_argument(
        "--ref_mode",
        default="AAA_ref",
        help="Reference vector mode; resolves _vectors/trial_vectors_<ref_mode>_<condition>.npy",
    )
    parser.add_argument("--conditions", default="AAA,BBB,BABA")
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--balance_trials", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_subdir",
        default=None,
        help="Optional output subdir under _pca_common (default: ref_mode)",
    )
    args = parser.parse_args()

    q_dir = Path(args.q_dir)
    vectors_dir = q_dir / "_vectors"
    meta_path = vectors_dir / "vector_extraction_meta.json"
    if not vectors_dir.exists():
        print(f"Missing vectors directory: {vectors_dir}")
        return 1
    if not meta_path.exists():
        print(f"Missing vector_extraction_meta.json: {meta_path}")
        return 1

    conditions = parse_conditions(args.conditions)
    meta = read_json(meta_path)
    meta_trial_ids = meta.get("trial_ids", {})
    if not isinstance(meta_trial_ids, dict):
        meta_trial_ids = {}

    arr_by_cond: Dict[str, np.ndarray] = {}
    trial_ids_by_cond: Dict[str, List[str]] = {}

    for cond in conditions:
        vec_path = vectors_dir / f"trial_vectors_{args.ref_mode}_{cond}.npy"
        if not vec_path.exists():
            print(f"Missing vector file for condition={cond}: {vec_path}")
            return 1
        arr = np.load(vec_path)
        if arr.ndim != 2:
            print(f"Vector array must be 2D: {vec_path} shape={arr.shape}")
            return 1
        arr_by_cond[cond] = arr.astype(np.float32, copy=False)
        raw_ids = meta_trial_ids.get(cond, [])
        if isinstance(raw_ids, list) and len(raw_ids) == arr.shape[0]:
            ids = [str(item) for item in raw_ids]
        else:
            ids = [f"t{i:06d}" for i in range(arr.shape[0])]
        trial_ids_by_cond[cond] = ids

    if args.balance_trials:
        min_n = min(arr.shape[0] for arr in arr_by_cond.values())
        if min_n < 1:
            print("No trials available after loading vectors.")
            return 1
        for cond in conditions:
            arr = arr_by_cond[cond]
            ids = trial_ids_by_cond[cond]
            if arr.shape[0] == min_n:
                continue
            idxs = list(range(arr.shape[0]))
            rng = random.Random(args.seed + sum(ord(c) for c in cond))
            rng.shuffle(idxs)
            idxs = sorted(idxs[:min_n])
            arr_by_cond[cond] = arr[idxs]
            trial_ids_by_cond[cond] = [ids[idx] for idx in idxs]

    rows: List[np.ndarray] = []
    labels: List[str] = []
    trial_id_flat: List[str] = []
    for cond in conditions:
        arr = arr_by_cond[cond]
        rows.append(arr)
        labels.extend([cond] * arr.shape[0])
        trial_id_flat.extend(trial_ids_by_cond[cond])
    matrix = np.concatenate(rows, axis=0).astype(np.float32, copy=False)

    if matrix.shape[0] < 2:
        print("Need at least 2 samples for PCA.")
        return 1
    if args.n_components < 1:
        print("n_components must be >= 1")
        return 1
    n_components = min(args.n_components, matrix.shape[0], matrix.shape[1])

    mean_vec = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean_vec
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    comps = vt[:n_components]
    points = centered @ comps.T

    denom = float(matrix.shape[0] - 1) if matrix.shape[0] > 1 else 1.0
    explained = (s**2) / denom
    total_var = float(np.sum(explained))
    explained_ratio = (
        (explained[:n_components] / total_var).tolist() if total_var > 0 else [0.0] * n_components
    )

    out_leaf = args.out_subdir.strip() if isinstance(args.out_subdir, str) else ""
    out_dir = q_dir / "_pca_common" / (out_leaf or args.ref_mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    point_rows: List[Dict[str, object]] = []
    for i in range(points.shape[0]):
        row = {
            "condition": labels[i],
            "trial_id": trial_id_flat[i],
        }
        for comp_idx in range(n_components):
            row[f"pc{comp_idx + 1}"] = float(points[i, comp_idx])
        point_rows.append(row)
    write_csv(out_dir / "pca_points.csv", point_rows)

    centroids: Dict[str, np.ndarray] = {}
    centroid_rows: List[Dict[str, object]] = []
    for cond in conditions:
        idxs = [i for i, value in enumerate(labels) if value == cond]
        arr = points[idxs]
        centroid = arr.mean(axis=0)
        centroids[cond] = centroid
        row = {"condition": cond}
        for comp_idx in range(n_components):
            row[f"pc{comp_idx + 1}"] = float(centroid[comp_idx])
        centroid_rows.append(row)
    write_csv(out_dir / "pca_centroids.csv", centroid_rows)

    distance_summary = {
        "ref_mode": args.ref_mode,
        "conditions": conditions,
        "pairwise_centroid_distance": pairwise_distances(centroids),
    }
    write_json(out_dir / "distance_summary.json", distance_summary)

    plotted = maybe_plot_scatter(
        out_path=out_dir / "scatter.png",
        points=points,
        labels=labels,
        title=f"Common PCA ({args.ref_mode})",
    )
    plotted_3d = maybe_plot_scatter_3d_html(
        out_path=out_dir / "scatter_3d_interactive.html",
        points=points,
        labels=labels,
        trial_ids=trial_id_flat,
        title=f"Common PCA 3D ({args.ref_mode})",
    )
    plotted_3d_png = maybe_plot_scatter_3d_png(
        out_path=out_dir / "scatter_3d.png",
        points=points,
        labels=labels,
        title=f"Common PCA 3D ({args.ref_mode})",
    )
    model_meta = {
        "created_at": utc_now(),
        "ref_mode": args.ref_mode,
        "out_subdir": (out_leaf or args.ref_mode),
        "conditions": conditions,
        "n_components": int(n_components),
        "n_samples": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "balance_trials": bool(args.balance_trials),
        "explained_variance_ratio": [float(x) for x in explained_ratio],
        "scatter_plotted": bool(plotted),
        "scatter_3d_interactive_plotted": bool(plotted_3d),
        "scatter_3d_png_plotted": bool(plotted_3d_png),
    }
    write_json(out_dir / "pca_model_meta.json", model_meta)
    print(f"saved pca artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
