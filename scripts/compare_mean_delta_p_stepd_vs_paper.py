#!/usr/bin/env python3
"""
Compare StepD mean_delta_p vs paper indirect_effect mean (Δp) by head.

Example:
  python scripts/compare_mean_delta_p_stepd_vs_paper.py \
    --stepd_csv results/antonym/aie_scores.csv \
    --paper_pt results/antonym/antonym_indirect_effect.pt \
    --out_dir results/compare_delta_p
"""

import argparse
import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {c: c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _rankdata(values: np.ndarray) -> np.ndarray:
    # Average ranks for ties.
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # handle ties
    sorted_vals = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        if end - start > 1:
            avg_rank = ranks[order[start:end]].mean()
            ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearsonr(rx, ry)


def _load_stepd(stepd_csv: str, stepd_col: str) -> Tuple[pd.DataFrame, str, str, str]:
    df = pd.read_csv(stepd_csv)
    layer_col = _find_column(df, ["layer", "layer_idx", "layer_id", "l"])
    head_col = _find_column(df, ["head", "head_idx", "head_id", "h"])
    if layer_col is None or head_col is None:
        raise ValueError(
            "Failed to find layer/head columns in StepD CSV. "
            f"columns={list(df.columns)}"
        )
    score_col = _find_column(df, [stepd_col, "mean_delta_p_target", "mean_delta_p"])
    if score_col is None:
        raise ValueError(
            "Failed to find StepD score column. "
            f"requested={stepd_col} columns={list(df.columns)}"
        )
    return df, layer_col, head_col, score_col


def _load_paper_tensor(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("indirect_effect", "ie", "effects"):
            if key in obj:
                obj = obj[key]
                break
    if isinstance(obj, torch.Tensor):
        return obj
    raise ValueError(f"Unsupported paper_pt content type: {type(obj)}")


def _reduce_paper_tensor(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 3:
        return t.mean(dim=0)
    if t.ndim == 2:
        return t
    raise ValueError(f"Unsupported indirect_effect shape: {tuple(t.shape)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare StepD mean_delta_p vs paper indirect_effect mean Δp."
    )
    parser.add_argument("--stepd_csv", required=True, help="Path to StepD aie_scores.csv")
    parser.add_argument("--paper_pt", required=True, help="Path to paper indirect_effect.pt")
    parser.add_argument("--out_dir", default="results/compare_delta_p", help="Output dir")
    parser.add_argument(
        "--stepd_col",
        default="mean_delta_p",
        help="StepD score column (fallback to mean_delta_p_target).",
    )
    parser.add_argument("--topk", type=int, default=20, help="Top-k by abs diff")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    stepd_df, layer_col, head_col, score_col = _load_stepd(args.stepd_csv, args.stepd_col)
    stepd_df = stepd_df[[layer_col, head_col, score_col]].copy()
    stepd_df.columns = ["layer", "head", "stepd_mean_delta_p"]

    paper_t = _load_paper_tensor(args.paper_pt)
    paper_mean = _reduce_paper_tensor(paper_t)
    paper_np = paper_mean.detach().cpu().numpy()

    rows = []
    for layer in range(paper_np.shape[0]):
        for head in range(paper_np.shape[1]):
            rows.append((layer, head, float(paper_np[layer, head])))
    paper_df = pd.DataFrame(rows, columns=["layer", "head", "paper_mean_delta_p"])

    merged = pd.merge(stepd_df, paper_df, on=["layer", "head"], how="inner")
    merged["diff"] = merged["stepd_mean_delta_p"] - merged["paper_mean_delta_p"]
    merged["abs_diff"] = merged["diff"].abs()
    merged_path = os.path.join(args.out_dir, "compare_delta_p_merged.csv")
    merged.to_csv(merged_path, index=False)

    topk = merged.sort_values("abs_diff", ascending=False).head(args.topk)
    topk_path = os.path.join(args.out_dir, "compare_delta_p_topk.csv")
    topk.to_csv(topk_path, index=False)

    x = merged["paper_mean_delta_p"].to_numpy()
    y = merged["stepd_mean_delta_p"].to_numpy()
    pearson_r = _pearsonr(x, y)
    spearman_r = _spearmanr(x, y)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=10, alpha=0.6)
    lim = [
        min(x.min(initial=0.0), y.min(initial=0.0)),
        max(x.max(initial=0.0), y.max(initial=0.0)),
    ]
    plt.plot(lim, lim, "k--", linewidth=1)
    plt.xlabel("paper_mean_delta_p")
    plt.ylabel("stepd_mean_delta_p")
    plt.title(f"Δp scatter (r={pearson_r:.3f}, ρ={spearman_r:.3f})")
    plt.tight_layout()
    scatter_path = os.path.join(args.out_dir, "compare_delta_p_scatter.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()

    max_layer = int(merged["layer"].max())
    max_head = int(merged["head"].max())
    grid = np.full((max_layer + 1, max_head + 1), np.nan, dtype=float)
    for row in merged.itertuples(index=False):
        grid[int(row.layer), int(row.head)] = float(row.diff)
    plt.figure(figsize=(8, 4))
    im = plt.imshow(grid, aspect="auto", cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("head")
    plt.ylabel("layer")
    plt.title("Δp diff (stepd - paper)")
    plt.tight_layout()
    heatmap_path = os.path.join(args.out_dir, "compare_delta_p_diff_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()

    print(
        f"StepD: rows={len(stepd_df)} layer_col={layer_col} head_col={head_col} score_col={score_col}"
    )
    print(f"Paper tensor shape: {tuple(paper_t.shape)}")
    print(f"Pearson r={pearson_r:.4f} Spearman rho={spearman_r:.4f}")
    if not topk.empty:
        top_preview = topk.head(min(5, len(topk)))[
            ["layer", "head", "stepd_mean_delta_p", "paper_mean_delta_p", "diff"]
        ]
        print("Top diff preview:")
        print(top_preview.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
