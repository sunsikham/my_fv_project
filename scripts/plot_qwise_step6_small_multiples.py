#!/usr/bin/env python3
"""Plot q-wise Step6 small-multiple line charts (non-heatmap)."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LAYER_RE = re.compile(r"^layer_(\d+)$")
Q_RE = re.compile(r"^Q(\d+)$")


def q_sort_key(qid: str) -> Tuple[int, str]:
    m = Q_RE.match(qid)
    if m:
        return (int(m.group(1)), qid)
    return (10**9, qid)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot q-wise Step6 small-multiple charts for acc and prob/delta."
    )
    p.add_argument(
        "--qwise_root",
        required=True,
        help="Q-wise root directory (e.g., results_fv/relation_qwise/relationB_ex)",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for figures",
    )
    p.add_argument("--ncols", type=int, default=4, help="Number of subplot columns")
    p.add_argument("--dpi", type=int, default=170, help="Figure DPI")
    return p.parse_args()


def load_layer_rows(step6_dir: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for layer_dir in sorted(step6_dir.iterdir(), key=lambda p: p.name):
        if not layer_dir.is_dir():
            continue
        m = LAYER_RE.match(layer_dir.name)
        if not m:
            continue
        layer = int(m.group(1))
        eval_path = layer_dir / "eval_summary.json"
        if not eval_path.exists():
            continue
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "layer": float(layer),
                "acc_with": float(payload.get("acc_with", 0.0)),
                "delta_acc": float(payload.get("delta_acc", 0.0)),
                "mean_p_with": float(payload.get("mean_p_with", 0.0)),
                "mean_delta_p": float(payload.get("mean_delta_p", 0.0)),
            }
        )
    return sorted(rows, key=lambda r: r["layer"])


def discover_qids(root: Path) -> List[str]:
    qids: List[str] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if not Q_RE.match(child.name):
            continue
        status_path = child / "artifacts" / "qid_status.json"
        if not status_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if status.get("status") == "completed":
            qids.append(child.name)
    return sorted(set(qids), key=q_sort_key)


def make_grid(n_items: int, ncols: int) -> Tuple[plt.Figure, List[plt.Axes], int, int]:
    ncols = max(1, ncols)
    nrows = int(math.ceil(n_items / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 2.8 * nrows))
    if nrows == 1 and ncols == 1:
        ax_list = [axes]
    elif nrows == 1:
        ax_list = list(axes)
    elif ncols == 1:
        ax_list = list(axes)
    else:
        ax_list = list(axes.flatten())
    return fig, ax_list, nrows, ncols


def plot_acc(qids: List[str], by_q: Dict[str, List[Dict[str, float]]], out_path: Path, ncols: int, dpi: int) -> None:
    fig, axes, nrows, ncols = make_grid(len(qids), ncols)
    for i, qid in enumerate(qids):
        ax = axes[i]
        rows = by_q.get(qid, [])
        if not rows:
            ax.set_title(f"{qid} (missing)")
            ax.axis("off")
            continue
        x = [int(r["layer"]) for r in rows]
        y_acc = [r["acc_with"] for r in rows]
        y_dacc = [r["delta_acc"] for r in rows]
        ax.plot(x, y_acc, marker="o", markersize=2.5, linewidth=1.2, label="acc_with")
        ax.plot(x, y_dacc, marker=".", markersize=2.0, linewidth=1.0, linestyle="--", label="delta_acc")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(qid, fontsize=9)
        ax.grid(alpha=0.25, linestyle=":")
        if i // ncols == nrows - 1:
            ax.set_xlabel("layer", fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel("acc", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc="best")
    for j in range(len(qids), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Q-wise Step6: Accuracy by Layer", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_prob_delta(
    qids: List[str],
    by_q: Dict[str, List[Dict[str, float]]],
    out_path: Path,
    ncols: int,
    dpi: int,
) -> None:
    fig, axes, nrows, ncols = make_grid(len(qids), ncols)
    for i, qid in enumerate(qids):
        ax = axes[i]
        rows = by_q.get(qid, [])
        if not rows:
            ax.set_title(f"{qid} (missing)")
            ax.axis("off")
            continue
        x = [int(r["layer"]) for r in rows]
        y_p = [r["mean_p_with"] for r in rows]
        y_dp = [r["mean_delta_p"] for r in rows]
        ax.plot(x, y_p, marker="o", markersize=2.5, linewidth=1.2, label="mean_p_with")
        ax.plot(x, y_dp, marker=".", markersize=2.0, linewidth=1.0, linestyle="--", label="mean_delta_p")
        ax.set_title(qid, fontsize=9)
        ax.grid(alpha=0.25, linestyle=":")
        if i // ncols == nrows - 1:
            ax.set_xlabel("layer", fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel("p / delta_p", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc="best")
    for j in range(len(qids), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Q-wise Step6: Mean p / Delta p by Layer", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = Path(args.qwise_root)
    out_dir = Path(args.out_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"missing qwise root: {root}")

    qids = discover_qids(root)
    if not qids:
        raise ValueError(f"no completed Q dirs under: {root}")

    by_q: Dict[str, List[Dict[str, float]]] = {}
    for qid in qids:
        step6_dir = root / qid / "artifacts" / "step6"
        if step6_dir.exists() and step6_dir.is_dir():
            by_q[qid] = load_layer_rows(step6_dir)
        else:
            by_q[qid] = []

    out_acc = out_dir / "qwise_small_multiples_acc.png"
    out_prob = out_dir / "qwise_small_multiples_meanp_delta.png"
    plot_acc(qids, by_q, out_acc, args.ncols, args.dpi)
    plot_prob_delta(qids, by_q, out_prob, args.ncols, args.dpi)

    print(f"saved: {out_acc}")
    print(f"saved: {out_prob}")
    print(f"q_count={len(qids)} q_ids={','.join(qids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
