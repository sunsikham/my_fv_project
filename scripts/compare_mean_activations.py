#!/usr/bin/env python3
"""Compare paper mean_head_activations_FIXED.pt vs StepD mean_activations.pt."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List

import torch

EPS = 1e-12


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_labels(raw: Any) -> List[str]:
    if isinstance(raw, dict):
        items = sorted(raw.items(), key=lambda kv: int(kv[0]))
        return [value for _key, value in items]
    if isinstance(raw, list):
        if not raw:
            return []
        first = raw[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            items = sorted(raw, key=lambda item: int(item[0]))
            return [item[1] for item in items]
        return raw
    raise ValueError("Unsupported dummy_labels format")


def _find_label_index(labels: List[str], label: str) -> int:
    try:
        return labels.index(label)
    except ValueError as exc:
        raise ValueError(f"Label '{label}' not found in dummy_labels") from exc


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    weight = pos - lo
    return float(sorted_vals[lo] * (1.0 - weight) + sorted_vals[hi] * weight)


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    mean_val = sum(values) / len(values)
    median_val = _quantile(values, 0.5)
    return {
        "mean": float(mean_val),
        "median": float(median_val),
        "p05": float(_quantile(values, 0.05)),
        "p95": float(_quantile(values, 0.95)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare mean activations between paper and StepD outputs.")
    parser.add_argument(
        "--paper_pt",
        default="results/antonym/antonym_mean_head_activations_FIXED.pt",
        help="Paper mean_head_activations_FIXED.pt",
    )
    parser.add_argument(
        "--paper_labels",
        default="results/antonym/antonym_dummy_labels.json",
        help="Paper dummy labels JSON",
    )
    parser.add_argument(
        "--stepd_pt",
        default="runs/<RUN_ID>/artifacts/mean_activations.pt",
        help="StepD mean_activations.pt",
    )
    parser.add_argument(
        "--stepd_meta",
        default="runs/<RUN_ID>/artifacts/mean_activations_meta.json",
        help="StepD mean_activations_meta.json",
    )
    parser.add_argument(
        "--out_dir",
        default="runs/<RUN_ID>/artifacts/compare_mean",
        help="Output directory",
    )
    parser.add_argument("--topk", type=int, default=30, help="Top-K heads to print")
    parser.add_argument("--dtype", default="fp32", choices=["fp32"], help="Tensor dtype (default: fp32)")
    args = parser.parse_args()

    paper_labels_raw = _load_json(args.paper_labels)
    labels_payload = paper_labels_raw.get("labels", paper_labels_raw)
    paper_dummy_labels = _normalize_labels(labels_payload)
    paper_query_slot = _find_label_index(paper_dummy_labels, "query_predictive_token")

    stepd_meta = _load_json(args.stepd_meta)
    if "slot_query_pred" not in stepd_meta:
        raise ValueError("stepd_meta missing slot_query_pred")
    stepd_query_slot = int(stepd_meta["slot_query_pred"])
    stepd_labels = stepd_meta.get("dummy_labels")
    if stepd_labels is None:
        raise ValueError("stepd_meta missing dummy_labels")
    if not (0 <= stepd_query_slot < len(stepd_labels)):
        raise ValueError("stepd_query_slot out of range for dummy_labels")
    stepd_label = stepd_labels[stepd_query_slot]
    if stepd_label not in ("QUERY_PRED", "query_predictive_token"):
        print(
            "WARNING: stepd dummy_labels at slot_query_pred is "
            f"{stepd_label!r} (expected 'QUERY_PRED' or 'query_predictive_token')"
        )

    paper = torch.load(args.paper_pt, map_location="cpu").to(dtype=torch.float32)
    stepd = torch.load(args.stepd_pt, map_location="cpu").to(dtype=torch.float32)

    if paper.dim() != 4:
        raise ValueError(f"paper tensor must be [L,H,S,D], got {tuple(paper.shape)}")
    if stepd.dim() != 4:
        raise ValueError(f"stepd tensor must be [L,H,S,D], got {tuple(stepd.shape)}")

    paper_shape = tuple(paper.shape)
    stepd_shape = tuple(stepd.shape)

    if stepd_shape[0] != paper_shape[0] or stepd_shape[1] != paper_shape[1] or stepd_shape[3] != paper_shape[3]:
        raise ValueError(
            "Shape mismatch (expected L/H/D match). "
            f"paper={paper_shape} stepd={stepd_shape}"
        )

    paper_vec = paper[:, :, paper_query_slot, :]
    stepd_vec = stepd[:, :, stepd_query_slot, :]

    dot = (paper_vec * stepd_vec).sum(dim=-1)
    norm_paper = torch.linalg.norm(paper_vec, dim=-1)
    norm_stepd = torch.linalg.norm(stepd_vec, dim=-1)
    cosine = dot / (norm_paper * norm_stepd + EPS)
    l2 = torch.linalg.norm(paper_vec - stepd_vec, dim=-1)
    norm_ratio = norm_stepd / (norm_paper + EPS)
    rel_l2 = l2 / (norm_paper + EPS)

    cosine_list = cosine.flatten().tolist()
    l2_list = l2.flatten().tolist()
    rel_l2_list = rel_l2.flatten().tolist()
    norm_ratio_list = norm_ratio.flatten().tolist()

    cosine_stats = _summary_stats(cosine_list)
    l2_stats = _summary_stats(l2_list)
    rel_l2_stats = _summary_stats(rel_l2_list)
    norm_ratio_stats = _summary_stats(norm_ratio_list)

    print("cosine: mean={mean:.6f} median={median:.6f} p05={p05:.6f} min={min:.6f}".format(**cosine_stats))
    print("l2: mean={mean:.6f} median={median:.6f} p95={p95:.6f} max={max:.6f}".format(**l2_stats))
    print("rel_l2: mean={mean:.6f} median={median:.6f} p95={p95:.6f} max={max:.6f}".format(**rel_l2_stats))
    print("norm_ratio: mean={mean:.6f} median={median:.6f} p05={p05:.6f} p95={p95:.6f}".format(**norm_ratio_stats))

    rows: List[Dict[str, Any]] = []
    n_layers, n_heads = paper_vec.shape[:2]
    for layer in range(n_layers):
        for head in range(n_heads):
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "cosine": float(cosine[layer, head].item()),
                    "l2": float(l2[layer, head].item()),
                    "rel_l2": float(rel_l2[layer, head].item()),
                    "norm_paper": float(norm_paper[layer, head].item()),
                    "norm_stepd": float(norm_stepd[layer, head].item()),
                    "norm_ratio": float(norm_ratio[layer, head].item()),
                }
            )

    topk = min(args.topk, len(rows))
    topk_cosine = sorted(rows, key=lambda r: r["cosine"])[:topk]
    topk_rel_l2 = sorted(rows, key=lambda r: r["rel_l2"], reverse=True)[:topk]

    print(f"topk cosine (lowest): {topk}")
    for row in topk_cosine:
        print(
            "layer={layer} head={head} cosine={cosine:.6f} l2={l2:.6f} "
            "rel_l2={rel_l2:.6f} norm_paper={norm_paper:.6f} "
            "norm_stepd={norm_stepd:.6f} norm_ratio={norm_ratio:.6f}".format(**row)
        )

    print(f"topk rel_l2 (highest): {topk}")
    for row in topk_rel_l2:
        print(
            "layer={layer} head={head} cosine={cosine:.6f} l2={l2:.6f} "
            "rel_l2={rel_l2:.6f} norm_paper={norm_paper:.6f} "
            "norm_stepd={norm_stepd:.6f} norm_ratio={norm_ratio:.6f}".format(**row)
        )

    os.makedirs(args.out_dir, exist_ok=True)

    summary = {
        "paper_shape": list(paper_shape),
        "stepd_shape": list(stepd_shape),
        "paper_query_slot": paper_query_slot,
        "paper_query_label": "query_predictive_token",
        "stepd_query_slot": stepd_query_slot,
        "stepd_query_label": stepd_label,
        "stats": {
            "cosine": cosine_stats,
            "l2": l2_stats,
            "rel_l2": rel_l2_stats,
            "norm_ratio": norm_ratio_stats,
        },
    }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    fields = [
        "layer",
        "head",
        "cosine",
        "l2",
        "rel_l2",
        "norm_paper",
        "norm_stepd",
        "norm_ratio",
    ]
    _write_csv(os.path.join(args.out_dir, "per_head.csv"), rows, fields)
    _write_csv(os.path.join(args.out_dir, "topk_cosine.csv"), topk_cosine, fields)
    _write_csv(os.path.join(args.out_dir, "topk_rel_l2.csv"), topk_rel_l2, fields)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
