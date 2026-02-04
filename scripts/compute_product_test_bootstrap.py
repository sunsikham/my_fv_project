#!/usr/bin/env python3
"""Product Test bootstrap for 5-edge scores.

Example:
  python scripts/compute_product_test_bootstrap.py \
    --in_csv results/pt_5edge_shot_sweep_gpt2.csv \
    --out_csv results/pt_bootstrap_summary.csv \
    --n_boot 10000
"""

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np


REQUIRED_COLS = {"q_id", "shot", "trial_index", "edge", "target_s_norm"}
EDGES = ("AB", "AC", "AD", "BC", "BD")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute product test bootstrap statistics from 5-edge scores."
    )
    parser.add_argument("--in_csv", required=True, help="Input CSV path")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument("--n_boot", type=int, default=10000, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--qid", required=False, default=None, help="Optional q_id filter")
    parser.add_argument(
        "--shot_list",
        required=False,
        default=None,
        help="Optional comma-separated shot filter",
    )
    parser.add_argument(
        "--stat",
        required=False,
        default="mean",
        choices=["mean", "median"],
        help="Statistic for edge aggregation",
    )
    parser.add_argument(
        "--eps", type=float, default=1e-12, help="Epsilon for PT denominator"
    )
    return parser.parse_args()


def _parse_shot_list(raw: str) -> List[int]:
    if raw is None:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def _load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Missing header row in input CSV")
        missing = REQUIRED_COLS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")
        return list(reader)


def _group_rows(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, int], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = {}
    for row in rows:
        q_id = row["q_id"]
        shot = int(row["shot"])
        grouped.setdefault((q_id, shot), []).append(row)
    return grouped


def _product_test(x: float, y: float, z: float, eps: float) -> float:
    min_val = max(min(x, y, z), eps)
    return (x * y * z) / (min_val ** 2)


def _percentile(arr: np.ndarray, pct: float) -> float:
    return float(np.percentile(arr, pct))


def main() -> int:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    shot_filter = set(_parse_shot_list(args.shot_list))

    rows = _load_rows(args.in_csv)
    if args.qid:
        rows = [row for row in rows if row["q_id"] == args.qid]
    if shot_filter:
        rows = [row for row in rows if int(row["shot"]) in shot_filter]

    grouped = _group_rows(rows)
    if not grouped:
        raise ValueError("No rows found after filtering")

    out_dir = os.path.dirname(args.out_csv) or "."
    if out_dir == "results":
        out_dir = os.path.join(out_dir, "pt_analysis")
        args.out_csv = os.path.join(out_dir, os.path.basename(args.out_csv))
    os.makedirs(out_dir, exist_ok=True)
    out_fields = [
        "q_id",
        "shot",
        "n_trials_used",
        "n_boot",
        "pt_abc_mean",
        "pt_abc_median",
        "pt_abd_mean",
        "pt_abd_median",
        "pt_abc_p_gt1",
        "pt_abd_p_gt1",
        "delta_mean",
        "delta_median",
        "delta_p_gt0",
        "pt_abc_ci2_low",
        "pt_abc_ci2_high",
        "pt_abd_ci2_low",
        "pt_abd_ci2_high",
        "delta_ci2_low",
        "delta_ci2_high",
        "pt_abc_ci1_low",
        "pt_abd_ci1_low",
        "delta_ci1_low",
    ]

    with open(args.out_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()

        for (q_id, shot), group in sorted(grouped.items()):
            # collect trials with all edges present
            edge_map: Dict[int, Dict[str, float]] = {}
            for row in group:
                edge = row["edge"]
                if edge not in EDGES:
                    continue
                val = float(row["target_s_norm"])
                if not (0.0 <= val <= 1.0):
                    raise ValueError(
                        f"target_s_norm out of range for q_id={q_id} shot={shot}"
                    )
                trial_idx = int(row["trial_index"])
                edge_map.setdefault(trial_idx, {})[edge] = val

            trials = [
                t for t, edges in edge_map.items() if all(e in edges for e in EDGES)
            ]
            if not trials:
                continue
            sAB, sAC, sAD, sBC, sBD = [], [], [], [], []
            for t in trials:
                edges = edge_map[t]
                sAB.append(edges["AB"])
                sAC.append(edges["AC"])
                sAD.append(edges["AD"])
                sBC.append(edges["BC"])
                sBD.append(edges["BD"])

            N = len(sAB)
            if N < 5:
                print(
                    f"[warn] skip q_id={q_id} shot={shot}: N={N} < 5 after drop"
                )
                continue

            sAB = np.array(sAB)
            sAC = np.array(sAC)
            sAD = np.array(sAD)
            sBC = np.array(sBC)
            sBD = np.array(sBD)

            pt_abc = np.zeros(args.n_boot, dtype=float)
            pt_abd = np.zeros(args.n_boot, dtype=float)
            delta = np.zeros(args.n_boot, dtype=float)

            for b in range(args.n_boot):
                idx = rng.integers(0, N, size=N)
                if args.stat == "mean":
                    sAB_bar = float(sAB[idx].mean())
                    sAC_bar = float(sAC[idx].mean())
                    sBC_bar = float(sBC[idx].mean())
                    sAD_bar = float(sAD[idx].mean())
                    sBD_bar = float(sBD[idx].mean())
                else:
                    sAB_bar = float(np.median(sAB[idx]))
                    sAC_bar = float(np.median(sAC[idx]))
                    sBC_bar = float(np.median(sBC[idx]))
                    sAD_bar = float(np.median(sAD[idx]))
                    sBD_bar = float(np.median(sBD[idx]))

                pt_abc_b = _product_test(sAB_bar, sAC_bar, sBC_bar, args.eps)
                pt_abd_b = _product_test(sAB_bar, sAD_bar, sBD_bar, args.eps)
                pt_abc[b] = pt_abc_b
                pt_abd[b] = pt_abd_b
                delta[b] = pt_abd_b - pt_abc_b

            row_out = {
                "q_id": q_id,
                "shot": shot,
                "n_trials_used": N,
                "n_boot": args.n_boot,
                "pt_abc_mean": float(pt_abc.mean()),
                "pt_abc_median": float(np.median(pt_abc)),
                "pt_abd_mean": float(pt_abd.mean()),
                "pt_abd_median": float(np.median(pt_abd)),
                "pt_abc_p_gt1": float((pt_abc > 1.0).mean()),
                "pt_abd_p_gt1": float((pt_abd > 1.0).mean()),
                "delta_mean": float(delta.mean()),
                "delta_median": float(np.median(delta)),
                "delta_p_gt0": float((delta > 0.0).mean()),
                "pt_abc_ci2_low": _percentile(pt_abc, 2.5),
                "pt_abc_ci2_high": _percentile(pt_abc, 97.5),
                "pt_abd_ci2_low": _percentile(pt_abd, 2.5),
                "pt_abd_ci2_high": _percentile(pt_abd, 97.5),
                "delta_ci2_low": _percentile(delta, 2.5),
                "delta_ci2_high": _percentile(delta, 97.5),
                "pt_abc_ci1_low": _percentile(pt_abc, 5.0),
                "pt_abd_ci1_low": _percentile(pt_abd, 5.0),
                "delta_ci1_low": _percentile(delta, 5.0),
            }
            writer.writerow(row_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
