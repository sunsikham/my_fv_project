#!/usr/bin/env python3
"""Compute product test bootstrap stats from human rating CSVs."""

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

EDGES = {"AB", "AC", "AD", "BC", "BD"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute product test bootstrap stats from human ratings."
    )
    parser.add_argument(
        "--in_csv",
        required=True,
        help="Human ratings CSV path or directory of CSVs",
    )
    parser.add_argument("--out_csv", required=True, help="Output summary CSV")
    parser.add_argument("--n_boot", type=int, default=10000, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--qid", required=False, default=None, help="Optional q_id filter")
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
    parser.add_argument(
        "--save_bootstrap",
        action="store_true",
        help="Save bootstrap samples (pt_abc/pt_abd/delta) per q_id",
    )
    parser.add_argument(
        "--bootstrap_out_dir",
        default=None,
        help="Output dir for bootstrap samples (default: alongside out_csv)",
    )
    parser.add_argument(
        "--n_perm",
        type=int,
        default=5000,
        help="Permutation iterations for sign-flip test (default: 5000)",
    )
    parser.add_argument(
        "--perm_stat",
        choices=["mean", "median"],
        default="mean",
        help="Statistic for permutation test (default: mean)",
    )
    return parser.parse_args()


def _rating_to_unit(score: float) -> float:
    return (score - 1.0) / 6.0


def _parse_qid(question_number: str) -> str:
    # Expect format Trial:2_4 or Trial:10_12
    m = re.search(r"Trial:(\d+)_", question_number)
    if not m:
        raise ValueError(f"Unrecognized Question Number: {question_number}")
    return m.group(1)


def _pair_to_edge(p1: str, p2: str) -> str | None:
    a = p1.strip().upper()
    b = p2.strip().upper()
    if not a or not b:
        raise ValueError("Empty pair values")
    pair = "".join(sorted([a, b]))
    if pair not in EDGES:
        return None
    return pair


def _product_test(x: float, y: float, z: float, eps: float) -> float:
    min_val = max(min(x, y, z), eps)
    return (x * y * z) / (min_val ** 2)


def _percentile(arr: np.ndarray, pct: float) -> float:
    return float(np.percentile(arr, pct))


def main() -> int:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    # q_id -> edge -> list[s_norm]
    edges_by_q: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    input_paths: List[str] = []
    if os.path.isdir(args.in_csv):
        for name in sorted(os.listdir(args.in_csv)):
            if not name.lower().endswith(".csv"):
                continue
            if name.startswith("."):
                continue
            input_paths.append(os.path.join(args.in_csv, name))
    else:
        input_paths.append(args.in_csv)

    required = {"Question Number", "Asked Pair One", "Asked Pair Two", "Score"}
    files_used = 0
    rows_used = 0
    for path in input_paths:
        with open(path, "r", encoding="utf-8", newline="") as f:
            raw_reader = csv.reader(f)
            rows = list(raw_reader)
        if not rows:
            continue
        header_idx = None
        header = None
        for i, row in enumerate(rows[:10]):
            normalized = [cell.strip() for cell in row]
            if required.issubset(set(normalized)):
                header_idx = i
                header = normalized
                break
        if header_idx is None or header is None:
            print(f"[warn] missing header in {path}; skipping")
            continue

        header_map = {name: idx for idx, name in enumerate(header)}
        data_rows = rows[header_idx + 1 :]

        for row in data_rows:
            if not row:
                continue
            qnum = (row[header_map["Question Number"]] if "Question Number" in header_map and header_map["Question Number"] < len(row) else "").strip()
            if not qnum:
                continue
            q_id = _parse_qid(qnum)
            if args.qid and q_id != args.qid:
                continue
            pair_one = row[header_map["Asked Pair One"]] if "Asked Pair One" in header_map and header_map["Asked Pair One"] < len(row) else ""
            pair_two = row[header_map["Asked Pair Two"]] if "Asked Pair Two" in header_map and header_map["Asked Pair Two"] < len(row) else ""
            edge = _pair_to_edge(pair_one, pair_two)
            if edge is None:
                continue
            try:
                score_str = row[header_map["Score"]] if "Score" in header_map and header_map["Score"] < len(row) else ""
                score = float(score_str)
            except ValueError:
                continue
            s_norm = _rating_to_unit(score)
            if not (0.0 <= s_norm <= 1.0):
                raise ValueError(f"Score out of range after norm: {score}")
            edges_by_q[q_id][edge].append(s_norm)
            rows_used += 1
        files_used += 1

    print(f"files_used={files_used} rows_used={rows_used}")

    # ensure all 5 edges present per q_id
    by_q = {}
    for q_id, edge_map in edges_by_q.items():
        if all(e in edge_map and edge_map[e] for e in EDGES):
            by_q[q_id] = edge_map

    out_dir = os.path.dirname(args.out_csv) or "."
    if out_dir == "results":
        out_dir = os.path.join(out_dir, "pt_analysis")
        args.out_csv = os.path.join(out_dir, os.path.basename(args.out_csv))
    os.makedirs(out_dir, exist_ok=True)
    bootstrap_dir = None
    if args.save_bootstrap:
        bootstrap_dir = args.bootstrap_out_dir
        if bootstrap_dir is None:
            bootstrap_dir = out_dir
        os.makedirs(bootstrap_dir, exist_ok=True)
    out_fields = [
        "q_id",
        "shot",
        "n_trials_used",
        "n_boot",
        "n_perm",
        "perm_stat",
        "pt_abc_mean",
        "pt_abc_median",
        "pt_abd_mean",
        "pt_abd_median",
        "pt_abc_p_gt1",
        "pt_abd_p_gt1",
        "delta_mean",
        "delta_median",
        "delta_p_gt0",
        "perm_p_one",
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

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()

        for q_id, edge_map in sorted(by_q.items()):
            edge_counts = {e: len(edge_map[e]) for e in EDGES}
            N = min(edge_counts.values())
            if N < 5:
                print(
                    f"[warn] skip q_id={q_id}: min_edge_count={N} < 5 "
                    f"counts={edge_counts}"
                )
                continue

            sAB = np.array(edge_map["AB"])
            sAC = np.array(edge_map["AC"])
            sAD = np.array(edge_map["AD"])
            sBC = np.array(edge_map["BC"])
            sBD = np.array(edge_map["BD"])

            pt_abc = np.zeros(args.n_boot, dtype=float)
            pt_abd = np.zeros(args.n_boot, dtype=float)
            delta = np.zeros(args.n_boot, dtype=float)

            for b in range(args.n_boot):
                idx_ab = rng.integers(0, len(sAB), size=N)
                idx_ac = rng.integers(0, len(sAC), size=N)
                idx_bc = rng.integers(0, len(sBC), size=N)
                idx_ad = rng.integers(0, len(sAD), size=N)
                idx_bd = rng.integers(0, len(sBD), size=N)
                if args.stat == "mean":
                    sAB_bar = float(sAB[idx_ab].mean())
                    sAC_bar = float(sAC[idx_ac].mean())
                    sBC_bar = float(sBC[idx_bc].mean())
                    sAD_bar = float(sAD[idx_ad].mean())
                    sBD_bar = float(sBD[idx_bd].mean())
                else:
                    sAB_bar = float(np.median(sAB[idx_ab]))
                    sAC_bar = float(np.median(sAC[idx_ac]))
                    sBC_bar = float(np.median(sBC[idx_bc]))
                    sAD_bar = float(np.median(sAD[idx_ad]))
                    sBD_bar = float(np.median(sBD[idx_bd]))

                pt_abc_b = _product_test(sAB_bar, sAC_bar, sBC_bar, args.eps)
                pt_abd_b = _product_test(sAB_bar, sAD_bar, sBD_bar, args.eps)
                pt_abc[b] = pt_abc_b
                pt_abd[b] = pt_abd_b
                delta[b] = pt_abd_b - pt_abc_b

            if args.save_bootstrap and bootstrap_dir is not None:
                out_path = os.path.join(bootstrap_dir, f"pt_bootstrap_{q_id}.npz")
                np.savez(
                    out_path,
                    pt_abc=pt_abc,
                    pt_abd=pt_abd,
                    delta=delta,
                    q_id=q_id,
                    n_trials=N,
                    n_boot=args.n_boot,
                )

            # permutation test on delta
            if args.perm_stat == "mean":
                t_obs = float(delta.mean())
            else:
                t_obs = float(np.median(delta))
            perm_count = 0
            for _ in range(args.n_perm):
                signs = rng.choice([-1.0, 1.0], size=delta.shape[0])
                delta_perm = delta * signs
                if args.perm_stat == "mean":
                    t_perm = float(delta_perm.mean())
                else:
                    t_perm = float(np.median(delta_perm))
                if t_perm >= t_obs:
                    perm_count += 1
            perm_p_one = (perm_count + 1) / (args.n_perm + 1)

            row_out = {
                "q_id": q_id,
                "shot": 1,
                "n_trials_used": N,
                "n_boot": args.n_boot,
                "n_perm": args.n_perm,
                "perm_stat": args.perm_stat,
                "pt_abc_mean": float(pt_abc.mean()),
                "pt_abc_median": float(np.median(pt_abc)),
                "pt_abd_mean": float(pt_abd.mean()),
                "pt_abd_median": float(np.median(pt_abd)),
                "pt_abc_p_gt1": float((pt_abc > 1.0).mean()),
                "pt_abd_p_gt1": float((pt_abd > 1.0).mean()),
                "delta_mean": float(delta.mean()),
                "delta_median": float(np.median(delta)),
                "delta_p_gt0": float((delta > 0.0).mean()),
                "perm_p_one": float(perm_p_one),
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
