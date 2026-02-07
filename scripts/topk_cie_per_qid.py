#!/usr/bin/env python3
"""Print top-k heads per q_id from cie_scores.csv."""

import argparse
import csv
import os
from collections import defaultdict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-k CIE heads per q_id")
    parser.add_argument("--in_csv", required=False, help="Path to cie_scores.csv")
    parser.add_argument(
        "--in_dir",
        required=False,
        default=None,
        help="Directory containing cie_scores.csv (default: results/attention_head)",
    )
    parser.add_argument("--topk", type=int, default=20, help="Top-k per q_id")
    parser.add_argument(
        "--score_key",
        default="mean_delta_p",
        help="Score column to rank by (default: mean_delta_p)",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Optional output dir for per-q_id CSVs (default: alongside in_csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    by_q = defaultdict(list)
    in_csv = args.in_csv
    if in_csv is None:
        base_dir = args.in_dir or os.path.join("results", "attention_head")
        in_csv = os.path.join(base_dir, "cie_scores.csv")

    with open(in_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Missing header row")
        if args.score_key not in reader.fieldnames:
            raise ValueError(f"score_key not found: {args.score_key}")
        for row in reader:
            row[args.score_key] = float(row[args.score_key])
            by_q[row["q_id"]].append(row)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(in_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    for q_id in sorted(by_q.keys()):
        rows = sorted(by_q[q_id], key=lambda r: r[args.score_key], reverse=True)
        print(f"\n{q_id} TOP {args.topk} ({args.score_key})")
        for r in rows[: args.topk]:
            print(
                f"layer={r['layer']} head={r['head']} {args.score_key}={r[args.score_key]:.6g}"
            )
        out_path = os.path.join(out_dir, f"topk_{q_id}.csv")
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows[: args.topk])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
