#!/usr/bin/env python3
"""Generate relation trials from CSV for StepD."""

import argparse
from pathlib import Path

from fv.relation_trials import generate_relation_trials, save_trials_json


def _format_overlap(overlaps):
    if not overlaps:
        return "n/a"
    avg = sum(overlaps) / len(overlaps)
    return f"mean={avg:.2f} min={min(overlaps)} max={max(overlaps)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate relation trials from CSV.")
    parser.add_argument(
        "--csv_path",
        default="datasets/relation/relationB_ex (1).csv",
        help="Path to relation CSV (default: datasets/relation/relationB_ex (1).csv)",
    )
    parser.add_argument(
        "--q_list",
        default=None,
        help="Comma-separated q list (default: all in CSV)",
    )
    parser.add_argument(
        "--n_trials_per_q",
        type=int,
        required=True,
        help="Number of trials to generate per q",
    )
    parser.add_argument(
        "--n_demos",
        type=int,
        default=10,
        help="Number of demos per trial (default: 10; reduced if needed)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--save_trials_json",
        type=int,
        default=0,
        help="Save trials JSON to --out_path (default: 0)",
    )
    parser.add_argument(
        "--out_path",
        default=None,
        help="Output path when --save_trials_json=1",
    )
    args = parser.parse_args()

    trials_data, stats = generate_relation_trials(
        csv_path=args.csv_path,
        q_list=args.q_list,
        n_trials_per_q=args.n_trials_per_q,
        n_demos=args.n_demos,
        seed=args.seed,
    )

    print("relation_trials: summary")
    print(f"  csv_path={args.csv_path}")
    print(f"  n_q_total={len(stats.q_counts)}")
    print(f"  n_q_skipped={len(stats.skipped_qs)}")
    if stats.skipped_qs:
        skipped = ", ".join(
            f"{q}({count})" for q, count in sorted(stats.skipped_qs.items())
        )
        print(f"  skipped_qs={skipped}")
    print(f"  n_demos_effective={stats.n_demos_effective}")
    for q_id in sorted(stats.q_counts.keys()):
        if q_id in stats.skipped_qs:
            continue
        overlap = _format_overlap(stats.shuffle_match_counts.get(q_id, []))
        print(
            f"  q={q_id} n_examples={stats.q_counts[q_id]} "
            f"n_demos={stats.q_demo_counts[q_id]} "
            f"n_trials={stats.q_trials[q_id]} "
            f"corruption_overlap={overlap}"
        )

    if args.save_trials_json:
        if not args.out_path:
            raise ValueError("--out_path is required when --save_trials_json=1")
        save_trials_json(trials_data, args.out_path)
        print(f"saved trials: {args.out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
