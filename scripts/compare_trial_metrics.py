#!/usr/bin/env python3
"""Compare trial metrics JSONL between paper and StepD dumps."""

import argparse
import csv
import json
import os
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float(0.5 * (values[mid - 1] + values[mid]))


def summarize(values: List[float]) -> dict:
    if not values:
        return {"max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "max": max(values),
        "mean": sum(values) / len(values),
        "median": median(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare StepD vs paper trial metrics.")
    parser.add_argument("--paper_jsonl", required=True)
    parser.add_argument("--stepd_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    paper_rows = load_jsonl(args.paper_jsonl)
    stepd_rows = load_jsonl(args.stepd_jsonl)

    paper_by_trial: Dict[int, dict] = {row["trial_idx"]: row for row in paper_rows}
    stepd_by_trial: Dict[int, dict] = {row["trial_idx"]: row for row in stepd_rows}

    paper_trials = set(paper_by_trial.keys())
    stepd_trials = set(stepd_by_trial.keys())
    missing_in_paper = sorted(stepd_trials - paper_trials)
    missing_in_stepd = sorted(paper_trials - stepd_trials)
    if missing_in_paper:
        print(f"WARNING: trials missing in paper dump: {missing_in_paper}")
    if missing_in_stepd:
        print(f"WARNING: trials missing in stepd dump: {missing_in_stepd}")

    merged = []
    diff_base = []
    diff_patch = []
    diff_delta = []

    for trial_idx in sorted(paper_trials & stepd_trials):
        p = paper_by_trial[trial_idx]
        s = stepd_by_trial[trial_idx]
        db = abs(s["p_base"] - p["p_base"])
        dp = abs(s["p_patch"] - p["p_patch"])
        dd = abs(s["delta_p"] - p["delta_p"])
        diff_base.append(db)
        diff_patch.append(dp)
        diff_delta.append(dd)
        merged.append(
            {
                "trial_idx": trial_idx,
                "p_base_paper": p["p_base"],
                "p_base_stepd": s["p_base"],
                "diff_base": db,
                "p_patch_paper": p["p_patch"],
                "p_patch_stepd": s["p_patch"],
                "diff_patch": dp,
                "delta_p_paper": p["delta_p"],
                "delta_p_stepd": s["delta_p"],
                "diff_delta": dd,
            }
        )

    merged_sorted = sorted(merged, key=lambda row: row["diff_delta"], reverse=True)

    os.makedirs(args.out_dir, exist_ok=True)
    merged_path = os.path.join(args.out_dir, "merged.csv")
    with open(merged_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trial_idx",
                "p_base_paper",
                "p_base_stepd",
                "diff_base",
                "p_patch_paper",
                "p_patch_stepd",
                "diff_patch",
                "delta_p_paper",
                "delta_p_stepd",
                "diff_delta",
            ],
        )
        writer.writeheader()
        writer.writerows(merged)

    summary = {
        "diff_p_base": summarize(diff_base),
        "diff_p_patch": summarize(diff_patch),
        "diff_delta_p": summarize(diff_delta),
        "missing_in_paper": missing_in_paper,
        "missing_in_stepd": missing_in_stepd,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("diff_delta_p:", summary["diff_delta_p"])
    print("diff_p_base:", summary["diff_p_base"])
    print("diff_p_patch:", summary["diff_p_patch"])
    print("top-10 diff_delta trials:")
    for row in merged_sorted[:10]:
        print(
            f"trial={row['trial_idx']} diff_delta={row['diff_delta']:.6g} "
            f"diff_base={row['diff_base']:.6g} diff_patch={row['diff_patch']:.6g}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
