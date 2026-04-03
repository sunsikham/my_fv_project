#!/usr/bin/env python3
"""
Summarize lexical top-k candidates across trials for each q_id/shot/edge.

Ranking is primarily by mean lexical candidate logprob.
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate pt_edge_topk.jsonl into q_id/shot/edge top-k summary tables."
    )
    p.add_argument("--in_jsonl", required=True, help="Input pt_edge_topk.jsonl")
    p.add_argument("--out_csv", required=True, help="Output CSV path")
    p.add_argument("--edges", default="AB,AD,BD", help="Comma-separated edges to include")
    p.add_argument("--top_n", type=int, default=20, help="Top N candidates per q_id/shot/edge")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    allowed_edges = {x.strip() for x in args.edges.split(",") if x.strip()}
    if not allowed_edges:
        raise ValueError("No edges provided")

    group_stats: Dict[Tuple[str, int, str, str], Dict[str, object]] = {}
    trial_sets: Dict[Tuple[str, int, str], set] = defaultdict(set)

    with open(args.in_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            edge = row["edge"]
            if edge not in allowed_edges:
                continue

            q_id = str(row["q_id"])
            shot = int(row["shot"])
            trial_index = int(row["trial_index"])
            query_input = str(row.get("query_input", ""))
            target_str = str(row.get("target_str", ""))
            trial_sets[(q_id, shot, edge)].add(trial_index)

            candidates: List[str] = list(row.get("lexical_candidates", []))
            canonicals: List[str] = list(row.get("lexical_candidate_canonical_forms", []))
            logprobs: List[float] = list(row.get("lexical_candidate_logprobs", []))
            probs: List[float] = list(row.get("lexical_candidate_probs", []))

            for rank_idx, (cand, canon, lp, prob) in enumerate(zip(candidates, canonicals, logprobs, probs), start=1):
                key = (q_id, shot, edge, canon)
                stat = group_stats.get(key)
                if stat is None:
                    stat = {
                        "q_id": q_id,
                        "shot": shot,
                        "edge": edge,
                        "query_input": query_input,
                        "target_str": target_str,
                        "candidate_canonical": canon,
                        "surface_counts": defaultdict(int),
                        "count": 0,
                        "sum_logprob": 0.0,
                        "sum_prob": 0.0,
                        "sum_rank": 0.0,
                        "top1_count": 0,
                        "trial_indices": set(),
                    }
                    group_stats[key] = stat
                stat["surface_counts"][cand] += 1
                stat["count"] += 1
                stat["sum_logprob"] += float(lp)
                stat["sum_prob"] += float(prob)
                stat["sum_rank"] += float(rank_idx)
                if rank_idx == 1:
                    stat["top1_count"] += 1
                stat["trial_indices"].add(trial_index)

    rows: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, int, str], List[Dict[str, object]]] = defaultdict(list)
    for (_, _, _, _), stat in group_stats.items():
        q_id = stat["q_id"]
        shot = stat["shot"]
        edge = stat["edge"]
        total_count = int(stat["count"])
        surface_counts = stat["surface_counts"]
        display_candidate = sorted(surface_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        n_trials = len(trial_sets[(q_id, shot, edge)])
        rowspec = {
            "q_id": q_id,
            "shot": shot,
            "edge": edge,
            "query_input": stat["query_input"],
            "target_str": stat["target_str"],
            "candidate_canonical": stat["candidate_canonical"],
            "display_candidate": display_candidate,
            "mean_logprob": float(stat["sum_logprob"]) / total_count,
            "mean_prob": float(stat["sum_prob"]) / total_count,
            "mean_rank_within_row": float(stat["sum_rank"]) / total_count,
            "occurrence_count": total_count,
            "trial_coverage_count": len(stat["trial_indices"]),
            "trial_coverage_frac": (len(stat["trial_indices"]) / n_trials) if n_trials else 0.0,
            "top1_count": int(stat["top1_count"]),
            "n_trials": n_trials,
        }
        grouped[(q_id, shot, edge)].append(rowspec)

    final_rows: List[Dict[str, object]] = []
    for group_key, entries in grouped.items():
        entries.sort(
            key=lambda r: (
                -float(r["mean_logprob"]),
                -int(r["occurrence_count"]),
                -float(r["mean_prob"]),
                str(r["display_candidate"]),
            )
        )
        for out_rank, entry in enumerate(entries[: args.top_n], start=1):
            entry = dict(entry)
            entry["summary_rank"] = out_rank
            final_rows.append(entry)

    df = pd.DataFrame(final_rows)
    if df.empty:
        raise ValueError("No summary rows produced")
    df = df.sort_values(["q_id", "shot", "edge", "summary_rank"]).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)
    print(f"saved_csv={args.out_csv}")
    print(f"rows={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
