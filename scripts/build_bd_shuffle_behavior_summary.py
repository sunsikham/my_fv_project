#!/usr/bin/env python3
"""Aggregate BD alternating vs shuffled behavior outputs into summary artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


BASELINE_BY_SIDE = {"D": "BDBDBD_D", "B": "DBDBDB_B"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BD shuffle behavior summaries.")
    parser.add_argument("--in_csv", required=True, help="Raw sweep CSV path")
    parser.add_argument("--edge_topk_jsonl", required=True, help="Edge top-k JSONL path")
    parser.add_argument("--regime_metrics_csv", required=True, help="Per-regime metrics CSV path")
    parser.add_argument("--case_deltas_csv", required=True, help="Per-case delta CSV path")
    parser.add_argument("--side_aggregate_csv", required=True, help="Side aggregate CSV path")
    parser.add_argument("--edge_topk_agg_csv", required=True, help="Aggregated lexical top-k CSV path")
    parser.add_argument("--summary_md", required=True, help="Markdown summary path")
    return parser.parse_args()


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row in {path}")
        return list(reader)


def _read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _float_list(values: Iterable[object]) -> List[float]:
    return [float(value) for value in values]


def _safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else float("nan")


def _safe_std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _write_csv(path: str, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_regime_metrics(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["q_id"]),
            str(row["query_side"]),
            str(row["regime_id"]),
            int(row["shot"]),
        )
        grouped[key].append(row)

    out_rows: List[Dict[str, object]] = []
    for (q_id, query_side, regime_id, shot), group_rows in sorted(grouped.items()):
        baseline_regime_id = str(group_rows[0]["baseline_regime_id"])
        case_kind = str(group_rows[0]["case_kind"])
        case_index = int(group_rows[0]["case_index"])
        layout_pattern = str(group_rows[0]["layout_pattern"])
        query_input = str(group_rows[0]["query_input"])
        target_str = str(group_rows[0]["target_str"])
        query_row_id = int(group_rows[0]["query_row_id"])
        trial_indices = sorted({int(row["trial_index"]) for row in group_rows})
        target_logprobs = _float_list(row["target_logprob_raw"] for row in group_rows)
        target_probs = _float_list(row["target_prob_raw"] for row in group_rows)
        target_logits = _float_list(row["target_logit"] for row in group_rows)
        target_ranks = _float_list(row["target_rank_in_vocab"] for row in group_rows)
        top1_count = sum(1 for row in group_rows if int(row["target_rank_in_vocab"]) == 1)
        n_trials = len(trial_indices)
        top1_accuracy = float(top1_count / n_trials) if n_trials else 0.0
        out_rows.append(
            {
                "q_id": q_id,
                "query_side": query_side,
                "regime_id": regime_id,
                "baseline_regime_id": baseline_regime_id,
                "case_kind": case_kind,
                "case_index": case_index,
                "layout_pattern": layout_pattern,
                "shot": shot,
                "query_input": query_input,
                "target_str": target_str,
                "query_row_id": query_row_id,
                "n_trials": n_trials,
                "top1_count": top1_count,
                "top1_accuracy": top1_accuracy,
                "mean_target_logprob": _safe_mean(target_logprobs),
                "std_target_logprob": _safe_std(target_logprobs),
                "mean_target_prob": _safe_mean(target_probs),
                "std_target_prob": _safe_std(target_probs),
                "mean_target_logit": _safe_mean(target_logits),
                "std_target_logit": _safe_std(target_logits),
                "mean_target_rank": _safe_mean(target_ranks),
                "std_target_rank": _safe_std(target_ranks),
            }
        )
    return out_rows


def _build_case_delta_rows(regime_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key = {
        (
            str(row["q_id"]),
            str(row["query_side"]),
            str(row["regime_id"]),
            int(row["shot"]),
        ): row
        for row in regime_rows
    }
    out_rows: List[Dict[str, object]] = []
    for row in regime_rows:
        if str(row["case_kind"]) != "shuffled":
            continue
        baseline_key = (
            str(row["q_id"]),
            str(row["query_side"]),
            str(row["baseline_regime_id"]),
            int(row["shot"]),
        )
        baseline = by_key.get(baseline_key)
        if baseline is None:
            raise ValueError(f"Missing baseline row for {baseline_key}")
        out_rows.append(
            {
                "q_id": row["q_id"],
                "query_side": row["query_side"],
                "shot": row["shot"],
                "regime_id": row["regime_id"],
                "baseline_regime_id": row["baseline_regime_id"],
                "case_index": row["case_index"],
                "layout_pattern": row["layout_pattern"],
                "n_trials": row["n_trials"],
                "baseline_top1_accuracy": baseline["top1_accuracy"],
                "shuffled_top1_accuracy": row["top1_accuracy"],
                "delta_top1_accuracy": float(row["top1_accuracy"]) - float(baseline["top1_accuracy"]),
                "baseline_mean_target_logprob": baseline["mean_target_logprob"],
                "shuffled_mean_target_logprob": row["mean_target_logprob"],
                "delta_mean_target_logprob": float(row["mean_target_logprob"]) - float(baseline["mean_target_logprob"]),
                "baseline_mean_target_prob": baseline["mean_target_prob"],
                "shuffled_mean_target_prob": row["mean_target_prob"],
                "delta_mean_target_prob": float(row["mean_target_prob"]) - float(baseline["mean_target_prob"]),
                "baseline_mean_target_logit": baseline["mean_target_logit"],
                "shuffled_mean_target_logit": row["mean_target_logit"],
                "delta_mean_target_logit": float(row["mean_target_logit"]) - float(baseline["mean_target_logit"]),
                "baseline_mean_target_rank": baseline["mean_target_rank"],
                "shuffled_mean_target_rank": row["mean_target_rank"],
                "delta_mean_target_rank": float(row["mean_target_rank"]) - float(baseline["mean_target_rank"]),
            }
        )
    return out_rows


def _build_side_aggregate_rows(case_delta_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in case_delta_rows:
        key = (str(row["q_id"]), str(row["query_side"]), int(row["shot"]))
        grouped[key].append(row)

    out_rows: List[Dict[str, object]] = []
    for (q_id, query_side, shot), group_rows in sorted(grouped.items()):
        top1_values = _float_list(row["delta_top1_accuracy"] for row in group_rows)
        lp_values = _float_list(row["delta_mean_target_logprob"] for row in group_rows)
        prob_values = _float_list(row["delta_mean_target_prob"] for row in group_rows)
        logit_values = _float_list(row["delta_mean_target_logit"] for row in group_rows)
        rank_values = _float_list(row["delta_mean_target_rank"] for row in group_rows)
        shuffled_top1_values = _float_list(row["shuffled_top1_accuracy"] for row in group_rows)
        baseline_top1 = float(group_rows[0]["baseline_top1_accuracy"])
        best_case = max(group_rows, key=lambda row: float(row["delta_top1_accuracy"]))
        worst_case = min(group_rows, key=lambda row: float(row["delta_top1_accuracy"]))
        out_rows.append(
            {
                "q_id": q_id,
                "query_side": query_side,
                "shot": shot,
                "baseline_regime_id": group_rows[0]["baseline_regime_id"],
                "n_cases": len(group_rows),
                "baseline_top1_accuracy": baseline_top1,
                "mean_shuffled_top1_accuracy": _safe_mean(shuffled_top1_values),
                "mean_delta_top1_accuracy": _safe_mean(top1_values),
                "std_delta_top1_accuracy": _safe_std(top1_values),
                "min_delta_top1_accuracy": min(top1_values),
                "max_delta_top1_accuracy": max(top1_values),
                "best_case_regime_id": best_case["regime_id"],
                "worst_case_regime_id": worst_case["regime_id"],
                "mean_delta_target_logprob": _safe_mean(lp_values),
                "std_delta_target_logprob": _safe_std(lp_values),
                "min_delta_target_logprob": min(lp_values),
                "max_delta_target_logprob": max(lp_values),
                "mean_delta_target_prob": _safe_mean(prob_values),
                "std_delta_target_prob": _safe_std(prob_values),
                "mean_delta_target_logit": _safe_mean(logit_values),
                "std_delta_target_logit": _safe_std(logit_values),
                "min_delta_target_logit": min(logit_values),
                "max_delta_target_logit": max(logit_values),
                "mean_delta_target_rank": _safe_mean(rank_values),
                "std_delta_target_rank": _safe_std(rank_values),
                "min_delta_target_rank": min(rank_values),
                "max_delta_target_rank": max(rank_values),
            }
        )
    return out_rows


def _aggregate_edge_topk(
    *,
    edge_rows: List[Dict[str, object]],
    n_trials_lookup: Dict[Tuple[str, str, int], int],
) -> List[Dict[str, object]]:
    trial_counts: Dict[Tuple[str, int, str], set] = defaultdict(set)
    by_key: Dict[Tuple[str, str, int, str], Dict[str, object]] = {}

    for row in edge_rows:
        q_id = str(row["q_id"])
        query_side = str(row["query_side"])
        shot = int(row["shot"])
        regime_id = str(row["regime_id"])
        trial_index = int(row["trial_index"])
        trial_counts[(q_id, shot, regime_id)].add(trial_index)
        candidates = list(row.get("lexical_candidates", []))
        canonicals = list(row.get("lexical_candidate_canonical_forms", []))
        logprobs = list(row.get("lexical_candidate_logprobs", []))
        probs = list(row.get("lexical_candidate_probs", []))
        for rank_idx, (cand, canon, lp, prob) in enumerate(
            zip(candidates, canonicals, logprobs, probs), start=1
        ):
            key = (q_id, query_side, shot, str(canon), regime_id)
            stat = by_key.get(key)
            if stat is None:
                stat = {
                    "q_id": q_id,
                    "query_side": query_side,
                    "shot": shot,
                    "regime_id": regime_id,
                    "query_input": str(row.get("query_input", "")),
                    "target_str": str(row.get("target_str", "")),
                    "candidate_canonical": str(canon),
                    "surface_counts": defaultdict(int),
                    "count": 0,
                    "sum_logprob": 0.0,
                    "sum_prob": 0.0,
                    "sum_rank": 0.0,
                    "top1_count": 0,
                    "trial_indices": set(),
                }
                by_key[key] = stat
            stat["surface_counts"][str(cand)] += 1
            stat["count"] += 1
            stat["sum_logprob"] += float(lp)
            stat["sum_prob"] += float(prob)
            stat["sum_rank"] += float(rank_idx)
            stat["trial_indices"].add(trial_index)
            if rank_idx == 1:
                stat["top1_count"] += 1

    grouped: Dict[Tuple[str, str, int, str], List[Dict[str, object]]] = defaultdict(list)
    for stat in by_key.values():
        q_id = stat["q_id"]
        query_side = stat["query_side"]
        shot = int(stat["shot"])
        regime_id = stat["regime_id"]
        n_trials = n_trials_lookup[(q_id, regime_id, shot)]
        display_candidate = sorted(
            stat["surface_counts"].items(), key=lambda item: (-item[1], item[0])
        )[0][0]
        grouped[(q_id, query_side, shot, regime_id)].append(
            {
                "q_id": q_id,
                "query_side": query_side,
                "shot": shot,
                "regime_id": regime_id,
                "query_input": stat["query_input"],
                "target_str": stat["target_str"],
                "candidate_canonical": stat["candidate_canonical"],
                "display_candidate": display_candidate,
                "mean_logprob": float(stat["sum_logprob"]) / stat["count"],
                "mean_prob": float(stat["sum_prob"]) / stat["count"],
                "mean_rank_within_row": float(stat["sum_rank"]) / stat["count"],
                "occurrence_count": int(stat["count"]),
                "trial_coverage_count": len(stat["trial_indices"]),
                "trial_coverage_frac": (len(stat["trial_indices"]) / n_trials) if n_trials else 0.0,
                "top1_count": int(stat["top1_count"]),
                "n_trials": n_trials,
            }
        )

    out_rows: List[Dict[str, object]] = []
    for _group_key, entries in sorted(grouped.items()):
        entries.sort(
            key=lambda row: (
                -float(row["mean_logprob"]),
                -int(row["occurrence_count"]),
                -float(row["mean_prob"]),
                str(row["display_candidate"]),
            )
        )
        for summary_rank, entry in enumerate(entries[:20], start=1):
            row = dict(entry)
            row["summary_rank"] = summary_rank
            out_rows.append(row)
    return out_rows


def _fmt(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "NA"


def _write_summary_md(
    *,
    path: str,
    regime_rows: List[Dict[str, object]],
    case_delta_rows: List[Dict[str, object]],
    side_aggregate_rows: List[Dict[str, object]],
) -> None:
    _ensure_parent(path)
    regime_lookup = {
        (str(row["q_id"]), str(row["regime_id"]), int(row["shot"])): row for row in regime_rows
    }
    side_lookup = {
        (str(row["q_id"]), str(row["query_side"]), int(row["shot"])): row
        for row in side_aggregate_rows
    }
    q_ids = sorted({str(row["q_id"]) for row in regime_rows})
    lines: List[str] = ["# BD Shuffle Summary", ""]
    for q_id in q_ids:
        lines.append(f"## {q_id}")
        lines.append("")
        for query_side in ("D", "B"):
            baseline_regime = BASELINE_BY_SIDE[query_side]
            lines.append(
                f"### Query Side {query_side} (`{baseline_regime}` vs 5 shuffled cases)"
            )
            lines.append("")
            lines.append(
                "| Shot | Baseline Acc | Mean Shuffled Acc | Mean Delta Acc | Min Delta Acc | Max Delta Acc | Mean Delta Logit | Mean Delta Logprob | Mean Delta Rank | Best Case | Worst Case |"
            )
            lines.append(
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
            )
            for shot in sorted({int(row["shot"]) for row in side_aggregate_rows if str(row["q_id"]) == q_id and str(row["query_side"]) == query_side}):
                agg = side_lookup[(q_id, query_side, shot)]
                baseline = regime_lookup[(q_id, baseline_regime, shot)]
                lines.append(
                    "| "
                    + f"{shot} | "
                    + f"{_fmt(baseline['top1_accuracy'])} | "
                    + f"{_fmt(agg['mean_shuffled_top1_accuracy'])} | "
                    + f"{_fmt(agg['mean_delta_top1_accuracy'])} | "
                    + f"{_fmt(agg['min_delta_top1_accuracy'])} | "
                    + f"{_fmt(agg['max_delta_top1_accuracy'])} | "
                    + f"{_fmt(agg['mean_delta_target_logit'])} | "
                    + f"{_fmt(agg['mean_delta_target_logprob'])} | "
                    + f"{_fmt(agg['mean_delta_target_rank'])} | "
                    + f"`{agg['best_case_regime_id']}` | "
                    + f"`{agg['worst_case_regime_id']}` |"
                )
            lines.append("")
        lines.append("### Per-Case Notes")
        lines.append("")
        for row in sorted(
            [item for item in case_delta_rows if str(item["q_id"]) == q_id],
            key=lambda item: (str(item["query_side"]), int(item["shot"]), int(item["case_index"])),
        ):
            lines.append(
                "- "
                + f"shot={row['shot']} side={row['query_side']} regime=`{row['regime_id']}` "
                + f"delta_acc={_fmt(row['delta_top1_accuracy'])} "
                + f"delta_logit={_fmt(row['delta_mean_target_logit'])} "
                + f"delta_lp={_fmt(row['delta_mean_target_logprob'])} "
                + f"delta_rank={_fmt(row['delta_mean_target_rank'])}"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def main() -> int:
    args = _parse_args()
    raw_rows = _read_csv_rows(args.in_csv)
    edge_rows = _read_jsonl(args.edge_topk_jsonl)
    if not raw_rows:
        raise ValueError("Input sweep CSV is empty")

    regime_rows = _aggregate_regime_metrics(raw_rows)
    regime_fieldnames = [
        "q_id",
        "query_side",
        "regime_id",
        "baseline_regime_id",
        "case_kind",
        "case_index",
        "layout_pattern",
        "shot",
        "query_input",
        "target_str",
        "query_row_id",
        "n_trials",
        "top1_count",
        "top1_accuracy",
        "mean_target_logprob",
        "std_target_logprob",
        "mean_target_prob",
        "std_target_prob",
        "mean_target_logit",
        "std_target_logit",
        "mean_target_rank",
        "std_target_rank",
    ]
    _write_csv(args.regime_metrics_csv, regime_fieldnames, regime_rows)

    case_delta_rows = _build_case_delta_rows(regime_rows)
    case_delta_fieldnames = [
        "q_id",
        "query_side",
        "shot",
        "regime_id",
        "baseline_regime_id",
        "case_index",
        "layout_pattern",
        "n_trials",
        "baseline_top1_accuracy",
        "shuffled_top1_accuracy",
        "delta_top1_accuracy",
        "baseline_mean_target_logprob",
        "shuffled_mean_target_logprob",
        "delta_mean_target_logprob",
        "baseline_mean_target_prob",
        "shuffled_mean_target_prob",
        "delta_mean_target_prob",
        "baseline_mean_target_logit",
        "shuffled_mean_target_logit",
        "delta_mean_target_logit",
        "baseline_mean_target_rank",
        "shuffled_mean_target_rank",
        "delta_mean_target_rank",
    ]
    _write_csv(args.case_deltas_csv, case_delta_fieldnames, case_delta_rows)

    side_aggregate_rows = _build_side_aggregate_rows(case_delta_rows)
    side_aggregate_fieldnames = [
        "q_id",
        "query_side",
        "shot",
        "baseline_regime_id",
        "n_cases",
        "baseline_top1_accuracy",
        "mean_shuffled_top1_accuracy",
        "mean_delta_top1_accuracy",
        "std_delta_top1_accuracy",
        "min_delta_top1_accuracy",
        "max_delta_top1_accuracy",
        "best_case_regime_id",
        "worst_case_regime_id",
        "mean_delta_target_logprob",
        "std_delta_target_logprob",
        "min_delta_target_logprob",
        "max_delta_target_logprob",
        "mean_delta_target_prob",
        "std_delta_target_prob",
        "mean_delta_target_logit",
        "std_delta_target_logit",
        "min_delta_target_logit",
        "max_delta_target_logit",
        "mean_delta_target_rank",
        "std_delta_target_rank",
        "min_delta_target_rank",
        "max_delta_target_rank",
    ]
    _write_csv(args.side_aggregate_csv, side_aggregate_fieldnames, side_aggregate_rows)

    n_trials_lookup = {
        (str(row["q_id"]), str(row["regime_id"]), int(row["shot"])): int(row["n_trials"])
        for row in regime_rows
    }
    edge_topk_agg_rows = _aggregate_edge_topk(
        edge_rows=edge_rows,
        n_trials_lookup=n_trials_lookup,
    )
    edge_topk_agg_fieldnames = [
        "q_id",
        "query_side",
        "shot",
        "regime_id",
        "query_input",
        "target_str",
        "candidate_canonical",
        "display_candidate",
        "mean_logprob",
        "mean_prob",
        "mean_rank_within_row",
        "occurrence_count",
        "trial_coverage_count",
        "trial_coverage_frac",
        "top1_count",
        "n_trials",
        "summary_rank",
    ]
    _write_csv(args.edge_topk_agg_csv, edge_topk_agg_fieldnames, edge_topk_agg_rows)

    _write_summary_md(
        path=args.summary_md,
        regime_rows=regime_rows,
        case_delta_rows=case_delta_rows,
        side_aggregate_rows=side_aggregate_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
