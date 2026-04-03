#!/usr/bin/env python3
"""Build a manual-review valid-answer scaffold from PT lexical top-k traces.

The scaffold is keyed by the actual query/target evaluation units that appear inside
each q_id, not by q_id alone. This avoids mixing distinct edges such as:
  - Q1 / dog -> puppy
  - Q1 / cow -> milk

The output is intended for manual review before a later set-based PT re-score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fv.pt_selected_targets import load_selected_target_artifact, normalize_selected_target


DEFAULT_FAMILIES = ("BASE_ABD", "CTX_ABD")
DEFAULT_SHOTS = (1, 3, 5, 7, 9)
DEFAULT_5EDGE_SHOTS = (1, 3, 5, 7, 10)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a manual-review valid-answer scaffold from PT lexical top-k traces."
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="PT run directory. Used for metadata and default top-k path resolution.",
    )
    parser.add_argument(
        "--topk_jsonl",
        default=None,
        help="Override top-k JSONL path. Defaults depend on row mode.",
    )
    parser.add_argument("--out_json", required=True, help="Output scaffold JSON path")
    parser.add_argument("--out_md", default=None, help="Optional output markdown summary path")
    parser.add_argument(
        "--families",
        default=None,
        help="Comma-separated family filter. Defaults depend on row mode.",
    )
    parser.add_argument(
        "--shots",
        default=None,
        help="Comma-separated shot filter. Defaults depend on row mode.",
    )
    parser.add_argument("--qid", default=None, help="Optional comma-separated q_id filter")
    parser.add_argument(
        "--top_n",
        type=int,
        default=30,
        help="Number of candidate suggestions to keep for each evaluation unit",
    )
    parser.add_argument(
        "--row_mode",
        default="auto",
        choices=["auto", "unified", "5edge"],
        help="Interpretation mode for top-k rows. Default: auto",
    )
    parser.add_argument(
        "--seed_selected_targets_json",
        default=None,
        help="Optional selected-target artifact used to prefill matching reviewed units.",
    )
    return parser.parse_args()


def _parse_csv_list(raw: str | None) -> List[str]:
    if raw is None:
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for part in str(raw).split(","):
        item = part.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_int_list(raw: str | None) -> List[int]:
    return [int(x) for x in _parse_csv_list(raw)]


def _infer_row_mode(topk_path: Path) -> str:
    for row in _load_topk_rows(topk_path):
        if "family_id" in row and "regime_id" in row:
            return "unified"
        return "5edge"
    raise ValueError(f"No rows found in top-k JSONL: {topk_path}")


def _qid_sort_key(qid: str) -> int:
    digits = "".join(ch for ch in str(qid) if ch.isdigit())
    return int(digits) if digits else 1_000_000


def _unit_sort_key(unit: Dict[str, object]) -> Tuple[int, str, str, str]:
    source_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    return (
        source_order.get(str(unit["query_source"]), 9),
        str(unit["query_input"]),
        str(unit["gold_target"]),
        str(unit["unit_id"]),
    )


def _candidate_sort_key(row: Dict[str, object]) -> Tuple[float, float, float, float, str]:
    return (
        -float(row["top1_count"]),
        -float(row["row_coverage_count"]),
        -float(row["mean_logprob"]),
        float(row["best_rank"]),
        str(row["display_candidate"]),
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _unit_id(q_id: str, query_source: str, query_input: str, gold_target: str) -> str:
    return f"{q_id}::{query_source}::{query_input}->{gold_target}"


def _load_topk_rows(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _build_scaffold(
    *,
    run_dir: Path,
    topk_path: Path,
    families: Sequence[str],
    shots: Sequence[int],
    qids: Sequence[str],
    top_n: int,
    row_mode: str,
) -> Dict[str, object]:
    family_filter = set(families)
    shot_filter = set(shots)
    qid_filter = set(qids)

    units: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    q_to_units: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for row in _load_topk_rows(topk_path):
        if row_mode == "unified":
            family_id = str(row["family_id"])
            regime_id = str(row["regime_id"])
        else:
            family_id = str(row.get("family_id") or "PT_5EDGE")
            regime_id = str(row.get("regime_id") or row.get("edge") or "")
        q_id = str(row["q_id"])
        shot = int(row["shot"])
        if family_filter and family_id not in family_filter:
            continue
        if shot_filter and shot not in shot_filter:
            continue
        if qid_filter and q_id not in qid_filter:
            continue

        query_source = str(row.get("query_source", ""))
        query_input = str(row.get("query_input", ""))
        gold_target = str(row.get("gold_target_str") or row.get("target_str", ""))
        unit_key = (q_id, query_source, query_input, gold_target)
        row_key = (
            family_id,
            regime_id,
            int(row["trial_index"]),
            shot,
            str(row["edge"]),
        )

        unit = units.get(unit_key)
        if unit is None:
            unit = {
                "q_id": q_id,
                "unit_id": _unit_id(q_id, query_source, query_input, gold_target),
                "query_source": query_source,
                "query_input": query_input,
                "gold_target": gold_target,
                "seed_valid_answers": [gold_target],
                "valid_answers": [gold_target],
                "review_status": "pending",
                "selected_target": gold_target,
                "selected_target_canonical": normalize_selected_target(gold_target),
                "notes": "",
                "families_seen": set(),
                "regimes_seen": set(),
                "shots_seen": set(),
                "row_keys": set(),
                "candidate_stats": {},
            }
            units[unit_key] = unit
            q_to_units[q_id].append(unit)

        unit["families_seen"].add(family_id)
        unit["regimes_seen"].add(regime_id)
        unit["shots_seen"].add(shot)
        unit["row_keys"].add(row_key)

        candidates = list(row.get("lexical_candidates", []))
        canonicals = list(row.get("lexical_candidate_canonical_forms", []))
        token_ids = list(row.get("lexical_candidate_token_ids", []))
        logits = list(row.get("lexical_candidate_logits", []))
        logprobs = list(row.get("lexical_candidate_logprobs", []))
        probs = list(row.get("lexical_candidate_probs", []))
        vocab_ranks = list(row.get("lexical_candidate_ranks", []))
        if len(token_ids) < len(candidates):
            token_ids = token_ids + [None] * (len(candidates) - len(token_ids))
        if len(logits) < len(candidates):
            logits = logits + [None] * (len(candidates) - len(logits))
        if len(vocab_ranks) < len(candidates):
            vocab_ranks = vocab_ranks + [None] * (len(candidates) - len(vocab_ranks))

        for rank_idx, (cand, canonical, token_id, logit, logprob, prob, vocab_rank) in enumerate(
            zip(candidates, canonicals, token_ids, logits, logprobs, probs, vocab_ranks),
            start=1,
        ):
            stats = unit["candidate_stats"].get(canonical)
            if stats is None:
                stats = {
                    "canonical": canonical,
                    "surface_counts": Counter(),
                    "occurrence_count": 0,
                    "row_keys": set(),
                    "top1_count": 0,
                    "sum_logit": 0.0,
                    "sum_logprob": 0.0,
                    "sum_prob": 0.0,
                    "best_rank": rank_idx,
                    "best_vocab_rank": (int(vocab_rank) if vocab_rank is not None else None),
                    "best_logit": (float(logit) if logit is not None else None),
                    "best_logprob": float(logprob),
                    "token_id_counts": Counter(),
                    "families_seen": set(),
                    "regimes_seen": set(),
                    "shots_seen": set(),
                }
                unit["candidate_stats"][canonical] = stats
            stats["surface_counts"][str(cand)] += 1
            if token_id is not None:
                stats["token_id_counts"][int(token_id)] += 1
            stats["occurrence_count"] += 1
            stats["row_keys"].add(row_key)
            if logit is not None:
                stats["sum_logit"] += float(logit)
            stats["sum_logprob"] += float(logprob)
            stats["sum_prob"] += float(prob)
            stats["best_rank"] = min(int(stats["best_rank"]), rank_idx)
            if vocab_rank is not None:
                if stats["best_vocab_rank"] is None:
                    stats["best_vocab_rank"] = int(vocab_rank)
                else:
                    stats["best_vocab_rank"] = min(int(stats["best_vocab_rank"]), int(vocab_rank))
            if logit is not None:
                if stats["best_logit"] is None:
                    stats["best_logit"] = float(logit)
                else:
                    stats["best_logit"] = max(float(stats["best_logit"]), float(logit))
            stats["best_logprob"] = max(float(stats["best_logprob"]), float(logprob))
            if rank_idx == 1:
                stats["top1_count"] += 1
            stats["families_seen"].add(family_id)
            stats["regimes_seen"].add(regime_id)
            stats["shots_seen"].add(shot)

    q_entries: List[Dict[str, object]] = []
    for q_id in sorted(q_to_units, key=_qid_sort_key):
        unit_entries: List[Dict[str, object]] = []
        for unit in sorted(q_to_units[q_id], key=_unit_sort_key):
            total_rows = len(unit["row_keys"])
            candidates: List[Dict[str, object]] = []
            for canonical, stats in unit["candidate_stats"].items():
                total_occ = int(stats["occurrence_count"])
                display_candidate = sorted(
                    stats["surface_counts"].items(),
                    key=lambda item: (-item[1], item[0]),
                )[0][0]
                row_coverage_count = len(stats["row_keys"])
                candidates.append(
                    {
                        "display_candidate": display_candidate,
                        "canonical": canonical,
                        "representative_token_id": (
                            sorted(stats["token_id_counts"].items(), key=lambda item: (-item[1], item[0]))[0][0]
                            if stats["token_id_counts"]
                            else None
                        ),
                        "row_coverage_count": row_coverage_count,
                        "row_coverage_frac": (row_coverage_count / total_rows) if total_rows else 0.0,
                        "occurrence_count": total_occ,
                        "top1_count": int(stats["top1_count"]),
                        "best_rank": int(stats["best_rank"]),
                        "best_vocab_rank": (int(stats["best_vocab_rank"]) if stats["best_vocab_rank"] is not None else None),
                        "best_logit": (float(stats["best_logit"]) if stats["best_logit"] is not None else None),
                        "mean_logit": (
                            float(stats["sum_logit"]) / total_occ
                            if total_occ and stats["best_logit"] is not None
                            else None
                        ),
                        "best_logprob": float(stats["best_logprob"]),
                        "mean_logprob": float(stats["sum_logprob"]) / total_occ,
                        "mean_prob": float(stats["sum_prob"]) / total_occ,
                        "families_seen": sorted(stats["families_seen"]),
                        "regimes_seen": sorted(stats["regimes_seen"]),
                        "shots_seen": sorted(stats["shots_seen"]),
                    }
                )
            candidates.sort(key=_candidate_sort_key)
            unit_entries.append(
                {
                    "unit_id": unit["unit_id"],
                    "query_source": unit["query_source"],
                    "query_input": unit["query_input"],
                    "gold_target": unit["gold_target"],
                    "seed_valid_answers": list(unit["seed_valid_answers"]),
                    "valid_answers": list(unit["valid_answers"]),
                    "review_status": unit["review_status"],
                    "selected_target": unit["selected_target"],
                    "selected_target_canonical": unit["selected_target_canonical"],
                    "notes": unit["notes"],
                    "families_seen": sorted(unit["families_seen"]),
                    "regimes_seen": sorted(unit["regimes_seen"]),
                    "shots_seen": sorted(unit["shots_seen"]),
                    "source_row_count": total_rows,
                    "candidate_suggestions": candidates[:top_n],
                }
            )
        q_entries.append({"q_id": q_id, "units": unit_entries})

    return {
        "format_version": 1,
        "artifact_kind": "pt_valid_answer_scaffold",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "canonical_root": str(run_dir),
        "source_run_dir": str(run_dir),
        "source_topk_jsonl": str(topk_path),
        "row_mode": row_mode,
        "families_included": list(families),
        "shots_included": list(shots),
        "qids_included": [entry["q_id"] for entry in q_entries],
        "questions": q_entries,
    }


def _apply_seed_selected_targets(scaffold: Dict[str, object], artifact_path: Path) -> int:
    artifact = load_selected_target_artifact(str(artifact_path))
    applied = 0
    for q_entry in scaffold["questions"]:
        for unit in q_entry["units"]:
            record = artifact.records_by_unit.get(str(unit["unit_id"]))
            if record is None:
                continue
            unit["selected_target"] = record.selected_target
            unit["selected_target_canonical"] = record.selected_target_canonical
            unit["review_status"] = "approved"
            existing_notes = str(unit.get("notes", "")).strip()
            seed_note = f"Seeded from {artifact.path}"
            unit["notes"] = seed_note if not existing_notes else f"{existing_notes}\n{seed_note}"
            valid_answers = [str(item) for item in unit.get("valid_answers", [])]
            if record.selected_target not in valid_answers:
                valid_answers.append(record.selected_target)
            unit["valid_answers"] = valid_answers
            applied += 1
    scaffold["seed_selected_targets_json"] = str(artifact.path)
    scaffold["seed_selected_target_units_applied"] = applied
    return applied


def _render_markdown(scaffold: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# PT Valid Answer Scaffold")
    lines.append("")
    lines.append(f"- `source_run_dir`: `{scaffold['source_run_dir']}`")
    lines.append(f"- `source_topk_jsonl`: `{scaffold['source_topk_jsonl']}`")
    lines.append(f"- `row_mode`: `{scaffold.get('row_mode', '')}`")
    lines.append(f"- `families_included`: `{', '.join(scaffold['families_included'])}`")
    lines.append(
        f"- `shots_included`: `{', '.join(str(x) for x in scaffold['shots_included'])}`"
    )
    if scaffold.get("seed_selected_targets_json"):
        lines.append(f"- `seed_selected_targets_json`: `{scaffold['seed_selected_targets_json']}`")
        lines.append(f"- `seed_selected_target_units_applied`: `{scaffold.get('seed_selected_target_units_applied', 0)}`")
    lines.append("")
    lines.append("Manual rule:")
    lines.append("- Set `selected_target` to the one relation-valid candidate you want PT to score for this unit.")
    lines.append("- Mark `review_status` as `approved` only after the selected target is final.")
    lines.append("")
    for q_entry in scaffold["questions"]:
        lines.append(f"## {q_entry['q_id']}")
        lines.append("")
        for unit in q_entry["units"]:
            lines.append(
                f"### {unit['query_source']} query: `{unit['query_input']}` -> gold `{unit['gold_target']}`"
            )
            lines.append("")
            lines.append(f"- `unit_id`: `{unit['unit_id']}`")
            lines.append(f"- `selected_target`: `{unit.get('selected_target', '')}`")
            lines.append(f"- `review_status`: `{unit.get('review_status', '')}`")
            lines.append(f"- `families_seen`: `{', '.join(unit['families_seen'])}`")
            lines.append(f"- `regimes_seen`: `{', '.join(unit['regimes_seen'])}`")
            lines.append(f"- `shots_seen`: `{', '.join(str(x) for x in unit['shots_seen'])}`")
            lines.append(f"- `source_row_count`: `{unit['source_row_count']}`")
            lines.append("")
            lines.append("| rank | candidate | coverage | top1 | best_rank | best_vocab_rank | mean_logit | mean_logprob | shots | regimes |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for rank_idx, cand in enumerate(unit["candidate_suggestions"], start=1):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(rank_idx),
                            str(cand["display_candidate"]),
                            f"{cand['row_coverage_count']}/{unit['source_row_count']} ({cand['row_coverage_frac']:.2f})",
                            str(cand["top1_count"]),
                            str(cand["best_rank"]),
                            (str(cand["best_vocab_rank"]) if cand.get("best_vocab_rank") is not None else ""),
                            (f"{cand['mean_logit']:.3f}" if cand.get("mean_logit") is not None else ""),
                            f"{cand['mean_logprob']:.3f}",
                            ",".join(str(x) for x in cand["shots_seen"]),
                            ",".join(cand["regimes_seen"]),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    row_mode = str(args.row_mode)
    if row_mode == "auto":
        if args.topk_jsonl:
            topk_path = Path(args.topk_jsonl).resolve()
        elif (run_dir / "pt_unified_edge_topk.jsonl").exists():
            topk_path = run_dir / "pt_unified_edge_topk.jsonl"
        elif (run_dir / "pt_edge_topk.jsonl").exists():
            topk_path = run_dir / "pt_edge_topk.jsonl"
        else:
            raise FileNotFoundError(f"Missing inferred top-k JSONL under {run_dir}")
        row_mode = _infer_row_mode(topk_path)
    else:
        if args.topk_jsonl:
            topk_path = Path(args.topk_jsonl).resolve()
        elif row_mode == "unified":
            topk_path = run_dir / "pt_unified_edge_topk.jsonl"
        else:
            topk_path = run_dir / "pt_edge_topk.jsonl"
    if not topk_path.exists():
        raise FileNotFoundError(f"Missing top-k JSONL: {topk_path}")

    if args.families is None:
        families = list(DEFAULT_FAMILIES) if row_mode == "unified" else ["PT_5EDGE"]
    else:
        families = _parse_csv_list(args.families)
    if args.shots is None:
        shots = list(DEFAULT_SHOTS) if row_mode == "unified" else list(DEFAULT_5EDGE_SHOTS)
    else:
        shots = _parse_int_list(args.shots)
    qids = _parse_csv_list(args.qid)
    if not shots:
        raise ValueError("No shots selected")
    if args.top_n < 1:
        raise ValueError("--top_n must be >= 1")

    scaffold = _build_scaffold(
        run_dir=run_dir,
        topk_path=topk_path,
        families=families,
        shots=shots,
        qids=qids,
        top_n=int(args.top_n),
        row_mode=row_mode,
    )
    if not scaffold["questions"]:
        raise ValueError("No scaffold entries produced after filtering")

    if args.seed_selected_targets_json:
        _apply_seed_selected_targets(scaffold, Path(args.seed_selected_targets_json).resolve())

    out_json = Path(args.out_json).resolve()
    _ensure_parent(out_json)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(scaffold, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    if args.out_md:
        out_md = Path(args.out_md).resolve()
        _ensure_parent(out_md)
        out_md.write_text(_render_markdown(scaffold), encoding="utf-8")

    print(f"saved_json={out_json}")
    if args.out_md:
        print(f"saved_md={Path(args.out_md).resolve()}")
    print(f"questions={len(scaffold['questions'])}")
    print(
        "units="
        + str(sum(len(question["units"]) for question in scaffold["questions"]))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
