#!/usr/bin/env python3
"""Recompute selected-target Unified PT sweep rows from cached edge top-k artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from transformers import AutoTokenizer

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in os.sys.path:
    os.sys.path.insert(0, _ROOT)

from fv.pt_selected_targets import (
    SelectedTargetArtifact,
    load_selected_target_artifact,
    normalize_selected_target,
    parse_family_ids,
    preflight_review_scope,
    resolve_selected_target,
)


FAMILY_ORDER = ("BASE_ABD", "CTX_ABD", "ZERO_CTRL", "A_ONLY")
APPROVED_REGIMES = (
    "BASE_AB",
    "BASE_AD",
    "BASE_BD",
    "CTX_ABABAB_B",
    "CTX_ADADAD_D",
    "CTX_BDBDBD_D",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute Unified PT selected-target sweep rows from cached edge top-k artifacts."
    )
    parser.add_argument("--source_run_dir", required=True, help="Source cache-build Unified PT run directory")
    parser.add_argument("--selected_targets_json", required=True, help="Canonical selected-target artifact")
    parser.add_argument("--out_csv", required=True, help="Output recomputed sweep CSV path")
    parser.add_argument("--source_sweep_csv", default=None, help="Override source pt_unified_shot_sweep.csv path")
    parser.add_argument("--topk_jsonl", default=None, help="Override source pt_unified_edge_topk.jsonl path")
    parser.add_argument("--family_ids", default="BASE_ABD,CTX_ABD", help="Comma-separated family filter")
    parser.add_argument("--shot_list", default="1,3,5,7,9", help="Comma-separated positive shot filter")
    parser.add_argument("--qid", default=None, help="Optional comma-separated q_id filter")
    return parser.parse_args()


def _parse_csv_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    out: List[str] = []
    seen = set()
    for part in str(raw).split(","):
        item = part.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_int_list(raw: Optional[str]) -> List[int]:
    return [int(x) for x in _parse_csv_list(raw)]


def _load_run_meta(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source run metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row in {path}")
        return list(reader)


def _load_topk_rows(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _edge_key_from_row(row: Dict[str, object]) -> Tuple[str, str, int, int, str]:
    return (
        str(row["family_id"]),
        str(row["q_id"]),
        int(row["trial_index"]),
        int(row["shot"]),
        str(row["edge"]),
    )


def _resolve_source_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    run_dir = Path(args.source_run_dir).resolve()
    sweep_csv = Path(args.source_sweep_csv).resolve() if args.source_sweep_csv else run_dir / "pt_unified_shot_sweep.csv"
    topk_jsonl = Path(args.topk_jsonl).resolve() if args.topk_jsonl else run_dir / "pt_unified_edge_topk.jsonl"
    meta_json = run_dir / "run_meta.json"
    if not sweep_csv.exists():
        raise FileNotFoundError(f"Missing source sweep CSV: {sweep_csv}")
    if not topk_jsonl.exists():
        raise FileNotFoundError(f"Missing source top-k JSONL: {topk_jsonl}")
    return run_dir, sweep_csv, topk_jsonl


def _select_target_first_token_id(tokenizer, selected_target: str) -> int:
    text = str(selected_target).strip()
    if not text:
        raise ValueError("selected_target is empty")
    ids = tokenizer.encode(" " + text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"selected_target produced no tokens: {selected_target!r}")
    return int(ids[0])


def _qid_from_artifact(artifact: SelectedTargetArtifact) -> List[str]:
    out: List[str] = []
    seen = set()
    for unit_id, record in artifact.records_by_unit.items():
        if record.q_id and record.q_id not in seen:
            seen.add(record.q_id)
            out.append(record.q_id)
    return out


def _normalize_rows(rows: List[Dict[str, object]]) -> None:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["family_id"]), str(row["q_id"])), []).append(row)

    for _, group_rows in grouped.items():
        raw = np.array([float(row["target_logprob_raw"]) for row in group_rows], dtype=float)
        p_low = float(np.percentile(raw, 5))
        p_high = float(np.percentile(raw, 95))
        if p_high == p_low:
            for row in group_rows:
                row["target_s_norm"] = 0.5
        else:
            for row in group_rows:
                s_val = (float(row["target_logprob_raw"]) - p_low) / (p_high - p_low)
                if s_val < 0.0:
                    s_val = 0.0
                elif s_val > 1.0:
                    s_val = 1.0
                row["target_s_norm"] = s_val
        for row in group_rows:
            row["norm_p_low"] = p_low
            row["norm_p_high"] = p_high
            row["norm_method"] = "robust_minmax_p05_p95"
            row["norm_scope"] = f"qid_family_{row['family_id']}_all_regimes_all_relevant_shots"


def _fieldnames(rows: Sequence[Dict[str, object]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _normalize_optional_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def main() -> int:
    args = _parse_args()
    family_ids = parse_family_ids(args.family_ids, allowed=FAMILY_ORDER)
    shot_list = _parse_int_list(args.shot_list)
    preflight_review_scope(family_ids=family_ids, shots=shot_list)

    selected_targets = load_selected_target_artifact(args.selected_targets_json)
    qid_filter = _parse_csv_list(args.qid)
    if not qid_filter:
        qid_filter = _qid_from_artifact(selected_targets)
    qid_filter_set = set(qid_filter)
    shot_filter = set(shot_list)
    family_filter = set(family_ids)
    approved_regimes = set(APPROVED_REGIMES)

    source_run_dir, source_sweep_csv, topk_jsonl = _resolve_source_paths(args)
    run_meta = _load_run_meta(source_run_dir / "run_meta.json")
    source_model = str(run_meta.get("model") or "").strip()
    source_model_spec = str(run_meta.get("model_spec") or "").strip()
    if not source_model or not source_model_spec:
        raise ValueError(f"Source run metadata missing model/model_spec: {source_run_dir / 'run_meta.json'}")

    tokenizer = AutoTokenizer.from_pretrained(source_model, use_fast=True)

    source_rows = _load_csv_rows(source_sweep_csv)
    source_lookup: Dict[Tuple[str, str, int, int, str], Dict[str, str]] = {}
    for row in source_rows:
        row_qid = str(row["q_id"])
        row_family = str(row["family_id"])
        row_shot = int(row["shot"])
        row_edge = str(row["edge"])
        if qid_filter_set and row_qid not in qid_filter_set:
            continue
        if row_family not in family_filter:
            continue
        if row_shot not in shot_filter:
            continue
        if row_edge not in approved_regimes:
            continue
        source_lookup[_edge_key_from_row(row)] = row

    if not source_lookup:
        raise ValueError("No source sweep rows remain after q/family/shot/regime filtering")

    topk_lookup: Dict[Tuple[str, str, int, int, str], Dict[str, object]] = {}
    for row in _load_topk_rows(topk_jsonl):
        row_qid = str(row["q_id"])
        row_family = str(row["family_id"])
        row_shot = int(row["shot"])
        row_edge = str(row["edge"])
        if qid_filter_set and row_qid not in qid_filter_set:
            continue
        if row_family not in family_filter:
            continue
        if row_shot not in shot_filter:
            continue
        if row_edge not in approved_regimes:
            continue
        key = _edge_key_from_row(row)
        topk_lookup[key] = row

    missing_topk = sorted(set(source_lookup) - set(topk_lookup))
    if missing_topk:
        raise ValueError(f"Missing edge-topk rows for {len(missing_topk)} source sweep rows; first={missing_topk[0]}")

    out_rows: List[Dict[str, object]] = []
    for key in sorted(source_lookup.keys()):
        source_row = dict(source_lookup[key])
        edge_row = topk_lookup[key]
        gold_target = str(source_row.get("gold_target_str") or source_row.get("target_str") or "").strip()
        query_source = str(source_row.get("query_source") or edge_row.get("query_source") or "").strip()
        query_input = str(source_row.get("query_input") or edge_row.get("query_input") or "").strip()
        record = resolve_selected_target(
            selected_targets,
            q_id=str(source_row["q_id"]),
            query_source=query_source,
            query_input=query_input,
            gold_target=gold_target,
        )
        target_id = _select_target_first_token_id(tokenizer, record.selected_target)

        token_ids = [int(x) for x in edge_row.get("lexical_candidate_token_ids", [])]
        logits = list(edge_row.get("lexical_candidate_logits", []))
        logprobs = list(edge_row.get("lexical_candidate_logprobs", []))
        probs = list(edge_row.get("lexical_candidate_probs", []))
        ranks = list(edge_row.get("lexical_candidate_ranks", []))
        if not (len(token_ids) == len(logits) == len(logprobs) == len(probs) == len(ranks)):
            raise ValueError(f"Candidate cache field length mismatch for row key={key}")
        lookup_source = ""
        target_logit = None
        target_logprob = None
        target_prob = None
        target_rank = None
        try:
            match_idx = token_ids.index(target_id)
            lookup_source = "lexical_topk"
            target_logit = float(logits[match_idx])
            target_logprob = float(logprobs[match_idx])
            target_prob = float(probs[match_idx])
            target_rank = int(ranks[match_idx])
        except ValueError:
            forced_token_id = edge_row.get("forced_selected_target_first_token_id")
            forced_token_id = (int(forced_token_id) if str(forced_token_id).strip() else None)
            forced_target_canonical = _normalize_optional_text(edge_row.get("forced_selected_target_canonical"))
            forced_target_str = _normalize_optional_text(edge_row.get("forced_selected_target_str"))
            if (
                forced_token_id is not None
                and forced_token_id == target_id
                and (
                    forced_target_canonical == record.selected_target_canonical
                    or normalize_selected_target(forced_target_str) == record.selected_target_canonical
                )
            ):
                lookup_source = "forced_selected_target"
                target_logit = float(edge_row["forced_selected_target_logit"])
                target_logprob = float(edge_row["forced_selected_target_logprob_raw"])
                target_prob = float(edge_row["forced_selected_target_prob_raw"])
                target_rank = int(edge_row["forced_selected_target_rank_in_vocab"])
            else:
                raise ValueError(
                    "Selected target first token missing from both cached lexical top-k and forced selected-target cache; "
                    f"q_id={source_row['q_id']} edge={source_row['edge']} shot={source_row['shot']} "
                    f"trial={source_row['trial_index']} selected_target={record.selected_target!r} "
                    f"target_id={target_id}"
                )

        source_row["gold_target_str"] = gold_target
        source_row["target_str"] = record.selected_target
        source_row["scored_target_str"] = record.selected_target
        source_row["scored_target_canonical"] = record.selected_target_canonical or normalize_selected_target(record.selected_target)
        source_row["selected_target_artifact"] = str(selected_targets.path)
        source_row["scoring_basis"] = "selected_target_offline_cache"
        source_row["target_resolution_status"] = "resolved_offline_cache"
        source_row["target_suffix_str"] = " " + record.selected_target
        source_row["target_first_token_id"] = target_id
        source_row["target_token_str"] = tokenizer.decode([target_id], skip_special_tokens=False)
        source_row["target_logit"] = float(target_logit)
        source_row["target_logprob_raw"] = float(target_logprob)
        source_row["target_prob_raw"] = float(target_prob)
        source_row["target_rank_in_vocab"] = int(target_rank)
        source_row["selected_target_lookup_source"] = lookup_source
        source_row["source_run_dir"] = str(source_run_dir)
        source_row["source_topk_jsonl"] = str(topk_jsonl)
        source_row["source_sweep_csv"] = str(source_sweep_csv)
        out_rows.append(source_row)

    _normalize_rows(out_rows)

    out_path = Path(args.out_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(out_rows)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"source_run_dir={source_run_dir}")
    print(f"source_sweep_csv={source_sweep_csv}")
    print(f"source_topk_jsonl={topk_jsonl}")
    print(f"selected_targets_json={Path(args.selected_targets_json).resolve()}")
    print(f"rows={len(out_rows)}")
    print(f"saved_csv={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
