#!/usr/bin/env python3
"""
Unified PT scorer:
  - baseline ABD family
  - mixed-context ABD family
  - zero-shot controls
  - A-only control
with family-aware eligibility, resume, and lexical top-k traces.
"""

import argparse
import csv
import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in os.sys.path:
    os.sys.path.insert(0, _ROOT)

from fv.hf_loader import load_hf_model_and_tokenizer
from fv.model_spec import get_model_spec
from fv.pt_selected_targets import (
    build_unit_id,
    load_selected_target_artifact,
    parse_family_ids,
    preflight_review_scope,
    resolve_selected_target,
)
from fv.prompting import build_prompt_qa
from scripts.score_cross_relation_target_logit import (
    _append_jsonl,
    _collect_edge_topk,
    _dedupe_by_edge_key,
    _file_sha256,
    _group_by_qid,
    _jaccard,
    _parse_shot_list,
    _percentile,
    _progress_line,
    _read_json,
    _read_jsonl,
    _read_relation_csv,
    _resume_dir,
    _select_query,
    _stable_hash,
    _target_first_token_id_with_checks,
    _target_rank_in_vocab,
    _write_json,
)


FAMILY_ORDER = ("BASE_ABD", "CTX_ABD", "ZERO_CTRL", "A_ONLY")
FAMILY_REGIMES = {
    "BASE_ABD": (
        {"regime_id": "BASE_AB", "edge_group": "AB", "mode": "prefix", "source": "A", "query_source": "B"},
        {"regime_id": "BASE_AD", "edge_group": "AD", "mode": "prefix", "source": "A", "query_source": "D"},
        {"regime_id": "BASE_BD", "edge_group": "BD", "mode": "prefix", "source": "B", "query_source": "D"},
    ),
    "CTX_ABD": (
        {"regime_id": "CTX_ABABAB_B", "edge_group": "AB", "mode": "alternating", "first_source": "A", "second_source": "B", "query_source": "B"},
        {"regime_id": "CTX_ADADAD_D", "edge_group": "AD", "mode": "alternating", "first_source": "A", "second_source": "D", "query_source": "D"},
        {"regime_id": "CTX_BDBDBD_D", "edge_group": "BD", "mode": "alternating", "first_source": "B", "second_source": "D", "query_source": "D"},
    ),
    "ZERO_CTRL": (
        {"regime_id": "ZERO_A", "edge_group": "ZA", "mode": "zero", "query_source": "A"},
        {"regime_id": "ZERO_B", "edge_group": "ZB", "mode": "zero", "query_source": "B"},
        {"regime_id": "ZERO_D", "edge_group": "ZD", "mode": "zero", "query_source": "D"},
    ),
    "A_ONLY": (
        {"regime_id": "AAAA_A", "edge_group": "AA", "mode": "prefix", "source": "AONLY", "query_source": "A"},
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified PT scorer with baseline, context drift, zero-shot, and A-only controls."
    )
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--model_spec", required=True, help="Model spec (e.g. llama3)")
    parser.add_argument("--device", required=True, help="Device (cpu/cuda)")
    parser.add_argument("--dtype", required=False, default=None, help="fp32/fp16/bf16")
    parser.add_argument("--quant", required=False, default="none", help="Quantization mode: none/4bit/8bit/auto")
    parser.add_argument("--relationA_ex_path", required=True, help="A demos CSV")
    parser.add_argument("--relationB_ex_path", required=True, help="B demos CSV")
    parser.add_argument("--relationD_ex_path", required=True, help="D demos CSV")
    parser.add_argument("--icl_B_path", required=True, help="B query CSV")
    parser.add_argument("--icl_D_path", required=True, help="D query CSV")
    parser.add_argument(
        "--shot_list",
        required=False,
        default="0,1,3,5,7,9",
        help="Comma-separated shot list (default: 0,1,3,5,7,9)",
    )
    parser.add_argument("--n_trials", type=int, required=True, help="Trials per q_id for non-zero families")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--qid", required=False, default=None, help="Optional q_id or comma-separated q_id list")
    parser.add_argument(
        "--family_ids",
        default=",".join(FAMILY_ORDER),
        help="Comma-separated family ids to execute. Default: BASE_ABD,CTX_ABD,ZERO_CTRL,A_ONLY",
    )
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument("--eligibility_csv", required=True, help="Output family eligibility CSV path")
    parser.add_argument(
        "--selected_targets_json",
        default=None,
        help="Optional canonical selected-target artifact. First pass supports only BASE_ABD,CTX_ABD and positive shots.",
    )
    parser.add_argument(
        "--forced_selected_targets_json",
        default=None,
        help="Optional canonical selected-target artifact used only to cache forced selected-target scores in edge-topk rows.",
    )
    parser.add_argument("--save_edge_topk", type=int, default=0, choices=[0, 1], help="If 1, save lexical top-k trace for all regimes")
    parser.add_argument("--edge_topk_k", type=int, default=10, help="Lexical top-k size (default: 10)")
    parser.add_argument("--edge_topk_jsonl", default=None, help="Optional output path for raw edge lexical top-k JSONL")
    parser.add_argument("--edge_topk_change_csv", default=None, help="Optional output path for adjacent-shot edge change summary CSV")
    return parser.parse_args()


def _parse_qid_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    seen = set()
    out: List[str] = []
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        out.append(part)
    return out


def _build_fingerprint(args: argparse.Namespace, *, family_order: Sequence[str]) -> str:
    payload = {
        "model": args.model,
        "model_spec": args.model_spec,
        "device": args.device,
        "dtype": args.dtype,
        "quant": args.quant,
        "relationA_ex_path": os.path.abspath(args.relationA_ex_path),
        "relationB_ex_path": os.path.abspath(args.relationB_ex_path),
        "relationD_ex_path": os.path.abspath(args.relationD_ex_path),
        "icl_B_path": os.path.abspath(args.icl_B_path),
        "icl_D_path": os.path.abspath(args.icl_D_path),
        "relationA_ex_sha256": _file_sha256(args.relationA_ex_path),
        "relationB_ex_sha256": _file_sha256(args.relationB_ex_path),
        "relationD_ex_sha256": _file_sha256(args.relationD_ex_path),
        "icl_B_sha256": _file_sha256(args.icl_B_path),
        "icl_D_sha256": _file_sha256(args.icl_D_path),
        "shot_list": list(_parse_shot_list(args.shot_list)),
        "n_trials": int(args.n_trials),
        "seed": int(args.seed),
        "qids": list(_parse_qid_list(args.qid)),
        "family_ids": list(family_order),
        "selected_targets_json": (os.path.abspath(args.selected_targets_json) if args.selected_targets_json else None),
        "selected_targets_sha256": (_file_sha256(args.selected_targets_json) if args.selected_targets_json else None),
        "forced_selected_targets_json": (
            os.path.abspath(args.forced_selected_targets_json) if args.forced_selected_targets_json else None
        ),
        "forced_selected_targets_sha256": (
            _file_sha256(args.forced_selected_targets_json) if args.forced_selected_targets_json else None
        ),
        "save_edge_topk": int(args.save_edge_topk),
        "edge_topk_k": int(args.edge_topk_k),
        "family_regimes": {
            family_id: [regime["regime_id"] for regime in FAMILY_REGIMES[family_id]]
            for family_id in family_order
        },
        "scorer_code_sha256": _file_sha256(__file__),
    }
    return _stable_hash(payload)


def _state_path(out_dir: str) -> str:
    return os.path.join(_resume_dir(out_dir), "pt_unified_resume_state.json")


def _trial_plan_path(out_dir: str) -> str:
    return os.path.join(_resume_dir(out_dir), "pt_unified_trial_plan.json")


def _unit_rows_path(out_dir: str, family_id: str, q_id: str) -> str:
    return os.path.join(_resume_dir(out_dir), f"raw_rows__{family_id}__{q_id}.jsonl")


def _unit_edge_topk_path(out_dir: str, family_id: str, q_id: str) -> str:
    return os.path.join(_resume_dir(out_dir), f"raw_edge_topk__{family_id}__{q_id}.jsonl")


def _unit_id(family_id: str, q_id: str) -> str:
    return f"{family_id}::{q_id}"


def _pattern_label(first_source: str, second_source: Optional[str], shot: int, mode: str) -> str:
    if mode == "zero":
        return "ZERO"
    if mode == "prefix":
        return first_source * shot
    if mode == "alternating" and second_source is not None:
        out: List[str] = []
        for idx in range(shot):
            out.append(first_source if idx % 2 == 0 else second_source)
        return "".join(out)
    raise ValueError(f"Unsupported pattern mode: {mode}")


def _build_demo_rows(regime: Dict[str, str], *, selected_rows: Dict[str, List[Dict[str, str]]], shot: int) -> List[Dict[str, str]]:
    mode = regime["mode"]
    if mode == "zero":
        return []
    if mode == "prefix":
        source_key = regime["source"]
        return selected_rows[source_key][:shot]
    if mode == "alternating":
        first_source = regime["first_source"]
        second_source = regime["second_source"]
        demos: List[Dict[str, str]] = []
        source_idx = {src: 0 for src in selected_rows}
        for pos in range(shot):
            src = first_source if pos % 2 == 0 else second_source
            demos.append(selected_rows[src][source_idx[src]])
            source_idx[src] += 1
        return demos
    raise ValueError(f"Unsupported regime mode: {mode}")


def _write_eligibility_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "q_id",
        "family_id",
        "eligible",
        "reason",
        "A_pool",
        "B_pool",
        "D_pool",
        "A_query",
        "B_query",
        "D_query",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_change_csv(path: str, rows: List[Dict[str, object]]) -> None:
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    fieldnames = list(rows[0].keys()) if rows else [
        "family_id",
        "q_id",
        "trial_index",
        "regime_id",
        "shot_from",
        "shot_to",
        "target_logprob_from",
        "target_logprob_to",
        "target_logprob_delta",
        "target_s_norm_from",
        "target_s_norm_to",
        "target_s_norm_delta",
        "target_rank_from",
        "target_rank_to",
        "target_rank_delta",
        "top1_candidate_from",
        "top1_candidate_to",
        "top1_changed",
        "lexical_token_id_jaccard",
        "lexical_text_jaccard",
        "lexical_overlap_count",
        "canonical_overlap_count",
    ]
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        if rows:
            writer.writerows(rows)
        handle.flush()


def _canonical_unit_counts(path: str, key_fields: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not os.path.exists(path):
        return counts
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle) if path.endswith(".csv") else None
        if reader is not None:
            for row in reader:
                unit = "::".join(str(row[field]) for field in key_fields)
                counts[unit] = counts.get(unit, 0) + 1
            return counts
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            unit = "::".join(str(row[field]) for field in key_fields)
            counts[unit] = counts.get(unit, 0) + 1
    return counts


def _completed_keys_from_raw(
    out_dir: str,
    family_id: str,
    q_id: str,
    *,
    edge_topk_enabled: bool,
) -> Tuple[set, int]:
    rows = _dedupe_by_edge_key(_read_jsonl(_unit_rows_path(out_dir, family_id, q_id)))
    row_keys = {(int(row["trial_index"]), int(row["shot"]), str(row["edge"])) for row in rows}
    if not edge_topk_enabled:
        return row_keys, len(row_keys)
    edge_rows = _dedupe_by_edge_key(_read_jsonl(_unit_edge_topk_path(out_dir, family_id, q_id)))
    edge_keys = {(int(row["trial_index"]), int(row["shot"]), str(row["edge"])) for row in edge_rows}
    common = row_keys & edge_keys
    return common, len(common)


def _query_repr(row: Optional[Dict[str, str]]) -> str:
    if row is None:
        return ""
    return f"{row['input']}->{row['output']}"


def _build_family_contexts(
    *,
    q_id: str,
    A_by: Dict[str, List[Dict[str, str]]],
    B_by: Dict[str, List[Dict[str, str]]],
    D_by: Dict[str, List[Dict[str, str]]],
    icl_B_by: Dict[str, List[Dict[str, str]]],
    icl_D_by: Dict[str, List[Dict[str, str]]],
    positive_max_shot: int,
    zero_enabled: bool,
) -> Dict[str, Dict[str, object]]:
    A_q = A_by.get(q_id, [])
    B_q = B_by.get(q_id, [])
    D_q = D_by.get(q_id, [])
    icl_B_q = icl_B_by.get(q_id, [])
    icl_D_q = icl_D_by.get(q_id, [])

    A_query = _select_query(A_q) if A_q else None
    B_query = _select_query(icl_B_q) if icl_B_q else None
    D_query = _select_query(icl_D_q) if icl_D_q else None

    out: Dict[str, Dict[str, object]] = {}

    # BASE_ABD
    base_reason = ""
    base_A_pool: List[Dict[str, str]] = []
    base_B_pool: List[Dict[str, str]] = []
    if positive_max_shot < 1:
        base_reason = "no positive shots requested"
    elif not A_q:
        base_reason = "missing relationA examples"
    elif not B_q:
        base_reason = "missing relationB examples"
    elif B_query is None:
        base_reason = "missing B query"
    elif D_query is None:
        base_reason = "missing D query"
    else:
        forbidden_A = {(B_query["input"], B_query["output"]), (D_query["input"], D_query["output"])}
        forbidden_B = {(D_query["input"], D_query["output"])}
        base_A_pool = [row for row in A_q if (row["input"], row["output"]) not in forbidden_A]
        base_B_pool = [row for row in B_q if (row["input"], row["output"]) not in forbidden_B]
        if len(base_A_pool) < positive_max_shot or len(base_B_pool) < positive_max_shot:
            base_reason = (
                "filtered demo pool too small "
                f"(A_pool={len(base_A_pool)}, B_pool={len(base_B_pool)})"
            )
    out["BASE_ABD"] = {
        "eligible": base_reason == "",
        "reason": base_reason or "eligible",
        "A_pool_rows": base_A_pool,
        "B_pool_rows": base_B_pool,
        "D_pool_rows": [],
        "queries": {"A": A_query, "B": B_query, "D": D_query},
    }

    # CTX_ABD
    ctx_reason = ""
    ctx_A_pool: List[Dict[str, str]] = []
    ctx_B_pool: List[Dict[str, str]] = []
    ctx_D_pool: List[Dict[str, str]] = []
    if positive_max_shot < 1:
        ctx_reason = "no positive shots requested"
    elif not A_q:
        ctx_reason = "missing relationA examples"
    elif not B_q:
        ctx_reason = "missing relationB examples"
    elif not D_q:
        ctx_reason = "missing relationD examples"
    elif B_query is None:
        ctx_reason = "missing B query"
    elif D_query is None:
        ctx_reason = "missing D query"
    else:
        forbidden_AB = {(B_query["input"], B_query["output"]), (D_query["input"], D_query["output"])}
        forbidden_D = {(D_query["input"], D_query["output"])}
        ctx_A_pool = [row for row in A_q if (row["input"], row["output"]) not in forbidden_AB]
        ctx_B_pool = [row for row in B_q if (row["input"], row["output"]) not in forbidden_AB]
        ctx_D_pool = [row for row in D_q if (row["input"], row["output"]) not in forbidden_D]
        if len(ctx_A_pool) < positive_max_shot or len(ctx_B_pool) < positive_max_shot or len(ctx_D_pool) < positive_max_shot:
            ctx_reason = (
                "filtered demo pool too small "
                f"(A_pool={len(ctx_A_pool)}, B_pool={len(ctx_B_pool)}, D_pool={len(ctx_D_pool)})"
            )
    out["CTX_ABD"] = {
        "eligible": ctx_reason == "",
        "reason": ctx_reason or "eligible",
        "A_pool_rows": ctx_A_pool,
        "B_pool_rows": ctx_B_pool,
        "D_pool_rows": ctx_D_pool,
        "queries": {"A": A_query, "B": B_query, "D": D_query},
    }

    # ZERO_CTRL
    zero_reason = ""
    if not zero_enabled:
        zero_reason = "shot 0 not requested"
    elif A_query is None:
        zero_reason = "missing A query"
    elif B_query is None:
        zero_reason = "missing B query"
    elif D_query is None:
        zero_reason = "missing D query"
    out["ZERO_CTRL"] = {
        "eligible": zero_reason == "",
        "reason": zero_reason or "eligible",
        "A_pool_rows": [],
        "B_pool_rows": [],
        "D_pool_rows": [],
        "queries": {"A": A_query, "B": B_query, "D": D_query},
    }

    # A_ONLY
    a_only_reason = ""
    a_only_pool: List[Dict[str, str]] = []
    if positive_max_shot < 1:
        a_only_reason = "no positive shots requested"
    elif A_query is None:
        a_only_reason = "missing A query"
    else:
        forbidden_A = {(A_query["input"], A_query["output"])}
        a_only_pool = [row for row in A_q if (row["input"], row["output"]) not in forbidden_A]
        if len(a_only_pool) < positive_max_shot:
            a_only_reason = f"filtered demo pool too small (A_pool={len(a_only_pool)})"
    out["A_ONLY"] = {
        "eligible": a_only_reason == "",
        "reason": a_only_reason or "eligible",
        "A_pool_rows": a_only_pool,
        "B_pool_rows": [],
        "D_pool_rows": [],
        "queries": {"A": A_query, "B": B_query, "D": D_query},
    }
    return out


def main() -> int:
    args = _parse_args()
    family_order = parse_family_ids(args.family_ids, allowed=FAMILY_ORDER)
    edge_topk_enabled = bool(args.save_edge_topk)
    if args.edge_topk_k < 1:
        raise ValueError("--edge_topk_k must be >= 1")
    if edge_topk_enabled:
        print("[edge-topk] enabled unified regimes k=" f"{args.edge_topk_k}", flush=True)

    out_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(_resume_dir(out_dir), exist_ok=True)
    os.makedirs(os.path.dirname(args.eligibility_csv) or ".", exist_ok=True)
    edge_topk_jsonl = args.edge_topk_jsonl or os.path.join(out_dir, "pt_unified_edge_topk.jsonl")
    edge_topk_change_csv = args.edge_topk_change_csv or os.path.join(out_dir, "pt_unified_edge_topk_change_summary.csv")

    A_rows = _read_relation_csv(args.relationA_ex_path)
    B_rows = _read_relation_csv(args.relationB_ex_path)
    D_rows = _read_relation_csv(args.relationD_ex_path)
    icl_B_rows = _read_relation_csv(args.icl_B_path)
    icl_D_rows = _read_relation_csv(args.icl_D_path)

    A_by = _group_by_qid(A_rows)
    B_by = _group_by_qid(B_rows)
    D_by = _group_by_qid(D_rows)
    icl_B_by = _group_by_qid(icl_B_rows)
    icl_D_by = _group_by_qid(icl_D_rows)

    requested_qids = _parse_qid_list(args.qid)
    if requested_qids:
        qids = requested_qids
    else:
        qids = sorted(set(A_by) | set(B_by) | set(D_by) | set(icl_B_by) | set(icl_D_by))
    if not qids:
        raise ValueError("No q_id available across unified sources")

    shots = _parse_shot_list(args.shot_list)
    if len(set(shots)) != len(shots):
        raise ValueError("shot_list contains duplicates")
    if any(shot < 0 for shot in shots):
        raise ValueError("shot_list includes negative shot")
    if any(shot > 9 for shot in shots):
        raise ValueError("shot_list includes value > 9")
    positive_shots = [shot for shot in shots if shot > 0]
    zero_enabled = 0 in shots
    positive_max_shot = max(positive_shots) if positive_shots else 0

    selected_targets = None
    if args.selected_targets_json:
        preflight_review_scope(family_ids=family_order, shots=shots)
        selected_targets = load_selected_target_artifact(args.selected_targets_json)
    forced_selected_targets = None
    if args.forced_selected_targets_json:
        preflight_review_scope(family_ids=family_order, shots=shots)
        forced_selected_targets = load_selected_target_artifact(args.forced_selected_targets_json)

    family_ctx_by_q: Dict[str, Dict[str, Dict[str, object]]] = {}
    eligibility_rows: List[Dict[str, object]] = []
    for q_id in qids:
        family_ctx = _build_family_contexts(
            q_id=q_id,
            A_by=A_by,
            B_by=B_by,
            D_by=D_by,
            icl_B_by=icl_B_by,
            icl_D_by=icl_D_by,
            positive_max_shot=positive_max_shot,
            zero_enabled=zero_enabled,
        )
        family_ctx_by_q[q_id] = family_ctx
        for family_id in family_order:
            info = family_ctx[family_id]
            queries = info["queries"]
            eligibility_rows.append(
                {
                    "q_id": q_id,
                    "family_id": family_id,
                    "eligible": int(bool(info["eligible"])),
                    "reason": info["reason"],
                    "A_pool": len(info["A_pool_rows"]),
                    "B_pool": len(info["B_pool_rows"]),
                    "D_pool": len(info["D_pool_rows"]),
                    "A_query": _query_repr(queries.get("A")),
                    "B_query": _query_repr(queries.get("B")),
                    "D_query": _query_repr(queries.get("D")),
                }
            )
    _write_eligibility_csv(args.eligibility_csv, eligibility_rows)

    config_fingerprint = _build_fingerprint(args, family_order=family_order)
    plan_path = _trial_plan_path(out_dir)
    if os.path.exists(plan_path):
        trial_plan = _read_json(plan_path)
        if trial_plan.get("config_fingerprint") != config_fingerprint:
            raise ValueError("Existing unified trial plan fingerprint mismatch for this out_dir")
    else:
        rng = random.Random(args.seed)
        plan_rows: List[Dict[str, object]] = []
        for q_id in qids:
            for family_id in family_order:
                info = family_ctx_by_q[q_id][family_id]
                if not info["eligible"]:
                    continue
                if family_id == "ZERO_CTRL":
                    plan_rows.append({"family_id": family_id, "q_id": q_id, "trial_index": 0})
                    continue
                for trial_index in range(args.n_trials):
                    row = {"family_id": family_id, "q_id": q_id, "trial_index": int(trial_index)}
                    if family_id == "BASE_ABD":
                        A_sample = rng.sample(info["A_pool_rows"], positive_max_shot)
                        B_sample = rng.sample(info["B_pool_rows"], positive_max_shot)
                        rng.shuffle(A_sample)
                        rng.shuffle(B_sample)
                        row["A10_row_ids"] = [int(r["row_id"]) for r in A_sample]
                        row["B10_row_ids"] = [int(r["row_id"]) for r in B_sample]
                    elif family_id == "CTX_ABD":
                        A_sample = rng.sample(info["A_pool_rows"], positive_max_shot)
                        B_sample = rng.sample(info["B_pool_rows"], positive_max_shot)
                        D_sample = rng.sample(info["D_pool_rows"], positive_max_shot)
                        rng.shuffle(A_sample)
                        rng.shuffle(B_sample)
                        rng.shuffle(D_sample)
                        row["A10_row_ids"] = [int(r["row_id"]) for r in A_sample]
                        row["B10_row_ids"] = [int(r["row_id"]) for r in B_sample]
                        row["D10_row_ids"] = [int(r["row_id"]) for r in D_sample]
                    elif family_id == "A_ONLY":
                        A_sample = rng.sample(info["A_pool_rows"], positive_max_shot)
                        rng.shuffle(A_sample)
                        row["A10_row_ids"] = [int(r["row_id"]) for r in A_sample]
                    else:
                        raise ValueError(f"Unexpected family_id for plan build: {family_id}")
                    plan_rows.append(row)
        trial_plan = {"config_fingerprint": config_fingerprint, "plan_rows": plan_rows}
        _write_json(plan_path, trial_plan)

    plan_by_unit: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in trial_plan["plan_rows"]:
        key = (str(row["family_id"]), str(row["q_id"]))
        plan_by_unit.setdefault(key, []).append(row)
    for key in plan_by_unit:
        plan_by_unit[key] = sorted(plan_by_unit[key], key=lambda row: int(row["trial_index"]))

    def expected_count_for_unit(family_id: str, q_id: str) -> int:
        plan_rows = plan_by_unit.get((family_id, q_id), [])
        if family_id == "ZERO_CTRL":
            return len(plan_rows) * (1 if zero_enabled else 0) * len(FAMILY_REGIMES[family_id])
        return len(plan_rows) * len(positive_shots) * len(FAMILY_REGIMES[family_id])

    state_path = _state_path(out_dir)
    if os.path.exists(state_path):
        state = _read_json(state_path)
        if state.get("config_fingerprint") != config_fingerprint:
            raise ValueError("Existing unified resume state fingerprint mismatch for this out_dir")
    else:
        state = {"config_fingerprint": config_fingerprint, "completed_units": [], "created_at": int(time.time())}

    inferred_completed: set = set()
    if os.path.exists(args.out_csv):
        score_counts = _canonical_unit_counts(args.out_csv, ("family_id", "q_id"))
        topk_counts = _canonical_unit_counts(edge_topk_jsonl, ("family_id", "q_id")) if edge_topk_enabled else {}
        for family_id, q_id in plan_by_unit.keys():
            unit = _unit_id(family_id, q_id)
            expected = expected_count_for_unit(family_id, q_id)
            if score_counts.get(unit, 0) >= expected and (not edge_topk_enabled or topk_counts.get(unit, 0) >= expected):
                inferred_completed.add(unit)

    completed_units = set(state.get("completed_units", [])) | inferred_completed
    state["completed_units"] = sorted(completed_units)
    _write_json(state_path, state)

    eligible_units = [(family_id, q_id) for family_id in family_order for q_id in qids if family_ctx_by_q[q_id][family_id]["eligible"]]
    if not eligible_units:
        raise ValueError("No eligible family/q_id units found")

    initial_done_counts: Dict[str, int] = {}
    for family_id, q_id in eligible_units:
        unit = _unit_id(family_id, q_id)
        expected = expected_count_for_unit(family_id, q_id)
        if unit in completed_units:
            initial_done_counts[unit] = expected
        else:
            _, done_count = _completed_keys_from_raw(out_dir, family_id, q_id, edge_topk_enabled=edge_topk_enabled)
            initial_done_counts[unit] = done_count

    total_prompt_evals = sum(expected_count_for_unit(family_id, q_id) for family_id, q_id in eligible_units)
    completed_prompt_evals = sum(initial_done_counts[_unit_id(family_id, q_id)] for family_id, q_id in eligible_units)
    progress_every = max(1, total_prompt_evals // 200) if total_prompt_evals else 1
    scorer_start = time.time()

    spec = get_model_spec(args.model_spec)
    tok_add_special = bool(spec.prepend_bos)
    model, tokenizer, diagnostics = load_hf_model_and_tokenizer(
        model_name=args.model,
        model_spec=args.model_spec,
        device=args.device,
        device_map=None,
        dtype=args.dtype,
        quant=args.quant,
    )
    model.eval()

    print(
        f"[resume] out_dir={out_dir} completed_units={len(completed_units)}/{len(eligible_units)} "
        f"total_prompt_evals={total_prompt_evals}",
        flush=True,
    )

    for q_idx, q_id in enumerate(qids, start=1):
        for family_id in family_order:
            info = family_ctx_by_q[q_id][family_id]
            if not info["eligible"]:
                print(f"[warn] skip family_id={family_id} q_id={q_id}: {info['reason']}", flush=True)
                continue

            unit = _unit_id(family_id, q_id)
            expected_count = expected_count_for_unit(family_id, q_id)
            if unit in completed_units:
                print(f"[unit-skip] family_id={family_id} q_id={q_id} already completed", flush=True)
                continue

            plan_rows = plan_by_unit.get((family_id, q_id), [])
            if not plan_rows:
                print(f"[warn] skip family_id={family_id} q_id={q_id}: no trial plan rows", flush=True)
                continue

            q_rows_path = _unit_rows_path(out_dir, family_id, q_id)
            q_edge_topk_path = _unit_edge_topk_path(out_dir, family_id, q_id)
            q_rows = _dedupe_by_edge_key(_read_jsonl(q_rows_path))
            edge_topk_rows = _dedupe_by_edge_key(_read_jsonl(q_edge_topk_path)) if edge_topk_enabled else []
            row_keys = {(int(row["trial_index"]), int(row["shot"]), str(row["edge"])) for row in q_rows}
            if edge_topk_enabled:
                edge_keys = {(int(row["trial_index"]), int(row["shot"]), str(row["edge"])) for row in edge_topk_rows}
                completed_eval_keys = row_keys & edge_keys
            else:
                completed_eval_keys = row_keys

            print(
                f"[unit-start] family_id={family_id} q_id={q_id} ({q_idx}/{len(qids)}) "
                f"reason=eligible A_pool={len(info['A_pool_rows'])} B_pool={len(info['B_pool_rows'])} D_pool={len(info['D_pool_rows'])}",
                flush=True,
            )
            print(
                f"[unit-resume] family_id={family_id} q_id={q_id} completed_regime_rows={len(completed_eval_keys)}/{expected_count}",
                flush=True,
            )

            row_id_maps = {
                "A": {int(row["row_id"]): row for row in info["A_pool_rows"]},
                "B": {int(row["row_id"]): row for row in info["B_pool_rows"]},
                "D": {int(row["row_id"]): row for row in info["D_pool_rows"]},
                "AONLY": {int(row["row_id"]): row for row in info["A_pool_rows"]},
            }
            query_by_source = info["queries"]
            family_shots = [0] if family_id == "ZERO_CTRL" else positive_shots

            for plan_row in plan_rows:
                trial_index = int(plan_row["trial_index"])
                selected_rows: Dict[str, List[Dict[str, str]]] = {}
                if family_id == "BASE_ABD":
                    selected_rows["A"] = [row_id_maps["A"][int(row_id)] for row_id in plan_row["A10_row_ids"]]
                    selected_rows["B"] = [row_id_maps["B"][int(row_id)] for row_id in plan_row["B10_row_ids"]]
                elif family_id == "CTX_ABD":
                    selected_rows["A"] = [row_id_maps["A"][int(row_id)] for row_id in plan_row["A10_row_ids"]]
                    selected_rows["B"] = [row_id_maps["B"][int(row_id)] for row_id in plan_row["B10_row_ids"]]
                    selected_rows["D"] = [row_id_maps["D"][int(row_id)] for row_id in plan_row["D10_row_ids"]]
                elif family_id == "A_ONLY":
                    selected_rows["AONLY"] = [row_id_maps["AONLY"][int(row_id)] for row_id in plan_row["A10_row_ids"]]

                for shot in family_shots:
                    for regime in FAMILY_REGIMES[family_id]:
                        regime_id = regime["regime_id"]
                        edge_key = (trial_index, int(shot), regime_id)
                        if edge_key in completed_eval_keys:
                            continue

                        query_source = regime["query_source"]
                        query = query_by_source[query_source]
                        if query is None:
                            raise ValueError(f"Missing query for family_id={family_id} regime_id={regime_id} q_id={q_id}")
                        demo_rows = _build_demo_rows(regime, selected_rows=selected_rows, shot=int(shot))
                        for demo in demo_rows:
                            if demo["input"] == query["input"] and demo["output"] == query["output"]:
                                raise ValueError(
                                    f"Query overlaps with demo for family_id={family_id} q_id={q_id} regime={regime_id} "
                                    f"trial={trial_index} shot={shot}"
                                )
                        demo_pairs = [(row["input"], row["output"]) for row in demo_rows]
                        gold_target_str = query["output"]
                        scoring_basis = "gold_target"
                        scored_target_str = gold_target_str
                        scored_target_canonical = ""
                        target_resolution_status = "gold_target_default"
                        forced_selected_record = None
                        if selected_targets is not None:
                            selected_record = resolve_selected_target(
                                selected_targets,
                                q_id=q_id,
                                query_source=query_source,
                                query_input=query["input"],
                                gold_target=gold_target_str,
                            )
                            scoring_basis = "selected_target"
                            scored_target_str = selected_record.selected_target
                            scored_target_canonical = selected_record.selected_target_canonical
                            target_resolution_status = "selected_target_resolved"
                        if forced_selected_targets is not None:
                            forced_selected_record = resolve_selected_target(
                                forced_selected_targets,
                                q_id=q_id,
                                query_source=query_source,
                                query_input=query["input"],
                                gold_target=gold_target_str,
                            )

                        query_pair = (query["input"], scored_target_str)
                        prefix_str, full_str = build_prompt_qa(
                            demo_pairs,
                            query_pair,
                            prepend_bos_token=False,
                            prepend_space=True,
                        )
                        if not full_str.startswith(prefix_str):
                            raise ValueError(
                                "Full prompt does not start with prefix: "
                                f"family_id={family_id} q_id={q_id} regime={regime_id} shot={shot} trial={trial_index}"
                            )
                        target_suffix_str = full_str[len(prefix_str):]
                        inputs = tokenizer(prefix_str, return_tensors="pt", add_special_tokens=tok_add_special)
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = model(**inputs)
                        next_logits = outputs.logits[0, -1, :]
                        next_logprobs = torch.log_softmax(next_logits, dim=-1)
                        prefix_ids_a = inputs["input_ids"][0].tolist()
                        target_id = _target_first_token_id_with_checks(
                            tokenizer,
                            prefix_str,
                            full_str,
                            tok_add_special,
                            prefix_ids_a,
                            q_id=q_id,
                            edge=regime_id,
                            shot=int(shot),
                            trial_index=trial_index,
                            spec_prepend_bos=spec.prepend_bos,
                        )
                        target_logit = float(next_logits[target_id].item())
                        target_logprob = float(next_logprobs[target_id].item())
                        target_prob = float(torch.exp(next_logprobs[target_id]).item())
                        forced_selected_payload: Dict[str, object] = {}
                        if forced_selected_record is not None:
                            forced_selected_str = forced_selected_record.selected_target
                            forced_full_str = f"{prefix_str} {forced_selected_str}"
                            forced_target_id = _target_first_token_id_with_checks(
                                tokenizer,
                                prefix_str,
                                forced_full_str,
                                tok_add_special,
                                prefix_ids_a,
                                q_id=q_id,
                                edge=regime_id,
                                shot=int(shot),
                                trial_index=trial_index,
                                spec_prepend_bos=spec.prepend_bos,
                            )
                            forced_selected_payload = {
                                "forced_selected_target_str": forced_selected_str,
                                "forced_selected_target_canonical": forced_selected_record.selected_target_canonical,
                                "forced_selected_target_artifact": str(forced_selected_targets.path),
                                "forced_selected_target_first_token_id": int(forced_target_id),
                                "forced_selected_target_token_str": tokenizer.decode([forced_target_id]),
                                "forced_selected_target_logit": float(next_logits[forced_target_id].item()),
                                "forced_selected_target_logprob_raw": float(next_logprobs[forced_target_id].item()),
                                "forced_selected_target_prob_raw": float(torch.exp(next_logprobs[forced_target_id]).item()),
                                "forced_selected_target_rank_in_vocab": _target_rank_in_vocab(next_logits, forced_target_id),
                                "forced_selected_target_resolution_status": "forced_selected_target_resolved",
                            }
                        if regime["mode"] == "alternating":
                            demo_pattern = _pattern_label(regime["first_source"], regime["second_source"], int(shot), regime["mode"])
                            demo_source = f"{regime['first_source']}/{regime['second_source']}"
                        elif regime["mode"] == "prefix":
                            demo_pattern = _pattern_label(regime["source"][0], None, int(shot), regime["mode"])
                            demo_source = regime["source"]
                        else:
                            demo_pattern = _pattern_label("ZERO", None, int(shot), regime["mode"])
                            demo_source = "ZERO"

                        row = {
                            "family_id": family_id,
                            "q_id": q_id,
                            "trial_index": trial_index,
                            "shot": int(shot),
                            "edge": regime_id,
                            "regime_id": regime_id,
                            "edge_group": regime["edge_group"],
                            "prompt_family": "unified_drift_control",
                            "demo_pattern": demo_pattern,
                            "query_target_source": query_source,
                            "seed": args.seed,
                            "model": args.model,
                            "model_spec": args.model_spec,
                            "quant": args.quant,
                            "dtype": args.dtype,
                            "device": args.device,
                            "query_source": query_source,
                            "query_input": query["input"],
                            "gold_target_str": gold_target_str,
                            "target_str": scored_target_str,
                            "scored_target_str": scored_target_str,
                            "scored_target_canonical": scored_target_canonical,
                            "scoring_basis": scoring_basis,
                            "selected_target_artifact": (str(selected_targets.path) if selected_targets is not None else ""),
                            "target_resolution_status": target_resolution_status,
                            "target_suffix_str": target_suffix_str,
                            "query_row_id": query["row_id"],
                            "demo_source": demo_source,
                            "demo_ids_used": json.dumps([int(d["row_id"]) for d in demo_rows]),
                            "demo_row_ids_used": json.dumps([int(d["row_id"]) for d in demo_rows]),
                            "target_first_token_id": target_id,
                            "target_token_str": tokenizer.decode([target_id]),
                            "target_logprob_raw": target_logprob,
                            "target_prob_raw": target_prob,
                            "target_logit": target_logit,
                            "prompt_len_tokens": int(inputs["input_ids"].shape[1]),
                        }
                        q_rows.append(row)
                        _append_jsonl(q_rows_path, row)
                        if edge_topk_enabled:
                            topk_payload = _collect_edge_topk(
                                tokenizer=tokenizer,
                                next_logits=next_logits,
                                next_logprobs=next_logprobs,
                                k=args.edge_topk_k,
                            )
                            if forced_selected_payload:
                                forced_selected_payload["forced_selected_target_in_lexical_topk"] = int(
                                    int(forced_selected_payload["forced_selected_target_first_token_id"])
                                    in {int(x) for x in topk_payload["lexical_candidate_token_ids"]}
                                )
                            edge_row = {
                                "family_id": family_id,
                                "q_id": q_id,
                                "trial_index": trial_index,
                                "shot": int(shot),
                                "edge": regime_id,
                                "regime_id": regime_id,
                                "edge_group": regime["edge_group"],
                                "prompt_family": "unified_drift_control",
                                "query_source": query_source,
                                "query_input": query["input"],
                                "gold_target_str": gold_target_str,
                                "target_str": scored_target_str,
                                "scored_target_str": scored_target_str,
                                "scored_target_canonical": scored_target_canonical,
                                "scoring_basis": scoring_basis,
                                "selected_target_artifact": (str(selected_targets.path) if selected_targets is not None else ""),
                                "target_resolution_status": target_resolution_status,
                                "target_first_token_id": target_id,
                                "target_token_str": tokenizer.decode([target_id]),
                                "target_logprob_raw": target_logprob,
                                "target_prob_raw": target_prob,
                                "target_logit": target_logit,
                                "target_rank_in_vocab": _target_rank_in_vocab(next_logits, target_id),
                                **topk_payload,
                                **forced_selected_payload,
                            }
                            edge_topk_rows.append(edge_row)
                            _append_jsonl(q_edge_topk_path, edge_row)

                        completed_prompt_evals += 1
                        completed_eval_keys.add(edge_key)
                        if (
                            completed_prompt_evals == 1
                            or completed_prompt_evals % progress_every == 0
                            or completed_prompt_evals == total_prompt_evals
                        ):
                            print(
                                _progress_line(
                                    q_id=q_id,
                                    q_idx=q_idx,
                                    q_total=len(qids),
                                    trial_index=trial_index,
                                    n_trials=max(1, args.n_trials),
                                    shot=int(shot),
                                    edge=f"{family_id}:{regime_id}",
                                    done=completed_prompt_evals,
                                    total=total_prompt_evals,
                                    elapsed_sec=time.time() - scorer_start,
                                ),
                                flush=True,
                            )

            q_rows = _dedupe_by_edge_key(q_rows)
            if edge_topk_enabled:
                edge_topk_rows = _dedupe_by_edge_key(edge_topk_rows)

            raw_logprobs = [float(row["target_logprob_raw"]) for row in q_rows]
            p_low = _percentile(raw_logprobs, 5)
            p_high = _percentile(raw_logprobs, 95)
            if p_high == p_low:
                for row in q_rows:
                    row["target_s_norm"] = 0.5
            else:
                for row in q_rows:
                    x = row["target_logprob_raw"]
                    s = (x - p_low) / (p_high - p_low)
                    if s < 0.0:
                        s = 0.0
                    elif s > 1.0:
                        s = 1.0
                    row["target_s_norm"] = s
            for row in q_rows:
                row["norm_p_low"] = p_low
                row["norm_p_high"] = p_high
                row["norm_method"] = "robust_minmax_p05_p95"
                row["norm_scope"] = f"qid_family_{family_id}_all_regimes_all_relevant_shots"

            if edge_topk_enabled:
                norm_lookup = {
                    (row["family_id"], row["q_id"], int(row["trial_index"]), int(row["shot"]), row["edge"]): row
                    for row in q_rows
                }
                for edge_row in edge_topk_rows:
                    key = (
                        edge_row["family_id"],
                        edge_row["q_id"],
                        int(edge_row["trial_index"]),
                        int(edge_row["shot"]),
                        edge_row["edge"],
                    )
                    norm_row = norm_lookup[key]
                    edge_row["target_s_norm"] = norm_row["target_s_norm"]
                    edge_row["norm_p_low"] = norm_row["norm_p_low"]
                    edge_row["norm_p_high"] = norm_row["norm_p_high"]
                    edge_row["norm_method"] = norm_row["norm_method"]
                    edge_row["norm_scope"] = norm_row["norm_scope"]

            write_header = not os.path.exists(args.out_csv) or os.path.getsize(args.out_csv) == 0
            fieldnames = list(q_rows[0].keys())
            with open(args.out_csv, "a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(q_rows)
                handle.flush()

            if edge_topk_enabled:
                with open(edge_topk_jsonl, "a", encoding="utf-8") as handle:
                    for row in edge_topk_rows:
                        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                    handle.flush()

                grouped_changes: Dict[Tuple[str, str, int, str], List[Dict[str, object]]] = {}
                for row in edge_topk_rows:
                    key = (str(row["family_id"]), str(row["q_id"]), int(row["trial_index"]), str(row["regime_id"]))
                    grouped_changes.setdefault(key, []).append(row)

                change_rows: List[Dict[str, object]] = []
                for (change_family_id, change_q_id, trial_index, regime_id), rows_for_key in sorted(grouped_changes.items()):
                    rows_sorted = sorted(rows_for_key, key=lambda r: int(r["shot"]))
                    for left, right in zip(rows_sorted[:-1], rows_sorted[1:]):
                        change_rows.append(
                            {
                                "family_id": change_family_id,
                                "q_id": change_q_id,
                                "trial_index": trial_index,
                                "regime_id": regime_id,
                                "shot_from": int(left["shot"]),
                                "shot_to": int(right["shot"]),
                                "target_logprob_from": left["target_logprob_raw"],
                                "target_logprob_to": right["target_logprob_raw"],
                                "target_logprob_delta": float(right["target_logprob_raw"]) - float(left["target_logprob_raw"]),
                                "target_s_norm_from": left["target_s_norm"],
                                "target_s_norm_to": right["target_s_norm"],
                                "target_s_norm_delta": float(right["target_s_norm"]) - float(left["target_s_norm"]),
                                "target_rank_from": int(left["target_rank_in_vocab"]),
                                "target_rank_to": int(right["target_rank_in_vocab"]),
                                "target_rank_delta": int(right["target_rank_in_vocab"]) - int(left["target_rank_in_vocab"]),
                                "top1_candidate_from": (left["lexical_candidates"][0] if left["lexical_candidates"] else ""),
                                "top1_candidate_to": (right["lexical_candidates"][0] if right["lexical_candidates"] else ""),
                                "top1_changed": (
                                    (left["lexical_candidates"][0] if left["lexical_candidates"] else "")
                                    != (right["lexical_candidates"][0] if right["lexical_candidates"] else "")
                                ),
                                "lexical_token_id_jaccard": _jaccard(left["lexical_candidate_token_ids"], right["lexical_candidate_token_ids"]),
                                "lexical_text_jaccard": _jaccard(left["lexical_candidates"], right["lexical_candidates"]),
                                "lexical_overlap_count": len(set(left["lexical_candidates"]) & set(right["lexical_candidates"])),
                                "canonical_overlap_count": len(
                                    set(left["lexical_candidate_canonical_forms"]) & set(right["lexical_candidate_canonical_forms"])
                                ),
                            }
                        )
                _append_change_csv(edge_topk_change_csv, change_rows)

            completed_units.add(unit)
            state["completed_units"] = sorted(completed_units)
            _write_json(state_path, state)
            print(
                f"[unit-done] family_id={family_id} q_id={q_id} "
                f"score_rows={len(q_rows)} edge_topk_rows={(len(edge_topk_rows) if edge_topk_enabled else 0)} "
                f"completed_units={len(completed_units)}/{len(eligible_units)}",
                flush=True,
            )

    if not completed_units:
        raise ValueError("No rows generated; check eligibility or input files.")

    with open(args.out_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        final_rows = list(reader)
    s_vals = [float(row["target_s_norm"]) for row in final_rows]
    print(
        "summary: "
        f"units_processed={len(completed_units)} "
        f"s_norm_min={min(s_vals):.4f} s_norm_max={max(s_vals):.4f}",
        flush=True,
    )
    if not all(0.0 <= s <= 1.0 for s in s_vals):
        raise AssertionError("target_s_norm out of [0,1] bounds")

    print(f"eligibility_csv={args.eligibility_csv}", flush=True)
    if edge_topk_enabled:
        print(f"edge_topk_jsonl={edge_topk_jsonl}", flush=True)
        print(f"edge_topk_change_csv={edge_topk_change_csv}", flush=True)
        print(
            f"[edge-topk] saved completed_units={len(completed_units)} "
            f"files=({os.path.basename(edge_topk_jsonl)},{os.path.basename(edge_topk_change_csv)})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
