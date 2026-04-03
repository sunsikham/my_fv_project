#!/usr/bin/env python3
"""
Mixed-context PT scorer: alternating demo regimes that induce relation drift.
"""

import argparse
import csv
import json
import os
import time
import random
from typing import Dict, List, Sequence, Tuple

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in os.sys.path:
    os.sys.path.insert(0, _ROOT)

from fv.hf_loader import load_hf_model_and_tokenizer
from fv.model_spec import get_model_spec
from fv.prompting import build_prompt_qa
from scripts.score_cross_relation_target_logit import (
    _append_jsonl,
    _build_config_fingerprint,
    _collect_edge_topk,
    _dedupe_by_edge_key,
    _jaccard,
    _file_sha256,
    _group_by_qid,
    _parse_shot_list,
    _percentile,
    _progress_line,
    _qid_edge_topk_raw_rows_path,
    _qid_raw_rows_path,
    _read_json,
    _read_jsonl,
    _read_relation_csv,
    _resume_dir,
    _resume_state_path,
    _select_query,
    _stable_hash,
    _target_first_token_id_with_checks,
    _target_rank_in_vocab,
    _trial_plan_path,
    _write_json,
)


PT_REGIMES = (
    ("ABABAB_B", "AB", "A", "B", "B"),
    ("ADADAD_D", "AD", "A", "D", "D"),
    ("BDBDBD_D", "BD", "B", "D", "D"),
)
PROBE_REGIMES = (
    ("ABABAB_A", "AB", "A", "B", "A"),
    ("ADADAD_A", "AD", "A", "D", "A"),
    ("BDBDBD_B", "BD", "B", "D", "B"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score mixed-context PT prompts with alternating relation demos."
    )
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--model_spec", required=True, help="Model spec (e.g. llama3)")
    parser.add_argument("--device", required=True, help="Device (cpu/cuda)")
    parser.add_argument("--dtype", required=False, default=None, help="fp32/fp16/bf16")
    parser.add_argument(
        "--quant",
        required=False,
        default="none",
        help="Quantization mode: none/4bit/8bit/auto",
    )
    parser.add_argument("--relationA_ex_path", required=True, help="A demos CSV")
    parser.add_argument("--relationB_ex_path", required=True, help="B demos CSV")
    parser.add_argument("--relationD_ex_path", required=True, help="D demos CSV")
    parser.add_argument("--icl_B_path", required=True, help="B query CSV")
    parser.add_argument("--icl_D_path", required=True, help="D query CSV")
    parser.add_argument(
        "--shot_list",
        required=False,
        default="1,3,5,7,10",
        help="Comma-separated shot list (default: 1,3,5,7,10)",
    )
    parser.add_argument("--n_trials", type=int, required=True, help="Trials per q_id")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--qid", required=False, default=None, help="Optional q_id")
    parser.add_argument(
        "--regime_mode",
        choices=["pt", "candidate_probe"],
        default="pt",
        help="Regime set: standard PT or candidate-probe variant.",
    )
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument(
        "--save_edge_topk",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, save lexical top-k trace for selected regimes.",
    )
    parser.add_argument(
        "--edge_topk_k",
        type=int,
        default=10,
        help="Lexical top-k size for regime trace (default: 10).",
    )
    parser.add_argument(
        "--edge_topk_jsonl",
        default=None,
        help="Optional output path for raw regime lexical top-k JSONL.",
    )
    parser.add_argument(
        "--edge_topk_change_csv",
        default=None,
        help="Optional output path for adjacent-shot regime change summary CSV.",
    )
    return parser.parse_args()


def _build_context_drift_fingerprint(args: argparse.Namespace) -> str:
    regimes = _get_regimes(args.regime_mode)
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
        "qid": args.qid,
        "regime_mode": args.regime_mode,
        "save_edge_topk": int(args.save_edge_topk),
        "edge_topk_k": int(args.edge_topk_k),
        "regimes": [regime_id for regime_id, *_rest in regimes],
        "scorer_code_sha256": _file_sha256(__file__),
    }
    return _stable_hash(payload)


def _get_regimes(mode: str):
    if mode == "pt":
        return PT_REGIMES
    if mode == "candidate_probe":
        return PROBE_REGIMES
    raise ValueError(f"Unsupported regime_mode: {mode}")


def _context_resume_state_path(out_dir: str) -> str:
    return os.path.join(_resume_dir(out_dir), "pt_context_drift_resume_state.json")


def _context_trial_plan_path(out_dir: str) -> str:
    return os.path.join(_resume_dir(out_dir), "pt_context_drift_trial_plan.json")


def _context_qid_raw_rows_path(out_dir: str, q_id: str) -> str:
    return os.path.join(_resume_dir(out_dir), f"context_drift_raw_rows_{q_id}.jsonl")


def _context_qid_edge_topk_raw_rows_path(out_dir: str, q_id: str) -> str:
    return os.path.join(_resume_dir(out_dir), f"context_drift_raw_edge_topk_{q_id}.jsonl")


def _row_key(row: Dict[str, object]) -> Tuple[int, int, str]:
    return (int(row["trial_index"]), int(row["shot"]), str(row["regime_id"]))


def _dedupe_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    latest: Dict[Tuple[int, int, str], Dict[str, object]] = {}
    for row in rows:
        latest[_row_key(row)] = row
    return [latest[key] for key in sorted(latest.keys())]


def _build_pattern_label(start_src: str, other_src: str, shot: int) -> str:
    seq: List[str] = []
    for pos in range(shot):
        if pos % 2 == 0:
            seq.append(start_src)
        else:
            seq.append(other_src)
    return "".join(seq)


def _alternating_demo_rows(
    source_rows: Dict[str, List[Dict[str, str]]],
    *,
    selected_rows: Dict[str, List[Dict[str, str]]],
    shot: int,
    first_source: str,
    second_source: str,
) -> List[Dict[str, str]]:
    demos: List[Dict[str, str]] = []
    source_indices = {src: 0 for src in selected_rows.keys()}
    for pos in range(shot):
        src = first_source if pos % 2 == 0 else second_source
        idx = source_indices[src]
        demos.append(selected_rows[src][idx])
        source_indices[src] += 1
    return demos


def _regime_query(query_by_source: Dict[str, Dict[str, str]], query_target_source: str) -> Dict[str, str]:
    return query_by_source[query_target_source]


def _append_context_change_csv(path: str, rows: List[Dict[str, object]]) -> None:
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    fieldnames = list(rows[0].keys()) if rows else [
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


def main() -> int:
    args = _parse_args()
    regimes = _get_regimes(args.regime_mode)
    edge_topk_enabled = bool(args.save_edge_topk)
    if args.edge_topk_k < 1:
        raise ValueError("--edge_topk_k must be >= 1")
    if edge_topk_enabled:
        print(
            "[edge-topk] enabled "
            f"regimes={','.join(regime_id for regime_id, *_rest in regimes)} "
            f"k={args.edge_topk_k}",
            flush=True,
        )

    out_dir = os.path.dirname(args.out_csv) or "."
    if out_dir == "results":
        out_dir = os.path.join(out_dir, "pt_analysis")
        args.out_csv = os.path.join(out_dir, os.path.basename(args.out_csv))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(_resume_dir(out_dir), exist_ok=True)
    edge_topk_jsonl = (
        args.edge_topk_jsonl
        if args.edge_topk_jsonl
        else os.path.join(out_dir, "pt_context_drift_edge_topk.jsonl")
    )
    edge_topk_change_csv = (
        args.edge_topk_change_csv
        if args.edge_topk_change_csv
        else os.path.join(out_dir, "pt_context_drift_edge_topk_change_summary.csv")
    )

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

    if args.qid:
        qids = [args.qid]
    else:
        if args.regime_mode == "pt":
            qids = sorted(set(A_by) & set(B_by) & set(D_by) & set(icl_B_by) & set(icl_D_by))
        else:
            qids = sorted(set(A_by) & set(B_by) & set(D_by) & set(icl_B_by))
    if not qids:
        raise ValueError("No q_id available after intersection")

    shots = _parse_shot_list(args.shot_list)
    if max(shots) > 10:
        raise ValueError("shot_list includes value > 10")

    config_fingerprint = _build_context_drift_fingerprint(args)
    state_path = _context_resume_state_path(out_dir)
    plan_path = _context_trial_plan_path(out_dir)
    if os.path.exists(state_path):
        state = _read_json(state_path)
        if state.get("config_fingerprint") != config_fingerprint:
            raise ValueError("Existing context-drift resume state fingerprint mismatch for this out_dir")
    else:
        inferred_completed_qids: List[str] = []
        if os.path.exists(args.out_csv):
            with open(args.out_csv, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                inferred_completed_qids = sorted({str(row["q_id"]) for row in reader if row.get("q_id")})
        state = {
            "config_fingerprint": config_fingerprint,
            "completed_qids": inferred_completed_qids,
            "created_at": int(time.time()),
        }
        _write_json(state_path, state)
    if os.path.exists(args.out_csv):
        with open(args.out_csv, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_completed_qids = sorted({str(row["q_id"]) for row in reader if row.get("q_id")})
        merged = sorted(set(state.get("completed_qids", [])) | set(existing_completed_qids))
        if merged != state.get("completed_qids", []):
            state["completed_qids"] = merged
            _write_json(state_path, state)

    if os.path.exists(plan_path):
        trial_plan = _read_json(plan_path)
        if trial_plan.get("config_fingerprint") != config_fingerprint:
            raise ValueError("Existing context-drift trial plan fingerprint mismatch for this out_dir")
    else:
        rng = random.Random(args.seed)
        plan_rows: List[Dict[str, object]] = []
        for q_id in qids:
            A_q = A_by.get(q_id, [])
            B_q = B_by.get(q_id, [])
            D_q = D_by.get(q_id, [])
            icl_B_q = icl_B_by.get(q_id, [])
            icl_D_q = icl_D_by.get(q_id, [])
            if args.regime_mode == "pt":
                if len(A_q) < 10 or len(B_q) < 10 or len(D_q) < 10 or len(icl_B_q) < 1 or len(icl_D_q) < 1:
                    continue
                B_query = _select_query(icl_B_q)
                D_query = _select_query(icl_D_q)
                forbidden_A = {(B_query["input"], B_query["output"]), (D_query["input"], D_query["output"])}
                forbidden_B = {(B_query["input"], B_query["output"]), (D_query["input"], D_query["output"])}
                forbidden_D = {(B_query["input"], B_query["output"]), (D_query["input"], D_query["output"])}
            else:
                if len(A_q) < 10 or len(B_q) < 10 or len(D_q) < 10 or len(icl_B_q) < 1:
                    continue
                A_query = _select_query(A_q)
                B_query = _select_query(icl_B_q)
                forbidden_A = {(A_query["input"], A_query["output"]), (B_query["input"], B_query["output"])}
                forbidden_B = {(B_query["input"], B_query["output"])}
                forbidden_D = {(A_query["input"], A_query["output"]), (B_query["input"], B_query["output"])}
            A_pool = [row for row in A_q if (row["input"], row["output"]) not in forbidden_A]
            B_pool = [row for row in B_q if (row["input"], row["output"]) not in forbidden_B]
            D_pool = [row for row in D_q if (row["input"], row["output"]) not in forbidden_D]
            if len(A_pool) < 10 or len(B_pool) < 10 or len(D_pool) < 10:
                continue
            for trial_index in range(args.n_trials):
                A10 = rng.sample(A_pool, 10)
                rng.shuffle(A10)
                B10 = rng.sample(B_pool, 10)
                rng.shuffle(B10)
                D10 = rng.sample(D_pool, 10)
                rng.shuffle(D10)
                plan_rows.append(
                    {
                        "q_id": q_id,
                        "trial_index": int(trial_index),
                        "A10_row_ids": [int(row["row_id"]) for row in A10],
                        "B10_row_ids": [int(row["row_id"]) for row in B10],
                        "D10_row_ids": [int(row["row_id"]) for row in D10],
                    }
                )
        trial_plan = {"config_fingerprint": config_fingerprint, "plan_rows": plan_rows}
        _write_json(plan_path, trial_plan)

    plan_by_q: Dict[str, List[Dict[str, object]]] = {}
    for row in trial_plan["plan_rows"]:
        plan_by_q.setdefault(str(row["q_id"]), []).append(row)
    for q_id in plan_by_q:
        plan_by_q[q_id] = sorted(plan_by_q[q_id], key=lambda row: int(row["trial_index"]))

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

    skipped_qids = 0
    completed_qids = set(state.get("completed_qids", []))
    eligible_qids = [q_id for q_id in qids if q_id in plan_by_q]
    total_prompt_evals = sum(len(plan_by_q[q_id]) * len(shots) * len(regimes) for q_id in eligible_qids)
    completed_prompt_evals = sum(len(plan_by_q[q_id]) * len(shots) * len(regimes) for q_id in completed_qids if q_id in plan_by_q)
    progress_every = max(1, total_prompt_evals // 200) if total_prompt_evals else 1
    scorer_start = time.time()

    print(
        f"[resume] out_dir={out_dir} completed_qids={len(completed_qids)}/{len(eligible_qids)} "
        f"total_prompt_evals={total_prompt_evals}",
        flush=True,
    )

    for q_idx, q_id in enumerate(qids, start=1):
        A_q = A_by.get(q_id, [])
        B_q = B_by.get(q_id, [])
        D_q = D_by.get(q_id, [])
        icl_B_q = icl_B_by.get(q_id, [])
        icl_D_q = icl_D_by.get(q_id, [])

        if args.regime_mode == "pt":
            if len(A_q) < 10 or len(B_q) < 10 or len(D_q) < 10 or len(icl_B_q) < 1 or len(icl_D_q) < 1:
                skipped_qids += 1
                continue
            B_query = _select_query(icl_B_q)
            D_query = _select_query(icl_D_q)
            forbidden_A = {(B_query["input"], B_query["output"]), (D_query["input"], D_query["output"])}
            forbidden_B = {(B_query["input"], B_query["output"]), (D_query["input"], D_query["output"])}
            forbidden_D = {(B_query["input"], B_query["output"]), (D_query["input"], D_query["output"])}
            query_by_source = {"B": B_query, "D": D_query}
        else:
            if len(A_q) < 10 or len(B_q) < 10 or len(D_q) < 10 or len(icl_B_q) < 1:
                skipped_qids += 1
                continue
            A_query = _select_query(A_q)
            B_query = _select_query(icl_B_q)
            forbidden_A = {(A_query["input"], A_query["output"]), (B_query["input"], B_query["output"])}
            forbidden_B = {(B_query["input"], B_query["output"])}
            forbidden_D = {(A_query["input"], A_query["output"]), (B_query["input"], B_query["output"])}
            query_by_source = {"A": A_query, "B": B_query}
        A_pool = [row for row in A_q if (row["input"], row["output"]) not in forbidden_A]
        B_pool = [row for row in B_q if (row["input"], row["output"]) not in forbidden_B]
        D_pool = [row for row in D_q if (row["input"], row["output"]) not in forbidden_D]
        if len(A_pool) < 10 or len(B_pool) < 10 or len(D_pool) < 10:
            skipped_qids += 1
            print(
                f"[warn] skip q_id={q_id}: filtered demo pool too small "
                f"(A_pool={len(A_pool)}, B_pool={len(B_pool)}, D_pool={len(D_pool)})",
                flush=True,
            )
            continue

        if q_id not in plan_by_q:
            skipped_qids += 1
            print(f"[warn] skip q_id={q_id}: no trial plan rows", flush=True)
            continue

        if q_id in completed_qids:
            print(f"[qid-skip] q_id={q_id} already completed", flush=True)
            continue

        print(
            f"[qid-start] q_id={q_id} "
            f"({q_idx}/{len(qids)}) "
            f"A_pool={len(A_pool)} B_pool={len(B_pool)} D_pool={len(D_pool)} "
            + (
                f"B_query=({B_query['input']}->{B_query['output']}) D_query=({D_query['input']}->{D_query['output']})"
                if args.regime_mode == "pt"
                else f"A_query=({A_query['input']}->{A_query['output']}) B_query=({B_query['input']}->{B_query['output']})"
            ),
            flush=True,
        )

        row_id_to_A = {int(row["row_id"]): row for row in A_pool}
        row_id_to_B = {int(row["row_id"]): row for row in B_pool}
        row_id_to_D = {int(row["row_id"]): row for row in D_pool}
        q_plan_rows = plan_by_q[q_id]
        q_rows_path = _context_qid_raw_rows_path(out_dir, q_id)
        q_edge_topk_path = _context_qid_edge_topk_raw_rows_path(out_dir, q_id)
        q_rows: List[Dict[str, object]] = _dedupe_rows(_read_jsonl(q_rows_path))
        edge_topk_rows: List[Dict[str, object]] = _dedupe_rows(_read_jsonl(q_edge_topk_path)) if edge_topk_enabled else []
        completed_eval_keys = {_row_key(row) for row in q_rows}
        raw_logprobs: List[float] = [float(row["target_logprob_raw"]) for row in q_rows]
        q_edge_topk_count = len(edge_topk_rows)

        print(
            f"[qid-resume] q_id={q_id} completed_regime_rows={len(completed_eval_keys)}/{len(q_plan_rows) * len(shots) * len(regimes)}",
            flush=True,
        )
        for plan_row in q_plan_rows:
            trial_index = int(plan_row["trial_index"])
            selected_rows = {
                "A": [row_id_to_A[int(row_id)] for row_id in plan_row["A10_row_ids"]],
                "B": [row_id_to_B[int(row_id)] for row_id in plan_row["B10_row_ids"]],
                "D": [row_id_to_D[int(row_id)] for row_id in plan_row["D10_row_ids"]],
            }

            for shot in shots:
                for regime_id, edge_group, first_source, second_source, query_target_source in regimes:
                    regime_key = (trial_index, int(shot), regime_id)
                    if regime_key in completed_eval_keys:
                        continue
                    demo_rows = _alternating_demo_rows(
                        source_rows={},
                        selected_rows=selected_rows,
                        shot=int(shot),
                        first_source=first_source,
                        second_source=second_source,
                    )
                    query = _regime_query(query_by_source, query_target_source)
                    for demo in demo_rows:
                        if demo["input"] == query["input"] and demo["output"] == query["output"]:
                            raise ValueError(
                                f"Query overlaps with demo for q_id={q_id} regime={regime_id} "
                                f"trial={trial_index} shot={shot}"
                            )
                    demo_pairs = [(d["input"], d["output"]) for d in demo_rows]
                    query_pair = (query["input"], query["output"])
                    prefix_str, full_str = build_prompt_qa(
                        demo_pairs,
                        query_pair,
                        prepend_bos_token=False,
                        prepend_space=True,
                    )
                    if not full_str.startswith(prefix_str):
                        raise ValueError(
                            "Full prompt does not start with prefix: "
                            f"q_id={q_id} regime={regime_id} shot={shot} trial={trial_index} "
                            f"prefix_tail={repr(prefix_str[-120:])}"
                        )
                    target_suffix_str = full_str[len(prefix_str) :]
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
                    target_token_str = tokenizer.decode([target_id])
                    row = {
                        "q_id": q_id,
                        "trial_index": trial_index,
                        "shot": int(shot),
                        "edge": regime_id,
                        "prompt_family": "context_drift",
                        "regime_id": regime_id,
                        "edge_group": edge_group,
                        "demo_pattern": _build_pattern_label(first_source, second_source, int(shot)),
                        "query_target_source": query_target_source,
                        "seed": args.seed,
                        "model": args.model,
                        "model_spec": args.model_spec,
                        "quant": args.quant,
                        "dtype": args.dtype,
                        "device": args.device,
                        "query_source": query_target_source,
                        "query_input": query["input"],
                        "target_str": query["output"],
                        "target_suffix_str": target_suffix_str,
                        "query_row_id": query["row_id"],
                        "demo_source": f"{first_source}/{second_source}",
                        "demo_ids_used": json.dumps([int(d["row_id"]) for d in demo_rows]),
                        "demo_row_ids_used": json.dumps([int(d["row_id"]) for d in demo_rows]),
                        "target_first_token_id": target_id,
                        "target_token_str": target_token_str,
                        "target_logprob_raw": target_logprob,
                        "target_prob_raw": target_prob,
                        "target_logit": target_logit,
                        "prompt_len_tokens": int(inputs["input_ids"].shape[1]),
                    }
                    q_rows.append(row)
                    _append_jsonl(q_rows_path, row)
                    raw_logprobs.append(target_logprob)
                    if edge_topk_enabled:
                        edge_row = {
                            "q_id": q_id,
                            "trial_index": trial_index,
                            "shot": int(shot),
                            "edge": regime_id,
                            "regime_id": regime_id,
                            "edge_group": edge_group,
                            "prompt_family": "context_drift",
                            "query_source": query_target_source,
                            "query_input": query["input"],
                            "target_str": query["output"],
                            "target_first_token_id": target_id,
                            "target_token_str": target_token_str,
                            "target_logprob_raw": target_logprob,
                            "target_prob_raw": target_prob,
                            "target_logit": target_logit,
                            "target_rank_in_vocab": _target_rank_in_vocab(next_logits, target_id),
                            **_collect_edge_topk(
                                tokenizer=tokenizer,
                                next_logits=next_logits,
                                next_logprobs=next_logprobs,
                                k=args.edge_topk_k,
                            ),
                        }
                        edge_topk_rows.append(edge_row)
                        _append_jsonl(q_edge_topk_path, edge_row)
                        q_edge_topk_count += 1
                    completed_eval_keys.add(regime_key)
                    completed_prompt_evals += 1
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
                                n_trials=args.n_trials,
                                shot=int(shot),
                                edge=regime_id,
                                done=completed_prompt_evals,
                                total=total_prompt_evals,
                                elapsed_sec=time.time() - scorer_start,
                            ),
                            flush=True,
                        )

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
            row["norm_scope"] = "qid_all_regimes_all_shots_this_run"

        if edge_topk_enabled:
            norm_lookup = {
                (row["q_id"], int(row["trial_index"]), int(row["shot"]), row["regime_id"]): row
                for row in q_rows
            }
            for edge_row in edge_topk_rows:
                key = (
                    edge_row["q_id"],
                    int(edge_row["trial_index"]),
                    int(edge_row["shot"]),
                    edge_row["regime_id"],
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

            grouped_changes: Dict[Tuple[str, int, str], List[Dict[str, object]]] = {}
            for row in edge_topk_rows:
                key = (str(row["q_id"]), int(row["trial_index"]), str(row["regime_id"]))
                grouped_changes.setdefault(key, []).append(row)

            change_rows: List[Dict[str, object]] = []
            for (q_edge_id, trial_index, regime_id), rows_for_key in sorted(grouped_changes.items()):
                rows_for_key = sorted(rows_for_key, key=lambda r: int(r["shot"]))
                for left, right in zip(rows_for_key[:-1], rows_for_key[1:]):
                    change_rows.append(
                        {
                            "q_id": q_edge_id,
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

            _append_context_change_csv(edge_topk_change_csv, change_rows)

        completed_qids.add(q_id)
        state["completed_qids"] = sorted(completed_qids)
        _write_json(state_path, state)
        print(
            f"[qid-done] q_id={q_id} score_rows={len(q_rows)} edge_topk_rows={q_edge_topk_count} "
            f"completed_qids={len(completed_qids)}/{len(eligible_qids)}",
            flush=True,
        )

    if not completed_qids and skipped_qids == len(qids):
        raise ValueError("No rows generated; check q_id filters or input files.")

    with open(args.out_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        final_rows = list(reader)
    s_vals = [float(row["target_s_norm"]) for row in final_rows]
    print(
        "summary: "
        f"qids_processed={len(completed_qids)} skipped={skipped_qids} "
        f"s_norm_min={min(s_vals):.4f} s_norm_max={max(s_vals):.4f}",
        flush=True,
    )
    if not all(0.0 <= s <= 1.0 for s in s_vals):
        raise AssertionError("target_s_norm out of [0,1] bounds")

    if edge_topk_enabled:
        print(f"edge_topk_jsonl={edge_topk_jsonl}", flush=True)
        print(f"edge_topk_change_csv={edge_topk_change_csv}", flush=True)
        print(
            f"[edge-topk] saved completed_qids={len(completed_qids)} "
            f"files=({os.path.basename(edge_topk_jsonl)},{os.path.basename(edge_topk_change_csv)})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
