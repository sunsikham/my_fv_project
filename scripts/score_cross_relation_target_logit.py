#!/usr/bin/env python3
"""
Product-test scorer: 5 edges x shot sweep with q_id-level normalization.
"""

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fv.hf_loader import load_hf_model_and_tokenizer
from fv.model_spec import get_model_spec
from fv.prompting import build_prompt_qa


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score 5-edge ICL prompts with shot sweeps and q_id-level normalization."
        )
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
    parser.add_argument("--icl_B_path", required=True, help="B query CSV")
    parser.add_argument("--icl_C_path", required=True, help="C query CSV")
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
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    return parser.parse_args()


def _resolve_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    col_set = {c.strip(): c for c in fieldnames}
    for cand in candidates:
        if cand in col_set:
            return col_set[cand]
    lower_map = {c.strip().lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _read_relation_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row in {path}")
        col_q = _resolve_column(reader.fieldnames, ["q_id", "q", "id", "qid"])
        col_in = _resolve_column(reader.fieldnames, ["input", "exA", "ex_A", "ex_a"])
        col_out = _resolve_column(reader.fieldnames, ["output", "exB", "ex_B", "ex_b"])
        if col_q is None or col_in is None or col_out is None:
            raise ValueError(
                "CSV missing required columns (q_id/q/id + input/exA + output/exB) "
                f"in {path}"
            )
        rows = []
        for row_idx, row in enumerate(reader):
            q_id = (row.get(col_q) or "").strip()
            inp = (row.get(col_in) or "").strip()
            out = (row.get(col_out) or "").strip()
            if not q_id or not inp or not out:
                continue
            rows.append(
                {
                    "row_id": row_idx,
                    "q_id": q_id,
                    "input": inp,
                    "output": out,
                }
            )
        return rows


def _group_by_qid(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["q_id"], []).append(row)
    return grouped


def _select_query(rows: List[Dict[str, str]]) -> Dict[str, str]:
    sorted_rows = sorted(rows, key=lambda x: (x["input"], x["output"]))
    return sorted_rows[0]


def _target_first_token_id_with_checks(
    tokenizer,
    prefix_str: str,
    full_str: str,
    tok_add_special: bool,
    prefix_ids_a: List[int],
    q_id: str,
    edge: str,
    shot: int,
    trial_index: int,
    spec_prepend_bos: Optional[bool] = None,
) -> int:
    prefix_ids_b = tokenizer.encode(prefix_str, add_special_tokens=tok_add_special)
    if prefix_ids_a != prefix_ids_b:
        tail = repr(prefix_str[-120:])
        raise ValueError(
            "Prefix token mismatch between model input and target-id calc: "
            f"q_id={q_id} edge={edge} shot={shot} trial={trial_index} "
            f"prefix_tail={tail} tok_add_special={tok_add_special} "
            f"spec_prepend_bos={spec_prepend_bos} "
            f"prefix_ids_a_tail={prefix_ids_a[-20:]} prefix_ids_b_tail={prefix_ids_b[-20:]}"
        )

    full_ids = tokenizer.encode(full_str, add_special_tokens=tok_add_special)
    if len(full_ids) <= len(prefix_ids_b):
        raise ValueError("Tokenization does not extend prefix for target.")
    if full_ids[: len(prefix_ids_b)] != prefix_ids_b:
        window = full_ids[max(0, len(prefix_ids_b) - 10) : len(prefix_ids_b) + 10]
        raise ValueError(
            "Prefix-invariance failed for full_ids prefix: "
            f"q_id={q_id} edge={edge} shot={shot} trial={trial_index} "
            f"prefix_tail={repr(prefix_str[-120:])} "
            f"target_suffix_str={repr(full_str[len(prefix_str):])} "
            f"tok_add_special={tok_add_special} spec_prepend_bos={spec_prepend_bos} "
            f"prefix_ids_tail={prefix_ids_b[-20:]} full_ids_win={window}"
        )
    return int(full_ids[len(prefix_ids_b)])


def _parse_shot_list(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    shots = [int(p) for p in parts]
    if not shots:
        raise ValueError("--shot_list is empty")
    return shots


def _percentile(values: List[float], pct: float) -> float:
    arr = np.array(values, dtype=float)
    return float(np.percentile(arr, pct))


def main() -> int:
    args = _parse_args()
    rng = random.Random(args.seed)

    A_rows = _read_relation_csv(args.relationA_ex_path)
    B_rows = _read_relation_csv(args.relationB_ex_path)
    icl_B_rows = _read_relation_csv(args.icl_B_path)
    icl_C_rows = _read_relation_csv(args.icl_C_path)
    icl_D_rows = _read_relation_csv(args.icl_D_path)

    A_by = _group_by_qid(A_rows)
    B_by = _group_by_qid(B_rows)
    icl_B_by = _group_by_qid(icl_B_rows)
    icl_C_by = _group_by_qid(icl_C_rows)
    icl_D_by = _group_by_qid(icl_D_rows)

    if args.qid:
        qids = [args.qid]
    else:
        qids = sorted(set(icl_B_by) & set(icl_C_by) & set(icl_D_by))

    if not qids:
        raise ValueError("No q_id available after intersection")

    shots = _parse_shot_list(args.shot_list)
    if max(shots) > 10:
        raise ValueError("shot_list includes value > 10")

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

    out_rows: List[Dict[str, object]] = []
    skipped_qids = 0

    for q_idx, q_id in enumerate(qids, start=1):
        A_q = A_by.get(q_id, [])
        B_q = B_by.get(q_id, [])
        icl_B_q = icl_B_by.get(q_id, [])
        icl_C_q = icl_C_by.get(q_id, [])
        icl_D_q = icl_D_by.get(q_id, [])

        if (
            len(A_q) < 10
            or len(B_q) < 10
            or len(icl_B_q) < 1
            or len(icl_C_q) < 1
            or len(icl_D_q) < 1
        ):
            skipped_qids += 1
            continue

        B_query = _select_query(icl_B_q)
        C_query = _select_query(icl_C_q)
        D_query = _select_query(icl_D_q)

        q_rows: List[Dict[str, object]] = []
        raw_logprobs: List[float] = []

        for trial_index in range(args.n_trials):
            A10 = rng.sample(A_q, 10)
            rng.shuffle(A10)
            B10 = rng.sample(B_q, 10)
            rng.shuffle(B10)

            for shot in shots:
                demos_A = A10[:shot]
                demos_B = B10[:shot]

                edges = [
                    ("AB", demos_A, B_query, "A", "B"),
                    ("AC", demos_A, C_query, "A", "C"),
                    ("AD", demos_A, D_query, "A", "D"),
                    ("BC", demos_B, C_query, "B", "C"),
                    ("BD", demos_B, D_query, "B", "D"),
                ]

                for edge, demos, query, demo_src, q_src in edges:
                    for demo in demos:
                        if demo["input"] == query["input"] and demo["output"] == query["output"]:
                            raise ValueError(
                                f"Query overlaps with demo for q_id={q_id} edge={edge} "
                                f"trial={trial_index} shot={shot} query_source={q_src}"
                            )
                    demo_pairs = [(d["input"], d["output"]) for d in demos]
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
                            f"q_id={q_id} edge={edge} shot={shot} trial={trial_index} "
                            f"prefix_tail={repr(prefix_str[-120:])}"
                        )
                    target_suffix_str = full_str[len(prefix_str) :]
                    inputs = tokenizer(
                        prefix_str, return_tensors="pt", add_special_tokens=tok_add_special
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                    next_logits = outputs.logits[0, -1, :]
                    prefix_ids_a = inputs["input_ids"][0].tolist()
                    target_id = _target_first_token_id_with_checks(
                        tokenizer,
                        prefix_str,
                        full_str,
                        tok_add_special,
                        prefix_ids_a,
                        q_id=q_id,
                        edge=edge,
                        shot=shot,
                        trial_index=trial_index,
                        spec_prepend_bos=spec.prepend_bos,
                    )
                    target_logit = float(next_logits[target_id].item())
                    target_logprob = float(
                        torch.log_softmax(next_logits, dim=-1)[target_id].item()
                    )
                    prompt_len_tokens = int(inputs["input_ids"].shape[1])
                    target_token_str = tokenizer.decode([target_id])

                    row = {
                        "q_id": q_id,
                        "trial_index": trial_index,
                        "shot": shot,
                        "edge": edge,
                        "seed": args.seed,
                        "model": args.model,
                        "model_spec": args.model_spec,
                        "quant": args.quant,
                        "dtype": args.dtype,
                        "device": args.device,
                        "query_source": q_src,
                        "query_input": query["input"],
                        "target_str": query["output"],
                        "target_suffix_str": target_suffix_str,
                        "query_row_id": query["row_id"],
                        "demo_source": demo_src,
                        "demo_ids_10": json.dumps([d["row_id"] for d in (A10 if demo_src == "A" else B10)]),
                        "demo_ids_used": json.dumps([d["row_id"] for d in demos]),
                        "target_first_token_id": target_id,
                        "target_token_str": target_token_str,
                        "target_logprob_raw": target_logprob,
                        "target_logit": target_logit,
                        "prompt_len_tokens": prompt_len_tokens,
                    }
                    q_rows.append(row)
                    raw_logprobs.append(target_logprob)

            if trial_index % max(1, args.n_trials // 10) == 0:
                print(
                    f"qid={q_id} trial={trial_index}/{args.n_trials} "
                    f"(qid {q_idx}/{len(qids)})"
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
            row["norm_scope"] = "qid_all_edges_all_shots_this_run"

        out_rows.extend(q_rows)

    if not out_rows:
        raise ValueError("No rows generated; check q_id filters or input files.")

    out_dir = os.path.dirname(args.out_csv) or "."
    if out_dir == "results":
        out_dir = os.path.join(out_dir, "pt_analysis")
        args.out_csv = os.path.join(out_dir, os.path.basename(args.out_csv))
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = list(out_rows[0].keys())
    with open(args.out_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    s_vals = [row["target_s_norm"] for row in out_rows]
    print(
        "summary: "
        f"qids_processed={len(qids) - skipped_qids} skipped={skipped_qids} "
        f"s_norm_min={min(s_vals):.4f} s_norm_max={max(s_vals):.4f}"
    )
    if not all(0.0 <= s <= 1.0 for s in s_vals):
        raise AssertionError("target_s_norm out of [0,1] bounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
