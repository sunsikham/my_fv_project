#!/usr/bin/env python3
"""
Example:
  python scripts/score_cross_relation_target_logit.py \
    --model gpt2 \
    --model_spec gpt2 \
    --device cpu \
    --demo_relation_path datasets/relation/relationA_ex.csv \
    --query_relation_path datasets/relation/relationB_ex.csv \
    --qid Q1 \
    --n_demos 3 \
    --n_trials 5 \
    --out_csv results/cross_relation_logits.csv
"""

import argparse
import csv
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fv.hf_loader import load_hf_model_and_tokenizer
from fv.model_spec import get_model_spec
from fv.prompting import build_prompt_qa


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score target first-token logits with cross-relation demos."
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
    parser.add_argument(
        "--demo_relation_path",
        required=True,
        help="CSV path for demo relation",
    )
    parser.add_argument(
        "--query_relation_path",
        required=True,
        help="CSV path for query relation",
    )
    parser.add_argument("--n_demos", type=int, required=False, help="Number of demos")
    parser.add_argument(
        "--n_demos_list",
        required=False,
        default=None,
        help="Comma-separated demo counts (e.g. 1,3,5)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        required=True,
        help="Trials per q_id",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
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
        for row in reader:
            q_id = (row.get(col_q) or "").strip()
            inp = (row.get(col_in) or "").strip()
            out = (row.get(col_out) or "").strip()
            if not q_id or not inp or not out:
                continue
            rows.append({"q_id": q_id, "input": inp, "output": out})
        return rows


def _group_by_qid(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    grouped: Dict[str, List[Tuple[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["q_id"], []).append((row["input"], row["output"]))
    return grouped


def _select_query(pair_list: List[Tuple[str, str]]) -> Tuple[str, str]:
    sorted_pairs = sorted(pair_list, key=lambda x: (x[0], x[1]))
    return sorted_pairs[0]


def _target_first_token_id(
    tokenizer, prefix_str: str, target_str: str, tok_add_special: bool
) -> int:
    boundary_prefix = prefix_str
    boundary_answer = target_str
    if boundary_prefix.endswith(" ") and not boundary_answer.startswith(" "):
        boundary_prefix = boundary_prefix[:-1]
        boundary_answer = f" {boundary_answer}"
    full_ids = tokenizer.encode(
        boundary_prefix + boundary_answer, add_special_tokens=tok_add_special
    )
    prefix_ids = tokenizer.encode(boundary_prefix, add_special_tokens=tok_add_special)
    if len(full_ids) <= len(prefix_ids):
        raise ValueError("Tokenization does not extend prefix for target.")
    return int(full_ids[len(prefix_ids)])


def _parse_demo_counts(args: argparse.Namespace) -> List[int]:
    if args.n_demos_list and args.n_demos is not None:
        raise ValueError("Use either --n_demos or --n_demos_list, not both.")
    if args.n_demos_list:
        parts = [p.strip() for p in args.n_demos_list.split(",") if p.strip()]
        counts = [int(p) for p in parts]
        if not counts:
            raise ValueError("--n_demos_list provided but empty")
        return counts
    if args.n_demos is None:
        raise ValueError("Provide --n_demos or --n_demos_list.")
    return [int(args.n_demos)]


def main() -> int:
    args = _parse_args()
    rng = random.Random(args.seed)

    demo_rows = _read_relation_csv(args.demo_relation_path)
    query_rows = _read_relation_csv(args.query_relation_path)
    demo_by_qid = _group_by_qid(demo_rows)
    query_by_qid = _group_by_qid(query_rows)

    qids = sorted(set(query_by_qid.keys()))
    demo_counts = _parse_demo_counts(args)
    if not qids:
        raise ValueError("No q_id found in query relation file")

    spec = get_model_spec(args.model_spec)
    tok_add_special = bool(spec.prepend_bos)

    model, tokenizer, diagnostics = load_hf_model_and_tokenizer(
        model_name=args.model,
        model_spec=args.model_spec,
        device=args.device,
        dtype=args.dtype,
        quant=args.quant,
        device_map=None,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    device = torch.device(args.device)
    resolved_quant = diagnostics.get("resolved_quant") if diagnostics else None
    if resolved_quant in {"4bit", "8bit"}:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            pass
    else:
        model.to(device)
    model.eval()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "q_id",
                "trial_index",
                "demo_relation_path",
                "query_relation_path",
                "n_demos",
                "query_input",
                "target_str",
                "target_first_token_id",
                "target_token_str",
                "target_logit",
                "target_logprob",
                "prompt_len_tokens",
                "seed",
            ]
        )

        for q_id in qids:
            query_pairs = query_by_qid.get(q_id, [])
            if not query_pairs:
                print(f"skip q_id={q_id}: no query rows")
                continue
            demo_pairs = demo_by_qid.get(q_id, [])

            query = _select_query(query_pairs)
            query_input, target_str = query

            for n_demos in demo_counts:
                if len(demo_pairs) < n_demos:
                    print(
                        f"skip q_id={q_id} n_demos={n_demos}: "
                        f"demo pool {len(demo_pairs)} < {n_demos}"
                    )
                    continue

                for trial_index in range(args.n_trials):
                    demos = rng.sample(demo_pairs, n_demos)
                    rng.shuffle(demos)
                    prefix_str, _full_str = build_prompt_qa(
                        demos,
                        query,
                        prepend_bos_token=False,
                        prepend_space=True,
                    )
                    target_id = _target_first_token_id(
                        tokenizer, prefix_str, target_str, tok_add_special
                    )
                    inputs = tokenizer(
                        prefix_str,
                        return_tensors="pt",
                        add_special_tokens=tok_add_special,
                    ).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    next_logits = outputs.logits[0, -1, :]
                    next_logprobs = torch.log_softmax(next_logits, dim=-1)
                    target_logit = float(next_logits[target_id].item())
                    target_logprob = float(next_logprobs[target_id].item())
                    prompt_len_tokens = int(inputs["input_ids"].shape[-1])
                    target_token_str = tokenizer.decode([target_id])
                    writer.writerow(
                        [
                            q_id,
                            trial_index,
                            args.demo_relation_path,
                            args.query_relation_path,
                            n_demos,
                            query_input,
                            target_str,
                            target_id,
                            target_token_str,
                            target_logit,
                            target_logprob,
                            prompt_len_tokens,
                            args.seed,
                        ]
                    )

                    if trial_index == 0:
                        print("[SMOKE]")
                        print(f"q_id={q_id} n_demos={n_demos}")
                        print(f"query_input={repr(query_input)}")
                        print(f"target_str={repr(target_str)}")
                        print(f"demos_sample={demos[:2]}")
                        print(f"prompt_tail={repr(prefix_str[-150:])}")
                        print(f"target_id={target_id}")
                        print(f"target_token_str={repr(target_token_str)}")
                        print(f"target_logit={target_logit}")
                        print(f"target_logprob={target_logprob}")

    print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
