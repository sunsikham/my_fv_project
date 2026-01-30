#!/usr/bin/env python3
"""
Debug: measure whether same-relation ICL demos raise target logit.

Example:
  python scripts/debug_same_relation_logit.py \
    --model gpt2 \
    --model_spec gpt2 \
    --device cpu \
    --relation_path datasets/relation/relationA_ex.csv \
    --n_demos 8 \
    --n_trials 50 \
    --out_csv results/debug_relationA_logits.csv
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
        description="Debug same-relation ICL logit lift."
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
        "--relation_path",
        required=True,
        help="CSV path for relation (used for both demos and queries).",
    )
    parser.add_argument("--n_demos", type=int, default=8, help="Demos per prompt")
    parser.add_argument("--n_trials", type=int, default=50, help="Trials to average")
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


def _read_relation_csv(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row in {path}")
        col_in = _resolve_column(reader.fieldnames, ["input", "exA", "ex_A", "ex_a"])
        col_out = _resolve_column(reader.fieldnames, ["output", "exB", "ex_B", "ex_b"])
        if col_in is None or col_out is None:
            raise ValueError(
                "CSV missing required columns (input/exA + output/exB) "
                f"in {path}"
            )
        rows: List[Tuple[str, str]] = []
        for row in reader:
            inp = (row.get(col_in) or "").strip()
            out = (row.get(col_out) or "").strip()
            if not inp or not out:
                continue
            rows.append((inp, out))
        return rows


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


def _score_query(
    model,
    tokenizer,
    device: torch.device,
    tok_add_special: bool,
    demos: List[Tuple[str, str]],
    query: Tuple[str, str],
) -> Tuple[float, float, int, str, int]:
    prefix_str, _full_str = build_prompt_qa(
        demos,
        query,
        prepend_bos_token=False,
        prepend_space=True,
    )
    target_str = query[1]
    target_id = _target_first_token_id(tokenizer, prefix_str, target_str, tok_add_special)
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
    return target_logit, target_logprob, prompt_len_tokens, target_token_str, target_id


def main() -> int:
    args = _parse_args()
    rng = random.Random(args.seed)
    rows = _read_relation_csv(args.relation_path)
    if len(rows) < args.n_demos + 1:
        raise ValueError(
            "Not enough rows for demos + query: "
            f"have={len(rows)} need={args.n_demos + 1}"
        )

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
                "trial_index",
                "query_index",
                "query_input",
                "target_str",
                "target_token_id",
                "target_token_str",
                "n_demos",
                "logit_no_demo",
                "logprob_no_demo",
                "logit_with_demo",
                "logprob_with_demo",
                "delta_logit",
                "delta_logprob",
                "prompt_len_no_demo",
                "prompt_len_with_demo",
                "seed",
                "relation_path",
            ]
        )

        delta_logit_vals: List[float] = []
        delta_logprob_vals: List[float] = []

        for trial_index in range(args.n_trials):
            shuffled = rows[:]
            rng.shuffle(shuffled)
            demos = shuffled[: args.n_demos]
            query = shuffled[args.n_demos]

            logit_no, logprob_no, len_no, tok_str_no, tok_id_no = _score_query(
                model,
                tokenizer,
                device,
                tok_add_special,
                [],
                query,
            )
            logit_with, logprob_with, len_with, tok_str_with, tok_id_with = _score_query(
                model,
                tokenizer,
                device,
                tok_add_special,
                demos,
                query,
            )
            if tok_id_no != tok_id_with:
                raise ValueError(
                    "Target token id mismatch between no-demo and demo prompt."
                )

            delta_logit = logit_with - logit_no
            delta_logprob = logprob_with - logprob_no
            delta_logit_vals.append(delta_logit)
            delta_logprob_vals.append(delta_logprob)

            writer.writerow(
                [
                    trial_index,
                    0,
                    query[0],
                    query[1],
                    tok_id_no,
                    tok_str_no,
                    args.n_demos,
                    logit_no,
                    logprob_no,
                    logit_with,
                    logprob_with,
                    delta_logit,
                    delta_logprob,
                    len_no,
                    len_with,
                    args.seed,
                    args.relation_path,
                ]
            )

        if delta_logit_vals:
            mean_delta_logit = sum(delta_logit_vals) / len(delta_logit_vals)
            mean_delta_logprob = sum(delta_logprob_vals) / len(delta_logprob_vals)
        else:
            mean_delta_logit = 0.0
            mean_delta_logprob = 0.0

    print(f"wrote {args.out_csv}")
    print(f"mean_delta_logit={mean_delta_logit:.6f}")
    print(f"mean_delta_logprob={mean_delta_logprob:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
