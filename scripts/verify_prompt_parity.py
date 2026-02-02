#!/usr/bin/env python3
"""Verify fv prompt parity against fixed_trials (string + input_ids)."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from fv.prompting import build_prompt_qa_paper


def _load_fixed(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify prompt string/token parity.")
    parser.add_argument("--fixed_trials_path", required=True)
    parser.add_argument("--max_trials", type=int, default=5)
    parser.add_argument("--model_name_for_tokenizer", default=None)
    args = parser.parse_args()

    fixed = _load_fixed(Path(args.fixed_trials_path))
    trials = fixed.get("trials", [])
    meta = fixed.get("meta", {})
    model_name = args.model_name_for_tokenizer or meta.get("model_name_for_tokenizer")
    if not model_name:
        raise ValueError("model_name_for_tokenizer missing (pass --model_name_for_tokenizer)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prepend_bos_token = bool(meta.get("prepend_bos_token_used", False))
    tok_add_special = bool(meta.get("model_prepend_bos", False))
    prefixes = meta.get("prefixes")
    separators = meta.get("separators")

    mismatches = []
    for t_idx, trial in enumerate(trials[: args.max_trials]):
        demos_clean = trial.get("demos_clean")
        demos_corrupted = trial.get("demos_corrupted")
        query = trial.get("query")
        if demos_clean is None or demos_corrupted is None or query is None:
            raise ValueError("fixed_trials missing demos/query")

        demos_clean_pairs = [(d["input"], d["output"]) for d in demos_clean]
        demos_corrupt_pairs = [(d["input"], d["output"]) for d in demos_corrupted]
        query_pair = (query["input"], query["output"])

        clean_prefix, _clean_full, _ = build_prompt_qa_paper(
            demos_clean_pairs,
            query_pair,
            prefixes=prefixes,
            separators=separators,
            prepend_bos_token=prepend_bos_token,
            prepend_space=True,
        )
        corrupted_prefix, _corrupted_full, _ = build_prompt_qa_paper(
            demos_corrupt_pairs,
            query_pair,
            prefixes=prefixes,
            separators=separators,
            prepend_bos_token=prepend_bos_token,
            prepend_space=True,
        )

        fixed_clean = trial.get("clean_prompt_str")
        fixed_corrupted = trial.get("corrupted_prompt_str")

        if fixed_clean != clean_prefix:
            mismatches.append((t_idx, "clean_prompt_str"))
        if fixed_corrupted != corrupted_prefix:
            mismatches.append((t_idx, "corrupted_prompt_str"))

        clean_ids = tokenizer(
            clean_prefix, return_tensors="pt", add_special_tokens=tok_add_special
        )["input_ids"]
        fixed_clean_ids = tokenizer(
            fixed_clean, return_tensors="pt", add_special_tokens=tok_add_special
        )["input_ids"]
        if clean_ids.shape != fixed_clean_ids.shape or (clean_ids != fixed_clean_ids).any():
            mismatches.append((t_idx, "clean_input_ids"))

        corrupt_ids = tokenizer(
            corrupted_prefix, return_tensors="pt", add_special_tokens=tok_add_special
        )["input_ids"]
        fixed_corrupt_ids = tokenizer(
            fixed_corrupted, return_tensors="pt", add_special_tokens=tok_add_special
        )["input_ids"]
        if corrupt_ids.shape != fixed_corrupt_ids.shape or (corrupt_ids != fixed_corrupt_ids).any():
            mismatches.append((t_idx, "corrupted_input_ids"))

    print("checked_trials:", min(len(trials), args.max_trials))
    print("mismatch_count:", len(mismatches))
    for item in mismatches[:10]:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
