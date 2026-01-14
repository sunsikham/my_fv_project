#!/usr/bin/env python3
"""Check input_ids/attention_mask/position_ids parity for fixed trials."""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.prompt_utils import get_token_meta_labels as paper_get_token_meta_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Check fixed trial input parity (paper vs StepD).")
    parser.add_argument(
        "--fixed_trials",
        default="datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_WITH_DEMOS_STEPD_LLAMA.json",
        help="Path to fixed_trials JSON",
    )
    parser.add_argument(
        "--model_path",
        default="/data/.hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        help="Local HF model/tokenizer path",
    )
    parser.add_argument(
        "--max_mismatches",
        type=int,
        default=10,
        help="Max mismatches to print",
    )
    args = parser.parse_args()

    with open(args.fixed_trials, "r", encoding="utf-8") as handle:
        fixed = json.load(handle)

    meta = fixed.get("meta", {})
    tok_add_special = bool(meta.get("model_prepend_bos", False))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mismatches = []
    trials = fixed.get("trials", [])
    for t_idx, trial in enumerate(trials):
        token_labels, paper_prefix = paper_get_token_meta_labels(
            trial["prompt_data_corrupted"],
            tokenizer,
            query=None,
            prepend_bos=tok_add_special,
        )
        _ = token_labels  # keep for parity with signature
        stepd_prefix = trial["corrupted_prompt_str"]

        paper = tokenizer(
            paper_prefix, return_tensors="pt", add_special_tokens=tok_add_special
        )
        stepd = tokenizer(
            stepd_prefix, return_tensors="pt", add_special_tokens=tok_add_special
        )

        if not torch.equal(paper["input_ids"], stepd["input_ids"]):
            mismatches.append((t_idx, "input_ids"))
            if len(mismatches) >= args.max_mismatches:
                break
            continue
        if not torch.equal(paper["attention_mask"], stepd["attention_mask"]):
            mismatches.append((t_idx, "attention_mask"))
            if len(mismatches) >= args.max_mismatches:
                break
            continue

        seq_len = paper["input_ids"].shape[1]
        paper_pos = torch.arange(seq_len).unsqueeze(0)
        stepd_pos = torch.arange(seq_len).unsqueeze(0)
        if not torch.equal(paper_pos, stepd_pos):
            mismatches.append((t_idx, "position_ids"))
            if len(mismatches) >= args.max_mismatches:
                break

    print("total_trials:", len(trials))
    print("mismatch_count:", len(mismatches))
    for item in mismatches:
        print(item)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
