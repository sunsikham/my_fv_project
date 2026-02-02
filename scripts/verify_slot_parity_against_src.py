#!/usr/bin/env python3
"""Verify slot/label/idx_map parity between src and fv implementations."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer

from fv.prompting import (
    compute_duplicated_labels as fv_compute_duplicated_labels,
    get_dummy_token_labels as fv_get_dummy_token_labels,
    get_token_meta_labels as fv_get_token_meta_labels,
)

from src.utils.prompt_utils import (
    compute_duplicated_labels as src_compute_duplicated_labels,
    get_dummy_token_labels as src_get_dummy_token_labels,
    get_token_meta_labels as src_get_token_meta_labels,
)


def _load_fixed_trials(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_token_labels(token_labels):
    return [(int(idx), str(tok), str(label)) for (idx, tok, label) in token_labels]


def _normalize_dummy_labels(dummy_labels):
    return [(int(idx), str(label)) for (idx, label) in dummy_labels]


def _normalize_idx_map(idx_map):
    return {int(k): int(v) for k, v in idx_map.items()}


def _normalize_idx_avg(idx_avg):
    return {str(k): (int(v[0]), int(v[1])) for k, v in idx_avg.items()}


def _compare(name, a, b, mismatches):
    if a != b:
        mismatches.append(name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify src vs fv slot parity.")
    parser.add_argument("--fixed_trials_path", required=True)
    parser.add_argument("--max_trials", type=int, default=5)
    parser.add_argument("--mode", choices=["clean", "corrupted"], default="corrupted")
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--assert_zero", action="store_true", default=True)
    args = parser.parse_args()

    fixed = _load_fixed_trials(Path(args.fixed_trials_path))
    trials = fixed.get("trials", [])
    meta = fixed.get("meta", {})
    prepend_bos = bool(meta.get("model_prepend_bos", False))
    prefixes = meta.get("prefixes")
    separators = meta.get("separators")

    tokenizer_name = args.tokenizer_name or meta.get("model_name_for_tokenizer")
    if not tokenizer_name:
        raise ValueError("tokenizer_name missing (pass --tokenizer_name)")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mismatches = []
    for t_idx, trial in enumerate(trials[: args.max_trials]):
        prompt_key = "prompt_data_clean" if args.mode == "clean" else "prompt_data_corrupted"
        prompt_data = trial.get(prompt_key)
        if prompt_data is None:
            raise ValueError(f"fixed_trials missing {prompt_key}")

        n_icl_examples = len(prompt_data.get("examples", []))
        model_config = {"prepend_bos": prepend_bos}

        src_dummy = src_get_dummy_token_labels(
            n_icl_examples,
            tokenizer=tokenizer,
            model_config=model_config,
            prefixes=prefixes,
            separators=separators,
        )
        fv_dummy = fv_get_dummy_token_labels(
            n_icl_examples,
            tokenizer=tokenizer,
            model_config=model_config,
            prefixes=prefixes,
            separators=separators,
        )

        src_token_labels, _src_prompt = src_get_token_meta_labels(
            prompt_data,
            tokenizer,
            query=None,
            prepend_bos=prepend_bos,
        )
        fv_token_labels, _fv_prompt = fv_get_token_meta_labels(
            prompt_data,
            tokenizer,
            query=None,
            prepend_bos=prepend_bos,
        )

        src_idx_map, src_idx_avg = src_compute_duplicated_labels(
            src_token_labels, src_dummy
        )
        fv_idx_map, fv_idx_avg = fv_compute_duplicated_labels(
            fv_token_labels, fv_dummy
        )

        trial_mismatches = []
        _compare(
            "dummy_labels",
            _normalize_dummy_labels(src_dummy),
            _normalize_dummy_labels(fv_dummy),
            trial_mismatches,
        )
        _compare(
            "token_meta_labels",
            _normalize_token_labels(src_token_labels),
            _normalize_token_labels(fv_token_labels),
            trial_mismatches,
        )
        _compare(
            "idx_map",
            _normalize_idx_map(src_idx_map),
            _normalize_idx_map(fv_idx_map),
            trial_mismatches,
        )
        _compare(
            "idx_avg",
            _normalize_idx_avg(src_idx_avg),
            _normalize_idx_avg(fv_idx_avg),
            trial_mismatches,
        )

        if trial_mismatches:
            mismatches.append((t_idx, trial_mismatches))

    print("checked_trials:", min(len(trials), args.max_trials))
    print("mismatch_count:", len(mismatches))
    for item in mismatches[:10]:
        print(item)

    if args.assert_zero and mismatches:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
