#!/usr/bin/env python3
"""Dump QUERY_PRED token alignment for paper vs StepD paths."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer

from fv.prompting import (
    compute_duplicated_labels as paper_compute_duplicated_labels,
    get_dummy_token_labels as paper_get_dummy_token_labels,
    get_token_meta_labels as paper_get_token_meta_labels,
)
from fv.slots import (
    build_prompt_segments,
    compute_duplicated_labels as stepd_compute_duplicated_labels,
    get_dummy_token_labels_and_slot_map,
    get_token_meta_labels as stepd_get_token_meta_labels,
    resolve_slot_seq_token_idx,
    segments_to_text_and_spans,
)


def load_fixed_trials(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_safe(value):
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump QUERY_PRED token alignment.")
    parser.add_argument("--fixed_trials_path", required=True)
    parser.add_argument("--trial_idx", type=int, default=0)
    parser.add_argument("--out_dir", default="runs/slot_check")
    parser.add_argument("--model_name_for_tokenizer", default=None)
    args = parser.parse_args()

    fixed = load_fixed_trials(args.fixed_trials_path)
    trials = fixed.get("trials", [])
    if args.trial_idx < 0 or args.trial_idx >= len(trials):
        raise ValueError("trial_idx out of range")
    trial = trials[args.trial_idx]

    meta = fixed.get("meta", {})
    model_name = args.model_name_for_tokenizer or meta.get("model_name_for_tokenizer")
    if not model_name:
        raise ValueError("model_name_for_tokenizer missing (pass --model_name_for_tokenizer)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(args.out_dir, exist_ok=True)

    # Paper path
    prompt_data = trial.get("prompt_data_corrupted")
    if prompt_data is None:
        raise ValueError("fixed_trials missing prompt_data_corrupted")
    token_labels, prompt_string = paper_get_token_meta_labels(
        prompt_data, tokenizer, query=None, prepend_bos=meta.get("model_prepend_bos", False)
    )
    prompt_ids = tokenizer(prompt_string).input_ids
    decoded_tokens = [tokenizer.decode([i]) for i in prompt_ids]

    n_shots = meta.get("n_shots")
    if n_shots is None:
        raise ValueError("fixed_trials meta missing n_shots")
    model_config = {"prepend_bos": bool(meta.get("model_prepend_bos", False))}
    dummy_labels = paper_get_dummy_token_labels(
        n_shots,
        tokenizer=tokenizer,
        model_config=model_config,
        prefixes=meta.get("prefixes"),
        separators=meta.get("separators"),
    )
    paper_idx_map, _ = paper_compute_duplicated_labels(token_labels, dummy_labels)
    query_label_idx = None
    for idx, label in dummy_labels:
        if label == "query_predictive_token":
            query_label_idx = idx
            break
    if query_label_idx is None:
        raise ValueError("query_predictive_token not found in dummy_labels")
    query_token_indices = [k for k, v in paper_idx_map.items() if v == query_label_idx]
    if not query_token_indices:
        raise ValueError("No prompt token mapped to query_predictive_token")
    paper_query_idx = max(query_token_indices)
    paper_query_token_id = prompt_ids[paper_query_idx]

    paper_payload = {
        "trial_idx": args.trial_idx,
        "prompt_string": prompt_string,
        "input_ids": prompt_ids,
        "decoded_tokens": decoded_tokens,
        "token_labels": [label for _idx, _tok, label in token_labels],
        "query_pred_index": paper_query_idx,
        "query_pred_token_id": paper_query_token_id,
        "query_pred_token": tokenizer.decode([paper_query_token_id]),
    }
    write_json(os.path.join(args.out_dir, f"paper_trial_{args.trial_idx}.json"), paper_payload)

    # StepD path
    demos = trial.get("demos_corrupted")
    query = trial.get("query")
    if demos is None or query is None:
        raise ValueError("fixed_trials missing demos_corrupted/query")
    demos_pairs = (
        [(d["input"], d["output"]) for d in demos]
        if demos and isinstance(demos[0], dict)
        else demos
    )
    query_pair = (
        (query["input"], query["output"]) if isinstance(query, dict) else query
    )
    segments = build_prompt_segments(demos_pairs, query_pair)
    prefix_str = trial["corrupted_prompt_str"]
    full_str = prefix_str + trial["target_str"]
    full_str_segments, _spans = segments_to_text_and_spans(segments)
    bos = tokenizer.bos_token or tokenizer.eos_token
    if bos and full_str.startswith(bos) and full_str[len(bos) :] == full_str_segments:
        segments = [("SPECIAL_BOS", bos)] + list(segments)
        full_str_segments, _spans = segments_to_text_and_spans(segments)
    if full_str_segments != full_str:
        raise ValueError("Prompt builder mismatch with SSOT")

    encoded = tokenizer(
        full_str,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    special_index_labels = {}
    for idx, (start, end) in enumerate(offsets):
        if start != end:
            continue
        token_id = input_ids[idx]
        if tokenizer.bos_token_id is not None and token_id == tokenizer.bos_token_id:
            special_index_labels[idx] = "SPECIAL_BOS"
        elif tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            special_index_labels[idx] = "SPECIAL_EOS"
        else:
            special_index_labels[idx] = "SPECIAL"

    token_meta_labels, _input_ids, _offsets = stepd_get_token_meta_labels(
        tokenizer,
        full_str,
        segments,
        False,
        special_index_labels,
    )
    dummy_labels_stepd, slot_index_map = get_dummy_token_labels_and_slot_map(
        len(demos_pairs),
        special_prefix=["SPECIAL_BOS"] if bos and full_str.startswith(bos) else None,
        special_suffix=None,
    )
    idx_map, _idx_avg = stepd_compute_duplicated_labels(
        token_meta_labels, dummy_labels_stepd, allow_empty_labels=["QUERY_OUT"]
    )
    slot_q = slot_index_map["QUERY_PRED"]
    stepd_query_idx = resolve_slot_seq_token_idx(idx_map, slot_q, require_single=False)
    stepd_query_token_id = input_ids[stepd_query_idx]

    prefix_ids = tokenizer(prefix_str, add_special_tokens=False).input_ids
    prefix_len = len(prefix_ids)
    if prefix_len <= 0 or prefix_len > len(input_ids):
        raise ValueError("Invalid prefix length for StepD dump")
    stepd_query_idx = prefix_len - 1
    stepd_query_token_id = prefix_ids[stepd_query_idx]

    stepd_payload = {
        "trial_idx": args.trial_idx,
        "prompt_string": prefix_str,
        "input_ids": prefix_ids,
        "decoded_tokens": [tokenizer.decode([i]) for i in prefix_ids],
        "token_labels": token_meta_labels[:prefix_len],
        "query_pred_index": stepd_query_idx,
        "query_pred_token_id": stepd_query_token_id,
        "query_pred_token": tokenizer.decode([stepd_query_token_id]),
    }
    write_json(os.path.join(args.out_dir, f"stepd_trial_{args.trial_idx}.json"), stepd_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
