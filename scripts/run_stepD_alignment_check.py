#!/usr/bin/env python3
"""StepD slot->sequence alignment sanity check."""

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.corrupt import make_corrupted_demos
from fv.dataset_loader import load_pairs_antonym, sample_demos_and_query
from fv.model_spec import get_model_spec
from fv.prompting import build_prompt_qa
from fv.slots import (
    build_prompt_segments,
    compute_duplicated_labels,
    get_dummy_token_labels_and_slot_map,
    get_token_meta_labels,
    infer_special_token_labels,
    resolve_slot_seq_token_idx,
    segments_to_text_and_spans,
    validate_idx_map,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="StepD slot alignment check.")
    parser.add_argument("--model", default="gpt2", help="Model name or path")
    parser.add_argument(
        "--model_spec",
        default="gpt2",
        help="Model spec name for adapter resolution",
    )
    parser.add_argument(
        "--dataset_path",
        default="datasets/processed/antonym.json",
        help="Path to antonym dataset JSON",
    )
    parser.add_argument("--n_trials", type=int, default=5, help="Trials to sample")
    parser.add_argument("--n_icl_examples", type=int, default=2, help="ICL demos per prompt")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    tok_group = parser.add_mutually_exclusive_group()
    tok_group.add_argument("--tok_add_special", action="store_true")
    tok_group.add_argument("--no_tok_add_special", action="store_true")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime import check
        print(f"Failed to import transformers: {exc}")
        return 1

    try:
        spec = get_model_spec(args.model_spec)
    except ValueError as exc:
        print(str(exc))
        return 1

    if args.tok_add_special:
        tok_add_special = True
    elif args.no_tok_add_special:
        tok_add_special = False
    else:
        tok_add_special = bool(spec.prepend_bos)

    try:
        pairs = load_pairs_antonym(args.dataset_path, canonical_by_input=True)
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        return 1

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as exc:  # pragma: no cover - runtime load check
        print(f"Failed to load tokenizer '{args.model}': {exc}")
        return 1

    dummy_labels_ref = None
    slot_index_map_ref = None
    special_index_labels_ref = None
    failures = 0
    sample_logged = False

    for trial_idx in range(args.n_trials):
        demos_orig, query = sample_demos_and_query(
            pairs, args.n_icl_examples, seed=args.seed + trial_idx
        )
        corrupted_demos = make_corrupted_demos(
            demos_orig,
            random.Random(args.seed + trial_idx),
            ensure_derangement=True,
        )
        prefix_str, full_str = build_prompt_qa(corrupted_demos, query)

        segments = build_prompt_segments(corrupted_demos, query)
        full_str_segments, _spans = segments_to_text_and_spans(segments)
        if full_str_segments != full_str:
            print("Prompt builder mismatch with SSOT")
            return 1

        encoded = tokenizer(
            full_str,
            return_offsets_mapping=True,
            add_special_tokens=tok_add_special,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        special_index_labels = infer_special_token_labels(tokenizer, input_ids, offsets)
        if (
            special_index_labels_ref is not None
            and special_index_labels != special_index_labels_ref
        ):
            print("Special token alignment mismatch across trials")
            return 1

        special_positions = sorted(special_index_labels.keys())
        special_prefix = []
        special_suffix = []
        if special_positions and special_positions[0] == 0:
            special_prefix.append(special_index_labels[special_positions[0]])
        if special_positions and special_positions[-1] == len(input_ids) - 1:
            if special_positions[-1] != special_positions[0]:
                special_suffix.append(special_index_labels[special_positions[-1]])

        dummy_labels, slot_index_map = get_dummy_token_labels_and_slot_map(
            args.n_icl_examples,
            special_prefix=special_prefix,
            special_suffix=special_suffix,
        )
        if dummy_labels_ref is None:
            dummy_labels_ref = dummy_labels
            slot_index_map_ref = slot_index_map
            special_index_labels_ref = special_index_labels
        elif dummy_labels != dummy_labels_ref or slot_index_map != slot_index_map_ref:
            print("dummy_labels/slot_index_map mismatch across trials")
            return 1

        token_meta_labels, _input_ids, _offsets = get_token_meta_labels(
            tokenizer,
            full_str,
            segments,
            tok_add_special,
            special_index_labels,
        )
        idx_map, _idx_avg = compute_duplicated_labels(
            token_meta_labels,
            dummy_labels,
            allow_empty_labels=["QUERY_OUT"],
        )
        validate_idx_map(idx_map, len(input_ids))

        slot_q = slot_index_map["QUERY_PRED"]
        seq_token_idx = resolve_slot_seq_token_idx(
            idx_map, slot_q, require_single=False
        )
        prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=tok_add_special)
        if seq_token_idx != len(prefix_ids) - 1:
            failures += 1

        if not sample_logged:
            decoded_full = tokenizer.decode([input_ids[seq_token_idx]])
            decoded_prefix = tokenizer.decode([prefix_ids[seq_token_idx]])
            print("sample_alignment:")
            print(
                f"  slot_idx={slot_q} seq_token_idx={seq_token_idx} "
                f"full_token={repr(decoded_full)} prefix_token={repr(decoded_prefix)}"
            )
            sample_logged = True

    print(f"alignment_checked: n_trials={args.n_trials} failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
