#!/usr/bin/env python3
"""Verify FV construction parity between src and fv implementations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.fv import compute_function_vector as fv_compute_function_vector
from src.utils.extract_utils import compute_function_vector as src_compute_function_vector
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify src vs fv FV parity.")
    parser.add_argument("--dataset_name", type=str, default="antonym")
    parser.add_argument(
        "--fixed_trials_id",
        type=str,
        default="fixed_trials_antonym_t10_s10_seed0",
        help="Fixed-trials identifier (folder stem).",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="Root directory containing canonical M1 artifacts.",
    )
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_top_heads", type=int, default=10)
    parser.add_argument("--token_class_idx", type=int, default=-1)
    parser.add_argument("--assert_zero", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = (
        PROJECT_ROOT
        / args.results_root
        / args.dataset_name
        / args.fixed_trials_id
    )
    mean_path = run_dir / f"{args.dataset_name}_mean_head_activations_FIXED.pt"
    ie_path = run_dir / f"{args.dataset_name}_indirect_effect.pt"

    missing = [str(p) for p in (mean_path, ie_path) if not p.exists()]
    if missing:
        print("missing_artifacts:")
        for path in missing:
            print(path)
        return 1

    set_seed(args.seed)
    torch.set_grad_enabled(False)
    model, _tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device
    )
    model.eval()
    src_model_config = dict(model_config)
    # src.compute_function_vector does not handle exact "gpt2" string.
    # Keep identical c_proj path behavior by mapping to the gpt2-xl branch.
    if str(src_model_config.get("name_or_path", "")).lower() == "gpt2":
        src_model_config["name_or_path"] = "gpt2-xl"

    mean_activations = torch.load(mean_path, map_location=args.device)
    indirect_effect = torch.load(ie_path, map_location=args.device)

    src_fv, src_top_heads = src_compute_function_vector(
        mean_activations=mean_activations,
        indirect_effect=indirect_effect,
        model=model,
        model_config=src_model_config,
        n_top_heads=args.n_top_heads,
        token_class_idx=args.token_class_idx,
    )
    fv_fv, fv_top_heads = fv_compute_function_vector(
        mean_activations=mean_activations,
        indirect_effect=indirect_effect,
        model=model,
        model_config=model_config,
        n_top_heads=args.n_top_heads,
        token_class_idx=args.token_class_idx,
    )

    top_heads_match = src_top_heads == fv_top_heads
    max_abs_diff = (src_fv - fv_fv).abs().max().item()
    fv_equal = torch.equal(src_fv, fv_fv)

    mismatch_count = 0
    if not top_heads_match:
        mismatch_count += 1
    if not fv_equal:
        mismatch_count += 1

    print(f"run_dir: {run_dir}")
    print(f"top_heads_count: {len(src_top_heads)}")
    print(f"top_heads_match: {top_heads_match}")
    if not top_heads_match:
        print(f"src_top_heads: {src_top_heads}")
        print(f"fv_top_heads: {fv_top_heads}")
    print(f"src_fv_shape: {tuple(src_fv.shape)}")
    print(f"fv_fv_shape: {tuple(fv_fv.shape)}")
    print(f"max_abs_diff: {max_abs_diff}")
    print(f"fv_equal: {fv_equal}")
    print(f"mismatch_count: {mismatch_count}")

    if args.assert_zero and mismatch_count != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
