#!/usr/bin/env python3
"""Verify M3 injection parity between src and fv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.fv import compute_function_vector as fv_compute_function_vector
from fv.intervene import function_vector_intervention as fv_function_vector_intervention
from src.utils.intervention_utils import (
    function_vector_intervention as src_function_vector_intervention,
)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify src vs fv injection parity.")
    parser.add_argument("--dataset_name", type=str, default="antonym")
    parser.add_argument(
        "--fixed_trials_path",
        type=str,
        default="datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json",
    )
    parser.add_argument(
        "--fixed_trials_id",
        type=str,
        default="fixed_trials_antonym_t10_s10_seed0",
    )
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--results_fv_root", type=str, default="results_fv")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_top_heads", type=int, default=10)
    parser.add_argument("--token_class_idx", type=int, default=-1)
    parser.add_argument("--edit_layer", type=int, default=9)
    parser.add_argument("--max_trials", type=int, default=5)
    parser.add_argument("--assert_zero", action="store_true")
    return parser.parse_args()


def _target_logprob(logits: torch.Tensor, target_id: int) -> float:
    log_probs = torch.log_softmax(logits, dim=-1)
    return float(log_probs[0, int(target_id)].item())


def _normalize_top_heads(top_heads):
    normalized = []
    for layer, head, score in top_heads:
        normalized.append([int(layer), int(head), float(score)])
    return normalized


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    torch.set_grad_enabled(False)

    fixed_trials_path = PROJECT_ROOT / args.fixed_trials_path
    with open(fixed_trials_path, "r", encoding="utf-8") as f:
        fixed_trials = json.load(f)
    trials = fixed_trials.get("trials", [])
    if not trials:
        raise ValueError("No trials found in fixed trials JSON")
    n_use = min(args.max_trials, len(trials))

    run_dir = (
        PROJECT_ROOT / args.results_root / args.dataset_name / args.fixed_trials_id
    )
    mean_path = run_dir / f"{args.dataset_name}_mean_head_activations_FIXED.pt"
    ie_path = run_dir / f"{args.dataset_name}_indirect_effect.pt"
    if not mean_path.exists() or not ie_path.exists():
        raise FileNotFoundError(
            f"Missing golden artifacts in {run_dir}; expected {mean_path.name} and {ie_path.name}"
        )
    mean_activations = torch.load(mean_path, map_location=args.device)
    indirect_effect = torch.load(ie_path, map_location=args.device)

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device
    )
    model.eval()

    function_vector, top_heads = fv_compute_function_vector(
        mean_activations=mean_activations,
        indirect_effect=indirect_effect,
        model=model,
        model_config=model_config,
        n_top_heads=args.n_top_heads,
        token_class_idx=args.token_class_idx,
    )

    out_dir = (
        PROJECT_ROOT / args.results_fv_root / args.dataset_name / args.fixed_trials_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_csv = out_dir / "injection_parity_trials.csv"
    report_json = out_dir / "injection_parity_report.json"

    rows = []
    mismatch_count = 0
    max_abs_diff_clean = 0.0
    max_abs_diff_with = 0.0
    for t_idx, trial in enumerate(trials[:n_use]):
        sentence = trial.get("corrupted_prompt_str")
        target = trial.get("target_str")
        target_id = trial.get("target_first_token_id")
        if sentence is None or target is None or target_id is None:
            raise ValueError(
                f"Trial {t_idx} missing one of corrupted_prompt_str/target_str/target_first_token_id"
            )

        src_clean, src_with = src_function_vector_intervention(
            sentence=sentence,
            target=target,
            edit_layer=args.edit_layer,
            function_vector=function_vector,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            compute_nll=False,
            generate_str=False,
        )
        fv_clean, fv_with = fv_function_vector_intervention(
            sentence=sentence,
            target=target,
            edit_layer=args.edit_layer,
            function_vector=function_vector,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            compute_nll=False,
            generate_str=False,
        )

        clean_src_logprob = _target_logprob(src_clean, target_id)
        clean_fv_logprob = _target_logprob(fv_clean, target_id)
        with_src_logprob = _target_logprob(src_with, target_id)
        with_fv_logprob = _target_logprob(fv_with, target_id)
        delta_src = with_src_logprob - clean_src_logprob
        delta_fv = with_fv_logprob - clean_fv_logprob

        clean_equal = clean_src_logprob == clean_fv_logprob
        with_equal = with_src_logprob == with_fv_logprob
        delta_equal = delta_src == delta_fv
        trial_match = clean_equal and with_equal and delta_equal
        if not trial_match:
            mismatch_count += 1

        clean_diff = float((src_clean - fv_clean).abs().max().item())
        with_diff = float((src_with - fv_with).abs().max().item())
        max_abs_diff_clean = max(max_abs_diff_clean, clean_diff)
        max_abs_diff_with = max(max_abs_diff_with, with_diff)

        rows.append(
            {
                "trial_idx": t_idx,
                "target_id": int(target_id),
                "clean_src_logprob": clean_src_logprob,
                "clean_fv_logprob": clean_fv_logprob,
                "with_src_logprob": with_src_logprob,
                "with_fv_logprob": with_fv_logprob,
                "delta_src_logprob": delta_src,
                "delta_fv_logprob": delta_fv,
                "clean_equal": int(clean_equal),
                "with_equal": int(with_equal),
                "delta_equal": int(delta_equal),
                "trial_match": int(trial_match),
                "clean_logits_max_abs_diff": clean_diff,
                "with_logits_max_abs_diff": with_diff,
            }
        )

    with open(trials_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "dataset_name": args.dataset_name,
        "fixed_trials_id": args.fixed_trials_id,
        "model_name": args.model_name,
        "device": args.device,
        "seed": args.seed,
        "checked_trials": n_use,
        "edit_layer": args.edit_layer,
        "n_top_heads": args.n_top_heads,
        "token_class_idx": args.token_class_idx,
        "top_heads": _normalize_top_heads(top_heads),
        "mismatch_count": mismatch_count,
        "max_abs_diff_clean_logits": max_abs_diff_clean,
        "max_abs_diff_with_logits": max_abs_diff_with,
        "trials_csv": str(trials_csv),
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"checked_trials: {n_use}")
    print(f"mismatch_count: {mismatch_count}")
    print(f"max_abs_diff_clean_logits: {max_abs_diff_clean}")
    print(f"max_abs_diff_with_logits: {max_abs_diff_with}")
    print(f"report_json: {report_json}")
    print(f"trials_csv: {trials_csv}")

    if args.assert_zero and mismatch_count != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
