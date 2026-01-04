#!/usr/bin/env python3
"""STEP E: Build top-k FV from StepD and run injection eval."""

import argparse
import csv
import os
import random
import statistics
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fv.corrupt import make_corrupted_demos
from fv.intervene import make_residual_injection_hook
from fv.io import prepare_run_dirs, resolve_out_dir, save_json
from fv.model_config import get_model_config
from fv.prompting import ANTONYM_PAIRS
from fv.slots import compute_query_predictive_slot


def make_logger(log_path: str):
    log_file = open(log_path, "w", encoding="utf-8")

    def log(message: str) -> None:
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    return log, log_file


def build_prompt(demos, query_pair):
    lines = ["Antonyms:"]
    for x_val, y_val in demos:
        lines.append(f"{x_val} -> {y_val}")
    lines.append(f"{query_pair[0]} ->")
    prompt = "\n".join(lines)
    prefix_str = f"{prompt} "
    full_str = f"{prefix_str}{query_pair[1]}"
    return prefix_str, full_str


def parse_layers(value: str, available_layers):
    if value == "all":
        return sorted(available_layers)
    layers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError("Invalid layer range")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP E top-k FV + eval.")
    parser.add_argument("--run_id_stepD", required=True, help="StepD run_id")
    parser.add_argument("--k", type=int, default=20, help="Top-k heads (default: 20)")
    parser.add_argument("--model", default="gpt2", help="Model name or path (default: gpt2)")
    parser.add_argument("--layers", default="all", help="Layer filter (default: all)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Injection scale (default: 1.0)")
    parser.add_argument(
        "--n_eval_trials",
        type=int,
        default=20,
        help="Evaluation trials (default: 20)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--run_id",
        default=None,
        help="Run identifier (default: auto timestamp)",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: runs/<run_id>/artifacts/)",
    )
    args = parser.parse_args()

    if args.k < 1:
        print("k must be >= 1")
        return 1
    if args.n_eval_trials < 1:
        print("n_eval_trials must be >= 1")
        return 1

    run_info = prepare_run_dirs(args.run_id)
    if args.out_dir:
        artifacts_dir = resolve_out_dir(args.out_dir)
    else:
        artifacts_dir = run_info["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    log_path = os.path.join(run_info["logs_dir"], "stepE.log")
    log, log_file = make_logger(log_path)

    log("stepE start")
    log(f"run_id: {run_info['run_id']}")
    log(f"artifacts_dir: {artifacts_dir}")
    log(f"log_path: {log_path}")
    log(f"run_id_stepD: {args.run_id_stepD}")
    log(f"model: {args.model}")
    log(f"k: {args.k}")
    log(f"layers: {args.layers}")
    log(f"alpha: {args.alpha}")
    log(f"n_eval_trials: {args.n_eval_trials}")
    log(f"seed: {args.seed}")

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime import check
        log(f"Failed to import required libraries: {exc}")
        log_file.close()
        return 1

    model_cfg = get_model_config(args.model)
    if model_cfg is None:
        log(f"No model config found for '{args.model}'")
        log_file.close()
        return 1

    stepd_dir = resolve_out_dir(os.path.join("runs", args.run_id_stepD, "artifacts"))
    scores_path = os.path.join(stepd_dir, "aie_scores.csv")
    clean_mean_path = os.path.join(stepd_dir, "clean_mean.pt")
    if not os.path.exists(scores_path):
        log(f"Missing aie_scores.csv: {scores_path}")
        log_file.close()
        return 1
    if not os.path.exists(clean_mean_path):
        log(f"Missing clean_mean.pt: {clean_mean_path}")
        log_file.close()
        return 1

    scores = []
    with open(scores_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(row)

    available_layers = sorted({int(row["layer"]) for row in scores})
    try:
        layer_filter = set(parse_layers(args.layers, available_layers))
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1
    layer_filter &= set(available_layers)
    if not layer_filter:
        log("No layers left after filtering")
        log_file.close()
        return 1

    filtered = [
        row
        for row in scores
        if int(row["layer"]) in layer_filter
    ]
    filtered.sort(key=lambda r: float(r["mean_cie"]), reverse=True)
    top_rows = filtered[: args.k]
    if not top_rows:
        log("No heads found for top-k selection")
        log_file.close()
        return 1

    top_heads = [
        {
            "layer": int(row["layer"]),
            "head": int(row["head"]),
            "mean_cie": float(row["mean_cie"]),
            "std_cie": float(row["std_cie"]),
            "n_trials": int(row["n_trials"]),
        }
        for row in top_rows
    ]

    top_heads_path = os.path.join(artifacts_dir, "top_heads.json")
    save_json(
        top_heads_path,
        {
            "meta": {"source_run_id": args.run_id_stepD, "k": args.k, "model": args.model},
            "heads": top_heads,
        },
    )
    log(f"saved top heads: {top_heads_path}")

    clean_mean_blob = torch.load(clean_mean_path, map_location="cpu")
    clean_mean = clean_mean_blob["clean_mean"]
    n_heads = int(clean_mean_blob["n_heads"])
    head_dim = int(clean_mean_blob["head_dim"])
    resid_dim = int(clean_mean_blob["resid_dim"])

    fv_by_layer = {layer: torch.zeros((resid_dim,), dtype=torch.float32) for layer in layer_filter}
    for entry in top_heads:
        layer = entry["layer"]
        head = entry["head"]
        if layer not in fv_by_layer:
            continue
        head_vec = clean_mean[layer][head].to(dtype=torch.float32)
        resid_vec = torch.zeros((resid_dim,), dtype=torch.float32)
        start = head * head_dim
        end = start + head_dim
        resid_vec[start:end] = head_vec
        fv_by_layer[layer] += resid_vec

    fv_by_layer_path = os.path.join(artifacts_dir, "fv_by_layer.pt")
    torch.save(
        {
            "n_heads": n_heads,
            "head_dim": head_dim,
            "resid_dim": resid_dim,
            "selected_k": args.k,
            "alpha_default": args.alpha,
            "fv_by_layer": {layer: fv_by_layer[layer] for layer in fv_by_layer},
        },
        fv_by_layer_path,
    )
    log(f"saved fv_by_layer: {fv_by_layer_path}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
    except Exception as exc:  # pragma: no cover - runtime load check
        log(f"Failed to load model '{args.model}': {exc}")
        log_file.close()
        return 1

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        log("Model does not expose transformer.h blocks.")
        log_file.close()
        return 1

    n_layers = len(model.transformer.h)
    for layer in fv_by_layer:
        if layer < 0 or layer >= n_layers:
            log("Layer index out of range")
            log_file.close()
            return 1

    if args.n_eval_trials + 1 > len(ANTONYM_PAIRS):
        log("Not enough antonym pairs for requested demos + query.")
        log_file.close()
        return 1

    rng = random.Random(args.seed)
    trials = []
    for trial_idx in range(args.n_eval_trials):
        pairs = rng.sample(ANTONYM_PAIRS, 4)
        demos = pairs[:3]
        query = pairs[-1]
        clean_prefix_str, clean_full_str = build_prompt(demos, query)
        corrupted_demos = make_corrupted_demos(demos, rng, ensure_derangement=True)
        corrupted_prefix_str, corrupted_full_str = build_prompt(corrupted_demos, query)

        try:
            clean_slot = compute_query_predictive_slot(
                clean_prefix_str, clean_full_str, tokenizer
            )
            corrupted_slot = compute_query_predictive_slot(
                corrupted_prefix_str, corrupted_full_str, tokenizer
            )
        except ValueError as exc:
            log(str(exc))
            log_file.close()
            return 1

        if clean_slot["target_id"] != corrupted_slot["target_id"]:
            log("target_id mismatch between clean and corrupted")
            log_file.close()
            return 1

        trials.append(
            {
                "trial_idx": trial_idx,
                "corrupted_prefix_str": corrupted_prefix_str,
                "target_id": clean_slot["target_id"],
            }
        )

    per_layer = {layer: [] for layer in fv_by_layer}
    all_deltas = []

    log("running eval")
    for trial in trials:
        prefix_str = trial["corrupted_prefix_str"]
        target_id = trial["target_id"]

        inputs = tokenizer(prefix_str, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        last_index = inputs["input_ids"].shape[1] - 1

        with torch.inference_mode():
            outputs = model(**inputs)
        baseline_logits = outputs.logits[0, last_index]
        baseline_logit = baseline_logits[target_id].item()

        for layer, fv_vec in fv_by_layer.items():
            injection_state = {
                "fv": fv_vec.to(device=device, dtype=baseline_logits.dtype),
                "alpha": args.alpha,
                "batch_indices": None,
                "last_indices": None,
            }
            batch_indices = torch.tensor([0], device=device)
            last_indices = torch.tensor([last_index], device=device)
            injection_state["batch_indices"] = batch_indices
            injection_state["last_indices"] = last_indices

            inject_hook = make_residual_injection_hook(injection_state)
            block = model.transformer.h[layer]
            handle = block.register_forward_hook(inject_hook)
            with torch.inference_mode():
                outputs_fv = model(**inputs)
            handle.remove()

            patched_logits = outputs_fv.logits[0, last_index]
            patched_logit = patched_logits[target_id].item()
            delta_logit = patched_logit - baseline_logit
            per_layer[layer].append(delta_logit)
            all_deltas.append(delta_logit)

    per_layer_summary = {}
    for layer, deltas in per_layer.items():
        per_layer_summary[str(layer)] = {
            "mean_delta_logit": mean(deltas),
            "std_delta_logit": std(deltas),
            "any_nonzero": any(abs(v) > 1e-12 for v in deltas),
            "n": len(deltas),
        }

    overall = {
        "mean_delta_logit": mean(all_deltas),
        "std_delta_logit": std(all_deltas),
        "any_nonzero": any(abs(v) > 1e-12 for v in all_deltas),
    }

    eval_path = os.path.join(artifacts_dir, "stepE_eval.json")
    save_json(
        eval_path,
        {
            "meta": {
                "model": args.model,
                "run_id_stepD": args.run_id_stepD,
                "k": args.k,
                "layers": sorted(layer_filter),
                "alpha": args.alpha,
                "n_eval_trials": args.n_eval_trials,
                "seed": args.seed,
                "token_idx": -1,
                "measure": "logit[target_id]",
            },
            "per_layer": per_layer_summary,
            "overall": overall,
        },
    )

    log(f"saved eval: {eval_path}")
    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
