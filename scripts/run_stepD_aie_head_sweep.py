#!/usr/bin/env python3
"""STEP D: AIE head sweep using clean-mean replacement on corrupted prompts."""

import argparse
import os
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.corrupt import make_corrupted_demos
from fv.hooks import get_c_proj_pre_hook, reshape_resid_to_heads
from fv.io import prepare_run_dirs, resolve_out_dir, save_csv, save_json
from fv.model_config import get_model_config
from fv.patch import make_cproj_head_replacer
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


def parse_layers(value: str):
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


def parse_heads(value: str, n_heads: int):
    if value == "all":
        return list(range(n_heads))
    heads = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        heads.append(int(part))
    return sorted(set(heads))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def mean_abs(values):
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def compute_clean_mean(
    prefixes,
    model,
    tokenizer,
    device,
    layers,
    n_heads,
    head_dim,
    resid_dim,
    layer_modules,
    log,
):
    import torch

    means = {layer: torch.zeros((n_heads, head_dim), device=device) for layer in layers}
    counts = {layer: 0 for layer in layers}

    for layer in layers:
        state = {"current": None, "errors": []}

        def pre_hook(_module, inputs):
            tensor = inputs[0] if inputs else None
            try:
                reshaped = reshape_resid_to_heads(tensor, n_heads, head_dim, resid_dim)
            except ValueError as exc:
                state["errors"].append(str(exc))
                return
            seq_len = reshaped.shape[1]
            t_idx = seq_len - 1
            state["current"] = reshaped[:, t_idx, :, :].detach().mean(dim=0)

        handle = layer_modules[layer].register_forward_pre_hook(pre_hook)
        for prefix in prefixes:
            state["current"] = None
            inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                _ = model(**inputs)
            if state["errors"]:
                handle.remove()
                raise ValueError("; ".join(state["errors"]))
            if state["current"] is None:
                handle.remove()
                raise ValueError("Failed to capture head activations")
            means[layer] += state["current"]
            counts[layer] += 1
        handle.remove()
        log(
            f"clean_mean layer={layer} shape=({n_heads},{head_dim}) count={counts[layer]}"
        )

    for layer in layers:
        if counts[layer] == 0:
            raise ValueError("No clean_mean samples captured")
        means[layer] = means[layer] / counts[layer]

    return means


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP D AIE head sweep.")
    parser.add_argument("--model", default="gpt2", help="Model name or path (default: gpt2)")
    parser.add_argument("--layers", default="0", help="Layer list (default: 0)")
    parser.add_argument("--heads", default="all", help="Head list or 'all' (default: all)")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials (default: 20)")
    parser.add_argument(
        "--n_icl_examples",
        type=int,
        default=3,
        help="Number of ICL demos per prompt (default: 3)",
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
    parser.add_argument(
        "--save_trials",
        type=int,
        default=0,
        help="Save trial-level CSV (default: 0)",
    )
    args = parser.parse_args()

    if args.n_trials < 1:
        print("n_trials must be >= 1")
        return 1
    if args.n_icl_examples < 1:
        print("n_icl_examples must be >= 1")
        return 1

    run_info = prepare_run_dirs(args.run_id)
    if args.out_dir:
        artifacts_dir = resolve_out_dir(args.out_dir)
    else:
        artifacts_dir = run_info["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    log_path = os.path.join(run_info["logs_dir"], "stepD_aie.log")
    log, log_file = make_logger(log_path)

    log("stepD AIE head sweep start")
    log(f"run_id: {run_info['run_id']}")
    log(f"artifacts_dir: {artifacts_dir}")
    log(f"log_path: {log_path}")
    log(f"model: {args.model}")
    log(f"layers: {args.layers}")
    log(f"heads: {args.heads}")
    log(f"n_trials: {args.n_trials}")
    log(f"n_icl_examples: {args.n_icl_examples}")
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

    try:
        layers = parse_layers(args.layers)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1

    n_heads = int(model_cfg["n_heads"])
    head_dim = int(model_cfg["head_dim"])
    resid_dim = int(model_cfg["resid_dim"])

    try:
        heads = parse_heads(args.heads, n_heads)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1
    if not heads:
        log("No heads selected")
        log_file.close()
        return 1

    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        log("Model does not expose transformer.h blocks.")
        log_file.close()
        return 1
    n_layers = len(model.transformer.h)
    for layer in layers:
        if layer < 0 or layer >= n_layers:
            log("Layer index out of range")
            log_file.close()
            return 1
    for head in heads:
        if head < 0 or head >= n_heads:
            log("Head index out of range")
            log_file.close()
            return 1

    if args.n_icl_examples + 1 > len(ANTONYM_PAIRS):
        log("Not enough antonym pairs for requested demos + query.")
        log_file.close()
        return 1

    rng = random.Random(args.seed)
    trials = []

    for trial_idx in range(args.n_trials):
        pairs = rng.sample(ANTONYM_PAIRS, args.n_icl_examples + 1)
        demos = pairs[: args.n_icl_examples]
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
                "clean_prefix_str": clean_prefix_str,
                "corrupted_prefix_str": corrupted_prefix_str,
                "target_id": clean_slot["target_id"],
                "target_token": clean_slot["target_token"],
            }
        )

    layer_modules = {}
    for layer in layers:
        target_module, _target_name = get_c_proj_pre_hook(model, layer)
        layer_modules[layer] = target_module

    log("computing clean_mean")
    try:
        clean_mean = compute_clean_mean(
            [t["clean_prefix_str"] for t in trials],
            model,
            tokenizer,
            device,
            layers,
            n_heads,
            head_dim,
            resid_dim,
            layer_modules,
            log,
        )
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1

    clean_mean_path = os.path.join(artifacts_dir, "clean_mean.pt")
    torch.save(
        {
            "layers": layers,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "resid_dim": resid_dim,
            "clean_mean": {layer: clean_mean[layer].cpu() for layer in layers},
        },
        clean_mean_path,
    )
    log(f"saved clean_mean: {clean_mean_path}")

    scores = []
    trials_rows = []

    log("starting AIE sweep")
    for layer in layers:
        for head in heads:
            cie_vals = []
            nonzero = False
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

                replace_vec = clean_mean[layer][head]
                hook = make_cproj_head_replacer(
                    layer_idx=layer,
                    head_idx=head,
                    token_idx=-1,
                    mode="replace",
                    replace_vec=replace_vec,
                    model_config=model_cfg,
                    logger=None,
                )
                handle = layer_modules[layer].register_forward_pre_hook(hook)
                with torch.inference_mode():
                    outputs_patched = model(**inputs)
                handle.remove()

                patched_logits = outputs_patched.logits[0, last_index]
                patched_logit = patched_logits[target_id].item()
                cie = patched_logit - baseline_logit
                cie_vals.append(cie)
                if abs(cie) > 1e-12:
                    nonzero = True

                if args.save_trials:
                    trials_rows.append(
                        {
                            "trial_idx": trial["trial_idx"],
                            "layer": layer,
                            "head": head,
                            "baseline_logit": baseline_logit,
                            "patched_logit": patched_logit,
                            "cie": cie,
                            "target_id": target_id,
                        }
                    )

            clean_norm = clean_mean[layer][head].norm().item()
            scores.append(
                {
                    "layer": layer,
                    "head": head,
                    "n_trials": args.n_trials,
                    "mean_cie": mean(cie_vals),
                    "std_cie": std(cie_vals),
                    "mean_abs_cie": mean_abs(cie_vals),
                    "any_nonzero": nonzero,
                    "clean_mean_norm": clean_norm,
                }
            )

    scores_sorted = sorted(scores, key=lambda row: row["mean_cie"], reverse=True)
    log("top-10 by mean_cie:")
    for row in scores_sorted[:10]:
        log(
            "top: "
            f"layer={row['layer']} "
            f"head={row['head']} "
            f"mean_cie={row['mean_cie']:.6f} "
            f"std_cie={row['std_cie']:.6f}"
        )

    scores_path = os.path.join(artifacts_dir, "aie_scores.csv")
    save_csv(scores_path, scores_sorted)

    scores_json_path = os.path.join(artifacts_dir, "aie_scores.json")
    save_json(
        scores_json_path,
        {
            "meta": {
                "model": args.model,
                "layers": layers,
                "heads": heads,
                "n_trials": args.n_trials,
                "n_icl_examples": args.n_icl_examples,
                "seed": args.seed,
                "token_idx": -1,
                "measure": "logit[target_id]",
            },
            "scores": scores_sorted,
        },
    )

    log(f"saved scores: {scores_path}")
    log(f"saved scores json: {scores_json_path}")

    if args.save_trials:
        trials_path = os.path.join(artifacts_dir, "aie_trials.csv")
        save_csv(trials_path, trials_rows)
        log(f"saved trials: {trials_path}")

    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
