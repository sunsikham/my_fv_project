#!/usr/bin/env python3
"""STEP 6: Evaluate FV injection on 0-shot antonym prompts."""

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.intervene import make_residual_injection_hook
from fv.io import (
    infer_step5_metadata_path,
    load_json,
    prepare_run_dirs,
    resolve_out_dir,
    save_step6_results,
)
from fv.model_config import get_model_config
from fv.prompting import build_zero_shot_prompt


def summarize_prompt(prompt: str, limit: int = 80) -> str:
    if len(prompt) <= limit:
        return prompt
    return prompt[: limit - 3] + "..."


def get_target_token(answer: str, tokenizer) -> Tuple[int, str, bool]:
    with_space = " " + answer
    ids_with_space = tokenizer.encode(with_space, add_special_tokens=False)
    leading_space = True
    token_ids = ids_with_space

    if not token_ids:
        token_ids = tokenizer.encode(answer, add_special_tokens=False)
        leading_space = False

    if not token_ids:
        raise ValueError(f"Failed to tokenize answer: '{answer}'")

    token_id = token_ids[0]
    token_str = tokenizer.convert_ids_to_tokens(token_id)
    return token_id, token_str, leading_space


def parse_answer_override(value: str, tokenizer) -> Dict[str, int]:
    if value == "auto":
        return {}
    if value.isdigit():
        token_id = int(value)
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        return {"override_id": token_id, "override_str": token_str}
    token_ids = tokenizer.encode(value, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Failed to tokenize --answer_first_token: '{value}'")
    token_id = token_ids[0]
    token_str = tokenizer.convert_ids_to_tokens(token_id)
    return {"override_id": token_id, "override_str": token_str}


def resolve_device(device_str: str, torch_module):
    if device_str == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if device_str == "cuda":
        if not torch_module.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        return torch_module.device("cuda")
    if device_str == "cpu":
        return torch_module.device("cpu")
    raise ValueError(f"Unknown device option: {device_str}")


def resolve_dtype(dtype_str: str, torch_module):
    mapping = {
        "fp32": torch_module.float32,
        "fp16": torch_module.float16,
        "bf16": torch_module.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype option: {dtype_str}")
    return mapping[dtype_str]


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP 6 FV injection eval.")
    parser.add_argument("--model", default="gpt2", help="Model name or path (default: gpt2)")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (default: 0)")
    parser.add_argument(
        "--fv_path",
        default="artifacts/step5/fv_gpt2_layer0_n20.pt",
        help="Path to FV tensor",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Injection scale (default: 1.0)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--n_eval", type=int, default=20, help="Number of eval prompts")
    parser.add_argument(
        "--answer_first_token",
        default="auto",
        help="Override target token (string or id), or 'auto'",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Model dtype (default: fp32)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for eval prompts (default: 1)",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: runs/<run_id>/artifacts/step6/)",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Run identifier (default: auto timestamp)",
    )
    args = parser.parse_args()

    run_info = None
    if args.out_dir:
        out_dir = resolve_out_dir(args.out_dir)
    else:
        run_info = prepare_run_dirs(args.run_id)
        out_dir = os.path.join(run_info["artifacts_dir"], "step6")
    os.makedirs(out_dir, exist_ok=True)

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime import check
        print(f"Failed to import required libraries: {exc}")
        return 1

    model_cfg = get_model_config(args.model)
    if model_cfg is None:
        print(f"No model config found for '{args.model}'")
        return 1

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        device = resolve_device(args.device, torch)
    except ValueError as exc:
        print(str(exc))
        return 1

    try:
        dtype = resolve_dtype(args.dtype, torch)
    except ValueError as exc:
        print(str(exc))
        return 1

    if device.type == "cpu" and dtype == torch.float16:
        print("fp16 is not supported on cpu")
        return 1

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"device: cuda ({gpu_name})")
    else:
        print("device: cpu")

    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"seed: {args.seed}")
    print(f"model: {args.model}")
    print(f"layer: {args.layer}")
    print(f"fv_path: {args.fv_path}")
    print(f"alpha: {args.alpha}")
    print(f"dtype: {args.dtype}")
    print(f"batch_size: {args.batch_size}")
    if run_info:
        print(f"run_id: {run_info['run_id']}")
    print(f"out_dir: {out_dir}")
    hook_target = model_cfg["hook_target"].format(layer=args.layer)
    reshape_rule = model_cfg["reshape"]
    print(
        "config: "
        f"n_heads={model_cfg['n_heads']} "
        f"head_dim={model_cfg['head_dim']} "
        f"resid_dim={model_cfg['resid_dim']} "
        f"hook_target={hook_target} "
        f"reshape={reshape_rule}"
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if args.dtype == "fp32":
            model = AutoModelForCausalLM.from_pretrained(args.model)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    except Exception as exc:  # pragma: no cover - runtime load check
        print(f"Failed to load model '{args.model}': {exc}")
        return 1

    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    try:
        fv = torch.load(args.fv_path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - runtime load check
        print(f"Failed to load FV from '{args.fv_path}': {exc}")
        return 1

    if fv.dim() != 1:
        print(f"FV must be 1D, got shape {tuple(fv.shape)}")
        return 1

    hidden_size = getattr(model.config, "n_embd", None)
    if hidden_size is None:
        hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        print("Missing hidden_size in model config.")
        return 1
    num_heads = getattr(model.config, "n_head", None)
    if num_heads is None:
        num_heads = getattr(model.config, "num_attention_heads", None)
    if num_heads is None:
        print("Missing num_heads in model config.")
        return 1

    if hidden_size != model_cfg["resid_dim"] or num_heads != model_cfg["n_heads"]:
        print("Model config does not match expected dimensions.")
        return 1

    if fv.shape[0] != hidden_size:
        print(f"FV size {fv.shape[0]} does not match hidden_size {hidden_size}")
        return 1

    fv = fv.to(device=device, dtype=dtype)
    fv_norm = fv.norm().item()
    print(f"fv shape: {tuple(fv.shape)}")
    print(f"fv norm: {fv_norm:.6f}")

    step5_metadata_path = infer_step5_metadata_path(args.fv_path)
    step5_metadata = None
    if step5_metadata_path and os.path.exists(step5_metadata_path):
        try:
            step5_metadata = load_json(step5_metadata_path)
            print(f"loaded step5 metadata: {step5_metadata_path}")
        except Exception as exc:
            print(f"Failed to load step5 metadata: {exc}")
            return 1
    else:
        print("step5 metadata not found")

    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        print("Model does not expose transformer.h blocks.")
        return 1
    if args.layer < 0 or args.layer >= len(model.transformer.h):
        print("Layer index out of range.")
        return 1

    block = model.transformer.h[args.layer]

    injection_state = {
        "fv": fv,
        "alpha": args.alpha,
        "batch_indices": None,
        "last_indices": None,
    }
    inject_hook = make_residual_injection_hook(injection_state)

    if args.batch_size < 1:
        print("batch_size must be >= 1")
        return 1

    rng = random.Random(args.seed)
    results = []
    sum_base = 0.0
    sum_with = 0.0
    sum_delta = 0.0
    sum_delta_logit = 0.0

    override_info = {}
    try:
        override_info = parse_answer_override(args.answer_first_token, tokenizer)
    except ValueError as exc:
        print(str(exc))
        return 1

    if override_info:
        print(
            "override target: "
            f"id={override_info['override_id']} "
            f"token={override_info['override_str']}"
        )

    print(f"n_eval start: {args.n_eval}")
    idx = 0
    while idx < args.n_eval:
        batch_prompts = []
        batch_answers = []
        batch_targets = []
        batch_target_tokens = []
        batch_leading_spaces = []
        batch_used_ids = []
        batch_used_tokens = []

        remaining = args.n_eval - idx
        batch_size = min(args.batch_size, remaining)
        for _ in range(batch_size):
            prompt, answer = build_zero_shot_prompt(rng)
            try:
                target_id, target_token, leading_space = get_target_token(answer, tokenizer)
            except ValueError as exc:
                print(str(exc))
                return 1

            target_used = target_id
            target_used_token = target_token
            if override_info:
                target_used = override_info["override_id"]
                target_used_token = override_info["override_str"]

            batch_prompts.append(prompt)
            batch_answers.append(answer)
            batch_targets.append(target_id)
            batch_target_tokens.append(target_token)
            batch_leading_spaces.append(leading_space)
            batch_used_ids.append(target_used)
            batch_used_tokens.append(target_used_token)

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            print("Missing attention_mask in tokenizer output.")
            return 1
        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(batch_size, device=device)
        target_ids_tensor = torch.tensor(batch_used_ids, device=device)

        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.logits
        last_logits = logits[batch_indices, last_indices]
        base_logit_vals = last_logits[batch_indices, target_ids_tensor]
        base_prob_vals = torch.softmax(last_logits.float(), dim=-1)[
            batch_indices, target_ids_tensor
        ]

        injection_state["last_indices"] = last_indices
        injection_state["batch_indices"] = batch_indices
        handle = block.register_forward_hook(inject_hook)
        with torch.inference_mode():
            outputs_fv = model(**inputs)
        handle.remove()

        logits_fv = outputs_fv.logits
        last_logits_fv = logits_fv[batch_indices, last_indices]
        with_logit_vals = last_logits_fv[batch_indices, target_ids_tensor]
        with_prob_vals = torch.softmax(last_logits_fv.float(), dim=-1)[
            batch_indices, target_ids_tensor
        ]

        for i in range(batch_size):
            base_prob = base_prob_vals[i].item()
            with_prob = with_prob_vals[i].item()
            base_logit = base_logit_vals[i].item()
            with_logit = with_logit_vals[i].item()
            delta_p = with_prob - base_prob
            delta_logit = with_logit - base_logit

            sum_base += base_prob
            sum_with += with_prob
            sum_delta += delta_p
            sum_delta_logit += delta_logit

            prompt_summary = summarize_prompt(batch_prompts[i])
            print(
                "sample: "
                f"idx={idx} "
                f"prompt='{prompt_summary}' "
                f"answer='{batch_answers[i]}' "
                f"target_id={batch_targets[i]} "
                f"target_token={batch_target_tokens[i]} "
                f"leading_space={batch_leading_spaces[i]} "
                f"used_id={batch_used_ids[i]} "
                f"used_token={batch_used_tokens[i]} "
                f"p_base={base_prob:.6f} "
                f"p_with={with_prob:.6f} "
                f"delta={delta_p:.6f}"
            )

            results.append(
                {
                    "idx": idx,
                    "prompt": batch_prompts[i],
                    "answer": batch_answers[i],
                    "target_id": batch_targets[i],
                    "target_token": batch_target_tokens[i],
                    "leading_space": batch_leading_spaces[i],
                    "used_id": batch_used_ids[i],
                    "used_token": batch_used_tokens[i],
                    "p_base": base_prob,
                    "p_with": with_prob,
                    "delta_p": delta_p,
                    "delta_logit": delta_logit,
                }
            )
            idx += 1

    mean_base = sum_base / args.n_eval if args.n_eval else 0.0
    mean_with = sum_with / args.n_eval if args.n_eval else 0.0
    mean_delta = sum_delta / args.n_eval if args.n_eval else 0.0
    mean_delta_logit = sum_delta_logit / args.n_eval if args.n_eval else 0.0

    print(f"mean(p_base): {mean_base:.6f}")
    print(f"mean(p_with): {mean_with:.6f}")
    print(f"mean(delta p): {mean_delta:.6f}")
    print(f"mean(delta logit): {mean_delta_logit:.6f}")

    step6_metadata = {
        "model": args.model,
        "layer": args.layer,
        "hook_target": hook_target,
        "slot": "query_predictive",
        "slot_name": "query_predictive",
        "slot_index_rule": "s=len(tokenize(prefix_str)); slot_index=s-1; target_id=input_ids[s]",
        "heads": step5_metadata.get("heads") if step5_metadata else None,
        "seed": args.seed,
        "n_eval": args.n_eval,
        "alpha": args.alpha,
        "fv_path": args.fv_path,
        "config": {
            "n_heads": model_cfg["n_heads"],
            "head_dim": model_cfg["head_dim"],
            "resid_dim": model_cfg["resid_dim"],
            "hook_type": model_cfg["hook_type"],
            "reshape": model_cfg["reshape"],
            "dtype": args.dtype,
            "device": str(device),
            "batch_size": args.batch_size,
        },
        "fv": {"shape": list(fv.shape), "norm": fv_norm},
        "step5_metadata_path": step5_metadata_path if step5_metadata else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    payload = {
        "config": {
            "model": args.model,
            "layer": args.layer,
            "fv_path": args.fv_path,
            "seed": args.seed,
            "n_eval": args.n_eval,
            "answer_first_token": args.answer_first_token,
            "alpha": args.alpha,
            "dtype": args.dtype,
            "device": str(device),
            "batch_size": args.batch_size,
        },
        "fv": {"shape": list(fv.shape), "norm": fv_norm},
        "summary": {
            "mean_p_base": mean_base,
            "mean_p_with": mean_with,
            "mean_delta_p": mean_delta,
            "mean_delta_logit": mean_delta_logit,
        },
        "metadata": step6_metadata,
        "step5_metadata": step5_metadata,
        "results": results,
    }

    saved_paths = save_step6_results(
        out_dir,
        args.model,
        args.layer,
        args.n_eval,
        payload,
        step6_metadata,
    )

    print(f"saved results: {saved_paths['results_path']}")
    print(f"metadata saved: {saved_paths['metadata_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
