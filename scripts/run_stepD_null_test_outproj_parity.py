#!/usr/bin/env python3
"""Null test for StepD out_proj recompute parity.

Example:
  python scripts/run_stepD_null_test_outproj_parity.py \
    --model meta-llama/Llama-3.1-8B \
    --model_spec llama3 \
    --dataset_path datasets/processed/antonym.json \
    --n_trials 50 \
    --layer 0 --head 0 \
    --quant 4bit \
    --device cuda --dtype fp16
"""

import argparse
import csv
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.adapters import infer_head_dims, resolve_blocks
from fv.corrupt import make_corrupted_demos
from fv.dataset_loader import load_pairs_antonym, sample_demos_and_query
from fv.hf_loader import load_hf_model_and_tokenizer
from fv.hooks import get_out_proj_pre_hook_target
from fv.io import prepare_run_dirs
from fv.model_spec import get_model_spec
from fv.patch import make_out_proj_head_output_overrider
from fv.prompting import build_prompt_qa
from fv.slots import get_target_first_token_id_from_boundary


def _boundary_prefix_and_answer_from_full(prefix_str: str, full_str: str) -> Tuple[str, str]:
    answer_str = full_str[len(prefix_str) :]
    if prefix_str.endswith(" ") and not answer_str.startswith(" "):
        return prefix_str[:-1], f" {answer_str}"
    return prefix_str, answer_str


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


@contextmanager
def out_proj_hook_ctx(module, hook_fn):
    handle = module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def compute_target_id(
    prefix_str: str,
    full_str: str,
    tokenizer,
    tok_add_special: bool,
) -> int:
    boundary_prefix, boundary_answer = _boundary_prefix_and_answer_from_full(
        prefix_str, full_str
    )
    return get_target_first_token_id_from_boundary(
        boundary_prefix,
        boundary_answer,
        tokenizer,
        tokenize_kwargs={"add_special_tokens": tok_add_special},
    )


def collect_trial_prompts(
    pairs: List[Tuple[str, str]],
    tokenizer,
    tok_add_special: bool,
    seed: int,
    n_icl_examples: int,
    n_trials: int,
) -> List[Dict[str, object]]:
    trials: List[Dict[str, object]] = []
    for trial_idx in range(n_trials):
        demos, query = sample_demos_and_query(
            pairs, n_icl_examples, seed=seed + trial_idx
        )
        corrupted_demos = make_corrupted_demos(
            demos, random.Random(seed + trial_idx), ensure_derangement=True
        )
        prefix_str, full_str = build_prompt_qa(corrupted_demos, query)
        target_id = compute_target_id(prefix_str, full_str, tokenizer, tok_add_special)
        trials.append(
            {
                "trial_idx": trial_idx,
                "prefix_str": prefix_str,
                "full_str": full_str,
                "target_id": target_id,
            }
        )
    return trials


def main() -> int:
    parser = argparse.ArgumentParser(description="StepD out_proj parity null test.")
    parser.add_argument("--run_id", default=None, help="Run identifier (default: auto)")
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
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--n_icl_examples", type=int, default=2, help="ICL demos per prompt")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--layer", type=int, default=0, help="Layer index")
    parser.add_argument("--head", type=int, default=0, help="Head index")
    parser.add_argument(
        "--quant",
        default="none",
        choices=["auto", "none", "4bit", "8bit"],
        help="Quantization option",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["fp32", "fp16", "bf16"],
        help="Model dtype (default: fp32 on CPU, fp16 on CUDA)",
    )
    parser.add_argument(
        "--device_map",
        default=None,
        help="Optional HF device_map (default: None or 'auto')",
    )
    tok_group = parser.add_mutually_exclusive_group()
    tok_group.add_argument(
        "--tok_add_special",
        action="store_true",
        help="Force add_special_tokens=True for prompts",
    )
    tok_group.add_argument(
        "--no_tok_add_special",
        action="store_true",
        help="Force add_special_tokens=False for prompts",
    )
    parser.add_argument(
        "--null_alpha",
        type=float,
        default=0.0,
        help="Alpha for null patch (0.0 should match self)",
    )
    parser.add_argument(
        "--warn_threshold",
        type=float,
        default=1e-4,
        help="Warn if max_abs_delta_logit_self_vs_null exceeds this",
    )
    args = parser.parse_args()

    run_info = prepare_run_dirs(args.run_id)
    out_dir = os.path.join(run_info["artifacts_dir"], "null_test")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "null_test_results.csv")

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - runtime import check
        print(f"Failed to import torch: {exc}")
        return 1

    try:
        spec = get_model_spec(args.model_spec)
    except ValueError as exc:
        print(str(exc))
        return 1

    if args.device in (None, "auto"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype is None:
        args.dtype = "fp16" if args.device == "cuda" else "fp32"

    try:
        device = resolve_device(args.device, torch)
    except ValueError as exc:
        print(str(exc))
        return 1

    if device.type == "cpu" and args.dtype in {"fp16", "bf16"}:
        print(f"cpu does not support {args.dtype}; forcing fp32")
        args.dtype = "fp32"

    try:
        _dtype = resolve_dtype(args.dtype, torch)
    except ValueError as exc:
        print(str(exc))
        return 1

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        pairs = load_pairs_antonym(args.dataset_path, canonical_by_input=True)
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        return 1
    if len(pairs) < args.n_icl_examples + 1:
        print(
            "Not enough dataset pairs for requested demos + query: "
            f"pairs={len(pairs)} n_icl_examples={args.n_icl_examples}"
        )
        return 1

    loader_device = None if args.device_map else args.device
    try:
        model, tokenizer, diagnostics = load_hf_model_and_tokenizer(
            model_name=args.model,
            model_spec=args.model_spec,
            device=loader_device,
            dtype=args.dtype,
            quant=args.quant,
            device_map=args.device_map,
        )
    except Exception as exc:  # pragma: no cover - runtime load check
        print(f"Failed to load model '{args.model}': {exc}")
        return 1

    resolved_quant = diagnostics.get("resolved_quant") if diagnostics else None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    if args.device_map:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            pass
    elif resolved_quant in {"4bit", "8bit"}:
        pass
    else:
        model.to(device)
    model.eval()

    try:
        dims = infer_head_dims(model, spec_name=args.model_spec)
    except ValueError as exc:
        print(str(exc))
        return 1

    n_heads = int(dims["n_heads"])
    head_dim = int(dims["head_dim"])
    resid_dim = int(dims["hidden_size"])

    try:
        blocks = resolve_blocks(model, spec, logger=print)
    except ValueError as exc:
        print(str(exc))
        return 1

    layer_count = len(blocks)
    if args.layer < 0 or args.layer >= layer_count:
        print(f"Layer index out of range: {args.layer}")
        return 1
    if args.head < 0 or args.head >= n_heads:
        print(f"Head index out of range: {args.head}")
        return 1

    if args.tok_add_special:
        tok_add_special = True
    elif args.no_tok_add_special:
        tok_add_special = False
    else:
        tok_add_special = bool(spec.prepend_bos)

    model_cfg = {
        "n_heads": n_heads,
        "head_dim": head_dim,
        "resid_dim": resid_dim,
        "n_layers": layer_count,
    }

    try:
        target_module, _target_name = get_out_proj_pre_hook_target(
            model, args.layer, spec_name=args.model_spec, logger=print
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    try:
        trials = collect_trial_prompts(
            pairs,
            tokenizer,
            tok_add_special,
            seed=args.seed,
            n_icl_examples=args.n_icl_examples,
            n_trials=args.n_trials,
        )
    except Exception as exc:
        print(str(exc))
        return 1

    replace_vec = torch.zeros((head_dim,), dtype=torch.float32)
    hook_state = {"mode": "self", "replace_vec": replace_vec, "alpha": 1.0}
    hook_fn = make_out_proj_head_output_overrider(
        layer_idx=args.layer,
        head_idx=args.head,
        token_idx=-1,
        mode="self",
        replace_vec=None,
        model_config=model_cfg,
        resolved_quant=resolved_quant,
        force_recompute_outproj=True,
        state=hook_state,
        logger=print,
    )

    fieldnames = [
        "trial_idx",
        "prompt_len",
        "target_id",
        "p_raw",
        "p_self",
        "p_null",
        "logit_raw",
        "logit_self",
        "logit_null",
        "max_abs_delta_logit_self_vs_null",
        "abs_delta_logit_target_self_vs_null",
        "abs_delta_prob_target_self_vs_null",
        "abs_delta_logprob_target_self_vs_null",
        "max_abs_delta_logit_raw_vs_self",
    ]

    max_self_null = []
    max_raw_self = []
    debug_printed = False

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trial in trials:
            prefix_str = trial["prefix_str"]
            target_id = trial["target_id"]

            inputs = tokenizer(
                prefix_str, return_tensors="pt", add_special_tokens=tok_add_special
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            seq_len = inputs["input_ids"].shape[1]
            prompt_len = int(seq_len)

            with torch.inference_mode():
                outputs_raw = model(**inputs)
            logits_raw = outputs_raw.logits[0, -1]

            hook_state["mode"] = "self"
            hook_state["replace_vec"] = replace_vec
            hook_state["alpha"] = 1.0
            with out_proj_hook_ctx(target_module, hook_fn):
                with torch.inference_mode():
                    outputs_self = model(**inputs)
            logits_self = outputs_self.logits[0, -1]

            hook_state["mode"] = "replace"
            hook_state["replace_vec"] = replace_vec
            hook_state["alpha"] = float(args.null_alpha)
            with out_proj_hook_ctx(target_module, hook_fn):
                with torch.inference_mode():
                    outputs_null = model(**inputs)
            logits_null = outputs_null.logits[0, -1]

            diff_self_null = (logits_self - logits_null).abs()
            diff_raw_self = (logits_raw - logits_self).abs()
            max_abs_self_null = diff_self_null.max().item()
            max_abs_raw_self = diff_raw_self.max().item()

            p_raw = F.softmax(logits_raw, dim=-1)[target_id].item()
            p_self = F.softmax(logits_self, dim=-1)[target_id].item()
            p_null = F.softmax(logits_null, dim=-1)[target_id].item()

            logit_raw = logits_raw[target_id].item()
            logit_self = logits_self[target_id].item()
            logit_null = logits_null[target_id].item()

            logprob_self = F.log_softmax(logits_self, dim=-1)[target_id].item()
            logprob_null = F.log_softmax(logits_null, dim=-1)[target_id].item()

            writer.writerow(
                {
                    "trial_idx": trial["trial_idx"],
                    "prompt_len": prompt_len,
                    "target_id": target_id,
                    "p_raw": p_raw,
                    "p_self": p_self,
                    "p_null": p_null,
                    "logit_raw": logit_raw,
                    "logit_self": logit_self,
                    "logit_null": logit_null,
                    "max_abs_delta_logit_self_vs_null": max_abs_self_null,
                    "abs_delta_logit_target_self_vs_null": abs(logit_self - logit_null),
                    "abs_delta_prob_target_self_vs_null": abs(p_self - p_null),
                    "abs_delta_logprob_target_self_vs_null": abs(
                        logprob_self - logprob_null
                    ),
                    "max_abs_delta_logit_raw_vs_self": max_abs_raw_self,
                }
            )

            max_self_null.append(max_abs_self_null)
            max_raw_self.append(max_abs_raw_self)

            if not debug_printed and max_abs_self_null > args.warn_threshold:
                diff_top = torch.topk(diff_self_null, k=5)
                idxs = diff_top.indices.tolist()
                vals = diff_top.values.tolist()
                print(
                    "WARN: self vs null delta exceeds threshold "
                    f"(trial={trial['trial_idx']} max={max_abs_self_null:.6g})"
                )
                print(
                    "target logits: "
                    f"id={target_id} "
                    f"raw={logit_raw:.6g} "
                    f"self={logit_self:.6g} "
                    f"null={logit_null:.6g}"
                )
                print("top diff ids:", list(zip(idxs, vals)))
                debug_printed = True

    mean_self_null = sum(max_self_null) / len(max_self_null) if max_self_null else 0.0
    max_self_null_val = max(max_self_null) if max_self_null else 0.0
    mean_raw_self = sum(max_raw_self) / len(max_raw_self) if max_raw_self else 0.0
    max_raw_self_val = max(max_raw_self) if max_raw_self else 0.0

    print("SELF vs NULL_PATCH max_abs_delta_logit:")
    print(f"  mean={mean_self_null:.6g} max={max_self_null_val:.6g}")
    print("RAW vs SELF max_abs_delta_logit:")
    print(f"  mean={mean_raw_self:.6g} max={max_raw_self_val:.6g}")
    if max_self_null_val > args.warn_threshold:
        print(
            "WARNING: SELF vs NULL_PATCH exceeds threshold "
            f"({max_self_null_val:.6g} > {args.warn_threshold:.6g})"
        )

    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
