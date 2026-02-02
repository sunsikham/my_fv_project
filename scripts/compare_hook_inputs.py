#!/usr/bin/env python3
"""Compare out_proj input tensors between paper and StepD pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.fixed_trials_adapter import gather_attn_activations
from fv.hf_loader import load_hf_model_and_tokenizer
from fv.hooks import get_out_proj_pre_hook_target
from fv.model_spec import get_model_spec
from fv.prompting import get_dummy_token_labels, get_token_meta_labels

EPS = 1e-12


def _infer_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _load_fixed_trial(path: str, trial_idx: int) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    trials = data.get("trials", data)
    if not trials:
        raise ValueError("fixed_trials has no trials")
    if trial_idx < 0 or trial_idx >= len(trials):
        raise ValueError(f"trial_idx {trial_idx} out of range (n={len(trials)})")
    return trials[trial_idx]


def _compute_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: paper={tuple(a.shape)} stepd={tuple(b.shape)}")
    a_flat = a.flatten()
    b_flat = b.flatten()
    diff = a_flat - b_flat
    max_abs = diff.abs().max().item()
    dot = torch.dot(a_flat, b_flat)
    norm_a = torch.linalg.norm(a_flat)
    norm_b = torch.linalg.norm(b_flat)
    cosine = (dot / (norm_a * norm_b + EPS)).item()
    rel_l2 = (torch.linalg.norm(diff) / (norm_a + EPS)).item()
    return {"max_abs_diff": max_abs, "cosine": cosine, "rel_l2": rel_l2}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare out_proj input tensors (paper vs StepD).")
    parser.add_argument(
        "--fixed_trials_path",
        default="datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0_WITH_DEMOS_STEPD_LLAMA.json",
        help="Path to fixed_trials JSON",
    )
    parser.add_argument("--trial_idx", type=int, default=0, help="Trial index to compare")
    parser.add_argument("--layer", type=int, default=1, help="Layer index to compare")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B", help="HF model name")
    parser.add_argument("--paper_device", default="cuda", help="Device for paper model loader")
    parser.add_argument("--stepd_device", default="cuda", help="Device for StepD model loader")
    parser.add_argument("--stepd_model_spec", default="llama3", help="Model spec for StepD loader")
    parser.add_argument("--stepd_dtype", default="fp16", help="StepD dtype (fp16/bf16/fp32)")
    parser.add_argument("--stepd_quant", default="4bit", help="StepD quant (none/4bit/8bit/auto)")
    parser.add_argument("--stepd_device_map", default=None, help="StepD device_map (e.g. auto)")
    parser.add_argument("--stepd_trust_remote_code", action="store_true", help="Enable trust_remote_code")
    args = parser.parse_args()

    trial = _load_fixed_trial(args.fixed_trials_path, args.trial_idx)
    prompt_data = trial.get("prompt_data_clean")
    if prompt_data is None:
        raise ValueError("fixed_trials missing prompt_data_clean")

    # Paper pipeline: load model/tokenizer and capture out_proj input via TraceDict.
    paper_model, paper_tokenizer, _diag = load_hf_model_and_tokenizer(
        model_name=args.model_name,
        model_spec=args.stepd_model_spec,
        device=args.paper_device,
        dtype=None,
        quant="none",
        device_map=None,
    )
    paper_model.eval()

    spec = get_model_spec(args.stepd_model_spec)
    paper_cfg = {
        "n_layers": spec.n_layers,
        "n_heads": spec.n_heads,
        "resid_dim": spec.hidden_size,
        "attn_hook_names": [
            f"{spec.blocks_path}.{i}.{spec.attn_path_in_block}.{spec.out_proj_path_in_attn}"
            for i in range(spec.n_layers)
        ],
        "prepend_bos": spec.prepend_bos,
    }

    n_icl_examples = len(prompt_data.get("examples", []))
    dummy_labels = get_dummy_token_labels(
        n_icl_examples,
        tokenizer=paper_tokenizer,
        model_config=paper_cfg,
        prefixes=prompt_data.get("prefixes"),
        separators=prompt_data.get("separators"),
    )

    layer_name = paper_cfg["attn_hook_names"][args.layer]
    td, _idx_map, _idx_avg = gather_attn_activations(
        prompt_data=prompt_data,
        layers=[layer_name],
        dummy_labels=dummy_labels,
        model=paper_model,
        tokenizer=paper_tokenizer,
        model_config=paper_cfg,
    )
    paper_tensor = td[layer_name].input
    if paper_tensor.dim() != 3:
        raise ValueError(f"paper hook tensor shape unexpected: {tuple(paper_tensor.shape)}")
    paper_tensor = paper_tensor[0].detach().float().cpu()

    # StepD pipeline: load model/tokenizer and capture out_proj input via pre-hook.
    stepd_model, stepd_tokenizer, _diag = load_hf_model_and_tokenizer(
        model_name=args.model_name,
        model_spec=args.stepd_model_spec,
        device=args.stepd_device,
        dtype=args.stepd_dtype,
        quant=args.stepd_quant,
        device_map=args.stepd_device_map,
        trust_remote_code=args.stepd_trust_remote_code,
    )
    stepd_model.eval()

    _token_labels, prompt_string = get_token_meta_labels(
        prompt_data,
        paper_tokenizer,
        prepend_bos=paper_cfg["prepend_bos"],
    )
    inputs = stepd_tokenizer(
        prompt_string,
        return_tensors="pt",
        add_special_tokens=True,
    )
    stepd_device = _infer_device(stepd_model)
    inputs = {key: value.to(stepd_device) for key, value in inputs.items()}

    hook_tensor = {}

    def _capture_pre_hook(_module, inputs_):
        tensor = inputs_[0] if inputs_ else None
        if tensor is not None:
            hook_tensor["value"] = tensor.detach()

    out_proj, _path = get_out_proj_pre_hook_target(
        stepd_model, args.layer, spec_name=args.stepd_model_spec
    )
    handle = out_proj.register_forward_pre_hook(_capture_pre_hook)

    with torch.inference_mode():
        _ = stepd_model(**inputs)

    handle.remove()
    if "value" not in hook_tensor:
        raise ValueError("StepD hook tensor missing")
    stepd_tensor = hook_tensor["value"]
    if stepd_tensor.dim() != 3:
        raise ValueError(f"stepd hook tensor shape unexpected: {tuple(stepd_tensor.shape)}")
    stepd_tensor = stepd_tensor[0].detach().float().cpu()

    metrics = _compute_metrics(paper_tensor, stepd_tensor)
    print(f"max_abs_diff: {metrics['max_abs_diff']:.6f}")
    print(f"cosine: {metrics['cosine']:.6f}")
    print(f"rel_l2: {metrics['rel_l2']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
