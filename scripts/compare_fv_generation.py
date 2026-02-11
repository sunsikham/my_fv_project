#!/usr/bin/env python3
"""Compare FV generation between src and fv paths and dump artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.fv import compute_function_vector as fv_compute_function_vector
from src.utils.extract_utils import compute_function_vector as src_compute_function_vector
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed


def _normalize_top_heads(top_heads):
    out = []
    for layer, head, score in top_heads:
        out.append([int(layer), int(head), float(score)])
    return out


def _load_runtime_fv(path: Path, map_location: str):
    blob = torch.load(path, map_location=map_location)
    if isinstance(blob, dict):
        if "fv_global_resid" in blob:
            return blob["fv_global_resid"].reshape(-1)
        if "fv" in blob:
            return blob["fv"].reshape(-1)
    if torch.is_tensor(blob):
        return blob.reshape(-1)
    raise ValueError(f"Unsupported runtime_fv format: {type(blob)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare src/fv FV generation.")
    parser.add_argument("--dataset_name", type=str, default="antonym")
    parser.add_argument(
        "--fixed_trials_id",
        type=str,
        default="fixed_trials_antonym_t10_s10_seed0_llama31_8b",
    )
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_top_heads", type=int, default=10)
    parser.add_argument("--token_class_idx", type=int, default=-1)
    parser.add_argument("--runtime_fv_path", type=str, default="")
    parser.add_argument("--out_dir", type=str, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    torch.set_grad_enabled(False)

    run_dir = PROJECT_ROOT / args.results_root / args.dataset_name / args.fixed_trials_id
    mean_path = run_dir / f"{args.dataset_name}_mean_head_activations_FIXED.pt"
    ie_path = run_dir / f"{args.dataset_name}_indirect_effect.pt"
    if not mean_path.exists() or not ie_path.exists():
        raise FileNotFoundError(
            f"Missing M1 artifacts in {run_dir}; expected {mean_path.name} and {ie_path.name}"
        )

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_activations = torch.load(mean_path, map_location=args.device)
    indirect_effect = torch.load(ie_path, map_location=args.device)

    model, _tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device
    )
    model.eval()

    src_fv, src_top_heads = src_compute_function_vector(
        mean_activations=mean_activations,
        indirect_effect=indirect_effect,
        model=model,
        model_config=model_config,
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

    src_fv_1d = src_fv.reshape(-1).detach().cpu()
    fv_fv_1d = fv_fv.reshape(-1).detach().cpu()

    src_top_heads_n = _normalize_top_heads(src_top_heads)
    fv_top_heads_n = _normalize_top_heads(fv_top_heads)
    top_heads_match = src_top_heads_n == fv_top_heads_n
    max_abs_diff_src_fv = float((src_fv_1d - fv_fv_1d).abs().max().item())
    cosine_src_fv = float(
        torch.nn.functional.cosine_similarity(src_fv_1d, fv_fv_1d, dim=0).item()
    )

    runtime_section = None
    if args.runtime_fv_path:
        runtime_path = PROJECT_ROOT / args.runtime_fv_path
        if not runtime_path.exists():
            raise FileNotFoundError(f"runtime_fv_path not found: {runtime_path}")
        runtime_fv_1d = _load_runtime_fv(runtime_path, map_location="cpu").float()
        runtime_section = {
            "runtime_fv_path": str(runtime_path),
            "runtime_fv_shape": list(runtime_fv_1d.shape),
            "runtime_fv_norm": float(runtime_fv_1d.norm().item()),
            "max_abs_diff_runtime_vs_src": float(
                (runtime_fv_1d - src_fv_1d.float()).abs().max().item()
            ),
            "max_abs_diff_runtime_vs_fv": float(
                (runtime_fv_1d - fv_fv_1d.float()).abs().max().item()
            ),
            "cosine_runtime_vs_src": float(
                torch.nn.functional.cosine_similarity(
                    runtime_fv_1d, src_fv_1d.float(), dim=0
                ).item()
            ),
            "cosine_runtime_vs_fv": float(
                torch.nn.functional.cosine_similarity(
                    runtime_fv_1d, fv_fv_1d.float(), dim=0
                ).item()
            ),
        }
        torch.save(runtime_fv_1d, out_dir / "runtime_fv.pt")

    torch.save(src_fv_1d, out_dir / "src_fv.pt")
    torch.save(fv_fv_1d, out_dir / "fv_fv.pt")
    with open(out_dir / "src_top_heads.json", "w", encoding="utf-8") as f:
        json.dump(src_top_heads_n, f, indent=2, ensure_ascii=True)
    with open(out_dir / "fv_top_heads.json", "w", encoding="utf-8") as f:
        json.dump(fv_top_heads_n, f, indent=2, ensure_ascii=True)

    report = {
        "dataset_name": args.dataset_name,
        "fixed_trials_id": args.fixed_trials_id,
        "model_name": args.model_name,
        "device": args.device,
        "seed": args.seed,
        "n_top_heads": args.n_top_heads,
        "token_class_idx": args.token_class_idx,
        "top_heads_match": top_heads_match,
        "max_abs_diff_src_vs_fv": max_abs_diff_src_fv,
        "cosine_src_vs_fv": cosine_src_fv,
        "src_fv_norm": float(src_fv_1d.norm().item()),
        "fv_fv_norm": float(fv_fv_1d.norm().item()),
        "runtime_compare": runtime_section,
        "artifacts": {
            "src_fv": str(out_dir / "src_fv.pt"),
            "fv_fv": str(out_dir / "fv_fv.pt"),
            "src_top_heads": str(out_dir / "src_top_heads.json"),
            "fv_top_heads": str(out_dir / "fv_top_heads.json"),
        },
    }
    with open(out_dir / "fv_generation_compare_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    print(f"out_dir: {out_dir}")
    print(f"top_heads_match: {top_heads_match}")
    print(f"max_abs_diff_src_vs_fv: {max_abs_diff_src_fv}")
    print(f"cosine_src_vs_fv: {cosine_src_fv}")
    if runtime_section:
        print(
            "runtime_max_abs_diff: "
            f"src={runtime_section['max_abs_diff_runtime_vs_src']} "
            f"fv={runtime_section['max_abs_diff_runtime_vs_fv']}"
        )
    print(f"report: {out_dir / 'fv_generation_compare_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
