#!/usr/bin/env python3
"""Generate and normalize M1 golden artifacts, then verify DoD."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run M1 golden artifact generation and DoD checks."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="antonym",
        help="Logical dataset name used by src scripts.",
    )
    parser.add_argument(
        "--fixed_trials_path",
        type=str,
        default="datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json",
        help="Path to fixed_trials json.",
    )
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_shots", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument(
        "--save_path_root",
        type=str,
        default="results",
        help="Base directory used by src scripts.",
    )
    parser.add_argument(
        "--python_bin",
        type=str,
        default=".venv/bin/python",
        help="Python executable used to run src scripts.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    fixed_trials_path = (repo_root / args.fixed_trials_path).resolve()
    if not fixed_trials_path.exists():
        raise FileNotFoundError(f"fixed_trials_path not found: {fixed_trials_path}")
    fixed_trials_id = fixed_trials_path.stem

    save_root = repo_root / args.save_path_root
    producer_dir = save_root / args.dataset_name
    canonical_run_dir = producer_dir / fixed_trials_id
    canonical_run_dir.mkdir(parents=True, exist_ok=True)

    common_env = "PYTHONPATH=."
    avg_cmd = [
        "env",
        common_env,
        args.python_bin,
        "src/compute_average_activations.py",
        "--dataset_name",
        args.dataset_name,
        "--model_name",
        args.model_name,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--n_shots",
        str(args.n_shots),
        "--n_trials",
        str(args.n_trials),
        "--fixed_trials_path",
        str(fixed_trials_path),
        "--save_path_root",
        args.save_path_root,
    ]

    ie_cmd = [
        "env",
        common_env,
        args.python_bin,
        "src/compute_indirect_effect.py",
        "--dataset_name",
        args.dataset_name,
        "--model_name",
        args.model_name,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--n_shots",
        str(args.n_shots),
        "--n_trials",
        str(args.n_trials),
        "--fixed_trials_path",
        str(fixed_trials_path),
        "--save_path_root",
        args.save_path_root,
        "--last_token_only",
        "True",
    ]

    run_cmd(avg_cmd, repo_root)
    run_cmd(ie_cmd, repo_root)

    required_src_files = [
        (
            f"{args.dataset_name}_mean_head_activations_FIXED.pt",
            producer_dir / f"{args.dataset_name}_mean_head_activations_FIXED.pt",
        ),
        (
            f"{args.dataset_name}_dummy_labels.json",
            producer_dir / f"{args.dataset_name}_dummy_labels.json",
        ),
        (
            f"{args.dataset_name}_indirect_effect.pt",
            producer_dir / f"{args.dataset_name}_indirect_effect.pt",
        ),
    ]
    for out_name, src_path in required_src_files:
        if not src_path.exists():
            raise FileNotFoundError(f"Expected source artifact missing: {src_path}")
        dst_path = canonical_run_dir / out_name
        shutil.copy2(src_path, dst_path)
        print(f"copied: {src_path} -> {dst_path}")

    mean_path = canonical_run_dir / f"{args.dataset_name}_mean_head_activations_FIXED.pt"
    ie_path = canonical_run_dir / f"{args.dataset_name}_indirect_effect.pt"

    mean = torch.load(mean_path, map_location="cpu")
    ie = torch.load(ie_path, map_location="cpu")

    if mean.ndim != 4:
        raise AssertionError(
            f"mean activations rank mismatch: expected 4, got {mean.ndim}"
        )
    if ie.ndim != 3:
        raise AssertionError(
            f"indirect effect rank mismatch: expected 3 (last_token_only), got {ie.ndim}"
        )

    print("DoD checks:")
    print(f"- canonical_run_dir: {canonical_run_dir}")
    print(f"- mean shape: {tuple(mean.shape)}")
    print(f"- indirect_effect shape: {tuple(ie.shape)}")
    print("- required files: OK (3/3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
