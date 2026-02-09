#!/usr/bin/env python3
"""Fail-fast preflight gate for model/spec changes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_or_fail(cmd, label: str) -> int:
    print(f"[preflight] {label}")
    print("[preflight] cmd: " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[preflight] FAILED at {label} (code={result.returncode})")
        return int(result.returncode)
    print(f"[preflight] PASS {label}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fail-fast model-change preflight")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--model", required=True, help="HF model name or path")
    parser.add_argument("--model_spec", required=True, help="Model spec key")
    parser.add_argument(
        "--out_dir",
        default="artifacts/model_change_preflight",
        help="Preflight artifact directory",
    )
    parser.add_argument(
        "--dataset_path",
        default="datasets/processed/antonym.json",
        help="Dataset path for StepD alignment check",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument(
        "--align_n_trials",
        type=int,
        default=5,
        help="StepD alignment check trials",
    )
    parser.add_argument(
        "--align_n_icl_examples",
        type=int,
        default=3,
        help="StepD alignment ICL demos",
    )
    parser.add_argument(
        "--run_step6_smoke",
        type=int,
        default=0,
        help="Run Step6 smoke gate (1/0)",
    )
    parser.add_argument(
        "--step6_fv_global_path",
        default=None,
        help="Step6 smoke: fv_global_resid.pt path",
    )
    parser.add_argument(
        "--step6_sampled_trials_path",
        default=None,
        help="Step6 smoke: sampled_trials.json path",
    )
    parser.add_argument(
        "--step6_edit_layer",
        type=int,
        default=0,
        help="Step6 smoke: edit layer",
    )
    parser.add_argument(
        "--step6_n_eval",
        type=int,
        default=4,
        help="Step6 smoke: n_eval",
    )
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir = out_dir / "hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    audit_cmd = [
        args.python,
        "scripts/run_m5_cpu_model_spec_audit.py",
        "--spec_keys",
        args.model_spec,
        "--l2_spec_keys",
        args.model_spec,
        "--out_dir",
        str(out_dir / "model_spec_audit"),
        "--hf_cache_dir",
        str(hf_cache_dir),
    ]
    rc = run_or_fail(audit_cmd, "M5-A tokenizer/resolver audit")
    if rc != 0:
        return rc

    align_cmd = [
        args.python,
        "scripts/run_stepD_alignment_check.py",
        "--model",
        args.model,
        "--model_spec",
        args.model_spec,
        "--dataset_path",
        args.dataset_path,
        "--n_trials",
        str(args.align_n_trials),
        "--n_icl_examples",
        str(args.align_n_icl_examples),
        "--seed",
        str(args.seed),
    ]
    rc = run_or_fail(align_cmd, "StepD slot alignment check")
    if rc != 0:
        return rc

    if int(args.run_step6_smoke) == 1:
        if not args.step6_fv_global_path or not args.step6_sampled_trials_path:
            print(
                "[preflight] step6 smoke requires "
                "--step6_fv_global_path and --step6_sampled_trials_path"
            )
            return 2
        step6_cmd = [
            args.python,
            "scripts/run_step6_fv_injection_eval.py",
            "--model",
            args.model,
            "--model_spec",
            args.model_spec,
            "--edit_layer",
            str(args.step6_edit_layer),
            "--fv_global_path",
            args.step6_fv_global_path,
            "--sampled_trials_path",
            args.step6_sampled_trials_path,
            "--n_eval",
            str(args.step6_n_eval),
            "--batch_size",
            "1",
            "--seed",
            str(args.seed),
            "--out_dir",
            str(out_dir / "step6_smoke"),
        ]
        rc = run_or_fail(step6_cmd, "Step6 smoke eval")
        if rc != 0:
            return rc

    print("[preflight] ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
