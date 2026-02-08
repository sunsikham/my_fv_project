#!/usr/bin/env python3
"""Run M0-M3 parity checks as a single suite with structured outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class StageResult:
    name: str
    cmd: list[str]
    exit_code: int
    mismatch_count: Optional[int]
    max_abs_diff: Optional[float]
    duration_sec: float
    status: str
    stdout: str
    stderr: str


def _parse_mismatch_count(output: str) -> Optional[int]:
    m = re.search(r"mismatch_count:\s*([0-9]+)", output)
    if not m:
        return None
    return int(m.group(1))


def _parse_max_abs_diff(output: str) -> Optional[float]:
    m = re.search(r"max_abs_diff:\s*([0-9eE+\-.]+)", output)
    if m:
        return float(m.group(1))
    m = re.search(r"max_abs_diff_with_logits:\s*([0-9eE+\-.]+)", output)
    if m:
        return float(m.group(1))
    return None


def _stage_status(exit_code: int, mismatch_count: Optional[int]) -> str:
    if exit_code != 0:
        return "FAIL"
    if mismatch_count is not None and mismatch_count != 0:
        return "FAIL"
    return "PASS"


def _run_stage(name: str, cmd: list[str]) -> StageResult:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    duration_sec = time.perf_counter() - start
    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    mismatch_count = _parse_mismatch_count(combined)
    max_abs_diff = _parse_max_abs_diff(combined)
    status = _stage_status(proc.returncode, mismatch_count)
    return StageResult(
        name=name,
        cmd=cmd,
        exit_code=proc.returncode,
        mismatch_count=mismatch_count,
        max_abs_diff=max_abs_diff,
        duration_sec=duration_sec,
        status=status,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M0-M3 parity suite.")
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
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--max_trials", type=int, default=5)
    parser.add_argument("--n_top_heads", type=int, default=10)
    parser.add_argument("--token_class_idx", type=int, default=-1)
    parser.add_argument("--edit_layer", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--results_fv_root", type=str, default="results_fv")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--fail_fast",
        type=int,
        default=0,
        choices=[0, 1],
        help="Stop at first failed stage if 1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = PROJECT_ROOT / args.results_fv_root / args.dataset_name / args.fixed_trials_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "parity_suite.log"
    report_path = out_dir / "parity_suite_report.json"
    csv_path = out_dir / "parity_suite_stages.csv"

    py = sys.executable
    stages: list[tuple[str, list[str]]] = [
        (
            "prompt",
            [
                py,
                "scripts/verify_prompt_parity.py",
                "--fixed_trials_path",
                args.fixed_trials_path,
                "--max_trials",
                str(args.max_trials),
                "--model_name_for_tokenizer",
                args.model_name,
            ],
        ),
        (
            "slot",
            [
                py,
                "scripts/verify_slot_parity_against_src.py",
                "--fixed_trials_path",
                args.fixed_trials_path,
                "--max_trials",
                str(args.max_trials),
                "--mode",
                "corrupted",
                "--tokenizer_name",
                args.model_name,
                "--assert_zero",
            ],
        ),
        (
            "fv",
            [
                py,
                "scripts/verify_fv_parity.py",
                "--dataset_name",
                args.dataset_name,
                "--fixed_trials_id",
                args.fixed_trials_id,
                "--results_root",
                args.results_root,
                "--model_name",
                args.model_name,
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--n_top_heads",
                str(args.n_top_heads),
                "--token_class_idx",
                str(args.token_class_idx),
                "--assert_zero",
            ],
        ),
        (
            "injection",
            [
                py,
                "scripts/verify_injection_parity.py",
                "--dataset_name",
                args.dataset_name,
                "--fixed_trials_path",
                args.fixed_trials_path,
                "--fixed_trials_id",
                args.fixed_trials_id,
                "--results_root",
                args.results_root,
                "--results_fv_root",
                args.results_fv_root,
                "--model_name",
                args.model_name,
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--n_top_heads",
                str(args.n_top_heads),
                "--token_class_idx",
                str(args.token_class_idx),
                "--edit_layer",
                str(args.edit_layer),
                "--max_trials",
                str(args.max_trials),
                "--assert_zero",
            ],
        ),
    ]

    results: list[StageResult] = []
    with log_path.open("w", encoding="utf-8") as logf:
        for name, cmd in stages:
            logf.write(f"== stage: {name}\n")
            logf.write(f"cmd: {' '.join(cmd)}\n")
            result = _run_stage(name, cmd)
            results.append(result)
            logf.write(f"exit_code: {result.exit_code}\n")
            logf.write(f"status: {result.status}\n")
            if result.mismatch_count is not None:
                logf.write(f"mismatch_count: {result.mismatch_count}\n")
            if result.max_abs_diff is not None:
                logf.write(f"max_abs_diff: {result.max_abs_diff}\n")
            logf.write("-- stdout --\n")
            logf.write(result.stdout)
            if not result.stdout.endswith("\n"):
                logf.write("\n")
            logf.write("-- stderr --\n")
            logf.write(result.stderr)
            if not result.stderr.endswith("\n"):
                logf.write("\n")
            logf.write("\n")
            logf.flush()
            print(
                f"[{name}] status={result.status} exit={result.exit_code} "
                f"mismatch_count={result.mismatch_count}"
            )
            if args.fail_fast == 1 and result.status == "FAIL":
                break

    earliest_failed_stage = next((r.name for r in results if r.status == "FAIL"), None)
    overall_status = "FAIL" if earliest_failed_stage else "PASS"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "status",
                "exit_code",
                "mismatch_count",
                "max_abs_diff",
                "duration_sec",
                "cmd",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "name": r.name,
                    "status": r.status,
                    "exit_code": r.exit_code,
                    "mismatch_count": r.mismatch_count,
                    "max_abs_diff": r.max_abs_diff,
                    "duration_sec": f"{r.duration_sec:.6f}",
                    "cmd": " ".join(r.cmd),
                }
            )

    report = {
        "status": overall_status,
        "dataset_name": args.dataset_name,
        "fixed_trials_id": args.fixed_trials_id,
        "model_name": args.model_name,
        "seed": args.seed,
        "earliest_failed_stage": earliest_failed_stage,
        "stages": [
            {
                "name": r.name,
                "status": r.status,
                "cmd": " ".join(r.cmd),
                "exit_code": r.exit_code,
                "mismatch_count": r.mismatch_count,
                "max_abs_diff": r.max_abs_diff,
                "duration_sec": r.duration_sec,
            }
            for r in results
        ],
        "paths": {
            "log": str(log_path),
            "csv": str(csv_path),
            "report": str(report_path),
        },
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"suite_status: {overall_status}")
    print(f"earliest_failed_stage: {earliest_failed_stage}")
    print(f"report_json: {report_path}")
    print(f"stages_csv: {csv_path}")
    print(f"log_path: {log_path}")
    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
