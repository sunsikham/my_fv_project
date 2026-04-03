#!/usr/bin/env python3
"""Wrapper for zero-shot FV injection with deterministic split manifest."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def storage_metadata(
    *,
    canonical_root: Path,
    sync_root: Path | None = None,
    sync_mode: str = "none",
    artifact_profile: str = "full",
) -> Dict[str, object]:
    return {
        "canonical_root": str(canonical_root),
        "sync_root": (str(sync_root) if sync_root is not None else None),
        "sync_mode": str(sync_mode),
        "artifact_profile": str(artifact_profile),
    }


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_layers(expr: str) -> List[int]:
    text = (expr or "").strip().lower()
    if not text:
        raise ValueError("layers expression is empty")
    layers: List[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid layer range: {token}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(token))
    uniq = sorted(set(layers))
    if not uniq:
        raise ValueError(f"No layers parsed from: {expr}")
    return uniq


def parse_alphas(text: str) -> List[float]:
    values = [part.strip() for part in (text or "").split(",") if part.strip()]
    if not values:
        raise ValueError("alpha_list must include at least one value")
    return [float(value) for value in values]


def _normalize_eval_trial(row: Dict[str, object], idx: int) -> Dict[str, object]:
    trial_id = str(row.get("trial_id", f"t{idx:06d}"))
    query = row.get("query")
    if not isinstance(query, dict):
        query = {}
    x_val = query.get("input")
    y_val = row.get("target_str") or query.get("output")
    if not isinstance(x_val, str):
        x_val = str(x_val or "")
    if not isinstance(y_val, str):
        y_val = str(y_val or "")
    target_str = f" {y_val.lstrip()}"
    prefix = f"Q: {x_val}\nA:"
    return {
        "q_id": str(row.get("q_id", "unknown")),
        "trial_idx": int(row.get("trial_idx", idx)),
        "trial_id": trial_id,
        "query": {"input": x_val, "output": y_val},
        "target_str": target_str,
        "corrupted_prompt_str": prefix,
        "corrupted_full_str": prefix + target_str,
    }


def build_split_manifest(
    *,
    source_trials: Sequence[Dict[str, object]],
    split_seed: int,
    eval_holdout_ratio: float,
) -> Dict[str, object]:
    if eval_holdout_ratio <= 0.0 or eval_holdout_ratio >= 1.0:
        raise ValueError("eval_holdout_ratio must be in (0, 1)")
    trial_ids: List[str] = []
    for idx, row in enumerate(source_trials):
        trial_ids.append(str(row.get("trial_id", f"t{idx:06d}")))
    uniq_ids = sorted(set(trial_ids))
    if len(uniq_ids) < 2:
        raise ValueError("Need at least 2 unique trial ids for split")

    rng = random.Random(split_seed)
    shuffled = uniq_ids[:]
    rng.shuffle(shuffled)
    eval_n = int(round(len(shuffled) * eval_holdout_ratio))
    eval_n = max(1, min(eval_n, len(shuffled) - 1))
    eval_ids = set(shuffled[:eval_n])
    train_ids = [trial_id for trial_id in shuffled if trial_id not in eval_ids]
    if not train_ids:
        raise ValueError("Empty train split after holdout")
    return {
        "created_at": utc_now(),
        "split_seed": int(split_seed),
        "eval_holdout_ratio": float(eval_holdout_ratio),
        "n_unique_trials": len(shuffled),
        "split_train_fv": sorted(train_ids),
        "split_eval_inject": sorted(eval_ids),
    }


def command_str(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_command(cmd: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] exec: {command_str(cmd)}\n")
        handle.flush()
        proc = subprocess.run(
            list(cmd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed rc={proc.returncode}: {command_str(cmd)}")


def detect_auto_layers(top_heads_path: Path, fallback_layer: int) -> List[int]:
    if not top_heads_path.exists():
        return [fallback_layer]
    payload = read_json(top_heads_path)
    heads = payload.get("heads", [])
    if not isinstance(heads, list) or not heads:
        return [fallback_layer]
    layers = sorted({int(row["layer"]) for row in heads if isinstance(row, dict) and "layer" in row})
    return layers if layers else [fallback_layer]


def _find_step6_result_json(layer_alpha_dir: Path) -> str:
    matches = sorted(glob.glob(str(layer_alpha_dir / "step6_results_*.json")))
    return matches[-1] if matches else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run zero-shot FV injection wrapper.")
    parser.add_argument("--q_dir", required=True, help="Per-q output directory")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_spec", default="llama3")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default=None, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quant", default="auto", choices=["auto", "none", "4bit", "8bit"])
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--fv_name", required=True, help="AAA|BBB|BABA")
    parser.add_argument("--fv_path", default=None, help="Override fv path (.npy)")
    parser.add_argument(
        "--source_trials_path",
        default=None,
        help="Source condition trial JSON (default: _trials/condition_AAA.json)",
    )
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--eval_holdout_ratio", type=float, default=0.3)
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--layer_mode", default="auto", choices=["auto", "list"])
    parser.add_argument("--layers", default="0")
    parser.add_argument("--alpha_list", default="0.5,1.0,1.5")
    parser.add_argument("--score_key", default="mean_delta_logprob")
    parser.add_argument("--resume", type=int, default=1, choices=[0, 1])
    parser.add_argument("--python_bin", default=sys.executable)
    args = parser.parse_args()

    q_dir = Path(args.q_dir)
    if not q_dir.exists():
        print(f"Missing q_dir: {q_dir}")
        return 1

    inj_root = q_dir / "_injection" / args.fv_name.upper()
    inj_root.mkdir(parents=True, exist_ok=True)
    logs_dir = q_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.source_trials_path:
        source_trials_path = Path(args.source_trials_path)
    else:
        source_trials_path = q_dir / "_trials" / "condition_AAA.json"
    if not source_trials_path.exists():
        print(f"Missing source_trials_path: {source_trials_path}")
        return 1

    source_payload = read_json(source_trials_path)
    source_trials = source_payload.get("trials", [])
    if not isinstance(source_trials, list) or not source_trials:
        print(f"No trials in source: {source_trials_path}")
        return 1

    split_manifest_path = inj_root / "split_manifest.json"
    if args.resume and split_manifest_path.exists():
        split_manifest = read_json(split_manifest_path)
    else:
        split_manifest = build_split_manifest(
            source_trials=source_trials,
            split_seed=args.split_seed,
            eval_holdout_ratio=args.eval_holdout_ratio,
        )
        split_manifest.update(storage_metadata(canonical_root=inj_root))
        write_json(split_manifest_path, split_manifest)

    eval_ids = set(split_manifest.get("split_eval_inject", []))
    train_ids = set(split_manifest.get("split_train_fv", []))
    if eval_ids & train_ids:
        print("split manifest invalid: train/eval overlap detected")
        return 1

    eval_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(source_trials):
        if not isinstance(row, dict):
            continue
        trial_id = str(row.get("trial_id", f"t{idx:06d}"))
        if trial_id in eval_ids:
            eval_rows.append(_normalize_eval_trial(row, idx))
    if not eval_rows:
        print("No eval rows after applying split manifest")
        return 1
    eval_trials_path = inj_root / "eval_trials.json"
    write_json(
        eval_trials_path,
        {
            "meta": {
                "source_trials_path": str(source_trials_path),
                "split_manifest_path": str(split_manifest_path),
                "n_eval_rows": len(eval_rows),
                "zero_shot_prompt_format": "Q: <query>\\nA:",
                **storage_metadata(canonical_root=inj_root),
            },
            "trials": eval_rows,
        },
    )

    if args.fv_path:
        fv_npy_path = Path(args.fv_path)
    else:
        fv_npy_path = q_dir / "_fv" / f"fv_{args.fv_name.upper()}.npy"
    if not fv_npy_path.exists():
        print(f"Missing fv path: {fv_npy_path}")
        return 1

    fv_arr = np.load(fv_npy_path)
    if fv_arr.ndim != 1:
        print(f"FV must be 1D: {fv_npy_path} shape={fv_arr.shape}")
        return 1
    try:
        import torch
    except Exception as exc:
        print(f"Failed to import torch: {exc}")
        return 1
    fv_pt_path = inj_root / "fv_global_resid.pt"
    fv_meta_path = inj_root / "fv_global_resid_meta.json"
    torch.save({"fv_global_resid": torch.tensor(fv_arr, dtype=torch.float32)}, fv_pt_path)
    qid = str(eval_rows[0].get("q_id", "unknown"))
    write_json(
        fv_meta_path,
        {
            "fv_type": "qid",
            "fv_source_qid": qid,
            "fv_name": args.fv_name.upper(),
            "token_position_rule": "t_idx = seq_len - 1 (last token of prefix)",
            "source_npy": str(fv_npy_path),
            "created_at": utc_now(),
            **storage_metadata(canonical_root=inj_root),
        },
    )

    if args.layer_mode == "list":
        layers = parse_layers(args.layers)
    else:
        top_heads_path = q_dir / "_top_heads" / f"top_heads_{args.fv_name.upper()}.json"
        layers = detect_auto_layers(top_heads_path, fallback_layer=0)
    alphas = parse_alphas(args.alpha_list)

    alpha_summary: List[Dict[str, object]] = []
    for layer in layers:
        for alpha in alphas:
            alpha_token = str(alpha).replace(".", "p")
            run_dir = inj_root / f"layer_{layer}" / f"alpha_{alpha_token}"
            run_dir.mkdir(parents=True, exist_ok=True)
            summary_path = run_dir / "eval_summary.json"
            result_json = _find_step6_result_json(run_dir)
            if args.resume and summary_path.exists() and result_json:
                alpha_summary.append(
                    {
                        "layer": int(layer),
                        "alpha": float(alpha),
                        "run_dir": str(run_dir),
                        "status": "skipped_resume",
                        "results_path": result_json,
                        "eval_summary_path": str(summary_path),
                    }
                )
                continue

            step6_cmd = [
                args.python_bin,
                str(PROJECT_ROOT / "scripts" / "run_step6_fv_injection_eval.py"),
                "--model",
                args.model,
                "--model_spec",
                args.model_spec,
                "--edit_layer",
                str(layer),
                "--fv_global_path",
                str(fv_pt_path),
                "--fv_global_meta_path",
                str(fv_meta_path),
                "--sampled_trials_path",
                str(eval_trials_path),
                "--eval_scope",
                "in_domain",
                "--fv_source_qid",
                qid,
                "--alpha",
                str(alpha),
                "--seed",
                str(args.split_seed),
                "--n_eval",
                str(min(args.n_eval, len(eval_rows))),
                "--score_key",
                args.score_key,
                "--out_dir",
                str(run_dir),
                "--device",
                args.device,
                "--quant",
                args.quant,
            ]
            if args.dtype:
                step6_cmd.extend(["--dtype", args.dtype])
            if args.device_map:
                step6_cmd.extend(["--device_map", args.device_map])
            log_path = logs_dir / f"injection_{args.fv_name}_L{layer}_a{alpha_token}.log"
            run_command(step6_cmd, log_path=log_path)
            result_json = _find_step6_result_json(run_dir)
            alpha_summary.append(
                {
                    "layer": int(layer),
                    "alpha": float(alpha),
                    "run_dir": str(run_dir),
                    "status": "completed",
                    "results_path": result_json,
                    "eval_summary_path": str(summary_path),
                    "log_path": str(log_path),
                }
            )

    write_json(
        inj_root / "alpha_sweep.json",
        {
            "created_at": utc_now(),
            "q_dir": str(q_dir),
            "fv_name": args.fv_name.upper(),
            "layers": layers,
            "alphas": alphas,
            "summary": alpha_summary,
            **storage_metadata(canonical_root=inj_root),
        },
    )
    print(f"saved injection sweep summary: {inj_root / 'alpha_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
