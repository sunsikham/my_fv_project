#!/usr/bin/env python3
"""Build super FV from selected qids and run Step6 injection eval on selected qids."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def parse_q_list(q_list: str) -> List[str]:
    parts = [part.strip() for part in q_list.split(",")]
    qids = [part for part in parts if part]
    if not qids:
        raise ValueError("q list must not be empty")
    return qids


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
            cwd=str(PROJECT_ROOT),
            check=False,
            env=os.environ.copy(),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed (rc={proc.returncode}): {command_str(cmd)}")


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV missing header: {path}")
        rows = list(reader)
    return list(reader.fieldnames), rows


def to_float_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_aie_scores(
    relation_root: Path,
    fv_qids: Sequence[str],
    out_csv: Path,
) -> List[int]:
    key_to_rows: Dict[Tuple[int, int], List[Dict[str, str]]] = {}
    all_headers: Optional[List[str]] = None

    for qid in fv_qids:
        path = relation_root / qid / "artifacts" / "aie_scores.csv"
        ensure_file(path, f"{qid} aie_scores.csv")
        headers, rows = load_csv_rows(path)
        if all_headers is None:
            all_headers = headers
        elif headers != all_headers:
            raise ValueError(f"aie_scores header mismatch for {qid}")
        for row in rows:
            layer = int(row["layer"])
            head = int(row["head"])
            key_to_rows.setdefault((layer, head), []).append(row)

    if all_headers is None:
        raise ValueError("No aie_scores rows were loaded")

    expected_n = len(fv_qids)
    keys = sorted(key_to_rows.keys())
    for key in keys:
        if len(key_to_rows[key]) != expected_n:
            raise ValueError(
                f"Incomplete head coverage at layer/head={key}: "
                f"{len(key_to_rows[key])} rows, expected {expected_n}"
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_headers)
        writer.writeheader()
        for layer, head in keys:
            rows = key_to_rows[(layer, head)]
            out_row: Dict[str, object] = {}
            for col in all_headers:
                if col == "layer":
                    out_row[col] = layer
                    continue
                if col == "head":
                    out_row[col] = head
                    continue
                if col == "n_qid":
                    out_row[col] = expected_n
                    continue
                vals = [to_float_or_none(row.get(col)) for row in rows]
                if all(v is not None for v in vals):
                    out_row[col] = float(sum(vals)) / float(len(vals))
                else:
                    out_row[col] = rows[0].get(col)
            writer.writerow(out_row)

    layers = sorted({layer for layer, _ in keys})
    return layers


def load_clean_mean_tensor(path: Path) -> torch.Tensor:
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "clean_mean" in blob:
        tensor = blob["clean_mean"]
    else:
        tensor = blob
    if not hasattr(tensor, "shape") or len(tensor.shape) != 3:
        raise ValueError(f"Unsupported clean_mean shape in {path}")
    return tensor.detach().cpu().to(dtype=torch.float32)


def aggregate_clean_mean(
    relation_root: Path,
    fv_qids: Sequence[str],
    out_path: Path,
) -> None:
    tensors: List[torch.Tensor] = []
    for qid in fv_qids:
        path = relation_root / qid / "artifacts" / "stepD_mean_acts" / "global_clean_mean.pt"
        ensure_file(path, f"{qid} global_clean_mean.pt")
        tensors.append(load_clean_mean_tensor(path))
    if not tensors:
        raise ValueError("No clean_mean tensors loaded")
    shape0 = tuple(tensors[0].shape)
    for idx, tensor in enumerate(tensors):
        if tuple(tensor.shape) != shape0:
            raise ValueError(
                f"clean_mean shape mismatch at index={idx}: {tuple(tensor.shape)} vs {shape0}"
            )
    stacked = torch.stack(tensors, dim=0)
    mean_tensor = stacked.mean(dim=0).to(dtype=torch.float32)

    n_layers = int(mean_tensor.shape[0])
    n_heads = int(mean_tensor.shape[1])
    head_dim = int(mean_tensor.shape[2])
    resid_dim = int(n_heads * head_dim)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "clean_mean": mean_tensor,
            "slot_q": None,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "resid_dim": resid_dim,
            "n_layers": n_layers,
            "n_qid_averaged": len(fv_qids),
            "source_qids": list(fv_qids),
        },
        out_path,
    )


def find_single_step6_results(layer_dir: Path) -> Optional[Path]:
    matches = sorted(layer_dir.glob("step6_results_*.json"))
    if not matches:
        return None
    return matches[-1]


def layer_output_complete(layer_dir: Path) -> bool:
    return find_single_step6_results(layer_dir) is not None and (layer_dir / "eval_summary.json").exists()


def build_step6_all_layers_summary(
    q_id: str,
    score_key: str,
    expected_layers: Sequence[int],
    step6_dir: Path,
    out_path: Path,
) -> Dict[str, object]:
    per_layer: Dict[str, object] = {}
    completed_layers: List[int] = []
    missing_layers: List[int] = []
    best_layer: Optional[int] = None
    best_score: Optional[float] = None

    for layer in expected_layers:
        layer_dir = step6_dir / f"layer_{layer}"
        results_path = find_single_step6_results(layer_dir)
        summary_path = layer_dir / "eval_summary.json"
        if results_path is None or not summary_path.exists():
            missing_layers.append(layer)
            continue
        score_val = None
        payload = json.load(results_path.open("r", encoding="utf-8"))
        summary = payload.get("summary", {})
        if isinstance(summary, dict) and score_key in summary:
            score_val = float(summary[score_key])
            if best_score is None or score_val > best_score:
                best_score = score_val
                best_layer = layer
        per_layer[str(layer)] = {
            "layer_dir": str(layer_dir),
            "results_path": str(results_path),
            "eval_summary_path": str(summary_path),
            "score_key": score_key,
            "score_value": score_val,
        }
        completed_layers.append(layer)

    payload = {
        "q_id": q_id,
        "score_key": score_key,
        "expected_layers": list(expected_layers),
        "completed_layers": completed_layers,
        "missing_layers": missing_layers,
        "best_layer": best_layer,
        "best_score": best_score,
        "per_layer": per_layer,
        "generated_at": utc_now(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    return payload


def load_eval_summary(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    data = json.load(path.open("r", encoding="utf-8"))
    keys = [
        "acc_base",
        "acc_with",
        "delta_acc",
        "mean_delta_logprob",
        "mean_delta_p",
        "mean_delta_logit",
    ]
    out: Dict[str, float] = {}
    for key in keys:
        val = data.get(key)
        if isinstance(val, (int, float)):
            out[key] = float(val)
        else:
            out[key] = float("nan")
    return out


def maybe_add_dtype_and_device_map(cmd: List[str], dtype: Optional[str], device_map: Optional[str]) -> None:
    if dtype:
        cmd.extend(["--dtype", dtype])
    if device_map:
        cmd.extend(["--device_map", device_map])


def export_contract_artifacts(
    out_root: Path,
    super_fv_artifacts_dir: Path,
    run_id: str,
    relation_root: Path,
    fv_qids: Sequence[str],
    eval_qids: Sequence[str],
    layers: Sequence[int],
    topk: int,
    score_key: str,
    alpha: float,
    seed: int,
    model: str,
    model_spec: str,
    device: str,
    dtype: Optional[str],
    quant: str,
    device_map: Optional[str],
    n_eval: int,
) -> None:
    src_to_dst = [
        ("fv_global_resid.pt", "super_fv_global_resid.pt"),
        ("fv_by_layer.pt", "super_fv_by_layer.pt"),
        ("top_heads.json", "super_top_heads.json"),
    ]
    for src_name, dst_name in src_to_dst:
        src = super_fv_artifacts_dir / src_name
        ensure_file(src, f"super FV artifact {src_name}")
        shutil.copy2(src, out_root / dst_name)

    fv_meta_src = super_fv_artifacts_dir / "fv_global_resid_meta.json"
    ensure_file(fv_meta_src, "super FV artifact fv_global_resid_meta.json")
    with fv_meta_src.open("r", encoding="utf-8") as handle:
        fv_meta = json.load(handle)

    super_meta = {
        "created_at": utc_now(),
        "run_id": run_id,
        "relation_root": str(relation_root),
        "fv_qids": list(fv_qids),
        "eval_qids": list(eval_qids),
        "expected_layers": list(layers),
        "topk": topk,
        "score_key": score_key,
        "alpha": alpha,
        "seed": seed,
        "n_eval": n_eval,
        "model": model,
        "model_spec": model_spec,
        "device": device,
        "dtype": dtype,
        "quant": quant,
        "device_map": device_map,
        "source_files": {
            "fv_global_resid": str(super_fv_artifacts_dir / "fv_global_resid.pt"),
            "fv_by_layer": str(super_fv_artifacts_dir / "fv_by_layer.pt"),
            "top_heads": str(super_fv_artifacts_dir / "top_heads.json"),
            "fv_global_resid_meta": str(fv_meta_src),
        },
        "fv_global_resid_meta": fv_meta,
    }
    with (out_root / "super_fv_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(super_meta, handle, ensure_ascii=True, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run q-subset super FV build and Step6 eval.")
    parser.add_argument(
        "--relation_root",
        default="results_fv/relation_qwise/relationB_ex",
        help="relation qwise root directory",
    )
    parser.add_argument("--fv_qids", required=True, help="Comma-separated qids for super FV build")
    parser.add_argument(
        "--eval_qids",
        default=None,
        help="Comma-separated qids for eval (default: same as fv_qids)",
    )
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--score_key", default="mean_delta_p")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_spec", default="llama3")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quant", default="4bit", choices=["auto", "none", "4bit", "8bit"])
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n_eval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--python_bin", default=sys.executable)
    args = parser.parse_args()

    relation_root = (PROJECT_ROOT / args.relation_root).resolve()
    if not relation_root.exists():
        print(f"Missing relation_root: {relation_root}")
        return 1

    fv_qids = parse_q_list(args.fv_qids)
    eval_qids = parse_q_list(args.eval_qids) if args.eval_qids else list(fv_qids)

    run_id = args.run_id or f"super_fv_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_root = relation_root / "_super_fv" / run_id
    logs_dir = out_root / "logs"
    super_stepd_dir = out_root / "super_stepd_artifacts"
    super_stepd_mean_dir = super_stepd_dir / "stepD_mean_acts"
    super_fv_artifacts_dir = out_root / "super_fv_artifacts"
    eval_root = out_root / "eval_by_q"
    for path in [logs_dir, super_stepd_dir, super_stepd_mean_dir, super_fv_artifacts_dir, eval_root]:
        path.mkdir(parents=True, exist_ok=True)

    # Preflight required files.
    for qid in fv_qids:
        ensure_file(relation_root / qid / "artifacts" / "aie_scores.csv", f"{qid} aie_scores.csv")
        ensure_file(
            relation_root / qid / "artifacts" / "stepD_mean_acts" / "global_clean_mean.pt",
            f"{qid} global_clean_mean.pt",
        )
    for qid in eval_qids:
        ensure_file(
            relation_root / qid / "artifacts" / "sampled_trials_zeroshot.json",
            f"{qid} sampled_trials_zeroshot.json",
        )

    # Aggregate StepD signals.
    layers = aggregate_aie_scores(
        relation_root=relation_root,
        fv_qids=fv_qids,
        out_csv=super_stepd_dir / "aie_scores.csv",
    )
    aggregate_clean_mean(
        relation_root=relation_root,
        fv_qids=fv_qids,
        out_path=super_stepd_mean_dir / "global_clean_mean.pt",
    )

    # Save run meta (orchestration-level metadata).
    run_meta = {
        "created_at": utc_now(),
        "relation_root": str(relation_root),
        "run_id": run_id,
        "fv_qids": fv_qids,
        "eval_qids": eval_qids,
        "topk": args.topk,
        "score_key": args.score_key,
        "model": args.model,
        "model_spec": args.model_spec,
        "device": args.device,
        "dtype": args.dtype,
        "quant": args.quant,
        "device_map": args.device_map,
        "alpha": args.alpha,
        "n_eval": args.n_eval,
        "seed": args.seed,
        "expected_layers": layers,
        "paths": {
            "out_root": str(out_root),
            "super_stepd_artifacts_dir": str(super_stepd_dir),
            "super_fv_artifacts_dir": str(super_fv_artifacts_dir),
            "eval_root": str(eval_root),
        },
    }
    with (out_root / "run_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(run_meta, handle, ensure_ascii=True, indent=2)

    # Build super FV via StepE (reuse existing logic, skip StepE eval).
    sample_trials_for_stepe = relation_root / eval_qids[0] / "artifacts" / "sampled_trials_zeroshot.json"
    step_e_cmd: List[str] = [
        args.python_bin,
        str(PROJECT_ROOT / "scripts" / "run_stepE_topk_fv_and_eval.py"),
        "--stepd_artifacts_dir",
        str(super_stepd_dir),
        "--sampled_trials_path",
        str(sample_trials_for_stepe),
        "--k",
        str(args.topk),
        "--score_key",
        args.score_key,
        "--model",
        args.model,
        "--model_spec",
        args.model_spec,
        "--device",
        args.device,
        "--quant",
        args.quant,
        "--alpha",
        str(args.alpha),
        "--seed",
        str(args.seed),
        "--skip_eval",
        "--out_dir",
        str(super_fv_artifacts_dir),
    ]
    maybe_add_dtype_and_device_map(step_e_cmd, args.dtype, args.device_map)
    run_command(step_e_cmd, logs_dir / "stepE_super_fv.log")

    fv_global_path = super_fv_artifacts_dir / "fv_global_resid.pt"
    fv_global_meta_path = super_fv_artifacts_dir / "fv_global_resid_meta.json"
    ensure_file(fv_global_path, "super fv_global_resid.pt")
    ensure_file(fv_global_meta_path, "super fv_global_resid_meta.json")
    export_contract_artifacts(
        out_root=out_root,
        super_fv_artifacts_dir=super_fv_artifacts_dir,
        run_id=run_id,
        relation_root=relation_root,
        fv_qids=fv_qids,
        eval_qids=eval_qids,
        layers=layers,
        topk=args.topk,
        score_key=args.score_key,
        alpha=args.alpha,
        seed=args.seed,
        model=args.model,
        model_spec=args.model_spec,
        device=args.device,
        dtype=args.dtype,
        quant=args.quant,
        device_map=args.device_map,
        n_eval=args.n_eval,
    )

    # Run Step6 for each eval q and each layer.
    failures: List[str] = []
    for qid in eval_qids:
        q_eval_dir = eval_root / qid
        q_eval_dir.mkdir(parents=True, exist_ok=True)
        sampled_trials = relation_root / qid / "artifacts" / "sampled_trials_zeroshot.json"
        for layer in layers:
            layer_dir = q_eval_dir / f"layer_{layer}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            if layer_output_complete(layer_dir):
                continue
            step6_cmd: List[str] = [
                args.python_bin,
                str(PROJECT_ROOT / "scripts" / "run_step6_fv_injection_eval.py"),
                "--model",
                args.model,
                "--model_spec",
                args.model_spec,
                "--fv_global_path",
                str(fv_global_path),
                "--fv_global_meta_path",
                str(fv_global_meta_path),
                "--sampled_trials_path",
                str(sampled_trials),
                "--edit_layer",
                str(layer),
                "--alpha",
                str(args.alpha),
                "--n_eval",
                str(args.n_eval),
                "--score_key",
                args.score_key,
                "--device",
                args.device,
                "--quant",
                args.quant,
                "--seed",
                str(args.seed),
                "--out_dir",
                str(layer_dir),
            ]
            maybe_add_dtype_and_device_map(step6_cmd, args.dtype, args.device_map)
            try:
                run_command(step6_cmd, logs_dir / f"{qid}_step6_layer_{layer}.log")
            except Exception as exc:  # pragma: no cover - runtime path
                failures.append(f"{qid}:layer_{layer}:{exc}")

        build_step6_all_layers_summary(
            q_id=qid,
            score_key=args.score_key,
            expected_layers=layers,
            step6_dir=q_eval_dir,
            out_path=q_eval_dir / "step6_all_layers_summary.json",
        )

    # Baseline vs super comparison.
    compare_rows: List[Dict[str, object]] = []
    for qid in eval_qids:
        for layer in layers:
            baseline_eval = relation_root / qid / "artifacts" / "step6" / f"layer_{layer}" / "eval_summary.json"
            super_eval = eval_root / qid / f"layer_{layer}" / "eval_summary.json"
            base = load_eval_summary(baseline_eval)
            sup = load_eval_summary(super_eval)
            row: Dict[str, object] = {
                "q_id": qid,
                "layer": layer,
                "baseline_layer": layer,
                "super_layer": layer,
            }
            for key in [
                "acc_base",
                "acc_with",
                "delta_acc",
                "mean_delta_logprob",
                "mean_delta_p",
                "mean_delta_logit",
            ]:
                base_val = float("nan") if base is None else base[key]
                sup_val = float("nan") if sup is None else sup[key]
                row[f"{key}_baseline"] = base_val
                row[f"{key}_super"] = sup_val
                row[f"diff_{key}"] = sup_val - base_val
            compare_rows.append(row)

    compare_csv = out_root / "comparison_vs_baseline.csv"
    compare_json = out_root / "comparison_vs_baseline.json"
    fields = [
        "q_id",
        "layer",
        "baseline_layer",
        "super_layer",
        "acc_base_baseline",
        "acc_base_super",
        "diff_acc_base",
        "acc_with_baseline",
        "acc_with_super",
        "diff_acc_with",
        "delta_acc_baseline",
        "delta_acc_super",
        "diff_delta_acc",
        "mean_delta_logprob_baseline",
        "mean_delta_logprob_super",
        "diff_mean_delta_logprob",
        "mean_delta_p_baseline",
        "mean_delta_p_super",
        "diff_mean_delta_p",
        "mean_delta_logit_baseline",
        "mean_delta_logit_super",
        "diff_mean_delta_logit",
    ]
    with compare_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(compare_rows)
    with compare_json.open("w", encoding="utf-8") as handle:
        json.dump(compare_rows, handle, ensure_ascii=True, indent=2)

    # Final status.
    status = {
        "created_at": utc_now(),
        "run_id": run_id,
        "out_root": str(out_root),
        "fv_qids": fv_qids,
        "eval_qids": eval_qids,
        "layers": layers,
        "num_rows_comparison": len(compare_rows),
        "num_failures": len(failures),
        "failures": failures,
    }
    with (out_root / "run_status.json").open("w", encoding="utf-8") as handle:
        json.dump(status, handle, ensure_ascii=True, indent=2)

    print(f"run_id={run_id}")
    print(f"out_root={out_root}")
    print(f"comparison_rows={len(compare_rows)}")
    if failures:
        print(f"step6_failures={len(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
