#!/usr/bin/env python3
"""Run relation q-wise pipeline (StepD -> Heatmap -> StepE -> Step6 layer sweep).

This orchestrator follows TECH_SPEC_RELATION_QWISE:
- relation CSV fixed per run
- per-q independent processing
- skip q with insufficient rows for (query 1 + demos N)
- fixed trial snapshot reuse
- zero-shot snapshot generation
- Step6 all-layer sweep with per-layer directories
- qid_status.json and step6_all_layers_summary.json persistence for resume safety
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.model_spec import get_model_spec
from fv.relation_trials import load_relation_csv


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def command_str(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_command(
    cmd: Sequence[str],
    log_path: Path,
    dry_run: bool,
    env: Dict[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] exec: {command_str(cmd)}\n")
        handle.flush()
        if dry_run:
            handle.write(f"[{utc_now()}] dry_run=1, skipped\n")
            handle.flush()
            return
        proc = subprocess.run(
            list(cmd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed (rc={proc.returncode}): {command_str(cmd)}")


def parse_q_list(q_list: Optional[str], available: Sequence[str]) -> List[str]:
    if not q_list:
        return sorted(set(available))
    raw = [part.strip() for part in q_list.split(",")]
    requested = [part for part in raw if part]
    return requested


def parse_layer_expr(expr: str) -> List[int]:
    value = (expr or "").strip()
    if not value or value.lower() == "all":
        raise ValueError("layer expression must not be empty/all for explicit parsing")
    layers: List[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            start = int(a.strip())
            end = int(b.strip())
            if end < start:
                raise ValueError(f"invalid layer range: {token}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(token))
    uniq = sorted(set(layers))
    if not uniq:
        raise ValueError(f"no layers parsed from '{expr}'")
    if any(layer < 0 for layer in uniq):
        raise ValueError(f"negative layer index in '{expr}'")
    return uniq


def boundary_target_id(
    tokenizer,
    prefix_str: str,
    target_str: str,
    tok_add_special: bool,
) -> int:
    boundary_prefix = prefix_str
    boundary_answer = target_str
    if boundary_prefix.endswith(" ") and not boundary_answer.startswith(" "):
        boundary_prefix = boundary_prefix[:-1]
        boundary_answer = f" {boundary_answer}"
    prefix_ids = tokenizer.encode(boundary_prefix, add_special_tokens=tok_add_special)
    full_ids = tokenizer.encode(
        boundary_prefix + boundary_answer, add_special_tokens=tok_add_special
    )
    if len(full_ids) <= len(prefix_ids):
        raise ValueError("Tokenization does not extend prefix for target boundary.")
    return int(full_ids[len(prefix_ids)])


def build_zeroshot_trials(
    sampled_trials_path: Path,
    out_path: Path,
    model_name: str,
    model_spec_name: str,
) -> None:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to import transformers tokenizer: {exc}") from exc

    payload = read_json(sampled_trials_path)
    trials = payload.get("trials", [])
    if not isinstance(trials, list) or not trials:
        raise ValueError("sampled_trials.json missing non-empty 'trials' list")

    spec = get_model_spec(model_spec_name)
    tok_add_special = bool(spec.prepend_bos)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    rows: List[Dict[str, object]] = []
    for idx, row in enumerate(trials):
        if not isinstance(row, dict):
            continue
        query = row.get("query")
        if not isinstance(query, dict):
            raise ValueError(f"trial[{idx}] missing query object")
        x_val = query.get("input")
        y_val = row.get("target_str") or query.get("output")
        if not isinstance(x_val, str) or not isinstance(y_val, str):
            raise ValueError(f"trial[{idx}] invalid query/target strings")
        prefix = f"Q: {x_val}\nA: "
        target_id = boundary_target_id(
            tokenizer=tokenizer,
            prefix_str=prefix,
            target_str=y_val,
            tok_add_special=tok_add_special,
        )
        new_row = {
            "q_id": row.get("q_id", "unknown"),
            "trial_idx": int(row.get("trial_idx", idx)),
            "query": query,
            "target_str": y_val,
            "target_first_token_id": target_id,
            "corrupted_prompt_str": prefix,
            "corrupted_full_str": prefix + y_val,
        }
        for key in ("query_source_index", "demo_source_indices", "demo_order"):
            if key in row:
                new_row[key] = row[key]
        rows.append(new_row)

    out_payload = {
        "meta": {
            "source_sampled_trials_path": str(sampled_trials_path),
            "zero_shot": True,
            "prompt_format": "Q: <query>\\nA: ",
            "model": model_name,
            "model_spec": model_spec_name,
            "tok_add_special": tok_add_special,
            "n_trials": len(rows),
            "generated_at": utc_now(),
        },
        "trials": rows,
    }
    write_json(out_path, out_payload)


def load_expected_layers_from_aie(aie_scores_path: Path) -> List[int]:
    if not aie_scores_path.exists():
        raise FileNotFoundError(f"Missing aie_scores.csv: {aie_scores_path}")
    layers = set()
    with aie_scores_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get("layer")
            if raw is None:
                continue
            layers.add(int(raw))
    if not layers:
        raise ValueError(f"No layer values in {aie_scores_path}")
    return sorted(layers)


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
        payload = read_json(results_path)
        summary = payload.get("summary", {})
        score_val = None
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
    write_json(out_path, payload)
    return payload


def required_stepd_outputs(artifacts_dir: Path) -> List[Path]:
    return [
        artifacts_dir / "sampled_trials.json",
        artifacts_dir / "aie_scores.csv",
        artifacts_dir / "trial_metrics.jsonl",
    ]


def required_stepe_outputs(artifacts_dir: Path) -> List[Path]:
    return [
        artifacts_dir / "top_heads.json",
        artifacts_dir / "fv_global_resid.pt",
        artifacts_dir / "fv_by_layer.pt",
    ]


def all_exist(paths: Sequence[Path]) -> bool:
    return all(path.exists() for path in paths)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run q-wise relation pipeline with resume safety.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--model_spec", default="llama3")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default=None, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quant", default="auto", choices=["auto", "none", "4bit", "8bit"])
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_trials_per_q", type=int, default=25)
    parser.add_argument("--n_demos", type=int, default=9)
    parser.add_argument("--score_key", default="mean_delta_p")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--stepd_layers", default="all", help="StepD layers arg (default: all)")
    parser.add_argument("--stepd_heads", default="all", help="StepD heads arg (default: all)")
    parser.add_argument(
        "--step6_layers",
        default="auto",
        help="Step6 layer sweep list/range (default: auto from aie_scores.csv)",
    )
    parser.add_argument(
        "--relation_csv_path",
        default="datasets/relation/relationA_ex.csv",
        help="Path to relation CSV",
    )
    parser.add_argument("--relation_name", default=None, help="Output relation key (default: csv stem)")
    parser.add_argument("--q_list", default=None, help="Comma-separated q_id list (default: all)")
    parser.add_argument(
        "--out_root",
        default="results_fv/relation_qwise",
        help="Root output directory",
    )
    parser.add_argument("--python_bin", default=sys.executable, help="Python executable for child scripts")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore existing completion status and rerun stages")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke profile: single layer/head, tiny trials/eval unless overridden",
    )
    parser.add_argument("--smoke_layers", default="0", help="Smoke layers/Step6 layers (default: 0)")
    parser.add_argument("--smoke_heads", default="0", help="Smoke heads (default: 0)")
    args = parser.parse_args()

    if args.smoke:
        if args.stepd_layers == "all":
            args.stepd_layers = args.smoke_layers
        if args.stepd_heads == "all":
            args.stepd_heads = args.smoke_heads
        if args.step6_layers == "auto":
            args.step6_layers = args.smoke_layers
        if args.n_trials_per_q == 25:
            args.n_trials_per_q = 2
        if args.topk == 20:
            args.topk = 2
        if args.n_eval == 50:
            args.n_eval = 4

    relation_csv = resolve_path(args.relation_csv_path)
    if not relation_csv.exists():
        print(f"Missing relation CSV: {relation_csv}")
        return 1
    relation_name = args.relation_name or relation_csv.stem

    out_root = resolve_path(args.out_root) / relation_name
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (f":{existing_pythonpath}" if existing_pythonpath else "")

    q_map = load_relation_csv(str(relation_csv))
    q_ids = parse_q_list(args.q_list, sorted(q_map.keys()))
    if not q_ids:
        print("No q_id selected.")
        return 1

    orchestrator_log = out_root / "qwise_orchestrator.log"
    with orchestrator_log.open("a", encoding="utf-8") as main_log:
        main_log.write(f"[{utc_now()}] start relation={relation_name} n_q={len(q_ids)}\n")

    processed = 0
    failed_q: List[str] = []
    skipped_q: List[str] = []

    for q_id in q_ids:
        processed += 1
        run_base = out_root / q_id
        artifacts_dir = run_base / "artifacts"
        logs_dir = run_base / "logs"
        step6_dir = artifacts_dir / "step6"
        status_path = artifacts_dir / "qid_status.json"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        step6_dir.mkdir(parents=True, exist_ok=True)

        q_count = len(q_map.get(q_id, []))
        min_required = args.n_demos + 1
        status: Dict[str, object] = (
            read_json(status_path) if status_path.exists() else {}
        )
        if not status:
            status = {
                "q_id": q_id,
                "status": "pending",
                "created_at": utc_now(),
            }
        status["updated_at"] = utc_now()
        status["config"] = {
            "model": args.model,
            "model_spec": args.model_spec,
            "relation_csv_path": str(relation_csv),
            "n_trials_per_q": args.n_trials_per_q,
            "n_demos": args.n_demos,
            "topk": args.topk,
            "score_key": args.score_key,
            "alpha": args.alpha,
            "n_eval": args.n_eval,
            "seed": args.seed,
        }
        status["q_row_count"] = q_count
        write_json(status_path, status)

        if q_count < min_required:
            status["status"] = "skipped"
            status["reason"] = f"insufficient rows ({q_count} < {min_required})"
            status["updated_at"] = utc_now()
            write_json(status_path, status)
            skipped_q.append(q_id)
            continue

        if status.get("status") == "completed" and not args.force:
            # verify completion contract from status + files
            expected_layers = status.get("expected_layers")
            if isinstance(expected_layers, list) and expected_layers:
                layers_ok = all(
                    layer_output_complete(step6_dir / f"layer_{int(layer)}")
                    for layer in expected_layers
                )
                if (
                    all_exist(required_stepd_outputs(artifacts_dir))
                    and all_exist(required_stepe_outputs(artifacts_dir))
                    and (step6_dir / "step6_all_layers_summary.json").exists()
                    and layers_ok
                ):
                    continue

        status["status"] = "running"
        status["updated_at"] = utc_now()
        write_json(status_path, status)

        try:
            # StepD
            if args.force or not all_exist(required_stepd_outputs(artifacts_dir)):
                cmd = [
                    args.python_bin,
                    str(PROJECT_ROOT / "scripts" / "run_stepD_aie_head_sweep.py"),
                    "--model",
                    args.model,
                    "--model_spec",
                    args.model_spec,
                    "--device",
                    args.device,
                    "--quant",
                    args.quant,
                    "--layers",
                    args.stepd_layers,
                    "--heads",
                    args.stepd_heads,
                    "--relation_csv_path",
                    str(relation_csv),
                    "--relation_q_list",
                    q_id,
                    "--relation_n_trials_per_q",
                    str(args.n_trials_per_q),
                    "--relation_n_demos",
                    str(args.n_demos),
                    "--seed",
                    str(args.seed),
                    "--score_key",
                    args.score_key,
                    "--out_base_dir",
                    str(run_base),
                ]
                if args.dtype:
                    cmd.extend(["--dtype", args.dtype])
                if args.device_map:
                    cmd.extend(["--device_map", args.device_map])
                run_command(cmd, logs_dir / "stepD_runner.log", args.dry_run, env)

            # Heatmap
            heatmap_name = f"aie_heatmap_{q_id}.png"
            heatmap_path = artifacts_dir / heatmap_name
            if args.force or not heatmap_path.exists():
                cmd = [
                    args.python_bin,
                    str(PROJECT_ROOT / "scripts" / "plot_stepD_aie_heatmap.py"),
                    "--in_dir",
                    str(artifacts_dir),
                    "--metric",
                    args.score_key,
                    "--topk",
                    str(args.topk),
                    "--out_name",
                    heatmap_name,
                ]
                run_command(cmd, logs_dir / "heatmap.log", args.dry_run, env)

            # StepE
            if args.force or not all_exist(required_stepe_outputs(artifacts_dir)):
                cmd = [
                    args.python_bin,
                    str(PROJECT_ROOT / "scripts" / "run_stepE_topk_fv_and_eval.py"),
                    "--stepd_artifacts_dir",
                    str(artifacts_dir),
                    "--sampled_trials_path",
                    str(artifacts_dir / "sampled_trials.json"),
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
                    "--out_dir",
                    str(artifacts_dir),
                ]
                if args.dtype:
                    cmd.extend(["--dtype", args.dtype])
                if args.device_map:
                    cmd.extend(["--device_map", args.device_map])
                run_command(cmd, logs_dir / "stepE_runner.log", args.dry_run, env)

            # Zero-shot snapshot
            zeroshot_path = artifacts_dir / "sampled_trials_zeroshot.json"
            if args.force or not zeroshot_path.exists():
                if args.dry_run:
                    with (logs_dir / "zeroshot_snapshot.log").open("a", encoding="utf-8") as handle:
                        handle.write(f"[{utc_now()}] dry_run: skipped zeroshot build\n")
                else:
                    build_zeroshot_trials(
                        sampled_trials_path=artifacts_dir / "sampled_trials.json",
                        out_path=zeroshot_path,
                        model_name=args.model,
                        model_spec_name=args.model_spec,
                    )

            if args.dry_run:
                status["status"] = "dry_run"
                status["reason"] = "planned commands only"
                status["updated_at"] = utc_now()
                write_json(status_path, status)
                continue

            if args.step6_layers.lower() == "auto":
                expected_layers = load_expected_layers_from_aie(artifacts_dir / "aie_scores.csv")
            else:
                expected_layers = parse_layer_expr(args.step6_layers)
            status["expected_layers"] = expected_layers
            write_json(status_path, status)

            # Step6 all-layer sweep
            failed_layers: List[int] = []
            for layer in expected_layers:
                layer_dir = step6_dir / f"layer_{layer}"
                layer_dir.mkdir(parents=True, exist_ok=True)
                if not args.force and layer_output_complete(layer_dir):
                    continue
                cmd = [
                    args.python_bin,
                    str(PROJECT_ROOT / "scripts" / "run_step6_fv_injection_eval.py"),
                    "--model",
                    args.model,
                    "--model_spec",
                    args.model_spec,
                    "--fv_global_path",
                    str(artifacts_dir / "fv_global_resid.pt"),
                    "--sampled_trials_path",
                    str(zeroshot_path),
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
                    "--out_dir",
                    str(layer_dir),
                ]
                if args.dtype:
                    cmd.extend(["--dtype", args.dtype])
                if args.device_map:
                    cmd.extend(["--device_map", args.device_map])
                try:
                    run_command(
                        cmd,
                        logs_dir / f"step6_layer_{layer}.log",
                        args.dry_run,
                        env,
                    )
                except Exception:
                    failed_layers.append(layer)
                    if args.stop_on_error:
                        raise

            # Aggregate all-layer summary
            summary_payload = build_step6_all_layers_summary(
                q_id=q_id,
                score_key=args.score_key,
                expected_layers=expected_layers,
                step6_dir=step6_dir,
                out_path=step6_dir / "step6_all_layers_summary.json",
            )

            missing_layers = summary_payload.get("missing_layers", [])
            if failed_layers or (isinstance(missing_layers, list) and missing_layers):
                status["status"] = "failed"
                status["reason"] = (
                    f"step6 incomplete failed_layers={failed_layers} "
                    f"missing_layers={missing_layers}"
                )
                failed_q.append(q_id)
            else:
                status["status"] = "completed"
                status["reason"] = "all stages completed"
            status["updated_at"] = utc_now()
            write_json(status_path, status)

        except Exception as exc:
            status["status"] = "failed"
            status["reason"] = str(exc)
            status["updated_at"] = utc_now()
            write_json(status_path, status)
            failed_q.append(q_id)
            if args.stop_on_error:
                print(f"[ERROR] q_id={q_id}: {exc}")
                return 1

    print(f"processed_q={processed} skipped_q={len(skipped_q)} failed_q={len(set(failed_q))}")
    if failed_q:
        uniq = sorted(set(failed_q))
        print("failed_q_ids=" + ",".join(uniq))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
