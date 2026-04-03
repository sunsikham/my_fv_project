#!/usr/bin/env python3
"""Condition q-wise pipeline: StepD -> top-heads -> vectors -> PCA -> FV -> optional injection."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import shlex
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.adapters import infer_head_dims
from fv.condition_trials import (
    count_rows_by_q,
    generate_condition_trials_for_q,
    load_trials_json,
    parse_conditions,
    parse_q_list,
    save_trials_json,
)
from fv.head_vector_extract import (
    build_rank_map,
    extract_condition_trial_vectors,
    jaccard,
    load_stepd_scores_csv,
    serialize_heads,
    select_topk,
    spearman_rank_correlation,
    unique_heads,
)
from fv.hf_loader import load_hf_model_and_tokenizer
from fv.hooks import get_out_proj_pre_hook_target
from fv.relation_trials import load_relation_csv
from fv.stepd_resume import compute_stepd_code_fingerprint
from fv.tokenization import resolve_prompt_add_special_tokens


CONDITION_DEFAULT = "AAA,BBB,BABA"
STAGE_KEYS = ["stepd_done", "topheads_done", "vectors_done", "pca_done", "fv_done", "inject_done"]
SUPPORTED_SCHEMA = "v3"
SYNC_MODE_CHOICES = ("none", "per_q", "end")
HOME_ARTIFACT_PROFILE_CHOICES = ("core", "full")


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash_hex(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_alpha_list(text: str) -> List[float]:
    values = [part.strip() for part in (text or "").split(",") if part.strip()]
    if not values:
        raise ValueError("alpha_list must include at least one value")
    return [float(value) for value in values]


def command_str(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_command(
    *,
    cmd: Sequence[str],
    log_path: Path,
    env: Dict[str, str],
    cwd: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] exec: {command_str(cmd)}\n")
        handle.flush()
        proc = subprocess.run(
            list(cmd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
            cwd=str(cwd),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed rc={proc.returncode}: {command_str(cmd)}")


def cleanup_gpu_memory(log: Optional[Callable[[str], None]] = None) -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        if log is not None:
            log(f"[warn] torch.cuda.empty_cache failed: {exc}")
    try:
        torch.cuda.ipc_collect()
    except Exception as exc:
        if log is not None:
            log(f"[warn] torch.cuda.ipc_collect failed: {exc}")


def detect_git_head() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            cwd=str(PROJECT_ROOT),
            text=True,
        )
    except Exception:
        proc = None
    if proc and proc.returncode == 0:
        return proc.stdout.strip()
    return ""


def resolve_relation_name(
    *,
    relation_name: Optional[str],
    relation_a_csv: Path,
    relation_b_csv: Path,
    conditions: Sequence[str],
) -> str:
    if relation_name:
        return relation_name
    hash_payload = {
        "relation_a_csv_abs": str(relation_a_csv.resolve()),
        "relation_b_csv_abs": str(relation_b_csv.resolve()),
        "relation_a_sha256": file_sha256(relation_a_csv),
        "relation_b_sha256": file_sha256(relation_b_csv),
        "conditions": list(conditions),
    }
    short_hash = stable_hash_hex(hash_payload)[:10]
    return (
        f"relA_{relation_a_csv.stem}"
        f"__relB_{relation_b_csv.stem}"
        f"__h{short_hash}"
    )


def storage_metadata(
    *,
    canonical_root: Path,
    sync_root: Optional[Path],
    sync_mode: str,
    artifact_profile: str,
) -> Dict[str, object]:
    return {
        "canonical_root": str(canonical_root),
        "sync_root": (str(sync_root) if sync_root is not None else None),
        "sync_mode": str(sync_mode),
        "artifact_profile": str(artifact_profile),
    }


def load_status(path: Path, conditions: Sequence[str]) -> Dict[str, object]:
    if not path.exists():
        payload: Dict[str, object] = {"created_at": utc_now(), "status": "pending"}
        for key in STAGE_KEYS:
            payload[key] = False
        payload["stepd_conditions_done"] = {str(cond): False for cond in conditions}
        return payload
    payload = read_json(path)
    for key in STAGE_KEYS:
        payload.setdefault(key, False)
    payload.setdefault("status", "pending")
    payload.setdefault("stepd_conditions_done", {})
    for cond in conditions:
        payload["stepd_conditions_done"].setdefault(str(cond), False)
    return payload


def invalidate_downstream(status: Dict[str, object], start_key: str) -> None:
    if start_key not in STAGE_KEYS:
        return
    start_idx = STAGE_KEYS.index(start_key)
    for key in STAGE_KEYS[start_idx:]:
        status[key] = False


def _current_stepd_condition_fingerprint(
    *,
    args: argparse.Namespace,
    q_fingerprint: Dict[str, object],
    trial_path: Path,
    condition: str,
) -> str:
    return stable_hash_hex(
        {
            "q_id": q_fingerprint.get("q_id"),
            "condition": condition,
            "trial_path": str(trial_path.resolve()),
            "trial_sha256": file_sha256(trial_path),
            "model": args.model,
            "model_spec": args.model_spec,
            "dtype": args.dtype,
            "quant": args.quant,
            "device_map": args.device_map,
            "stepd_layers": args.stepd_layers,
            "stepd_heads": args.stepd_heads,
            "n_trials_per_q": args.n_trials_per_q,
            "n_demos": args.n_demos,
            "score_key": args.score_key,
            "seed": args.seed,
            "compute_prob_scores": 1,
            "code_fingerprint": compute_stepd_code_fingerprint(PROJECT_ROOT),
            "config_fingerprint": q_fingerprint.get("config_fingerprint"),
            "input_fingerprint": q_fingerprint.get("input_fingerprint"),
        }
    )


def _legacy_stepd_compatible(q_fingerprint: Dict[str, object], q_fp_path: Path) -> bool:
    if not q_fp_path.exists():
        return False
    old_fp = read_json(q_fp_path)
    return (
        old_fp.get("schema_version") == q_fingerprint.get("schema_version")
        and old_fp.get("config_fingerprint") == q_fingerprint.get("config_fingerprint")
        and old_fp.get("input_fingerprint") == q_fingerprint.get("input_fingerprint")
    )


def _stepd_outputs_ready(
    *,
    q_dir: Path,
    condition: str,
    expected_fingerprint: str,
    q_fingerprint: Dict[str, object],
    q_fp_path: Path,
    score_key: str,
) -> bool:
    score_path = q_dir / "_stepd" / f"aie_scores_{condition}.csv"
    if not score_path.exists():
        return False
    try:
        load_stepd_scores_csv(str(score_path), score_key=score_key)
    except Exception:
        return False
    meta_path = q_dir / "_stepd" / f"stepd_meta_{condition}.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        if meta.get("stepd_fingerprint") == expected_fingerprint:
            return True
    return _legacy_stepd_compatible(q_fingerprint, q_fp_path)


def build_config_payload(args: argparse.Namespace, relation_name: str) -> Dict[str, object]:
    return {
        "schema_version": args.schema_version,
        "model": args.model,
        "model_spec": args.model_spec,
        "device": args.device,
        "dtype": args.dtype,
        "quant": args.quant,
        "device_map": args.device_map,
        "conditions": args.conditions,
        "n_trials_per_q": args.n_trials_per_q,
        "n_demos": args.n_demos,
        "topk": args.topk,
        "score_key": args.score_key,
        "enable_union_ref": bool(args.enable_union_ref),
        "fv_streaming_mean": bool(args.fv_streaming_mean),
        "run_injection": bool(args.run_injection),
        "layer_mode": args.layer_mode,
        "injection_layers": args.injection_layers,
        "alpha_list": args.alpha_list,
        "split_seed": args.split_seed,
        "eval_holdout_ratio": args.eval_holdout_ratio,
        "seed": args.seed,
        "relation_name": relation_name,
        "sync_mode": args.sync_mode,
        "home_artifact_profile": args.home_artifact_profile,
        "sync_on_failure": bool(args.sync_on_failure),
    }


def build_fingerprint_payload(
    *,
    args: argparse.Namespace,
    relation_a_csv: Path,
    relation_b_csv: Path,
    selected_qids: Sequence[str],
    relation_name: str,
) -> Dict[str, object]:
    config_payload = build_config_payload(args, relation_name=relation_name)
    input_payload = {
        "relation_a_csv_path": str(relation_a_csv.resolve()),
        "relation_b_csv_path": str(relation_b_csv.resolve()),
        "relation_a_csv_sha256": file_sha256(relation_a_csv),
        "relation_b_csv_sha256": file_sha256(relation_b_csv),
        "selected_qids": list(selected_qids),
    }
    code_payload = {
        "git_head": detect_git_head(),
        "pipeline_script_sha256": file_sha256(Path(__file__)),
        "condition_trials_sha256": file_sha256(PROJECT_ROOT / "fv" / "condition_trials.py"),
        "head_vector_extract_sha256": file_sha256(
            PROJECT_ROOT / "fv" / "head_vector_extract.py"
        ),
    }
    return {
        "schema_version": args.schema_version,
        "config_fingerprint": stable_hash_hex(config_payload),
        "input_fingerprint": stable_hash_hex(input_payload),
        "code_fingerprint": stable_hash_hex(code_payload),
        "config_payload": config_payload,
        "input_payload": input_payload,
        "code_payload": code_payload,
    }


def acquire_lock(lock_path: Path, ttl_sec: int) -> str:
    owner = f"{socket.gethostname()}:{os.getpid()}:{int(time.time())}:{random.randint(0, 999999)}"
    now_ts = time.time()
    payload = {
        "owner_id": owner,
        "owner_pid": os.getpid(),
        "owner_host": socket.gethostname(),
        "start_ts": now_ts,
        "heartbeat_ts": now_ts,
        "ttl_sec": int(ttl_sec),
    }
    if lock_path.exists():
        try:
            current = read_json(lock_path)
        except Exception:
            current = {}
        hb = float(current.get("heartbeat_ts", 0.0))
        cur_ttl = int(current.get("ttl_sec", ttl_sec))
        if now_ts - hb <= cur_ttl:
            raise RuntimeError(
                f"Active lock exists: {lock_path} owner={current.get('owner_id')}"
            )
    write_json(lock_path, payload)
    return owner


def heartbeat_lock(lock_path: Path, owner_id: str, ttl_sec: int) -> None:
    if not lock_path.exists():
        raise RuntimeError(f"Lock file disappeared: {lock_path}")
    payload = read_json(lock_path)
    current_owner = str(payload.get("owner_id", ""))
    if current_owner != owner_id:
        raise RuntimeError(
            f"Lock ownership mismatch: expected={owner_id} found={current_owner}"
        )
    payload["heartbeat_ts"] = time.time()
    payload["ttl_sec"] = int(ttl_sec)
    write_json(lock_path, payload)


def release_lock(lock_path: Path, owner_id: str) -> None:
    if not lock_path.exists():
        return
    try:
        payload = read_json(lock_path)
    except Exception:
        payload = {}
    if str(payload.get("owner_id", "")) == owner_id:
        lock_path.unlink(missing_ok=True)


def copy_file_if_exists(src: Path, dst: Path) -> int:
    if not src.exists() or not src.is_file():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return 1


def copy_tree_if_exists(src: Path, dst: Path) -> int:
    if not src.exists() or not src.is_dir():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return 1


def copy_glob_files(src_dir: Path, pattern: str, dst_dir: Path) -> int:
    if not src_dir.exists() or not src_dir.is_dir():
        return 0
    copied = 0
    for src in sorted(src_dir.glob(pattern)):
        copied += copy_file_if_exists(src, dst_dir / src.name)
    return copied


def sync_q_outputs_to_home(
    *,
    q_dir: Path,
    home_q_dir: Path,
    profile: str,
) -> Dict[str, int]:
    files_copied = 0
    trees_copied = 0
    if profile == "full":
        copy_tree_if_exists(q_dir, home_q_dir)
        return {"files_copied": 0, "trees_copied": 1}

    trees_copied += copy_tree_if_exists(q_dir / "_status", home_q_dir / "_status")

    logs_dir = q_dir / "logs"
    home_logs_dir = home_q_dir / "logs"
    files_copied += copy_file_if_exists(logs_dir / "orchestrator.log", home_logs_dir / "orchestrator.log")
    files_copied += copy_file_if_exists(
        logs_dir / "d_extension_orchestrator.log",
        home_logs_dir / "d_extension_orchestrator.log",
    )
    files_copied += copy_glob_files(logs_dir, "pca*.log", home_logs_dir)

    files_copied += copy_glob_files(q_dir / "_trials", "condition_*.json", home_q_dir / "_trials")

    stepd_dir = q_dir / "_stepd"
    home_stepd_dir = home_q_dir / "_stepd"
    files_copied += copy_glob_files(stepd_dir, "aie_scores_*.csv", home_stepd_dir)
    files_copied += copy_glob_files(stepd_dir, "sampled_trials_*.json", home_stepd_dir)
    files_copied += copy_glob_files(stepd_dir, "trial_metrics_*.jsonl", home_stepd_dir)
    files_copied += copy_glob_files(stepd_dir, "stepd_meta_*.json", home_stepd_dir)

    trees_copied += copy_tree_if_exists(q_dir / "_top_heads", home_q_dir / "_top_heads")

    vectors_dir = q_dir / "_vectors"
    home_vectors_dir = home_q_dir / "_vectors"
    files_copied += copy_glob_files(vectors_dir, "trial_vectors_AAA_ref_*.npy", home_vectors_dir)
    files_copied += copy_glob_files(vectors_dir, "trial_vectors_union_ref_*.npy", home_vectors_dir)
    files_copied += copy_file_if_exists(
        vectors_dir / "vector_extraction_meta.json",
        home_vectors_dir / "vector_extraction_meta.json",
    )

    trees_copied += copy_tree_if_exists(q_dir / "_pca_common", home_q_dir / "_pca_common")
    trees_copied += copy_tree_if_exists(q_dir / "_fv", home_q_dir / "_fv")
    trees_copied += copy_tree_if_exists(q_dir / "_inject", home_q_dir / "_inject")
    return {"files_copied": files_copied, "trees_copied": trees_copied}


def sync_q_with_status(
    *,
    q_id: str,
    q_dir: Path,
    home_relation_root: Path,
    sync_mode: str,
    profile: str,
    reason: str,
    log: Callable[[str], None],
) -> bool:
    home_q_dir = home_relation_root / q_id
    payload: Dict[str, object] = {
        "updated_at": utc_now(),
        "status": "ok",
        "q_id": q_id,
        "reason": reason,
        "profile": profile,
        "source_q_dir": str(q_dir),
        "home_q_dir": str(home_q_dir),
    }
    payload.update(
        storage_metadata(
            canonical_root=q_dir,
            sync_root=home_q_dir,
            sync_mode=sync_mode,
            artifact_profile=profile,
        )
    )
    ok = True
    try:
        stats = sync_q_outputs_to_home(q_dir=q_dir, home_q_dir=home_q_dir, profile=profile)
        payload.update(stats)
    except Exception as exc:
        ok = False
        payload["status"] = "failed"
        payload["error"] = str(exc)
        log(f"[warn] home sync failed: {exc}")
    status_path = q_dir / "_status" / "home_sync_status.json"
    write_json(status_path, payload)
    if ok:
        copy_file_if_exists(status_path, home_q_dir / "_status" / "home_sync_status.json")
    return ok


def sync_relation_status_to_home(
    *,
    out_root: Path,
    home_relation_root: Path,
    log: Callable[[str], None],
) -> bool:
    try:
        copy_tree_if_exists(out_root / "_status", home_relation_root / "_status")
        return True
    except Exception as exc:
        log(f"[warn] relation _status sync failed: {exc}")
        return False


def serialize_top_rows(rows) -> List[Dict[str, object]]:
    out = []
    for row in rows:
        out.append(
            {
                "layer": int(row.layer),
                "head": int(row.head),
                "score_key": row.score_key,
                "rank_score": float(row.score),
            }
        )
    return out


def build_condition_top_heads(
    *,
    stepd_dir: Path,
    top_heads_dir: Path,
    conditions: Sequence[str],
    topk: int,
    score_key: str,
    enable_union_ref: bool,
) -> Dict[str, object]:
    stepd_rows: Dict[str, list] = {}
    topk_rows: Dict[str, list] = {}
    for cond in conditions:
        score_path = stepd_dir / f"aie_scores_{cond}.csv"
        if not score_path.exists():
            raise FileNotFoundError(f"Missing StepD score CSV: {score_path}")
        rows = load_stepd_scores_csv(str(score_path), score_key=score_key)
        stepd_rows[cond] = rows
        topk_rows[cond] = select_topk(rows, k=topk)
        write_json(
            top_heads_dir / f"top_heads_{cond}.json",
            {
                "meta": {
                    "condition": cond,
                    "score_key": score_key,
                    "k": int(topk),
                    "source_scores_path": str(score_path),
                },
                "heads": serialize_top_rows(topk_rows[cond]),
            },
        )

    sets_dir = top_heads_dir / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        sets_dir / "top_heads_ref_AAA.json",
        {
            "meta": {
                "source": "AAA_topk",
                "score_key": score_key,
                "k": int(topk),
            },
            "heads": serialize_top_rows(topk_rows["AAA"]),
        },
    )

    rank_maps = {cond: build_rank_map(rows) for cond, rows in stepd_rows.items()}
    top_sets = {
        cond: set((row.layer, row.head) for row in topk_rows[cond]) for cond in conditions
    }
    union_candidate = set().union(*top_sets.values())
    union_rows: List[Dict[str, object]] = []
    for layer, head in sorted(union_candidate):
        ranks: Dict[str, int] = {}
        for cond in conditions:
            rank = rank_maps[cond].get((layer, head))
            if rank is None:
                raise ValueError(
                    f"Missing rank for union candidate head layer={layer} head={head} cond={cond}"
                )
            ranks[cond] = int(rank)
        agg_rank = sum(ranks.values()) / float(len(conditions))
        union_rows.append(
            {
                "layer": int(layer),
                "head": int(head),
                "agg_rank": float(agg_rank),
                "ranks": ranks,
            }
        )
    union_rows.sort(key=lambda row: (row["agg_rank"], row["layer"], row["head"]))
    union_top = union_rows[:topk]
    if enable_union_ref:
        write_json(
            sets_dir / "top_heads_ref_union.json",
            {
                "meta": {
                    "source": "rank_mean_union",
                    "score_key": score_key,
                    "k": int(topk),
                    "conditions": list(conditions),
                },
                "heads": union_top,
            },
        )

    diag_dir = top_heads_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    overlap_payload = {
        "AAA_vs_BBB": jaccard(top_sets["AAA"], top_sets["BBB"]),
        "AAA_vs_BABA": jaccard(top_sets["AAA"], top_sets["BABA"]),
        "BBB_vs_BABA": jaccard(top_sets["BBB"], top_sets["BABA"]),
    }
    write_json(diag_dir / "tophead_overlap.json", overlap_payload)
    corr_payload = {
        "AAA_vs_BBB": spearman_rank_correlation(rank_maps["AAA"], rank_maps["BBB"]),
        "AAA_vs_BABA": spearman_rank_correlation(rank_maps["AAA"], rank_maps["BABA"]),
        "BBB_vs_BABA": spearman_rank_correlation(rank_maps["BBB"], rank_maps["BABA"]),
    }
    write_json(diag_dir / "score_correlation.json", corr_payload)
    return {
        "topk_rows": topk_rows,
        "union_top": union_top,
        "union_candidate": sorted(union_candidate),
    }


def stage_vector_extract(
    *,
    args: argparse.Namespace,
    q_dir: Path,
    conditions: Sequence[str],
    tophead_state: Dict[str, object],
    log: Callable[[str], None],
) -> Dict[str, object]:
    vectors_dir = q_dir / "_vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    fv_dir = q_dir / "_fv"
    fv_dir.mkdir(parents=True, exist_ok=True)

    tok_add_special = resolve_prompt_add_special_tokens(args.model, args.model_spec)
    topk_rows = tophead_state["topk_rows"]
    aaa_ref_heads = set((row.layer, row.head) for row in topk_rows["AAA"])
    cond_heads = {
        cond: set((row.layer, row.head) for row in topk_rows[cond]) for cond in conditions
    }
    union_ref_heads: Optional[Set[Tuple[int, int]]] = None
    if args.enable_union_ref:
        union_ref_heads = set(
            (int(row["layer"]), int(row["head"])) for row in tophead_state["union_top"]
        )
    capture_heads = unique_heads(tophead_state["union_candidate"])
    capture_layers = sorted(set(layer for layer, _head in capture_heads))
    layer_modules: Dict[int, object] = {}
    model = None
    tokenizer = None
    cleanup_gpu_memory(log)
    try:
        model, tokenizer, diagnostics = load_hf_model_and_tokenizer(
            model_name=args.model,
            model_spec=args.model_spec,
            device=args.device,
            dtype=args.dtype,
            quant=args.quant,
            device_map=args.device_map,
        )
        log(f"[vector] model loaded diagnostics={json.dumps(diagnostics, ensure_ascii=True)}")
        model.eval()

        dims = infer_head_dims(model, spec_name=args.model_spec)
        n_heads = int(dims["n_heads"])
        head_dim = int(dims["head_dim"])
        resid_dim = int(dims["hidden_size"])

        for layer in capture_layers:
            module, _path = get_out_proj_pre_hook_target(
                model, layer_idx=layer, spec_name=args.model_spec, logger=None
            )
            layer_modules[layer] = module

        trial_ids_meta: Dict[str, List[str]] = {}
        seq_idx_meta: Dict[str, List[int]] = {}
        for cond in conditions:
            payload = load_trials_json(str(q_dir / "_trials" / f"condition_{cond}.json"))
            trials = payload["trials"]
            log(f"[vector] start condition={cond} n_trials={len(trials)}")
            extracted = extract_condition_trial_vectors(
                model=model,
                tokenizer=tokenizer,
                tok_add_special=bool(tok_add_special),
                layer_modules=layer_modules,
                n_heads=n_heads,
                head_dim=head_dim,
                resid_dim=resid_dim,
                trials=trials,
                capture_heads=capture_heads,
                aaa_ref_heads=aaa_ref_heads,
                union_ref_heads=union_ref_heads,
                cond_topk_heads=cond_heads[cond],
                logger=log,
            )
            np.save(vectors_dir / f"trial_vectors_AAA_ref_{cond}.npy", extracted["aaa_ref"])
            if args.enable_union_ref and extracted["union_ref"] is not None:
                np.save(vectors_dir / f"trial_vectors_union_ref_{cond}.npy", extracted["union_ref"])
            np.save(
                vectors_dir / f"trial_vectors_capture_headwise_{cond}.npy",
                extracted["capture_headwise"],
            )
            cond_arr = extracted["cond_topk"]
            if args.fv_streaming_mean:
                count = int(cond_arr.shape[0])
                mean_vec = (
                    cond_arr.astype(np.float32, copy=False).mean(axis=0).tolist()
                    if count > 0
                    else []
                )
                write_json(
                    fv_dir / f"fv_running_stats_{cond}.json",
                    {
                        "condition": cond,
                        "count": count,
                        "dtype": "float32",
                        "mean": mean_vec,
                        "source_fingerprint": stable_hash_hex(
                            {
                                "q_dir": str(q_dir),
                                "condition": cond,
                                "capture_heads": capture_heads,
                            }
                        ),
                    },
                )
            else:
                np.save(vectors_dir / f"trial_vectors_cond_topk_{cond}.npy", cond_arr)
            trial_ids_meta[cond] = [str(x) for x in extracted["trial_ids"]]
            seq_idx_meta[cond] = [int(x) for x in extracted["seq_token_indices"]]

        vector_meta = {
            "created_at": utc_now(),
            "tok_add_special": bool(tok_add_special),
            "n_heads": n_heads,
            "head_dim": head_dim,
            "resid_dim": resid_dim,
            "capture_heads_count": len(capture_heads),
            "capture_heads": [{"layer": layer, "head": head} for layer, head in capture_heads],
            "query_pred_definition": "last_token_of_input_prefix",
            "query_pred_contract": "stepd_patch == stage3_capture == injection_add",
            "trial_ids": trial_ids_meta,
            "seq_token_indices": seq_idx_meta,
            "headwise_sets": {
                "capture": {
                    "file_pattern": "trial_vectors_capture_headwise_<COND>.npy",
                    "heads": serialize_heads(capture_heads),
                    "head_count": int(len(capture_heads)),
                    "dtype": "float16",
                    "shape_note": "(n_trials, n_capture_heads, resid_dim)",
                    "conditions": list(conditions),
                    "source": "union_candidate_capture",
                    "created_at": utc_now(),
                }
            },
        }
        write_json(vectors_dir / "vector_extraction_meta.json", vector_meta)
        return vector_meta
    finally:
        layer_modules.clear()
        if tokenizer is not None:
            del tokenizer
        if model is not None:
            del model
        cleanup_gpu_memory(log)


def stage_build_fv(
    *,
    args: argparse.Namespace,
    q_dir: Path,
    conditions: Sequence[str],
    vector_meta: Dict[str, object],
) -> Dict[str, object]:
    vectors_dir = q_dir / "_vectors"
    fv_dir = q_dir / "_fv"
    fv_dir.mkdir(parents=True, exist_ok=True)
    fv_meta: Dict[str, object] = {
        "created_at": utc_now(),
        "fv_streaming_mean": bool(args.fv_streaming_mean),
        "conditions": list(conditions),
        "source_vector_meta": str(vectors_dir / "vector_extraction_meta.json"),
        "fv_files": {},
    }
    for cond in conditions:
        if args.fv_streaming_mean:
            stats_path = fv_dir / f"fv_running_stats_{cond}.json"
            if not stats_path.exists():
                raise FileNotFoundError(f"Missing running stats for cond={cond}: {stats_path}")
            stats = read_json(stats_path)
            mean_list = stats.get("mean", [])
            fv_vec = np.asarray(mean_list, dtype=np.float32)
        else:
            cond_path = vectors_dir / f"trial_vectors_cond_topk_{cond}.npy"
            if not cond_path.exists():
                raise FileNotFoundError(f"Missing condition vector file: {cond_path}")
            arr = np.load(cond_path).astype(np.float32, copy=False)
            if arr.ndim != 2:
                raise ValueError(f"Invalid cond vector shape for {cond}: {arr.shape}")
            if arr.shape[0] < 1:
                raise ValueError(f"No trials in cond vector file for {cond}: {cond_path}")
            fv_vec = arr.mean(axis=0)
        if fv_vec.ndim != 1:
            raise ValueError(f"FV must be 1D for cond={cond}, got shape={fv_vec.shape}")
        fv_path = fv_dir / f"fv_{cond}.npy"
        np.save(fv_path, fv_vec.astype(np.float32, copy=False))
        fv_meta["fv_files"][cond] = str(fv_path)

    write_json(fv_dir / "fv_meta.json", fv_meta)
    return fv_meta


def run_pca_stage(
    *,
    args: argparse.Namespace,
    q_dir: Path,
    log_dir: Path,
    env: Dict[str, str],
) -> None:
    pca_script = PROJECT_ROOT / "scripts" / "run_condition_common_pca.py"
    for ref_mode in (["AAA_ref", "union_ref"] if args.enable_union_ref else ["AAA_ref"]):
        cmd = [
            args.python_bin,
            str(pca_script),
            "--q_dir",
            str(q_dir),
            "--ref_mode",
            ref_mode,
            "--conditions",
            args.conditions,
            "--n_components",
            str(args.n_components),
            "--balance_trials",
            str(args.balance_trials),
            "--seed",
            str(args.seed),
        ]
        run_command(
            cmd=cmd,
            log_path=log_dir / f"pca_{ref_mode}.log",
            env=env,
            cwd=PROJECT_ROOT,
        )


def run_injection_stage(
    *,
    args: argparse.Namespace,
    q_dir: Path,
    log_dir: Path,
    env: Dict[str, str],
) -> None:
    inject_script = PROJECT_ROOT / "scripts" / "run_zero_shot_injection.py"
    fv_names = [part.strip().upper() for part in args.inject_fv_names.split(",") if part.strip()]
    if not fv_names:
        raise ValueError("inject_fv_names must include at least one FV condition name")
    for fv_name in fv_names:
        cmd = [
            args.python_bin,
            str(inject_script),
            "--q_dir",
            str(q_dir),
            "--model",
            args.model,
            "--model_spec",
            args.model_spec,
            "--device",
            args.device,
            "--quant",
            args.quant,
            "--fv_name",
            fv_name,
            "--split_seed",
            str(args.split_seed),
            "--eval_holdout_ratio",
            str(args.eval_holdout_ratio),
            "--n_eval",
            str(args.n_eval),
            "--layer_mode",
            args.layer_mode,
            "--layers",
            args.injection_layers,
            "--alpha_list",
            args.alpha_list,
            "--score_key",
            args.score_key,
            "--resume",
            str(args.resume),
            "--python_bin",
            args.python_bin,
        ]
        if args.dtype:
            cmd.extend(["--dtype", args.dtype])
        if args.device_map:
            cmd.extend(["--device_map", args.device_map])
        run_command(
            cmd=cmd,
            log_path=log_dir / f"injection_{fv_name}.log",
            env=env,
            cwd=PROJECT_ROOT,
        )


def collect_qids(
    *,
    relation_a_csv: Path,
    relation_b_csv: Path,
    q_list_text: Optional[str],
) -> List[str]:
    by_q_a = load_relation_csv(str(relation_a_csv))
    by_q_b = load_relation_csv(str(relation_b_csv))
    common_qids = sorted(set(by_q_a.keys()) & set(by_q_b.keys()))
    selected = parse_q_list(q_list_text, common_qids)
    missing = [q_id for q_id in selected if q_id not in common_qids]
    if missing:
        raise ValueError(
            "Requested q_ids missing in A∩B relation set: " + ",".join(missing)
        )
    return selected


def stage_generate_trials_for_q(
    *,
    args: argparse.Namespace,
    q_dir: Path,
    q_id: str,
    conditions: Sequence[str],
    relation_a_csv: Path,
    relation_b_csv: Path,
    tokenizer,
    tok_add_special: bool,
) -> Dict[str, object]:
    trials_dir = q_dir / "_trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    by_q_a = load_relation_csv(str(relation_a_csv))
    by_q_b = load_relation_csv(str(relation_b_csv))
    payloads = generate_condition_trials_for_q(
        relation_a_pairs=by_q_a[q_id],
        relation_b_pairs=by_q_b[q_id],
        relation_a_csv=str(relation_a_csv),
        relation_b_csv=str(relation_b_csv),
        q_id=q_id,
        conditions=conditions,
        n_trials_per_q=args.n_trials_per_q,
        n_demos=args.n_demos,
        seed=args.seed,
        tokenizer=tokenizer,
        tok_add_special=tok_add_special,
        prepend_bos_token=bool(tok_add_special),
        enforce_shared_query_target=bool(args.strict_shared_query_target),
    )
    for cond, payload in payloads.items():
        save_trials_json(payload, str(trials_dir / f"condition_{cond}.json"))
    return {"conditions": list(payloads.keys()), "n_trials": args.n_trials_per_q}


def copy_stepd_outputs(
    *,
    q_dir: Path,
    condition: str,
    stepd_run_base: Path,
) -> None:
    stepd_dir = q_dir / "_stepd"
    stepd_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = stepd_run_base / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Missing StepD artifacts dir: {artifacts_dir}")
    score_src = artifacts_dir / "aie_scores.csv"
    sampled_src = artifacts_dir / "sampled_trials.json"
    trial_metrics_src = artifacts_dir / "trial_metrics.jsonl"
    if not score_src.exists():
        raise FileNotFoundError(f"Missing StepD score csv: {score_src}")
    run_meta_path = artifacts_dir / "run_meta.json"
    run_meta = read_json(run_meta_path) if run_meta_path.exists() else {}
    shutil.copy2(score_src, stepd_dir / f"aie_scores_{condition}.csv")
    if sampled_src.exists():
        shutil.copy2(sampled_src, stepd_dir / f"sampled_trials_{condition}.json")
    if trial_metrics_src.exists():
        shutil.copy2(trial_metrics_src, stepd_dir / f"trial_metrics_{condition}.jsonl")
    write_json(
        stepd_dir / f"stepd_meta_{condition}.json",
        {
            "condition": condition,
            "stepd_run_base": str(stepd_run_base),
            "copied_at": utc_now(),
            "stepd_fingerprint": run_meta.get("stepd_fingerprint"),
            "resume_source": "run_base_copy",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run condition q-wise pipeline.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_spec", default="llama3")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default=None, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quant", default="auto", choices=["auto", "none", "4bit", "8bit"])
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--relation_a_csv", required=True)
    parser.add_argument("--relation_b_csv", required=True)
    parser.add_argument("--relation_name", default=None)
    parser.add_argument("--conditions", default=CONDITION_DEFAULT)
    parser.add_argument("--q_list", default=None)
    parser.add_argument("--n_trials_per_q", type=int, default=25)
    parser.add_argument("--n_demos", type=int, default=9)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--score_key", default="mean_delta_p")
    parser.add_argument("--enable_union_ref", type=int, default=0, choices=[0, 1])
    parser.add_argument("--fv_streaming_mean", type=int, default=0, choices=[0, 1])
    parser.add_argument("--run_injection", type=int, default=0, choices=[0, 1])
    parser.add_argument("--inject_fv_names", default="BBB")
    parser.add_argument("--layer_mode", default="auto", choices=["auto", "list"])
    parser.add_argument("--injection_layers", default="0")
    parser.add_argument("--alpha_list", default="0.5,1.0,1.5")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--eval_holdout_ratio", type=float, default=0.3)
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--resume", type=int, default=1, choices=[0, 1])
    parser.add_argument("--stop_on_error", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_root", default="results_fv/relation_condition_qwise")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--balance_trials", type=int, default=1, choices=[0, 1])
    parser.add_argument("--stepd_layers", default="all")
    parser.add_argument("--stepd_heads", default="all")
    parser.add_argument("--schema_version", default=SUPPORTED_SCHEMA)
    parser.add_argument("--lock_ttl_sec", type=int, default=1800)
    parser.add_argument("--strict_shared_query_target", type=int, default=0, choices=[0, 1])
    parser.add_argument("--sync_home_root", default=None)
    parser.add_argument("--sync_mode", default="none", choices=list(SYNC_MODE_CHOICES))
    parser.add_argument(
        "--home_artifact_profile",
        default="core",
        choices=list(HOME_ARTIFACT_PROFILE_CHOICES),
    )
    parser.add_argument("--sync_on_failure", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    if args.schema_version != SUPPORTED_SCHEMA:
        print(
            f"Unsupported schema_version={args.schema_version} "
            f"(expected={SUPPORTED_SCHEMA})"
        )
        return 1
    if args.n_trials_per_q < 2:
        print("n_trials_per_q must be >= 2")
        return 1
    if args.n_demos < 1:
        print("n_demos must be >= 1")
        return 1
    if args.topk < 1:
        print("topk must be >= 1")
        return 1
    try:
        parse_alpha_list(args.alpha_list)
    except ValueError as exc:
        print(str(exc))
        return 1
    conditions = parse_conditions(args.conditions)
    if set(conditions) != {"AAA", "BBB", "BABA"}:
        print("conditions must include AAA,BBB,BABA in this pipeline version")
        return 1

    relation_a_csv = Path(args.relation_a_csv).expanduser().resolve()
    relation_b_csv = Path(args.relation_b_csv).expanduser().resolve()
    if not relation_a_csv.exists():
        print(f"Missing relation_a_csv: {relation_a_csv}")
        return 1
    if not relation_b_csv.exists():
        print(f"Missing relation_b_csv: {relation_b_csv}")
        return 1

    selected_qids = collect_qids(
        relation_a_csv=relation_a_csv,
        relation_b_csv=relation_b_csv,
        q_list_text=args.q_list,
    )
    if not selected_qids:
        print("No q_ids selected.")
        return 1

    relation_name = resolve_relation_name(
        relation_name=args.relation_name,
        relation_a_csv=relation_a_csv,
        relation_b_csv=relation_b_csv,
        conditions=conditions,
    )
    out_root = Path(args.out_root).expanduser().resolve() / relation_name
    out_root.mkdir(parents=True, exist_ok=True)
    sync_mode = str(args.sync_mode).strip().lower()
    home_artifact_profile = str(args.home_artifact_profile).strip().lower()
    home_relation_root: Optional[Path] = None
    if args.sync_home_root:
        home_relation_root = Path(args.sync_home_root).expanduser().resolve() / relation_name
        home_relation_root.mkdir(parents=True, exist_ok=True)
    if sync_mode != "none" and home_relation_root is None:
        print("sync_mode requested but sync_home_root is empty; disabling home sync")
        sync_mode = "none"
    effective_artifact_profile = home_artifact_profile if sync_mode != "none" else "full"

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (f":{existing_pythonpath}" if existing_pythonpath else "")

    global_fingerprint = build_fingerprint_payload(
        args=args,
        relation_a_csv=relation_a_csv,
        relation_b_csv=relation_b_csv,
        selected_qids=selected_qids,
        relation_name=relation_name,
    )
    write_json(out_root / "_status" / "config_fingerprint.json", global_fingerprint)
    write_json(
        out_root / "_status" / "run_meta.json",
        {
            "created_at": utc_now(),
            "relation_name": relation_name,
            "n_qids": len(selected_qids),
            "schema_version": args.schema_version,
            "model": args.model,
            "model_spec": args.model_spec,
            "out_root": str(out_root),
            "sync_mode": sync_mode,
            "home_artifact_profile": home_artifact_profile,
            "sync_home_root": (str(home_relation_root) if home_relation_root else None),
            **storage_metadata(
                canonical_root=out_root,
                sync_root=home_relation_root,
                sync_mode=sync_mode,
                artifact_profile=effective_artifact_profile,
            ),
        },
    )

    row_counts_a = count_rows_by_q(str(relation_a_csv))
    row_counts_b = count_rows_by_q(str(relation_b_csv))

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        print(f"Failed to import transformers tokenizer: {exc}")
        return 1
    tok_add_special = resolve_prompt_add_special_tokens(args.model, args.model_spec)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    failures: List[str] = []
    home_sync_failures: List[str] = []
    processed = 0
    for q_idx, q_id in enumerate(selected_qids, start=1):
        processed += 1
        q_dir = out_root / q_id
        status_dir = q_dir / "_status"
        logs_dir = q_dir / "logs"
        status_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        orchestrator_log_path = logs_dir / "orchestrator.log"

        def log(message: str) -> None:
            line = f"[{utc_now()}][{q_id}] {message}"
            print(line, flush=True)
            with orchestrator_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

        log(f"start q {q_idx}/{len(selected_qids)}")
        lock_path = status_dir / "lock"
        owner_id = ""
        status_path = status_dir / "qid_status.json"
        fp_path = status_dir / "config_fingerprint.json"
        status = load_status(status_path, conditions)
        try:
            owner_id = acquire_lock(lock_path, ttl_sec=args.lock_ttl_sec)
            heartbeat_lock(lock_path, owner_id, ttl_sec=args.lock_ttl_sec)

            if row_counts_a.get(q_id, 0) < (args.n_demos + 1) or row_counts_b.get(q_id, 0) < (
                args.n_demos + 1
            ):
                status["status"] = "skipped"
                status["reason"] = (
                    "insufficient rows for demos+query "
                    f"A={row_counts_a.get(q_id, 0)} B={row_counts_b.get(q_id, 0)} "
                    f"required={args.n_demos + 1}"
                )
                status["updated_at"] = utc_now()
                write_json(status_path, status)
                log(str(status["reason"]))
                if (
                    sync_mode == "per_q"
                    and home_relation_root is not None
                    and bool(args.sync_on_failure)
                ):
                    ok = sync_q_with_status(
                        q_id=q_id,
                        q_dir=q_dir,
                        home_relation_root=home_relation_root,
                        sync_mode=sync_mode,
                        profile=home_artifact_profile,
                        reason="skipped",
                        log=log,
                    )
                    if not ok:
                        home_sync_failures.append(q_id)
                continue

            q_fingerprint = {
                **global_fingerprint,
                "q_id": q_id,
            }
            mismatch = False
            if fp_path.exists():
                old_fp = read_json(fp_path)
                for key in (
                    "schema_version",
                    "config_fingerprint",
                    "input_fingerprint",
                    "code_fingerprint",
                ):
                    if old_fp.get(key) != q_fingerprint.get(key):
                        mismatch = True
                        break
            if mismatch:
                log("fingerprint mismatch detected: invalidating all stage flags")
                for key in STAGE_KEYS:
                    status[key] = False
                status["stepd_conditions_done"] = {str(cond): False for cond in conditions}
                status["fingerprint_mismatch_at"] = utc_now()
            write_json(fp_path, q_fingerprint)

            status["status"] = "running"
            status["updated_at"] = utc_now()
            status["q_position"] = {"index": q_idx, "total": len(selected_qids)}
            status["q_root"] = str(q_dir)
            status.update(
                storage_metadata(
                    canonical_root=q_dir,
                    sync_root=(home_relation_root / q_id if home_relation_root is not None else None),
                    sync_mode=sync_mode,
                    artifact_profile=effective_artifact_profile,
                )
            )
            write_json(status_path, status)

            # Stage 0: trials
            heartbeat_lock(lock_path, owner_id, ttl_sec=args.lock_ttl_sec)
            stage_generate_trials_for_q(
                args=args,
                q_dir=q_dir,
                q_id=q_id,
                conditions=conditions,
                relation_a_csv=relation_a_csv,
                relation_b_csv=relation_b_csv,
                tokenizer=tokenizer,
                tok_add_special=bool(tok_add_special),
            )
            log("stage0 trials ready")

            # Stage 1: StepD
            if not (args.resume and status.get("stepd_done")):
                stepd_conditions_done = {
                    str(cond): bool(status.get("stepd_conditions_done", {}).get(str(cond), False))
                    for cond in conditions
                }
                for cond in conditions:
                    heartbeat_lock(lock_path, owner_id, ttl_sec=args.lock_ttl_sec)
                    trial_path = q_dir / "_trials" / f"condition_{cond}.json"
                    stepd_run_base = q_dir / "_stepd" / f"run_{cond}"
                    expected_stepd_fingerprint = _current_stepd_condition_fingerprint(
                        args=args,
                        q_fingerprint=q_fingerprint,
                        trial_path=trial_path,
                        condition=cond,
                    )
                    if args.resume and _stepd_outputs_ready(
                        q_dir=q_dir,
                        condition=cond,
                        expected_fingerprint=expected_stepd_fingerprint,
                        q_fingerprint=q_fingerprint,
                        q_fp_path=fp_path,
                        score_key=args.score_key,
                    ):
                        stepd_conditions_done[str(cond)] = True
                        status["stepd_conditions_done"] = stepd_conditions_done
                        status["updated_at"] = utc_now()
                        write_json(status_path, status)
                        log(f"stage1 stepd skip cond={cond} (resume)")
                        continue
                    run_base_score = stepd_run_base / "artifacts" / "aie_scores.csv"
                    if args.resume and run_base_score.exists():
                        copy_stepd_outputs(q_dir=q_dir, condition=cond, stepd_run_base=stepd_run_base)
                        if _stepd_outputs_ready(
                            q_dir=q_dir,
                            condition=cond,
                            expected_fingerprint=expected_stepd_fingerprint,
                            q_fingerprint=q_fingerprint,
                            q_fp_path=fp_path,
                            score_key=args.score_key,
                        ):
                            stepd_conditions_done[str(cond)] = True
                            status["stepd_conditions_done"] = stepd_conditions_done
                            status["updated_at"] = utc_now()
                            write_json(status_path, status)
                            log(f"stage1 stepd salvage cond={cond} (resume)")
                            continue
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
                        "--n_trials",
                        str(args.n_trials_per_q),
                        "--n_icl_examples",
                        str(args.n_demos),
                        "--score_key",
                        args.score_key,
                        "--seed",
                        str(args.seed),
                        "--fixed_trials_path",
                        str(trial_path),
                        "--fixed_out_dir",
                        str(stepd_run_base),
                    ]
                    if args.dtype:
                        cmd.extend(["--dtype", args.dtype])
                    if args.device_map:
                        cmd.extend(["--device_map", args.device_map])
                    run_command(
                        cmd=cmd,
                        log_path=logs_dir / f"stepD_{cond}.log",
                        env=env,
                        cwd=PROJECT_ROOT,
                    )
                    copy_stepd_outputs(q_dir=q_dir, condition=cond, stepd_run_base=stepd_run_base)
                    if not _stepd_outputs_ready(
                        q_dir=q_dir,
                        condition=cond,
                        expected_fingerprint=expected_stepd_fingerprint,
                        q_fingerprint=q_fingerprint,
                        q_fp_path=fp_path,
                        score_key=args.score_key,
                    ):
                        raise RuntimeError(f"StepD output validation failed for cond={cond}")
                    stepd_conditions_done[str(cond)] = True
                    status["stepd_conditions_done"] = stepd_conditions_done
                    status["updated_at"] = utc_now()
                    write_json(status_path, status)
                    log(f"stage1 stepd done cond={cond}")
                status["stepd_conditions_done"] = stepd_conditions_done
                status["stepd_done"] = all(stepd_conditions_done.get(str(cond), False) for cond in conditions)
                if not status["stepd_done"]:
                    raise RuntimeError("StepD stage incomplete after resume/run loop")
                invalidate_downstream(status, "topheads_done")
                status["updated_at"] = utc_now()
                write_json(status_path, status)
                log("stage1 stepd done")
            else:
                log("stage1 stepd skipped (resume)")

            # Stage 2: top-heads
            tophead_state = None
            if not (args.resume and status.get("topheads_done")):
                heartbeat_lock(lock_path, owner_id, ttl_sec=args.lock_ttl_sec)
                tophead_state = build_condition_top_heads(
                    stepd_dir=q_dir / "_stepd",
                    top_heads_dir=q_dir / "_top_heads",
                    conditions=conditions,
                    topk=args.topk,
                    score_key=args.score_key,
                    enable_union_ref=bool(args.enable_union_ref),
                )
                status["topheads_done"] = True
                invalidate_downstream(status, "vectors_done")
                status["updated_at"] = utc_now()
                write_json(status_path, status)
                log("stage2 topheads done")
            else:
                top_heads_dir = q_dir / "_top_heads"
                # lightweight reload for downstream stages
                tophead_state = {
                    "topk_rows": {
                        cond: select_topk(
                            load_stepd_scores_csv(
                                str(q_dir / "_stepd" / f"aie_scores_{cond}.csv"),
                                score_key=args.score_key,
                            ),
                            k=args.topk,
                        )
                        for cond in conditions
                    },
                    "union_top": (
                        read_json(top_heads_dir / "sets" / "top_heads_ref_union.json").get("heads", [])
                        if args.enable_union_ref
                        else []
                    ),
                    "union_candidate": unique_heads(
                        [
                            (int(row.layer), int(row.head))
                            for cond in conditions
                            for row in select_topk(
                                load_stepd_scores_csv(
                                    str(q_dir / "_stepd" / f"aie_scores_{cond}.csv"),
                                    score_key=args.score_key,
                                ),
                                k=args.topk,
                            )
                        ]
                    ),
                }
                log("stage2 topheads skipped (resume)")

            # Stage 3: vectors
            vector_meta: Dict[str, object] = {}
            if not (args.resume and status.get("vectors_done")):
                heartbeat_lock(lock_path, owner_id, ttl_sec=args.lock_ttl_sec)
                vector_meta = stage_vector_extract(
                    args=args,
                    q_dir=q_dir,
                    conditions=conditions,
                    tophead_state=tophead_state,
                    log=log,
                )
                status["vectors_done"] = True
                invalidate_downstream(status, "pca_done")
                status["updated_at"] = utc_now()
                write_json(status_path, status)
                log("stage3 vectors done")
            else:
                vector_meta_path = q_dir / "_vectors" / "vector_extraction_meta.json"
                vector_meta = read_json(vector_meta_path) if vector_meta_path.exists() else {}
                log("stage3 vectors skipped (resume)")

            # Stage 4: PCA
            if not (args.resume and status.get("pca_done")):
                heartbeat_lock(lock_path, owner_id, ttl_sec=args.lock_ttl_sec)
                run_pca_stage(args=args, q_dir=q_dir, log_dir=logs_dir, env=env)
                status["pca_done"] = True
                invalidate_downstream(status, "fv_done")
                status["updated_at"] = utc_now()
                write_json(status_path, status)
                log("stage4 pca done")
            else:
                log("stage4 pca skipped (resume)")

            # Stage 5: FV
            if not (args.resume and status.get("fv_done")):
                heartbeat_lock(lock_path, owner_id, ttl_sec=args.lock_ttl_sec)
                stage_build_fv(
                    args=args,
                    q_dir=q_dir,
                    conditions=conditions,
                    vector_meta=vector_meta,
                )
                status["fv_done"] = True
                invalidate_downstream(status, "inject_done")
                status["updated_at"] = utc_now()
                write_json(status_path, status)
                log("stage5 fv done")
            else:
                log("stage5 fv skipped (resume)")

            # Stage 6: optional injection
            if args.run_injection:
                if not (args.resume and status.get("inject_done")):
                    heartbeat_lock(lock_path, owner_id, ttl_sec=args.lock_ttl_sec)
                    run_injection_stage(args=args, q_dir=q_dir, log_dir=logs_dir, env=env)
                    status["inject_done"] = True
                    status["updated_at"] = utc_now()
                    write_json(status_path, status)
                    log("stage6 injection done")
                else:
                    log("stage6 injection skipped (resume)")
            else:
                status["inject_done"] = False
                status["updated_at"] = utc_now()
                write_json(status_path, status)
                log("stage6 injection disabled")

            status["status"] = "completed"
            status["reason"] = "all required stages completed"
            status["updated_at"] = utc_now()
            write_json(status_path, status)
            log("q completed")
            if sync_mode == "per_q" and home_relation_root is not None:
                ok = sync_q_with_status(
                    q_id=q_id,
                    q_dir=q_dir,
                    home_relation_root=home_relation_root,
                    sync_mode=sync_mode,
                    profile=home_artifact_profile,
                    reason="completed",
                    log=log,
                )
                if not ok:
                    home_sync_failures.append(q_id)
        except Exception as exc:
            status["status"] = "failed"
            status["reason"] = str(exc)
            status["updated_at"] = utc_now()
            write_json(status_path, status)
            failures.append(q_id)
            log(f"FAILED: {exc}")
            if (
                sync_mode == "per_q"
                and home_relation_root is not None
                and bool(args.sync_on_failure)
            ):
                ok = sync_q_with_status(
                    q_id=q_id,
                    q_dir=q_dir,
                    home_relation_root=home_relation_root,
                    sync_mode=sync_mode,
                    profile=home_artifact_profile,
                    reason="failed",
                    log=log,
                )
                if not ok:
                    home_sync_failures.append(q_id)
            if args.stop_on_error:
                release_lock(lock_path, owner_id)
                break
        finally:
            cleanup_gpu_memory(log)
            release_lock(lock_path, owner_id)

    summary = {
        "created_at": utc_now(),
        "relation_name": relation_name,
        "n_selected_qids": len(selected_qids),
        "processed": processed,
        "failures": failures,
        "out_root": str(out_root),
        "sync_mode": sync_mode,
        "home_artifact_profile": home_artifact_profile,
        "home_sync_failures": sorted(set(home_sync_failures)),
        **storage_metadata(
            canonical_root=out_root,
            sync_root=home_relation_root,
            sync_mode=sync_mode,
            artifact_profile=effective_artifact_profile,
        ),
    }
    write_json(out_root / "_status" / "run_summary.json", summary)
    if sync_mode == "end" and home_relation_root is not None:
        for q_id in selected_qids:
            q_dir = out_root / q_id
            if not q_dir.exists():
                continue
            try:
                q_status = read_json(q_dir / "_status" / "qid_status.json")
            except Exception:
                q_status = {}
            q_state = str(q_status.get("status", "unknown"))
            if q_state in {"failed", "skipped"} and not bool(args.sync_on_failure):
                continue
            ok = sync_q_with_status(
                q_id=q_id,
                q_dir=q_dir,
                home_relation_root=home_relation_root,
                sync_mode=sync_mode,
                profile=home_artifact_profile,
                reason=f"end_sync:{q_state}",
                log=lambda msg: print(f"[{utc_now()}][{q_id}] {msg}", flush=True),
            )
            if not ok:
                home_sync_failures.append(q_id)
        summary["home_sync_failures"] = sorted(set(home_sync_failures))
        write_json(out_root / "_status" / "run_summary.json", summary)
    if sync_mode != "none" and home_relation_root is not None:
        sync_relation_status_to_home(
            out_root=out_root,
            home_relation_root=home_relation_root,
            log=lambda msg: print(f"[{utc_now()}][sync] {msg}", flush=True),
        )
    if failures:
        print(f"completed with failures: {','.join(failures)}")
        return 1
    print("completed all qids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
