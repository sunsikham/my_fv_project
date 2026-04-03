#!/usr/bin/env python3
"""Add DDD/DADA condition artifacts to existing q-wise outputs and run 5-condition PCA."""

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
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.adapters import infer_head_dims
from fv.head_vector_extract import (
    extract_condition_trial_vectors,
    load_stepd_scores_csv,
    serialize_heads,
    select_topk,
    unique_heads,
)
from fv.hf_loader import load_hf_model_and_tokenizer
from fv.hooks import get_out_proj_pre_hook_target
from fv.prompting import build_prompt_qa, word_pairs_to_prompt_data
from fv.relation_trials import load_relation_csv
from fv.tokenization import resolve_prompt_add_special_tokens


DEFAULT_D_CONDITIONS = ("DDD", "DADA")
DEFAULT_PCA_CONDITIONS = ("AAA", "BBB", "BABA", "DDD", "DADA")
DEFAULT_PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
DEFAULT_SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}
SYNC_MODE_CHOICES = ("none", "per_q", "end")
HOME_ARTIFACT_PROFILE_CHOICES = ("core", "full")


Head = Tuple[int, int]


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


def parse_csv_list(text: str) -> List[str]:
    vals = [part.strip() for part in (text or "").split(",") if part.strip()]
    out: List[str] = []
    seen: Set[str] = set()
    for val in vals:
        key = val.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


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
    except Exception as exc:  # pragma: no cover
        if log is not None:
            log(f"[warn] torch.cuda.empty_cache failed: {exc}")
    try:
        torch.cuda.ipc_collect()
    except Exception as exc:  # pragma: no cover
        if log is not None:
            log(f"[warn] torch.cuda.ipc_collect failed: {exc}")


def _stable_seed(*parts: object) -> int:
    joined = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


def _normalize_pair(pair: Tuple[str, str]) -> Tuple[str, str]:
    left = pair[0].lstrip() if isinstance(pair[0], str) else str(pair[0])
    right = pair[1].lstrip() if isinstance(pair[1], str) else str(pair[1])
    return (left, right)


def _shuffle_outputs(
    demos: Sequence[Tuple[str, str]],
    rng: random.Random,
) -> List[Tuple[str, str]]:
    outputs = [out for _inp, out in demos]
    shuffled = outputs[:]
    if len(shuffled) > 1:
        for _ in range(8):
            rng.shuffle(shuffled)
            if shuffled != outputs:
                break
        if shuffled == outputs:
            shuffled = shuffled[1:] + shuffled[:1]
    return [(inp, out) for (inp, _old), out in zip(demos, shuffled)]


def _alternating_demos(
    pairs_a: Sequence[Tuple[str, str]],
    pairs_d: Sequence[Tuple[str, str]],
    demo_indices: Sequence[int],
    start_with: str,
) -> List[Tuple[str, str]]:
    demos: List[Tuple[str, str]] = []
    for pos, idx in enumerate(demo_indices):
        use_d = (pos % 2 == 0 and start_with == "D") or (pos % 2 == 1 and start_with == "A")
        source = pairs_d if use_d else pairs_a
        demos.append(source[idx])
    return demos


def _compute_target_token_fields(
    tokenizer,
    tok_add_special: bool,
    prefix_str: str,
    target_str: str,
) -> Dict[str, object]:
    boundary_prefix = prefix_str
    boundary_answer = target_str
    if boundary_prefix.endswith(" ") and not boundary_answer.startswith(" "):
        boundary_prefix = boundary_prefix[:-1]
        boundary_answer = f" {boundary_answer}"
    full_ids = tokenizer.encode(
        boundary_prefix + boundary_answer,
        add_special_tokens=tok_add_special,
    )
    prefix_ids = tokenizer.encode(
        boundary_prefix,
        add_special_tokens=tok_add_special,
    )
    if len(full_ids) <= len(prefix_ids):
        raise ValueError("Tokenization does not extend prefix for target boundary.")
    answer_ids = full_ids[len(prefix_ids) :]
    return {
        "target_first_token_id": int(answer_ids[0]),
        "answer_ids": [int(x) for x in answer_ids],
    }


def validate_relation_d_csv(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing relation_d_csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(
                f"relationD CSV has no header. expected columns include id/ex_A/ex_B: {path}"
            )
        rows = list(reader)
    if len(rows) < 1:
        raise ValueError(
            "relationD CSV is empty. Fill datasets/relation/relationD_ex.csv with id,ex_A,ex_B rows."
        )
    required = {"id", "ex_A", "ex_B"}
    cols = set(reader.fieldnames or [])
    if not required.issubset(cols):
        raise ValueError(
            f"relationD CSV missing required columns {sorted(required)} (found={sorted(cols)})."
        )


def load_condition_template(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing condition template: {path}")
    payload = read_json(path)
    if not isinstance(payload.get("trials"), list):
        raise ValueError(f"Invalid condition template (missing trials list): {path}")
    return payload


def build_d_payload(
    *,
    q_id: str,
    cond: str,
    template_payload: Dict[str, object],
    pairs_a: Sequence[Tuple[str, str]],
    pairs_d: Sequence[Tuple[str, str]],
    relation_a_csv: Path,
    relation_d_csv: Path,
    tokenizer,
    tok_add_special: bool,
) -> Dict[str, object]:
    template_meta = template_payload.get("meta", {}) if isinstance(template_payload, dict) else {}
    template_trials = template_payload.get("trials", [])
    if not isinstance(template_meta, dict) or not isinstance(template_trials, list):
        raise ValueError("Invalid template payload format")

    seed = int(template_meta.get("seed", 0))
    prefixes = template_meta.get("prefixes", DEFAULT_PREFIXES)
    separators = template_meta.get("separators", DEFAULT_SEPARATORS)
    if not isinstance(prefixes, dict):
        prefixes = DEFAULT_PREFIXES
    if not isinstance(separators, dict):
        separators = DEFAULT_SEPARATORS
    prepend_bos = bool(template_meta.get("prepend_bos_token_used", False))

    out_trials: List[Dict[str, object]] = []
    for idx, trial in enumerate(template_trials):
        if not isinstance(trial, dict):
            raise ValueError(f"trial entry must be object: q_id={q_id} idx={idx}")
        trial_id = str(trial.get("trial_id", f"t{idx:06d}"))
        query_source_index = int(trial["query_source_index"])
        demo_source_indices = [int(x) for x in trial["demo_source_indices"]]
        if query_source_index < 0:
            raise ValueError(f"query_source_index must be >=0: q_id={q_id} trial_id={trial_id}")
        if query_source_index >= len(pairs_a) or query_source_index >= len(pairs_d):
            raise ValueError(
                f"query_source_index out of range for D build: q_id={q_id} trial_id={trial_id}"
            )
        for demo_idx in demo_source_indices:
            if demo_idx < 0 or demo_idx >= len(pairs_a) or demo_idx >= len(pairs_d):
                raise ValueError(
                    f"demo_source_indices out of range for D build: q_id={q_id} trial_id={trial_id}"
                )

        if cond == "DDD":
            demos_clean = [pairs_d[demo_idx] for demo_idx in demo_source_indices]
            query_pair = pairs_d[query_source_index]
        elif cond == "DADA":
            demos_clean = _alternating_demos(
                pairs_a=pairs_a,
                pairs_d=pairs_d,
                demo_indices=demo_source_indices,
                start_with="D",
            )
            query_pair = pairs_a[query_source_index]
        else:
            raise ValueError(f"Unsupported D condition: {cond}")

        cond_rng = random.Random(_stable_seed(seed, q_id, trial_id, cond, "corr"))
        demos_corrupted = _shuffle_outputs(demos_clean, cond_rng)

        clean_prefix_str, _clean_full = build_prompt_qa(
            demos_clean,
            query_pair,
            prefixes=prefixes,
            separators=separators,
            prepend_bos_token=prepend_bos,
            prepend_space=True,
        )
        corrupted_prefix_str, _corr_full = build_prompt_qa(
            demos_corrupted,
            query_pair,
            prefixes=prefixes,
            separators=separators,
            prepend_bos_token=prepend_bos,
            prepend_space=True,
        )

        prompt_data_clean = word_pairs_to_prompt_data(
            {"input": [x for x, _y in demos_clean], "output": [y for _x, y in demos_clean]},
            query_target_pair={"input": [query_pair[0]], "output": [query_pair[1]]},
            prepend_bos_token=prepend_bos,
            prefixes=prefixes,
            separators=separators,
            shuffle_labels=False,
            prepend_space=True,
        )
        prompt_data_corrupted = word_pairs_to_prompt_data(
            {"input": [x for x, _y in demos_corrupted], "output": [y for _x, y in demos_corrupted]},
            query_target_pair={"input": [query_pair[0]], "output": [query_pair[1]]},
            prepend_bos_token=prepend_bos,
            prefixes=prefixes,
            separators=separators,
            shuffle_labels=False,
            prepend_space=True,
        )
        target_str = str(prompt_data_clean["query_target"]["output"])
        token_fields = _compute_target_token_fields(
            tokenizer=tokenizer,
            tok_add_special=tok_add_special,
            prefix_str=corrupted_prefix_str,
            target_str=target_str,
        )

        query_anchor = trial.get(
            "query_anchor",
            {"input": pairs_a[query_source_index][0], "output": pairs_a[query_source_index][1]},
        )
        out_trials.append(
            {
                "q_id": q_id,
                "condition": cond,
                "trial_idx": int(trial.get("trial_idx", idx)),
                "trial_id": trial_id,
                "query_source_index": query_source_index,
                "demo_source_indices": demo_source_indices,
                "demo_order": [int(x) for x in trial.get("demo_order", list(range(len(demo_source_indices))))],
                "demos_clean": [{"input": x, "output": y} for x, y in demos_clean],
                "demos_corrupted": [{"input": x, "output": y} for x, y in demos_corrupted],
                "query": {"input": query_pair[0], "output": query_pair[1]},
                "query_anchor": query_anchor,
                "clean_prompt_str": clean_prefix_str,
                "corrupted_prompt_str": corrupted_prefix_str,
                "prompt_data_clean": prompt_data_clean,
                "prompt_data_corrupted": prompt_data_corrupted,
                "target_str": target_str,
                "target_first_token_id": int(token_fields["target_first_token_id"]),
                "answer_ids": [int(x) for x in token_fields["answer_ids"]],
            }
        )

    meta = dict(template_meta)
    meta.update(
        {
            "condition": cond,
            "q_id": q_id,
            "relation_a_csv": str(relation_a_csv),
            "relation_b_csv": str(relation_d_csv),
            "source": "d_extension_from_template_v1",
        }
    )
    return {"meta": meta, "trials": out_trials}


def copy_stepd_outputs(*, q_dir: Path, condition: str, stepd_run_base: Path) -> None:
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
        },
    )


def load_aaa_ref_heads(q_dir: Path) -> Set[Head]:
    ref_path = q_dir / "_top_heads" / "sets" / "top_heads_ref_AAA.json"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing AAA_ref top-head set: {ref_path}")
    payload = read_json(ref_path)
    rows = payload.get("heads", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Invalid AAA_ref top-head set file: {ref_path}")
    out: Set[Head] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.add((int(row["layer"]), int(row["head"])))
    if not out:
        raise ValueError(f"No heads parsed from AAA_ref set: {ref_path}")
    return out


def extract_d_vectors(
    *,
    args: argparse.Namespace,
    q_dir: Path,
    d_conditions: Sequence[str],
    tok_add_special: bool,
    log: Callable[[str], None],
) -> Dict[str, object]:
    vectors_dir = q_dir / "_vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    aaa_ref_heads = load_aaa_ref_heads(q_dir)
    topk_by_cond: Dict[str, Set[Head]] = {}
    for cond in d_conditions:
        score_path = q_dir / "_stepd" / f"aie_scores_{cond}.csv"
        rows = load_stepd_scores_csv(str(score_path), score_key=args.score_key)
        top_rows = select_topk(rows, k=args.topk)
        topk_by_cond[cond] = set((row.layer, row.head) for row in top_rows)

    capture_heads = unique_heads(set(aaa_ref_heads).union(*topk_by_cond.values()))
    model = None
    tokenizer = None
    layer_modules: Dict[int, object] = {}
    trial_ids_out: Dict[str, List[str]] = {}
    seq_idx_out: Dict[str, List[int]] = {}
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
        log(f"[d-vector] model loaded diagnostics={json.dumps(diagnostics, ensure_ascii=True)}")
        model.eval()
        dims = infer_head_dims(model, spec_name=args.model_spec)
        n_heads = int(dims["n_heads"])
        head_dim = int(dims["head_dim"])
        resid_dim = int(dims["hidden_size"])

        capture_layers = sorted(set(layer for layer, _head in capture_heads))
        for layer in capture_layers:
            module, _path = get_out_proj_pre_hook_target(
                model, layer_idx=layer, spec_name=args.model_spec, logger=None
            )
            layer_modules[layer] = module

        for cond in d_conditions:
            trial_path = q_dir / "_trials" / f"condition_{cond}.json"
            payload = read_json(trial_path)
            trials = payload.get("trials", [])
            if not isinstance(trials, list):
                raise ValueError(f"Invalid trials payload: {trial_path}")
            log(f"[d-vector] start condition={cond} n_trials={len(trials)}")
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
                union_ref_heads=None,
                cond_topk_heads=topk_by_cond[cond],
                logger=log,
            )
            np.save(vectors_dir / f"trial_vectors_AAA_ref_{cond}.npy", extracted["aaa_ref"])
            np.save(
                vectors_dir / f"trial_vectors_d_extension_capture_headwise_{cond}.npy",
                extracted["capture_headwise"],
            )
            trial_ids_out[cond] = [str(x) for x in extracted["trial_ids"]]
            seq_idx_out[cond] = [int(x) for x in extracted["seq_token_indices"]]

        meta_path = vectors_dir / "vector_extraction_meta.json"
        if meta_path.exists():
            meta = read_json(meta_path)
        else:
            meta = {"created_at": utc_now()}
        trial_ids = meta.get("trial_ids", {})
        seq_ids = meta.get("seq_token_indices", {})
        if not isinstance(trial_ids, dict):
            trial_ids = {}
        if not isinstance(seq_ids, dict):
            seq_ids = {}
        for cond in d_conditions:
            trial_ids[cond] = trial_ids_out[cond]
            seq_ids[cond] = seq_idx_out[cond]
        meta["trial_ids"] = trial_ids
        meta["seq_token_indices"] = seq_ids
        headwise_sets = meta.get("headwise_sets", {})
        if not isinstance(headwise_sets, dict):
            headwise_sets = {}
        headwise_sets["d_extension_capture"] = {
            "file_pattern": "trial_vectors_d_extension_capture_headwise_<COND>.npy",
            "heads": serialize_heads(capture_heads),
            "head_count": int(len(capture_heads)),
            "dtype": "float16",
            "shape_note": "(n_trials, n_capture_heads, resid_dim)",
            "conditions": list(d_conditions),
            "source": "d_extension_capture",
            "created_at": utc_now(),
        }
        meta["headwise_sets"] = headwise_sets
        meta["d_extension_updated_at"] = utc_now()
        write_json(meta_path, meta)
        return meta
    finally:
        layer_modules.clear()
        if tokenizer is not None:
            del tokenizer
        if model is not None:
            del model
        cleanup_gpu_memory(log)


def load_status(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {
            "created_at": utc_now(),
            "status": "pending",
            "ddd_trials_done": False,
            "dada_trials_done": False,
            "ddd_stepd_done": False,
            "dada_stepd_done": False,
            "d_vectors_done": False,
            "pca5_done": False,
        }
    payload = read_json(path)
    payload.setdefault("status", "pending")
    for key in (
        "ddd_trials_done",
        "dada_trials_done",
        "ddd_stepd_done",
        "dada_stepd_done",
        "d_vectors_done",
        "pca5_done",
    ):
        payload.setdefault(key, False)
    return payload


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
    home_base_root: Path,
    profile: str,
    reason: str,
    log: Callable[[str], None],
) -> bool:
    home_q_dir = home_base_root / q_id
    payload: Dict[str, object] = {
        "updated_at": utc_now(),
        "status": "ok",
        "q_id": q_id,
        "reason": reason,
        "profile": profile,
        "source_q_dir": str(q_dir),
        "home_q_dir": str(home_q_dir),
    }
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


def sync_base_status_to_home(
    *,
    base_root: Path,
    home_base_root: Path,
    log: Callable[[str], None],
) -> bool:
    try:
        copy_tree_if_exists(base_root / "_status", home_base_root / "_status")
        return True
    except Exception as exc:
        log(f"[warn] base _status sync failed: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run D extension (DDD/DADA) on existing q-wise outputs.")
    parser.add_argument("--base_root", required=True)
    parser.add_argument("--relation_a_csv", required=True)
    parser.add_argument("--relation_d_csv", required=True)
    parser.add_argument("--q_list", required=True, help="CSV q ids, e.g. Q11,Q13,Q16")
    parser.add_argument("--d_conditions", default="DDD,DADA")
    parser.add_argument("--pca_conditions", default="AAA,BBB,BABA,DDD,DADA")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_spec", default="llama3")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default=None, choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quant", default="auto", choices=["auto", "none", "4bit", "8bit"])
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--n_trials_per_q", type=int, default=25)
    parser.add_argument("--n_demos", type=int, default=9)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--score_key", default="mean_delta_p")
    parser.add_argument("--stepd_layers", default="16-55")
    parser.add_argument("--stepd_heads", default="all")
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--balance_trials", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pca_ref_mode", default="AAA_ref", choices=["AAA_ref", "union_ref"])
    parser.add_argument("--pca_out_subdir", default="AAA_ref_with_D")
    parser.add_argument("--resume", type=int, default=1, choices=[0, 1])
    parser.add_argument("--stop_on_error", type=int, default=0, choices=[0, 1])
    parser.add_argument("--on_busy", default="skip", choices=["skip", "fail"])
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--sync_home_root", default=None)
    parser.add_argument("--sync_mode", default="none", choices=list(SYNC_MODE_CHOICES))
    parser.add_argument(
        "--home_artifact_profile",
        default="core",
        choices=list(HOME_ARTIFACT_PROFILE_CHOICES),
    )
    parser.add_argument("--sync_on_failure", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    base_root = Path(args.base_root).expanduser().resolve()
    relation_a_csv = Path(args.relation_a_csv).expanduser().resolve()
    relation_d_csv = Path(args.relation_d_csv).expanduser().resolve()
    if not base_root.exists():
        print(f"Missing base_root: {base_root}")
        return 1
    if not relation_a_csv.exists():
        print(f"Missing relation_a_csv: {relation_a_csv}")
        return 1
    try:
        validate_relation_d_csv(relation_d_csv)
    except Exception as exc:
        print(str(exc))
        return 1

    q_list = parse_csv_list(args.q_list)
    if not q_list:
        print("q_list must not be empty")
        return 1
    d_conditions = parse_csv_list(args.d_conditions)
    if tuple(d_conditions) != DEFAULT_D_CONDITIONS:
        print(f"d_conditions must be exactly {','.join(DEFAULT_D_CONDITIONS)}")
        return 1
    pca_conditions = parse_csv_list(args.pca_conditions)
    if tuple(pca_conditions) != DEFAULT_PCA_CONDITIONS:
        print(f"pca_conditions must be exactly {','.join(DEFAULT_PCA_CONDITIONS)}")
        return 1
    sync_mode = str(args.sync_mode).strip().lower()
    home_artifact_profile = str(args.home_artifact_profile).strip().lower()
    home_base_root: Optional[Path] = None
    if args.sync_home_root:
        home_base_root = Path(args.sync_home_root).expanduser().resolve()
        home_base_root.mkdir(parents=True, exist_ok=True)
    if sync_mode != "none" and home_base_root is None:
        print("sync_mode requested but sync_home_root is empty; disabling home sync")
        sync_mode = "none"

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        print(f"Failed to import transformers tokenizer: {exc}")
        return 1
    tok_add_special = resolve_prompt_add_special_tokens(args.model, args.model_spec)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    relation_a_by_q = {q: [(_normalize_pair(pair)) for pair in pairs] for q, pairs in load_relation_csv(str(relation_a_csv)).items()}
    relation_d_by_q = {q: [(_normalize_pair(pair)) for pair in pairs] for q, pairs in load_relation_csv(str(relation_d_csv)).items()}

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (f":{existing_pythonpath}" if existing_pythonpath else "")

    failures: List[str] = []
    completed: List[str] = []
    skipped: List[str] = []
    home_sync_failures: List[str] = []

    for q_id in q_list:
        q_dir = base_root / q_id
        logs_dir = q_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        status_path = q_dir / "_status" / "d_extension_status.json"
        status = load_status(status_path)

        def log(msg: str) -> None:
            line = f"[{utc_now()}][{q_id}] {msg}"
            print(line, flush=True)
            with (logs_dir / "d_extension_orchestrator.log").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

        if not q_dir.exists():
            failures.append(q_id)
            log(f"FAILED: q_dir missing: {q_dir}")
            status["status"] = "failed"
            status["reason"] = f"q_dir missing: {q_dir}"
            status["updated_at"] = utc_now()
            write_json(status_path, status)
            if (
                sync_mode == "per_q"
                and home_base_root is not None
                and bool(args.sync_on_failure)
            ):
                ok = sync_q_with_status(
                    q_id=q_id,
                    q_dir=q_dir,
                    home_base_root=home_base_root,
                    profile=home_artifact_profile,
                    reason="failed:q_dir_missing",
                    log=log,
                )
                if not ok:
                    home_sync_failures.append(q_id)
            if args.stop_on_error:
                break
            continue

        active_lock = q_dir / "_status" / "lock"
        if active_lock.exists():
            reason = f"active lock exists: {active_lock}"
            if args.on_busy == "fail":
                failures.append(q_id)
                log(f"FAILED: {reason}")
                status["status"] = "failed"
                status["reason"] = reason
                status["updated_at"] = utc_now()
                write_json(status_path, status)
                if (
                    sync_mode == "per_q"
                    and home_base_root is not None
                    and bool(args.sync_on_failure)
                ):
                    ok = sync_q_with_status(
                        q_id=q_id,
                        q_dir=q_dir,
                        home_base_root=home_base_root,
                        profile=home_artifact_profile,
                        reason="failed:on_busy",
                        log=log,
                    )
                    if not ok:
                        home_sync_failures.append(q_id)
                if args.stop_on_error:
                    break
                continue
            skipped.append(q_id)
            log(f"SKIP: {reason}")
            status["status"] = "skipped"
            status["reason"] = reason
            status["updated_at"] = utc_now()
            write_json(status_path, status)
            if (
                sync_mode == "per_q"
                and home_base_root is not None
                and bool(args.sync_on_failure)
            ):
                ok = sync_q_with_status(
                    q_id=q_id,
                    q_dir=q_dir,
                    home_base_root=home_base_root,
                    profile=home_artifact_profile,
                    reason="skipped:on_busy",
                    log=log,
                )
                if not ok:
                    home_sync_failures.append(q_id)
            continue

        try:
            log("start D extension")
            status["status"] = "running"
            status["updated_at"] = utc_now()
            write_json(status_path, status)

            trials_dir = q_dir / "_trials"
            template_bbb = load_condition_template(trials_dir / "condition_BBB.json")
            template_baba = load_condition_template(trials_dir / "condition_BABA.json")
            if q_id not in relation_a_by_q:
                raise ValueError(f"q_id missing in relationA CSV: {q_id}")
            if q_id not in relation_d_by_q:
                raise ValueError(f"q_id missing in relationD CSV: {q_id}")
            pairs_a = relation_a_by_q[q_id]
            pairs_d = relation_d_by_q[q_id]

            if not (args.resume and (trials_dir / "condition_DDD.json").exists()):
                ddd_payload = build_d_payload(
                    q_id=q_id,
                    cond="DDD",
                    template_payload=template_bbb,
                    pairs_a=pairs_a,
                    pairs_d=pairs_d,
                    relation_a_csv=relation_a_csv,
                    relation_d_csv=relation_d_csv,
                    tokenizer=tokenizer,
                    tok_add_special=bool(tok_add_special),
                )
                write_json(trials_dir / "condition_DDD.json", ddd_payload)
            status["ddd_trials_done"] = True

            if not (args.resume and (trials_dir / "condition_DADA.json").exists()):
                dada_payload = build_d_payload(
                    q_id=q_id,
                    cond="DADA",
                    template_payload=template_baba,
                    pairs_a=pairs_a,
                    pairs_d=pairs_d,
                    relation_a_csv=relation_a_csv,
                    relation_d_csv=relation_d_csv,
                    tokenizer=tokenizer,
                    tok_add_special=bool(tok_add_special),
                )
                write_json(trials_dir / "condition_DADA.json", dada_payload)
            status["dada_trials_done"] = True
            status["updated_at"] = utc_now()
            write_json(status_path, status)
            log("D trials ready")

            for cond in d_conditions:
                score_out = q_dir / "_stepd" / f"aie_scores_{cond}.csv"
                if args.resume and score_out.exists():
                    log(f"StepD skip cond={cond} (resume)")
                    status[f"{cond.lower()}_stepd_done"] = True
                    continue
                trial_path = trials_dir / f"condition_{cond}.json"
                stepd_run_base = q_dir / "_stepd" / f"run_{cond}"
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
                status[f"{cond.lower()}_stepd_done"] = True
                status["updated_at"] = utc_now()
                write_json(status_path, status)
                log(f"StepD done cond={cond}")

            d_vector_paths = [q_dir / "_vectors" / f"trial_vectors_AAA_ref_{cond}.npy" for cond in d_conditions]
            if not (args.resume and all(path.exists() for path in d_vector_paths)):
                extract_d_vectors(
                    args=args,
                    q_dir=q_dir,
                    d_conditions=d_conditions,
                    tok_add_special=bool(tok_add_special),
                    log=log,
                )
            status["d_vectors_done"] = True
            status["updated_at"] = utc_now()
            write_json(status_path, status)
            log("D vectors done")

            pca_out = q_dir / "_pca_common" / args.pca_out_subdir / "pca_model_meta.json"
            if not (args.resume and pca_out.exists()):
                cmd = [
                    args.python_bin,
                    str(PROJECT_ROOT / "scripts" / "run_condition_common_pca.py"),
                    "--q_dir",
                    str(q_dir),
                    "--ref_mode",
                    args.pca_ref_mode,
                    "--conditions",
                    ",".join(pca_conditions),
                    "--n_components",
                    str(args.n_components),
                    "--balance_trials",
                    str(args.balance_trials),
                    "--seed",
                    str(args.seed),
                    "--out_subdir",
                    args.pca_out_subdir,
                ]
                run_command(
                    cmd=cmd,
                    log_path=logs_dir / "pca_AAA_ref_with_D.log",
                    env=env,
                    cwd=PROJECT_ROOT,
                )
            status["pca5_done"] = True
            status["status"] = "completed"
            status["reason"] = "DDD/DADA extension and 5-condition PCA complete"
            status["updated_at"] = utc_now()
            write_json(status_path, status)
            completed.append(q_id)
            log("q completed")
            if sync_mode == "per_q" and home_base_root is not None:
                ok = sync_q_with_status(
                    q_id=q_id,
                    q_dir=q_dir,
                    home_base_root=home_base_root,
                    profile=home_artifact_profile,
                    reason="completed",
                    log=log,
                )
                if not ok:
                    home_sync_failures.append(q_id)
        except Exception as exc:
            failures.append(q_id)
            status["status"] = "failed"
            status["reason"] = str(exc)
            status["updated_at"] = utc_now()
            write_json(status_path, status)
            log(f"FAILED: {exc}")
            if (
                sync_mode == "per_q"
                and home_base_root is not None
                and bool(args.sync_on_failure)
            ):
                ok = sync_q_with_status(
                    q_id=q_id,
                    q_dir=q_dir,
                    home_base_root=home_base_root,
                    profile=home_artifact_profile,
                    reason="failed",
                    log=log,
                )
                if not ok:
                    home_sync_failures.append(q_id)
            if args.stop_on_error:
                break
        finally:
            cleanup_gpu_memory(log)

    summary = {
        "created_at": utc_now(),
        "base_root": str(base_root),
        "q_list": q_list,
        "completed": completed,
        "skipped": skipped,
        "failures": failures,
        "sync_mode": sync_mode,
        "home_artifact_profile": home_artifact_profile,
        "home_sync_root": (str(home_base_root) if home_base_root else None),
        "home_sync_failures": sorted(set(home_sync_failures)),
    }
    write_json(base_root / "_status" / "d_extension_run_summary.json", summary)
    if sync_mode == "end" and home_base_root is not None:
        for q_id in q_list:
            q_dir = base_root / q_id
            if not q_dir.exists():
                continue
            try:
                q_status = read_json(q_dir / "_status" / "d_extension_status.json")
            except Exception:
                q_status = {}
            q_state = str(q_status.get("status", "unknown"))
            if q_state in {"failed", "skipped"} and not bool(args.sync_on_failure):
                continue
            ok = sync_q_with_status(
                q_id=q_id,
                q_dir=q_dir,
                home_base_root=home_base_root,
                profile=home_artifact_profile,
                reason=f"end_sync:{q_state}",
                log=lambda msg: print(f"[{utc_now()}][{q_id}] {msg}", flush=True),
            )
            if not ok:
                home_sync_failures.append(q_id)
        summary["home_sync_failures"] = sorted(set(home_sync_failures))
        write_json(base_root / "_status" / "d_extension_run_summary.json", summary)
    if sync_mode != "none" and home_base_root is not None:
        sync_base_status_to_home(
            base_root=base_root,
            home_base_root=home_base_root,
            log=lambda msg: print(f"[{utc_now()}][sync] {msg}", flush=True),
        )
    if failures:
        print(f"completed with failures: {','.join(failures)}")
        return 1
    print("completed all requested q_ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
