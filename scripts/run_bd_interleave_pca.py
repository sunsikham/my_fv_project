#!/usr/bin/env python3
"""Run Q1-style BD interleave PCA using a shared BD_ref measurement basis.

Reuse-first rules:
- Reuse existing BBB / DDD trial payloads and StepD score CSVs.
- Generate only the missing mixed-BD trial payloads.
- Reuse the existing StepD runner for those new mixed conditions.
- Build a shared BD_ref from BBB, DDD, BDBDBD_D, and DBDBDB_B scores.
- Re-extract vectors under BD_ref for all four comparison conditions.
"""

from __future__ import annotations

import argparse
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
    build_rank_map,
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


DEFAULT_PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
DEFAULT_SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}
PURE_CONDITIONS = ("BBB", "DDD")
MIXED_CONDITIONS = ("BDBDBD_D", "DBDBDB_B")
ALL_BD_CONDITIONS = PURE_CONDITIONS + MIXED_CONDITIONS

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


def storage_metadata(
    *,
    canonical_root: str,
    sync_root: Optional[str] = None,
    sync_mode: str = "none",
    artifact_profile: str = "full",
) -> Dict[str, object]:
    return {
        "canonical_root": str(canonical_root),
        "sync_root": (str(sync_root) if sync_root is not None else None),
        "sync_mode": str(sync_mode),
        "artifact_profile": str(artifact_profile),
    }


def make_logger(log_path: Path) -> Callable[[str], None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


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
    pairs_b: Sequence[Tuple[str, str]],
    pairs_d: Sequence[Tuple[str, str]],
    demo_indices: Sequence[int],
    start_with: str,
) -> List[Tuple[str, str]]:
    demos: List[Tuple[str, str]] = []
    for pos, idx in enumerate(demo_indices):
        use_d = (pos % 2 == 0 and start_with == "D") or (pos % 2 == 1 and start_with == "B")
        source = pairs_d if use_d else pairs_b
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


def load_condition_template(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing condition template: {path}")
    payload = read_json(path)
    if not isinstance(payload.get("trials"), list):
        raise ValueError(f"Invalid condition template (missing trials list): {path}")
    return payload


def validate_template_alignment(
    *,
    template_b: Dict[str, object],
    template_d: Dict[str, object],
    q_id: str,
) -> None:
    trials_b = template_b.get("trials", [])
    trials_d = template_d.get("trials", [])
    if not isinstance(trials_b, list) or not isinstance(trials_d, list):
        raise ValueError("Template payloads must contain trials list")
    if len(trials_b) != len(trials_d):
        raise ValueError(
            f"Q={q_id}: BBB and DDD templates differ in length "
            f"({len(trials_b)} vs {len(trials_d)})"
        )
    for idx, (trial_b, trial_d) in enumerate(zip(trials_b, trials_d)):
        if not isinstance(trial_b, dict) or not isinstance(trial_d, dict):
            raise ValueError(f"Q={q_id}: invalid template trial at index={idx}")
        key_b = (
            str(trial_b.get("trial_id", f"t{idx:06d}")),
            int(trial_b.get("trial_idx", idx)),
            int(trial_b["query_source_index"]),
            [int(x) for x in trial_b["demo_source_indices"]],
        )
        key_d = (
            str(trial_d.get("trial_id", f"t{idx:06d}")),
            int(trial_d.get("trial_idx", idx)),
            int(trial_d["query_source_index"]),
            [int(x) for x in trial_d["demo_source_indices"]],
        )
        if key_b != key_d:
            raise ValueError(
                f"Q={q_id}: BBB/DDD template mismatch at trial index={idx} "
                f"BBB={key_b} DDD={key_d}"
            )


def _serialize_top_rows(rows: Sequence[object]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "layer": int(row.layer),
                "head": int(row.head),
                "score": float(row.score),
                "score_key": str(row.score_key),
            }
        )
    return out


def build_mixed_bd_payload(
    *,
    q_id: str,
    cond: str,
    template_payload: Dict[str, object],
    pairs_b: Sequence[Tuple[str, str]],
    pairs_d: Sequence[Tuple[str, str]],
    relation_b_csv: Path,
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
        if query_source_index >= len(pairs_b) or query_source_index >= len(pairs_d):
            raise ValueError(
                f"query_source_index out of range for BD build: q_id={q_id} trial_id={trial_id}"
            )
        for demo_idx in demo_source_indices:
            if demo_idx < 0 or demo_idx >= len(pairs_b) or demo_idx >= len(pairs_d):
                raise ValueError(
                    f"demo_source_indices out of range for BD build: q_id={q_id} trial_id={trial_id}"
                )

        if cond == "BDBDBD_D":
            demos_clean = _alternating_demos(
                pairs_b=pairs_b,
                pairs_d=pairs_d,
                demo_indices=demo_source_indices,
                start_with="B",
            )
            query_pair = pairs_d[query_source_index]
        elif cond == "DBDBDB_B":
            demos_clean = _alternating_demos(
                pairs_b=pairs_b,
                pairs_d=pairs_d,
                demo_indices=demo_source_indices,
                start_with="D",
            )
            query_pair = pairs_b[query_source_index]
        else:
            raise ValueError(f"Unsupported BD mixed condition: {cond}")

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

        query_anchor = trial.get("query_anchor")
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
            "relation_b_csv": str(relation_b_csv),
            "relation_d_csv": str(relation_d_csv),
            "source": "bd_interleave_from_template_v1",
            "layout_rule": cond,
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


def build_bd_ref(
    *,
    q_dir: Path,
    conditions: Sequence[str],
    score_key: str,
    topk: int,
    ref_name: str,
    log: Callable[[str], None],
) -> Set[Head]:
    top_heads_dir = q_dir / "_top_heads"
    top_heads_dir.mkdir(parents=True, exist_ok=True)
    sets_dir = top_heads_dir / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)

    stepd_rows: Dict[str, list] = {}
    topk_rows: Dict[str, list] = {}
    for cond in conditions:
        score_path = q_dir / "_stepd" / f"aie_scores_{cond}.csv"
        rows = load_stepd_scores_csv(str(score_path), score_key=score_key)
        stepd_rows[cond] = rows
        topk_rows[cond] = select_topk(rows, k=topk)
        if cond in MIXED_CONDITIONS or cond == "DDD":
            write_json(
                top_heads_dir / f"top_heads_{cond}.json",
                {
                    "meta": {
                        "condition": cond,
                        "score_key": score_key,
                        "k": int(topk),
                        "source_scores_path": str(score_path),
                    },
                    "heads": _serialize_top_rows(topk_rows[cond]),
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
                    f"Missing rank for BD ref candidate head layer={layer} head={head} cond={cond}"
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
    ref_tag = str(ref_name).strip()
    if ref_tag.endswith("_ref"):
        ref_tag = ref_tag[: -len("_ref")]
    if not ref_tag:
        ref_tag = "BD"
    out_path = sets_dir / f"top_heads_ref_{ref_tag}.json"
    write_json(
        out_path,
        {
            "meta": {
                "source": "rank_mean_bd_ref",
                "ref_name": str(ref_name),
                "score_key": score_key,
                "k": int(topk),
                "conditions": list(conditions),
                "created_at": utc_now(),
            },
            "heads": union_top,
        },
    )
    log(f"saved BD ref set: {out_path} n_heads={len(union_top)}")
    return set((int(row["layer"]), int(row["head"])) for row in union_top)


def extract_bd_vectors(
    *,
    args: argparse.Namespace,
    q_dir: Path,
    conditions: Sequence[str],
    bd_ref_heads: Set[Head],
    ref_name: str,
    tok_add_special: bool,
    log: Callable[[str], None],
) -> None:
    vectors_dir = q_dir / "_vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    topk_by_cond: Dict[str, Set[Head]] = {}
    for cond in conditions:
        score_path = q_dir / "_stepd" / f"aie_scores_{cond}.csv"
        rows = load_stepd_scores_csv(str(score_path), score_key=args.score_key)
        top_rows = select_topk(rows, k=args.topk)
        topk_by_cond[cond] = set((row.layer, row.head) for row in top_rows)

    capture_heads = unique_heads(set(bd_ref_heads).union(*topk_by_cond.values()))
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
        log(f"[bd-vector] model loaded diagnostics={json.dumps(diagnostics, ensure_ascii=True)}")
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

        for cond in conditions:
            trial_path = q_dir / "_trials" / f"condition_{cond}.json"
            payload = read_json(trial_path)
            trials = payload.get("trials", [])
            if not isinstance(trials, list):
                raise ValueError(f"Invalid trials payload: {trial_path}")
            log(f"[bd-vector] start condition={cond} n_trials={len(trials)}")
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
                aaa_ref_heads=bd_ref_heads,
                union_ref_heads=None,
                cond_topk_heads=topk_by_cond[cond],
                logger=log,
            )
            np.save(vectors_dir / f"trial_vectors_{ref_name}_{cond}.npy", extracted["aaa_ref"])
            np.save(
                vectors_dir / f"trial_vectors_{ref_name}_capture_headwise_{cond}.npy",
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
        for cond in conditions:
            trial_ids[cond] = trial_ids_out[cond]
            seq_ids[cond] = seq_idx_out[cond]
        meta["trial_ids"] = trial_ids
        meta["seq_token_indices"] = seq_ids
        capture_map = meta.get("capture_heads_by_ref", {})
        if not isinstance(capture_map, dict):
            capture_map = {}
        capture_map[str(ref_name)] = [
            {"layer": int(layer), "head": int(head)} for layer, head in capture_heads
        ]
        meta["capture_heads_by_ref"] = capture_map
        headwise_sets = meta.get("headwise_sets", {})
        if not isinstance(headwise_sets, dict):
            headwise_sets = {}
        headwise_sets[f"{ref_name}_capture"] = {
            "file_pattern": f"trial_vectors_{ref_name}_capture_headwise_<COND>.npy",
            "heads": serialize_heads(capture_heads),
            "head_count": int(len(capture_heads)),
            "dtype": "float16",
            "shape_note": "(n_trials, n_capture_heads, resid_dim)",
            "conditions": list(conditions),
            "source": f"{ref_name}_capture",
            "created_at": utc_now(),
        }
        meta["headwise_sets"] = headwise_sets
        ref_meta = meta.get("ref_meta", {})
        if not isinstance(ref_meta, dict):
            ref_meta = {}
        ref_meta[str(ref_name)] = {
            "conditions": list(conditions),
            "capture_heads_count": int(len(capture_heads)),
            "capture_heads": serialize_heads(capture_heads),
            "updated_at": utc_now(),
        }
        meta["ref_meta"] = ref_meta
        write_json(meta_path, meta)
    finally:
        layer_modules.clear()
        if tokenizer is not None:
            del tokenizer
        if model is not None:
            del model
        cleanup_gpu_memory(log)


def load_status(path: Path) -> Dict[str, object]:
    if path.exists():
        payload = read_json(path)
        if isinstance(payload, dict):
            return payload
    return {}


def _resolve_template_counts(template_payload: Dict[str, object]) -> Tuple[int, int]:
    meta = template_payload.get("meta", {})
    trials = template_payload.get("trials", [])
    if not isinstance(meta, dict) or not isinstance(trials, list) or not trials:
        raise ValueError("Template payload missing meta/trials")
    n_trials = int(meta.get("n_trials", len(trials)))
    n_demos = int(meta.get("n_demos", meta.get("n_shots", 0)))
    if n_trials < 1 or n_demos < 1:
        raise ValueError("Template payload missing positive n_trials/n_demos")
    return n_trials, n_demos


def _normalize_by_q(csv_path: Path) -> Dict[str, List[Tuple[str, str]]]:
    return {q: [_normalize_pair(pair) for pair in pairs] for q, pairs in load_relation_csv(str(csv_path)).items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BD interleave PCA sanity check for one q_dir.")
    parser.add_argument("--q_dir", required=True)
    parser.add_argument("--q_id", default=None)
    parser.add_argument("--relation_b_csv", required=True)
    parser.add_argument("--relation_d_csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_spec", default="llama3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--quant", default="4bit")
    parser.add_argument("--device_map", default="")
    parser.add_argument("--score_key", default="mean_delta_p")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--stepd_layers", default="16-55")
    parser.add_argument("--stepd_heads", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--balance_trials", type=int, default=1, choices=[0, 1])
    parser.add_argument("--pca_out_subdir", default="BD_ref_BD_compare")
    parser.add_argument("--ref_name", default="BD_ref")
    parser.add_argument("--run_label", default="")
    parser.add_argument("--resume", type=int, default=1, choices=[0, 1])
    parser.add_argument("--stop_on_error", type=int, default=1, choices=[0, 1])
    parser.add_argument("--canonical_root", default="/scratch/sunsik/my_fv_project")
    args = parser.parse_args()

    q_dir = Path(args.q_dir).expanduser().resolve()
    q_id = str(args.q_id or q_dir.name)
    relation_b_csv = Path(args.relation_b_csv).expanduser().resolve()
    relation_d_csv = Path(args.relation_d_csv).expanduser().resolve()
    ref_name = str(args.ref_name).strip() or "BD_ref"
    run_label = str(args.run_label).strip()
    logs_dir = q_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_name = "bd_interleave_pca_orchestrator.log" if not run_label else f"bd_interleave_pca_{run_label}.log"
    status_name = "bd_interleave_pca_status.json" if not run_label else f"bd_interleave_pca_{run_label}_status.json"
    log = make_logger(logs_dir / log_name)
    status_path = q_dir / "_status" / status_name
    status = load_status(status_path)
    status.update(
        {
            "q_id": q_id,
            "status": "running",
            "updated_at": utc_now(),
            "storage": storage_metadata(canonical_root=args.canonical_root),
            "conditions": list(ALL_BD_CONDITIONS),
            "ref_mode": ref_name,
        }
    )
    write_json(status_path, status)

    if not q_dir.exists():
        status["status"] = "failed"
        status["reason"] = f"q_dir missing: {q_dir}"
        status["updated_at"] = utc_now()
        write_json(status_path, status)
        log(f"FAILED: q_dir missing: {q_dir}")
        return 1

    active_lock = q_dir / "_status" / "lock"
    if active_lock.exists():
        status["status"] = "failed"
        status["reason"] = f"active lock exists: {active_lock}"
        status["updated_at"] = utc_now()
        write_json(status_path, status)
        log(f"FAILED: active lock exists: {active_lock}")
        return 1

    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        status["status"] = "failed"
        status["reason"] = f"transformers tokenizer unavailable: {exc}"
        status["updated_at"] = utc_now()
        write_json(status_path, status)
        log(f"FAILED: transformers tokenizer unavailable: {exc}")
        return 1

    tok_add_special = resolve_prompt_add_special_tokens(args.model, args.model_spec)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    relation_b_by_q = _normalize_by_q(relation_b_csv)
    relation_d_by_q = _normalize_by_q(relation_d_csv)
    if q_id not in relation_b_by_q:
        raise ValueError(f"q_id missing in relationB CSV: {q_id}")
    if q_id not in relation_d_by_q:
        raise ValueError(f"q_id missing in relationD CSV: {q_id}")
    pairs_b = relation_b_by_q[q_id]
    pairs_d = relation_d_by_q[q_id]

    template_bbb = load_condition_template(q_dir / "_trials" / "condition_BBB.json")
    template_ddd = load_condition_template(q_dir / "_trials" / "condition_DDD.json")
    validate_template_alignment(template_b=template_bbb, template_d=template_ddd, q_id=q_id)
    n_trials_per_q, n_demos = _resolve_template_counts(template_bbb)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (f":{existing_pythonpath}" if existing_pythonpath else "")

    try:
        trials_dir = q_dir / "_trials"
        stepd_dir = q_dir / "_stepd"
        stepd_dir.mkdir(parents=True, exist_ok=True)

        for cond in MIXED_CONDITIONS:
            trial_path = trials_dir / f"condition_{cond}.json"
            if bool(args.resume) and trial_path.exists():
                log(f"trial payload skip cond={cond} (resume)")
                continue
            payload = build_mixed_bd_payload(
                q_id=q_id,
                cond=cond,
                template_payload=template_bbb,
                pairs_b=pairs_b,
                pairs_d=pairs_d,
                relation_b_csv=relation_b_csv,
                relation_d_csv=relation_d_csv,
                tokenizer=tokenizer,
                tok_add_special=bool(tok_add_special),
            )
            write_json(trial_path, payload)
            log(f"trial payload written cond={cond}: {trial_path}")
        status["mixed_trials_done"] = True
        status["updated_at"] = utc_now()
        write_json(status_path, status)

        for cond in MIXED_CONDITIONS:
            score_out = stepd_dir / f"aie_scores_{cond}.csv"
            if bool(args.resume) and score_out.exists():
                log(f"StepD skip cond={cond} (resume)")
                continue
            trial_path = trials_dir / f"condition_{cond}.json"
            stepd_run_base = stepd_dir / f"run_{cond}"
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
                str(n_trials_per_q),
                "--n_icl_examples",
                str(n_demos),
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
            log(f"StepD done cond={cond}")
        status["mixed_stepd_done"] = True
        status["updated_at"] = utc_now()
        write_json(status_path, status)

        bd_ref_heads = build_bd_ref(
            q_dir=q_dir,
            conditions=ALL_BD_CONDITIONS,
            score_key=args.score_key,
            topk=args.topk,
            ref_name=ref_name,
            log=log,
        )
        status["bd_ref_done"] = True
        status["updated_at"] = utc_now()
        write_json(status_path, status)

        bd_vector_paths = [q_dir / "_vectors" / f"trial_vectors_{ref_name}_{cond}.npy" for cond in ALL_BD_CONDITIONS]
        if not (bool(args.resume) and all(path.exists() for path in bd_vector_paths)):
            extract_bd_vectors(
                args=args,
                q_dir=q_dir,
                conditions=ALL_BD_CONDITIONS,
                bd_ref_heads=bd_ref_heads,
                ref_name=ref_name,
                tok_add_special=bool(tok_add_special),
                log=log,
            )
        else:
            log("BD vector extraction skip (resume)")
        status["bd_vectors_done"] = True
        status["updated_at"] = utc_now()
        write_json(status_path, status)

        pca_out = q_dir / "_pca_common" / args.pca_out_subdir / "pca_model_meta.json"
        if not (bool(args.resume) and pca_out.exists()):
            cmd = [
                args.python_bin,
                str(PROJECT_ROOT / "scripts" / "run_condition_common_pca.py"),
                "--q_dir",
                str(q_dir),
                "--ref_mode",
                ref_name,
                "--conditions",
                ",".join(ALL_BD_CONDITIONS),
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
                log_path=logs_dir / "pca_BD_ref_BD_compare.log",
                env=env,
                cwd=PROJECT_ROOT,
            )
        else:
            log("PCA skip (resume)")
        status["pca_done"] = True
        status["status"] = "completed"
        status["reason"] = "BD interleave PCA sanity check complete"
        status["updated_at"] = utc_now()
        write_json(status_path, status)
        log("completed")
        return 0
    except Exception as exc:
        status["status"] = "failed"
        status["reason"] = str(exc)
        status["updated_at"] = utc_now()
        write_json(status_path, status)
        log(f"FAILED: {exc}")
        if bool(args.stop_on_error):
            return 1
        return 0
    finally:
        del tokenizer


if __name__ == "__main__":
    raise SystemExit(main())
