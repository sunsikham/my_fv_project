#!/usr/bin/env python3
"""STEP D: AIE head sweep using mean_activations replacement on corrupted prompts."""

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.corrupt import make_corrupted_demos
from fv.dataset_loader import load_pairs_antonym, sample_demos_and_query
from fv.fixed_trials import load_fixed_trials
from fv.hooks import get_out_proj_pre_hook_target
from fv.io import prepare_run_dirs, resolve_out_dir, save_csv, save_json
from fv.adapters import infer_head_dims, resolve_blocks
from fv.hf_loader import load_hf_model_and_tokenizer
from fv.mean_activations import (
    compute_mean_activations_fixed_trials_ns,
    compute_mean_activations_ns,
)
from fv.model_spec import get_model_spec
from fv.patch import make_out_proj_head_output_overrider
from fv.prompting import (
    build_prompt_qa_paper,
    compute_duplicated_labels as paper_compute_duplicated_labels,
    get_dummy_token_labels as paper_get_dummy_token_labels,
    get_token_meta_labels as paper_get_token_meta_labels,
)
from fv.slots import (
    compute_query_predictive_slot,
    resolve_slot_seq_token_idx,
    get_target_first_token_id_from_boundary,
)
from fv.mean_activations import paper_labels_to_slot_map
from fv.relation_trials import generate_relation_trials, save_trials_json
from fv.stepd_resume import compute_stepd_code_fingerprint, stable_hash_hex


def storage_metadata(
    *,
    canonical_root: str,
    sync_root: str | None = None,
    sync_mode: str = "none",
    artifact_profile: str = "full",
):
    return {
        "canonical_root": str(canonical_root),
        "sync_root": (str(sync_root) if sync_root is not None else None),
        "sync_mode": str(sync_mode),
        "artifact_profile": str(artifact_profile),
    }


def make_logger(log_path: str):
    log_file = open(log_path, "w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    return log, log_file


def _json_safe(obj):
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def parse_layers(value: str):
    layers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError("Invalid layer range")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def parse_heads(value: str, n_heads: int):
    if value == "all":
        return list(range(n_heads))
    heads = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        heads.append(int(part))
    return sorted(set(heads))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def mean_abs(values):
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def _mean_vec_for_head(mean_acts, layer: int, head: int, slot_q: int):
    if hasattr(mean_acts, "dim") and mean_acts.dim() == 4:
        return mean_acts[layer, head, slot_q]
    return mean_acts[layer, head]


def _build_stepd_input_fingerprint(fixed_trials) -> str:
    return stable_hash_hex(fixed_trials)


def _build_stepd_resume_fingerprint(args, fixed_trials, *, layers, heads) -> str:
    return stable_hash_hex(
        {
            "mode": "relation_qid" if args.relation_csv_path else "single",
            "model": args.model,
            "model_spec": args.model_spec,
            "dtype": args.dtype,
            "quant": args.quant,
            "device_map": args.device_map,
            "layers": list(layers),
            "heads": list(heads),
            "n_trials": int(args.n_trials),
            "n_icl_examples": int(args.n_icl_examples),
            "n_mean_trials": int(args.n_mean_trials),
            "score_key": args.score_key,
            "seed": int(args.seed),
            "successful_icl_only": int(args.successful_icl_only),
            "shuffle_labels": int(args.shuffle_labels),
            "compute_prob_scores": int(args.compute_prob_scores),
            "baseline_recompute_outproj": True,
            "input_fingerprint": _build_stepd_input_fingerprint(fixed_trials),
            "code_fingerprint": compute_stepd_code_fingerprint(PROJECT_ROOT),
        }
    )


def _resume_paths(artifacts_dir: str) -> dict:
    resume_dir = os.path.join(artifacts_dir, "_resume")
    return {
        "dir": resume_dir,
        "state": os.path.join(resume_dir, "resume_state.json"),
        "scores": os.path.join(resume_dir, "aie_scores.resume.jsonl"),
        "trial_metrics": os.path.join(resume_dir, "trial_metrics.resume.jsonl"),
        "dump": os.path.join(resume_dir, "dump_trial_metrics.resume.jsonl"),
    }


def _ensure_resume_root_state(*, artifacts_dir: str, resume_fingerprint: str, log) -> dict:
    from fv.io import load_json

    paths = _resume_paths(artifacts_dir)
    os.makedirs(paths["dir"], exist_ok=True)
    state = load_json(paths["state"]) if os.path.exists(paths["state"]) else None
    if state is not None and state.get("resume_fingerprint") != resume_fingerprint:
        for key in ("scores", "trial_metrics", "dump", "state"):
            if os.path.exists(paths[key]):
                os.remove(paths[key])
        state = None
        log("resume fingerprint mismatch: cleared prior _resume state")
    if state is None:
        save_json(
            paths["state"],
            {
                "resume_fingerprint": resume_fingerprint,
                "created_at": int(time.time()),
            },
        )
    return paths


def _jsonl_rows(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _append_jsonl_rows(path: str, rows) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=_json_safe) + "\n")
        handle.flush()


def _dedupe_rows(rows, key_fn):
    out = {}
    for row in rows:
        out[key_fn(row)] = row
    return list(out.values())


def _score_row_key(row):
    return (row.get("q_id"), int(row["layer"]), int(row["head"]))


def _head_key(*, layer: int, head: int, q_id=None):
    return (q_id, int(layer), int(head))


def _trial_row_key(row):
    return (
        row.get("q_id"),
        int(row["layer"]),
        int(row["head"]),
        int(row["trial_idx"]),
    )


def _filter_rows_for_q(rows, q_id=None):
    if q_id is None:
        return [row for row in rows if row.get("q_id") in (None, "__all__")]
    return [row for row in rows if row.get("q_id") == q_id]


def _completed_head_keys(rows):
    return {_head_key(layer=int(row["layer"]), head=int(row["head"]), q_id=row.get("q_id")) for row in rows}


def _build_score_row_from_trial_rows(
    *,
    trial_rows,
    layer: int,
    head: int,
    n_trials_for_row: int,
    mean_acts,
    slot_q: int,
    compute_prob_scores: bool,
    q_id=None,
):
    metric_lists = _empty_metric_lists()
    nonzero = False
    for trial_row in trial_rows:
        for key in metric_lists:
            value = trial_row.get(key)
            if value is not None:
                metric_lists[key].append(float(value))
        if compute_prob_scores and trial_row.get("delta_p") is not None:
            if abs(float(trial_row["delta_p"])) > 1e-12:
                nonzero = True
    mean_vec = _mean_vec_for_head(mean_acts, layer=layer, head=head, slot_q=slot_q)
    row = {
        "layer": int(layer),
        "head": int(head),
        "n_trials": int(n_trials_for_row),
        "mean_delta_p": mean(metric_lists["delta_p"]),
        "std_delta_p": std(metric_lists["delta_p"]),
        "mean_abs_delta_p": mean_abs(metric_lists["delta_p"]),
        "mean_p_base": mean(metric_lists["p_base"]),
        "std_p_base": std(metric_lists["p_base"]),
        "mean_p_patch": mean(metric_lists["p_patch"]),
        "std_p_patch": std(metric_lists["p_patch"]),
        "mean_p_base_target": mean(metric_lists["p_base"]),
        "mean_p_patch_target": mean(metric_lists["p_patch"]),
        "mean_delta_p_target": mean(metric_lists["delta_p"]),
        "mean_delta_logit": mean(metric_lists["delta_logit"]),
        "std_delta_logit": std(metric_lists["delta_logit"]),
        "mean_abs_delta_logit": mean_abs(metric_lists["delta_logit"]),
        "mean_logit_base": mean(metric_lists["logit_base"]),
        "mean_logit_patch": mean(metric_lists["logit_patch"]),
        "mean_delta_logprob": mean(metric_lists["delta_logprob"]),
        "std_delta_logprob": std(metric_lists["delta_logprob"]),
        "mean_abs_delta_logprob": mean_abs(metric_lists["delta_logprob"]),
        "mean_logprob_base": mean(metric_lists["logprob_base"]),
        "mean_logprob_patch": mean(metric_lists["logprob_patch"]),
        "any_nonzero": nonzero,
        "mean_act_norm": mean_vec.norm().item(),
        "clean_mean_norm": mean_vec.norm().item(),
    }
    if q_id is not None:
        row["q_id"] = q_id
    return row


def _recover_legacy_completed_heads(
    *,
    legacy_trial_metrics_path: str,
    q_id,
    n_trials_for_row: int,
    mean_acts,
    slot_q: int,
    compute_prob_scores: bool,
):
    rows = _filter_rows_for_q(_jsonl_rows(legacy_trial_metrics_path), q_id=q_id)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), int(row["head"]))].append(row)
    recovered_score_rows = []
    recovered_trial_rows = []
    for (layer, head), head_rows in sorted(grouped.items()):
        deduped_head_rows = _dedupe_rows(head_rows, _trial_row_key)
        unique_trial_ids = {int(row["trial_idx"]) for row in deduped_head_rows}
        if len(unique_trial_ids) != int(n_trials_for_row):
            continue
        recovered_score_rows.append(
            _build_score_row_from_trial_rows(
                trial_rows=deduped_head_rows,
                layer=layer,
                head=head,
                n_trials_for_row=n_trials_for_row,
                mean_acts=mean_acts,
                slot_q=slot_q,
                compute_prob_scores=compute_prob_scores,
                q_id=q_id,
            )
        )
        recovered_trial_rows.extend(deduped_head_rows)
    return recovered_score_rows, recovered_trial_rows


def _prepare_resume_state(
    *,
    artifacts_dir: str,
    resume_fingerprint: str,
    q_id,
    n_trials_for_row: int,
    mean_acts,
    slot_q: int,
    compute_prob_scores: bool,
    log,
):
    paths = _resume_paths(artifacts_dir)

    score_rows = _dedupe_rows(_filter_rows_for_q(_jsonl_rows(paths["scores"]), q_id=q_id), _score_row_key)
    trial_rows = _dedupe_rows(_filter_rows_for_q(_jsonl_rows(paths["trial_metrics"]), q_id=q_id), _trial_row_key)
    if not score_rows:
        legacy_trial_metrics_path = os.path.join(artifacts_dir, "trial_metrics.jsonl")
        if os.path.exists(legacy_trial_metrics_path):
            recovered_scores, recovered_trials = _recover_legacy_completed_heads(
                legacy_trial_metrics_path=legacy_trial_metrics_path,
                q_id=q_id,
                n_trials_for_row=n_trials_for_row,
                mean_acts=mean_acts,
                slot_q=slot_q,
                compute_prob_scores=compute_prob_scores,
            )
            if recovered_scores:
                _append_jsonl_rows(paths["trial_metrics"], recovered_trials)
                _append_jsonl_rows(paths["scores"], recovered_scores)
                score_rows = _dedupe_rows(recovered_scores, _score_row_key)
                trial_rows = _dedupe_rows(recovered_trials, _trial_row_key)
                log(
                    "recovered legacy StepD progress: "
                    f"q_id={q_id if q_id is not None else '__all__'} "
                    f"completed_heads={len(score_rows)}"
                )
    return {
        "paths": paths,
        "score_rows": score_rows,
        "trial_rows": trial_rows,
        "completed_head_keys": _completed_head_keys(score_rows),
    }


def _write_canonical_jsonl(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=_json_safe) + "\n")


def _materialize_final_outputs(
    *,
    artifacts_dir: str,
    score_rows,
    dump_final_path: str | None,
    save_trials_enabled: bool,
):
    paths = _resume_paths(artifacts_dir)
    completed_keys = _completed_head_keys(score_rows)
    trial_rows = []
    for row in _dedupe_rows(_jsonl_rows(paths["trial_metrics"]), _trial_row_key):
        if _head_key(layer=int(row["layer"]), head=int(row["head"]), q_id=row.get("q_id")) in completed_keys:
            trial_rows.append(row)
    _write_canonical_jsonl(os.path.join(artifacts_dir, "trial_metrics.jsonl"), trial_rows)
    if save_trials_enabled and trial_rows:
        trial_fields = [
            "trial_idx",
            "layer",
            "head",
            "target_id",
            "p_base",
            "p_patch",
            "delta_p",
            "p_base_target",
            "p_patch_target",
            "delta_p_target",
            "logit_base",
            "logit_patch",
            "delta_logit",
            "logprob_base",
            "logprob_patch",
            "delta_logprob",
        ]
        aie_trials_rows = [{key: row.get(key) for key in trial_fields} for row in trial_rows]
        save_csv(os.path.join(artifacts_dir, "aie_trials.csv"), aie_trials_rows)
    if dump_final_path:
        dump_rows = []
        for row in _dedupe_rows(_jsonl_rows(paths["dump"]), _trial_row_key):
            if _head_key(layer=int(row["layer"]), head=int(row["head"]), q_id=row.get("q_id")) in completed_keys:
                dump_rows.append(row)
        _write_canonical_jsonl(dump_final_path, dump_rows)


DEFAULT_PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
DEFAULT_SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}


def _strip_leading_space(text):
    return text[1:] if isinstance(text, str) and text.startswith(" ") else text


def _normalize_demos(demos):
    if demos and isinstance(demos[0], dict):
        return [
            (_strip_leading_space(d["input"]), _strip_leading_space(d["output"]))
            for d in demos
        ]
    return [(_strip_leading_space(x), _strip_leading_space(y)) for x, y in demos]


def _normalize_query(query):
    if isinstance(query, dict):
        return (_strip_leading_space(query["input"]), _strip_leading_space(query["output"]))
    return (_strip_leading_space(query[0]), _strip_leading_space(query[1]))


def _resolve_prompt_meta(fixed_meta, tok_add_special: bool):
    prefixes = (fixed_meta or {}).get("prefixes") or DEFAULT_PREFIXES
    separators = (fixed_meta or {}).get("separators") or DEFAULT_SEPARATORS
    prepend_bos_token_used = (fixed_meta or {}).get("prepend_bos_token_used")
    if prepend_bos_token_used is None:
        prepend_bos_token_used = False if tok_add_special else True
    return {
        "prefixes": prefixes,
        "separators": separators,
        "prepend_bos_token_used": bool(prepend_bos_token_used),
    }


def run_stepd_debug(
    *,
    model,
    tokenizer,
    device,
    pairs,
    model_cfg,
    layer,
    head,
    tok_add_special: bool,
    baseline_recompute_outproj: bool,
    resolved_quant,
    seed: int,
    n_icl_examples: int,
    n_trials: int,
    model_spec: str,
    prompt_meta: dict,
    log,
) -> None:
    import torch
    import torch.nn.functional as F

    target_module, _target_name = get_out_proj_pre_hook_target(
        model, layer, spec_name=model_spec, logger=log
    )
    hook_state = {"mode": "self", "replace_vec": None, "seq_token_idx": None}
    hook_state["debug_capture"] = True
    hook = make_out_proj_head_output_overrider(
        layer_idx=layer,
        head_idx=head,
        seq_token_idx=0,
        mode="self",
        replace_vec=None,
        model_config=model_cfg,
        resolved_quant=resolved_quant,
        force_recompute_outproj=baseline_recompute_outproj,
        state=hook_state,
        logger=log,
    )
    replace_vec = torch.full(
        (int(model_cfg["head_dim"]),), 0.123, dtype=torch.float32, device=device
    )

    def _fmt(value):
        if value is None:
            return "None"
        return f"{value:.3e}"

    def _format_stats(label, stats):
        if not stats:
            log(f"[StepD debug] {label}: missing stats")
            return
        target_before = stats.get("target_before", {})
        target_after = stats.get("target_after", {})
        replace_stats = stats.get("replace_vec", None)
        log(
            "[StepD debug] "
            f"{label} mode={stats.get('mode')} "
            f"target_before(mean={_fmt(target_before.get('mean'))} "
            f"norm={_fmt(target_before.get('norm'))}) "
            f"target_after(mean={_fmt(target_after.get('mean'))} "
            f"norm={_fmt(target_after.get('norm'))}) "
            f"target_diff_max={_fmt(stats.get('target_diff_max'))} "
            f"other_heads_diff_max={_fmt(stats.get('other_heads_diff_max'))} "
            f"other_tokens_diff_max={_fmt(stats.get('other_tokens_diff_max'))}"
        )
        if replace_stats is not None:
            log(
                "[StepD debug] "
                f"{label} replace_vec(mean={_fmt(replace_stats.get('mean'))} "
                f"norm={_fmt(replace_stats.get('norm'))})"
            )

    max_abs_diffs = []
    p_target_diffs = []
    for trial_idx in range(n_trials):
        demos, query = sample_demos_and_query(
            pairs, n_icl_examples, seed=seed + trial_idx
        )
        demos_norm = _normalize_demos(demos)
        query_norm = _normalize_query(query)
        prefix_str, full_str, _prompt_data = build_prompt_qa_paper(
            demos_norm,
            query_norm,
            prefixes=prompt_meta["prefixes"],
            separators=prompt_meta["separators"],
            prepend_bos_token=prompt_meta["prepend_bos_token_used"],
            prepend_space=True,
        )
        boundary_prefix, boundary_answer = _boundary_prefix_and_answer_from_full(
            prefix_str, full_str
        )
        target_id = get_target_first_token_id_from_boundary(
            boundary_prefix,
            boundary_answer,
            tokenizer,
            tokenize_kwargs={"add_special_tokens": tok_add_special},
        )
        if trial_idx < 3:
            target_token = tokenizer.convert_ids_to_tokens(target_id)
            try:
                slot_info = compute_query_predictive_slot(
                    prefix_str,
                    full_str,
                    tokenizer,
                    add_special_tokens=tok_add_special,
                )
                src_id = slot_info["target_id"]
                src_token = slot_info["target_token"]
                token_id_match = src_id == target_id
                token_match = src_token == target_token
                if not token_match and src_token.lstrip() == target_token.lstrip():
                    mismatch_tag = "leading_space_mismatch"
                elif not token_match:
                    mismatch_tag = "token_mismatch"
                else:
                    mismatch_tag = "match"
                log(
                    "[StepD debug] "
                    f"trial={trial_idx} src_token_id_of_interest={src_id} "
                    f"src_token={repr(src_token)} "
                    f"target_id={target_id} target_token={repr(target_token)} "
                    f"match_id={token_id_match} match_token={token_match} "
                    f"tag={mismatch_tag}"
                )
            except Exception as exc:  # pragma: no cover - debug logging only
                log(f"[StepD debug] trial={trial_idx} src_compare_failed: {exc}")

        inputs = tokenizer(
            prefix_str, return_tensors="pt", add_special_tokens=tok_add_special
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        hook_state["seq_token_idx"] = int(inputs["input_ids"].shape[1] - 1)
        prefix_ids = inputs["input_ids"][0].tolist()
        full_ids = tokenizer(
            full_str, add_special_tokens=tok_add_special
        )["input_ids"]
        if hook_state["seq_token_idx"] != len(prefix_ids) - 1:
            log(
                "[StepD debug] slot alignment mismatch: "
                f"seq_token_idx={hook_state['seq_token_idx']} "
                f"prefix_last_idx={len(prefix_ids) - 1}"
            )
        if hook_state["seq_token_idx"] < len(full_ids):
            decoded_prefix = tokenizer.decode(
                [prefix_ids[hook_state["seq_token_idx"]]]
            )
            decoded_full = tokenizer.decode(
                [full_ids[hook_state["seq_token_idx"]]]
            )
            if decoded_prefix != decoded_full:
                if decoded_prefix.lstrip() == decoded_full.lstrip():
                    mismatch_tag = "leading_space_mismatch"
                else:
                    mismatch_tag = "token_mismatch"
                log(
                    "[StepD debug] "
                    f"patch_token_mismatch={mismatch_tag} "
                    f"decoded_prefix={repr(decoded_prefix)} "
                    f"decoded_full={repr(decoded_full)}"
                )
        else:
            log(
                "[StepD debug] "
                f"full_ids too short for seq_token_idx={hook_state['seq_token_idx']}"
            )

        with torch.inference_mode():
            outputs_raw = model(**inputs)
        hook_state["mode"] = "self"
        hook_state["replace_vec"] = None
        hook_state["debug_stats"] = None
        handle = target_module.register_forward_hook(hook)
        with torch.inference_mode():
            outputs_self = model(**inputs)
        handle.remove()
        if trial_idx < 3:
            _format_stats("baseline(self)", hook_state.get("debug_stats"))

        raw_logits = outputs_raw.logits[0, -1]
        self_logits = outputs_self.logits[0, -1]
        max_abs_diff = (raw_logits - self_logits).abs().max().item()
        max_abs_diffs.append(max_abs_diff)

        p_raw = F.softmax(raw_logits.float(), dim=-1)[target_id].item()
        p_self = F.softmax(self_logits.float(), dim=-1)[target_id].item()
        p_diff = abs(p_raw - p_self)
        p_target_diffs.append(p_diff)

        if trial_idx == 0:
            raw_argmax = int(torch.argmax(raw_logits).item())
            self_argmax = int(torch.argmax(self_logits).item())
            log(
                "[StepD debug] "
                f"max_abs_diff={max_abs_diff:.6e} "
                f"raw_argmax={raw_argmax} self_argmax={self_argmax}"
            )
        log(
            "[StepD debug] "
            f"trial={trial_idx} |p_target(raw)-p_target(self)|={p_diff:.6e}"
        )
        if trial_idx < 3:
            hook_state["mode"] = "replace"
            hook_state["replace_vec"] = replace_vec
            hook_state["alpha"] = 1.0
            hook_state["debug_stats"] = None
            handle = target_module.register_forward_hook(hook)
            with torch.inference_mode():
                _ = model(**inputs)
            handle.remove()
            _format_stats("patch(replace)", hook_state.get("debug_stats"))

    max_logit_diff = max(max_abs_diffs) if max_abs_diffs else 0.0
    mean_p_diff = mean(p_target_diffs) if p_target_diffs else 0.0
    max_p_diff = max(p_target_diffs) if p_target_diffs else 0.0
    log(
        "[StepD debug] "
        f"p_target_diff_mean={mean_p_diff:.6e} "
        f"p_target_diff_max={max_p_diff:.6e}"
    )
    if max_logit_diff != 0.0:
        log(
            "[StepD debug] baseline contamination detected: "
            "max_abs_diff != 0 (hook path mismatch)"
        )
    else:
        log("[StepD debug] baseline clean: max_abs_diff == 0")


def _make_demo_only_shuffle(demos, perm):
    outputs = [y for _x, y in demos]
    shuffled = [outputs[i] for i in perm]
    fixed_points = sum(1 for i, j in enumerate(perm) if i == j)
    shuffled_demos = [(demos[i][0], shuffled[i]) for i in range(len(demos))]
    return shuffled_demos, outputs, shuffled, fixed_points


def _boundary_prefix_and_answer_from_full(prefix_str: str, full_str: str):
    answer_str = full_str[len(prefix_str) :]
    if prefix_str.endswith(" ") and not answer_str.startswith(" "):
        return prefix_str[:-1], f" {answer_str}"
    return prefix_str, answer_str


def compute_trial_metrics(logits_base, logits_patch, target_id, compute_prob_scores: bool = True):
    import torch.nn.functional as F

    if compute_prob_scores:
        p_base = F.softmax(logits_base, dim=-1)[target_id].item()
        p_patch = F.softmax(logits_patch, dim=-1)[target_id].item()
        delta_p = p_patch - p_base
    else:
        p_base = None
        p_patch = None
        delta_p = None

    logit_base = logits_base[target_id].item()
    logit_patch = logits_patch[target_id].item()
    delta_logit = logit_patch - logit_base

    logprob_base = F.log_softmax(logits_base, dim=-1)[target_id].item()
    logprob_patch = F.log_softmax(logits_patch, dim=-1)[target_id].item()
    delta_logprob = logprob_patch - logprob_base

    return {
        "p_base": p_base,
        "p_patch": p_patch,
        "delta_p": delta_p,
        "logit_base": logit_base,
        "logit_patch": logit_patch,
        "delta_logit": delta_logit,
        "logprob_base": logprob_base,
        "logprob_patch": logprob_patch,
        "delta_logprob": delta_logprob,
    }



def _empty_metric_lists():
    return {
        "p_base": [],
        "p_patch": [],
        "delta_p": [],
        "logit_base": [],
        "logit_patch": [],
        "delta_logit": [],
        "logprob_base": [],
        "logprob_patch": [],
        "delta_logprob": [],
    }


def _prepare_trials_for_sweep(
    trials,
    tokenizer,
    device,
    tok_add_special: bool,
    cache_tokenized_inputs: bool,
):
    prepared = []
    tokenize_prefetch_calls = 0
    for pos, trial in enumerate(trials):
        prepared_trial = {
            "pos": pos,
            "trial": trial,
            "prefix_str": trial["corrupted_prefix_str"],
            "target_id": int(trial["target_id"]),
            "seq_token_idx": int(trial["seq_token_idx"]),
            "inputs": None,
        }
        if cache_tokenized_inputs:
            inputs = tokenizer(
                prepared_trial["prefix_str"],
                return_tensors="pt",
                add_special_tokens=tok_add_special,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            prepared_trial["inputs"] = inputs
            tokenize_prefetch_calls += 1
        prepared.append(prepared_trial)
    return prepared, {"tokenize_prefetch_calls": tokenize_prefetch_calls}


def _run_aie_sweep_for_trials(
    *,
    args,
    model,
    tokenizer,
    device,
    tok_add_special: bool,
    layer_modules,
    layers,
    heads,
    trials,
    mean_acts,
    slot_q: int,
    model_cfg,
    resolved_quant,
    baseline_recompute_outproj: bool,
    log,
    artifacts_dir: str,
    trial_metrics_file,
    progress_desc: str,
    n_trials_for_row: int,
    q_id=None,
    save_trials_rows=None,
    dump_handle=None,
    dump_expected_keys=None,
    log_patch_debug: bool = False,
):
    import torch

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    cache_inputs = bool(args.cache_tokenized_inputs)
    persistent_hook = bool(args.persistent_layer_hook)
    baseline_scope = args.baseline_cache_scope

    prepared_trials, prep_stats = _prepare_trials_for_sweep(
        trials=trials,
        tokenizer=tokenizer,
        device=device,
        tok_add_special=tok_add_special,
        cache_tokenized_inputs=cache_inputs,
    )

    pairs = [(layer, head) for layer in layers for head in heads]
    total = len(pairs)
    log_every = max(1, total // 100)
    patch_token_logged = False
    sweep_start = time.time()
    resume_state = _prepare_resume_state(
        artifacts_dir=artifacts_dir,
        resume_fingerprint=args._resume_fingerprint,
        q_id=q_id,
        n_trials_for_row=n_trials_for_row,
        mean_acts=mean_acts,
        slot_q=slot_q,
        compute_prob_scores=bool(args.compute_prob_scores),
        log=log,
    )
    score_rows = list(resume_state["score_rows"])
    completed_head_keys = set(resume_state["completed_head_keys"])
    pair_idx = len(completed_head_keys)

    perf_stats = {
        "progress_desc": progress_desc,
        "baseline_cache_scope": baseline_scope,
        "cache_tokenized_inputs": cache_inputs,
        "persistent_layer_hook": persistent_hook,
        "n_layers": len(layers),
        "n_heads": len(heads),
        "n_trials": len(prepared_trials),
        "pairs_total": total,
        "tokenize_prefetch_calls": prep_stats["tokenize_prefetch_calls"],
        "tokenize_runtime_calls": 0,
        "baseline_forward_calls": 0,
        "baseline_cache_hits": 0,
        "patched_forward_calls": 0,
    }

    pbar = (
        tqdm(total=total, desc=progress_desc, unit="head", initial=pair_idx)
        if tqdm is not None
        else None
    )
    global_baseline_cache = {} if baseline_scope == "trial" else None

    for layer in layers:
        remaining_heads = [
            int(head)
            for head in heads
            if _head_key(layer=int(layer), head=int(head), q_id=q_id) not in completed_head_keys
        ]
        if not remaining_heads:
            log(f"[StepD] layer={layer} fully satisfied by resume; skipping")
            continue
        layer_module = layer_modules[layer]
        baseline_head_idx = int(remaining_heads[0])

        layer_hook_state = None
        layer_hook_handle = None
        baseline_hook_state = None
        baseline_hook = None

        if persistent_hook:
            layer_hook_state = {
                "mode": "self",
                "replace_vec": None,
                "seq_token_idx": None,
                "head_idx": baseline_head_idx,
            }
            layer_hook = make_out_proj_head_output_overrider(
                layer_idx=layer,
                head_idx=baseline_head_idx,
                seq_token_idx=0,
                mode="self",
                replace_vec=None,
                model_config=model_cfg,
                resolved_quant=resolved_quant,
                force_recompute_outproj=baseline_recompute_outproj,
                state=layer_hook_state,
                logger=log,
            )
            layer_hook_handle = layer_module.register_forward_hook(layer_hook)
        else:
            baseline_hook_state = {
                "mode": "self",
                "replace_vec": None,
                "seq_token_idx": None,
                "head_idx": baseline_head_idx,
            }
            baseline_hook = make_out_proj_head_output_overrider(
                layer_idx=layer,
                head_idx=baseline_head_idx,
                seq_token_idx=0,
                mode="self",
                replace_vec=None,
                model_config=model_cfg,
                resolved_quant=resolved_quant,
                force_recompute_outproj=baseline_recompute_outproj,
                state=baseline_hook_state,
                logger=log,
            )

        baseline_cache = (
            global_baseline_cache if baseline_scope == "trial" else {}
        )

        for prepared in prepared_trials:
            cache_key = int(prepared["pos"])
            if cache_key in baseline_cache:
                perf_stats["baseline_cache_hits"] += 1
                continue

            inputs = prepared["inputs"]
            if inputs is None:
                inputs = tokenizer(
                    prepared["prefix_str"],
                    return_tensors="pt",
                    add_special_tokens=tok_add_special,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                perf_stats["tokenize_runtime_calls"] += 1

            seq_token_idx = prepared["seq_token_idx"]
            if persistent_hook:
                layer_hook_state["mode"] = "self"
                layer_hook_state["replace_vec"] = None
                layer_hook_state["seq_token_idx"] = seq_token_idx
                layer_hook_state["head_idx"] = baseline_head_idx
                with torch.inference_mode():
                    outputs = model(**inputs)
            else:
                baseline_hook_state["mode"] = "self"
                baseline_hook_state["replace_vec"] = None
                baseline_hook_state["seq_token_idx"] = seq_token_idx
                baseline_hook_state["head_idx"] = baseline_head_idx
                handle = layer_module.register_forward_hook(baseline_hook)
                with torch.inference_mode():
                    outputs = model(**inputs)
                handle.remove()

            baseline_cache[cache_key] = outputs.logits[0, -1].detach()
            perf_stats["baseline_forward_calls"] += 1

        log(
            "baseline cache ready: "
            f"layer={layer} scope={baseline_scope} size={len(baseline_cache)}"
        )

        for head in remaining_heads:
            pair_idx += 1
            if pbar is not None:
                if q_id is None:
                    pbar.set_postfix(
                        {"layer": layer, "head": head, "trials": len(prepared_trials)}
                    )
                else:
                    pbar.set_postfix({"layer": layer, "head": head})
            if pair_idx == 1 or pair_idx % log_every == 0 or pair_idx == total:
                elapsed = time.time() - sweep_start
                rate = pair_idx / elapsed if elapsed > 0 else 0.0
                remaining = total - pair_idx
                eta = remaining / rate if rate > 0 else 0.0
                percent = (pair_idx / total) * 100 if total else 100.0
                log(
                    "[StepD] "
                    f"{percent:.1f}% ({pair_idx}/{total}) "
                    f"layer={layer} head={head} "
                    f"elapsed={format_duration(elapsed)} "
                    f"ETA={format_duration(eta)}"
                )

            metric_lists = _empty_metric_lists()
            nonzero = False
            replace_vec = _mean_vec_for_head(mean_acts, layer=layer, head=int(head), slot_q=slot_q)

            patch_hook_state = None
            patch_hook = None
            if not persistent_hook:
                patch_hook_state = {
                    "mode": "replace",
                    "replace_vec": None,
                    "seq_token_idx": None,
                    "head_idx": int(head),
                }
                patch_hook = make_out_proj_head_output_overrider(
                    layer_idx=layer,
                    head_idx=int(head),
                    seq_token_idx=0,
                    mode="replace",
                    replace_vec=None,
                    model_config=model_cfg,
                    resolved_quant=resolved_quant,
                    force_recompute_outproj=baseline_recompute_outproj,
                    state=patch_hook_state,
                    logger=log,
                )

            for prepared in prepared_trials:
                trial = prepared["trial"]
                prefix_str = prepared["prefix_str"]
                target_id = prepared["target_id"]
                seq_token_idx = prepared["seq_token_idx"]
                cache_key = int(prepared["pos"])

                inputs = prepared["inputs"]
                if inputs is None:
                    inputs = tokenizer(
                        prefix_str,
                        return_tensors="pt",
                        add_special_tokens=tok_add_special,
                    )
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    perf_stats["tokenize_runtime_calls"] += 1

                baseline_logits = baseline_cache[cache_key]

                if log_patch_debug and pair_idx == 1 and trial["trial_idx"] == 0:
                    target_token = tokenizer.convert_ids_to_tokens(target_id)
                    prompt_tail = repr(prefix_str[-20:])
                    log(
                        "target_debug: "
                        f"prompt_tail={prompt_tail} "
                        f"target_token={target_token} target_id={target_id}"
                    )

                if log_patch_debug and not patch_token_logged:
                    prefix_ids = inputs["input_ids"][0].detach().cpu().tolist()
                    full_ids = trial.get("full_input_ids", [])
                    decoded_full = (
                        tokenizer.decode([full_ids[seq_token_idx]])
                        if full_ids
                        else None
                    )
                    decoded_prefix = tokenizer.decode([prefix_ids[seq_token_idx]])
                    log(
                        "PATCH slot/seq idx decoded: "
                        f"slot_idx={slot_q} seq_token_idx={seq_token_idx} "
                        f"full_token={repr(decoded_full)} "
                        f"prefix_token={repr(decoded_prefix)}"
                    )
                    patch_token_logged = True

                if persistent_hook:
                    layer_hook_state["mode"] = "replace"
                    layer_hook_state["replace_vec"] = replace_vec
                    layer_hook_state["seq_token_idx"] = seq_token_idx
                    layer_hook_state["head_idx"] = int(head)
                    with torch.inference_mode():
                        outputs_patched = model(**inputs)
                else:
                    patch_hook_state["mode"] = "replace"
                    patch_hook_state["replace_vec"] = replace_vec
                    patch_hook_state["seq_token_idx"] = seq_token_idx
                    patch_hook_state["head_idx"] = int(head)
                    handle = layer_module.register_forward_hook(patch_hook)
                    with torch.inference_mode():
                        outputs_patched = model(**inputs)
                    handle.remove()

                perf_stats["patched_forward_calls"] += 1
                patched_logits = outputs_patched.logits[0, -1]
                trial_metrics = compute_trial_metrics(
                    baseline_logits,
                    patched_logits,
                    target_id,
                    compute_prob_scores=bool(args.compute_prob_scores),
                )

                if args.compute_prob_scores:
                    metric_lists["p_base"].append(trial_metrics["p_base"])
                    metric_lists["p_patch"].append(trial_metrics["p_patch"])
                    metric_lists["delta_p"].append(trial_metrics["delta_p"])
                    if abs(trial_metrics["delta_p"]) > 1e-12:
                        nonzero = True
                metric_lists["logit_base"].append(trial_metrics["logit_base"])
                metric_lists["logit_patch"].append(trial_metrics["logit_patch"])
                metric_lists["delta_logit"].append(trial_metrics["delta_logit"])
                metric_lists["logprob_base"].append(trial_metrics["logprob_base"])
                metric_lists["logprob_patch"].append(trial_metrics["logprob_patch"])
                metric_lists["delta_logprob"].append(trial_metrics["delta_logprob"])

                trial_row = {
                    "trial_idx": trial["trial_idx"],
                    "layer": layer,
                    "head": head,
                    "target_id": target_id,
                    "target_token": tokenizer.convert_ids_to_tokens(target_id),
                    "seed_global": args.seed,
                    "shuffle_labels": bool(args.shuffle_labels),
                    "shuffle_derangement": False,
                    "n_icl_examples": args.n_icl_examples,
                    "demo_perm": trial.get("demo_perm"),
                    "demo_fixed_points": trial.get("demo_fixed_points"),
                    "demo_outputs_before": trial.get("demo_outputs_before"),
                    "demo_outputs_after": trial.get("demo_outputs_after"),
                    "p_base": trial_metrics["p_base"],
                    "p_patch": trial_metrics["p_patch"],
                    "delta_p": trial_metrics["delta_p"],
                    "p_base_target": trial_metrics["p_base"],
                    "p_patch_target": trial_metrics["p_patch"],
                    "delta_p_target": trial_metrics["delta_p"],
                    "logit_base": trial_metrics["logit_base"],
                    "logit_patch": trial_metrics["logit_patch"],
                    "delta_logit": trial_metrics["delta_logit"],
                    "logprob_base": trial_metrics["logprob_base"],
                    "logprob_patch": trial_metrics["logprob_patch"],
                    "delta_logprob": trial_metrics["delta_logprob"],
                    "prompt_tail_repr": repr(prefix_str[-30:]),
                    "prompt_ends_with_space": prefix_str.endswith(" "),
                    "slot_idx": slot_q,
                    "seq_token_idx": seq_token_idx,
                }
                if q_id is not None:
                    trial_row["q_id"] = q_id
                trial_metrics_file.write(
                    json.dumps(trial_row, ensure_ascii=True, default=_json_safe) + "\n"
                )

                if dump_handle is not None:
                    if (args.dump_layer is None or args.dump_layer == layer) and (
                        args.dump_head is None or args.dump_head == head
                    ):
                        if args.dump_max_trials < 0 or trial["trial_idx"] < args.dump_max_trials:
                            ids_list = inputs["input_ids"][0].detach().cpu().tolist()
                            dump_row = {
                                "trial_idx": trial["trial_idx"],
                                "layer": layer,
                                "head": head,
                                "target_id": target_id,
                                "target_token": tokenizer.convert_ids_to_tokens(target_id),
                                "seed_global": args.seed,
                                "n_icl_examples": args.n_icl_examples,
                                "p_base": trial_metrics["p_base"],
                                "p_patch": trial_metrics["p_patch"],
                                "delta_p": trial_metrics["delta_p"],
                                "p_base_target": trial_metrics["p_base"],
                                "p_patch_target": trial_metrics["p_patch"],
                                "delta_p_target": trial_metrics["delta_p"],
                                "logit_base": trial_metrics["logit_base"],
                                "logit_patch": trial_metrics["logit_patch"],
                                "delta_logit": trial_metrics["delta_logit"],
                                "logprob_base": trial_metrics["logprob_base"],
                                "logprob_patch": trial_metrics["logprob_patch"],
                                "delta_logprob": trial_metrics["delta_logprob"],
                                "prompt_tail_repr": (
                                    repr(prefix_str[-60:]) if args.dump_include_prompt else ""
                                ),
                                "prompt_ends_with_space": prefix_str.endswith(" "),
                                "seq_token_idx": seq_token_idx,
                                "ids_len": len(ids_list),
                                "ids_last10": ids_list[-10:],
                                "slot_idx": slot_q,
                            }
                            if set(dump_row.keys()) != set(dump_expected_keys):
                                missing = set(dump_expected_keys) - set(dump_row.keys())
                                extra = set(dump_row.keys()) - set(dump_expected_keys)
                                raise ValueError(
                                    f"dump keys mismatch missing={sorted(missing)} extra={sorted(extra)}"
                                )
                            dump_handle.write(
                                json.dumps(dump_row, ensure_ascii=True, default=_json_safe)
                                + "\n"
                            )

                if save_trials_rows is not None:
                    save_trials_rows.append(
                        {
                            "trial_idx": trial["trial_idx"],
                            "layer": layer,
                            "head": head,
                            "target_id": target_id,
                            "p_base": trial_metrics["p_base"],
                            "p_patch": trial_metrics["p_patch"],
                            "delta_p": trial_metrics["delta_p"],
                            "p_base_target": trial_metrics["p_base"],
                            "p_patch_target": trial_metrics["p_patch"],
                            "delta_p_target": trial_metrics["delta_p"],
                            "logit_base": trial_metrics["logit_base"],
                            "logit_patch": trial_metrics["logit_patch"],
                            "delta_logit": trial_metrics["delta_logit"],
                            "logprob_base": trial_metrics["logprob_base"],
                            "logprob_patch": trial_metrics["logprob_patch"],
                            "delta_logprob": trial_metrics["delta_logprob"],
                        }
                    )

            mean_act_norm = _mean_vec_for_head(mean_acts, layer=layer, head=int(head), slot_q=slot_q).norm().item()
            mean_delta_p = mean(metric_lists["delta_p"])
            std_delta_p = std(metric_lists["delta_p"])
            mean_abs_delta_p = mean_abs(metric_lists["delta_p"])
            mean_p_base = mean(metric_lists["p_base"])
            std_p_base = std(metric_lists["p_base"])
            mean_p_patch = mean(metric_lists["p_patch"])
            std_p_patch = std(metric_lists["p_patch"])
            if args.compute_prob_scores and not (-1.000001 <= mean_delta_p <= 1.000001):
                log(
                    f"[WARN] mean_delta_p out of range: {mean_delta_p:.6f} "
                    f"(layer={layer} head={head})"
                )
                assert -1.000001 <= mean_delta_p <= 1.000001

            mean_delta_logit = mean(metric_lists["delta_logit"])
            std_delta_logit = std(metric_lists["delta_logit"])
            mean_abs_delta_logit = mean_abs(metric_lists["delta_logit"])
            mean_logit_base = mean(metric_lists["logit_base"])
            mean_logit_patch = mean(metric_lists["logit_patch"])

            mean_delta_logprob = mean(metric_lists["delta_logprob"])
            std_delta_logprob = std(metric_lists["delta_logprob"])
            mean_abs_delta_logprob = mean_abs(metric_lists["delta_logprob"])
            mean_logprob_base = mean(metric_lists["logprob_base"])
            mean_logprob_patch = mean(metric_lists["logprob_patch"])

            row = {
                "layer": layer,
                "head": head,
                "n_trials": n_trials_for_row,
                "mean_delta_p": mean_delta_p,
                "std_delta_p": std_delta_p,
                "mean_abs_delta_p": mean_abs_delta_p,
                "mean_p_base": mean_p_base,
                "std_p_base": std_p_base,
                "mean_p_patch": mean_p_patch,
                "std_p_patch": std_p_patch,
                "mean_p_base_target": mean_p_base,
                "mean_p_patch_target": mean_p_patch,
                "mean_delta_p_target": mean_delta_p,
                "mean_delta_logit": mean_delta_logit,
                "std_delta_logit": std_delta_logit,
                "mean_abs_delta_logit": mean_abs_delta_logit,
                "mean_logit_base": mean_logit_base,
                "mean_logit_patch": mean_logit_patch,
                "mean_delta_logprob": mean_delta_logprob,
                "std_delta_logprob": std_delta_logprob,
                "mean_abs_delta_logprob": mean_abs_delta_logprob,
                "mean_logprob_base": mean_logprob_base,
                "mean_logprob_patch": mean_logprob_patch,
                "any_nonzero": nonzero,
                "mean_act_norm": mean_act_norm,
                "clean_mean_norm": mean_act_norm,
            }
            if q_id is not None:
                row["q_id"] = q_id
            score_rows.append(row)
            completed_head_keys.add(_head_key(layer=int(layer), head=int(head), q_id=q_id))
            _append_jsonl_rows(resume_state["paths"]["scores"], [row])
            if pbar is not None:
                pbar.update(1)

        if layer_hook_handle is not None:
            layer_hook_handle.remove()

    if pbar is not None:
        pbar.close()

    perf_stats["elapsed_sec"] = time.time() - sweep_start
    perf_stats["sec_per_head"] = (
        perf_stats["elapsed_sec"] / total if total > 0 else 0.0
    )
    return score_rows, perf_stats


def format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def compute_slot_with_fallback(
    prefix_str: str, full_str: str, tokenizer, log, add_special_tokens: bool = False
):
    try:
        return compute_query_predictive_slot(
            prefix_str,
            full_str,
            tokenizer,
            add_special_tokens=add_special_tokens,
        )
    except ValueError as exc:
        message = str(exc)
        if "Target id mismatch" not in message:
            raise
        if prefix_str.endswith(" "):
            raise
        trimmed_prefix = prefix_str.rstrip(" ")
        if trimmed_prefix == prefix_str:
            raise
        log("retrying slot computation with trimmed prefix space")
        return compute_query_predictive_slot(
            trimmed_prefix,
            full_str,
            tokenizer,
            add_special_tokens=add_special_tokens,
        )


def compute_trial_idx_map(
    demos,
    query,
    prefix_str: str,
    full_str: str,
    tokenizer,
    tok_add_special: bool,
    n_icl_examples: int,
    dummy_labels_ref=None,
    slot_index_map_ref=None,
    special_index_labels_ref=None,
    prompt_data=None,
):
    if prompt_data is None:
        _prefix_str, _full_str, prompt_data = build_prompt_qa_paper(
            demos,
            query,
            prefixes=None,
            separators=None,
            prepend_bos_token=not tok_add_special,
            prepend_space=True,
        )

    token_labels, prompt_string = paper_get_token_meta_labels(
        prompt_data, tokenizer, prepend_bos=tok_add_special
    )
    if prompt_string != prefix_str:
        raise ValueError("Prompt builder mismatch with paper prompt_string")
    prefix_ids = tokenizer(prefix_str, add_special_tokens=tok_add_special)["input_ids"]
    if len(prefix_ids) != len(token_labels):
        raise AssertionError(
            f"prefix length mismatch: prefix_ids={len(prefix_ids)} token_labels={len(token_labels)}"
        )
    seq_token_idx = len(prefix_ids) - 1
    if not (0 <= seq_token_idx < len(token_labels)):
        raise AssertionError(
            f"seq_token_idx out of range: seq_token_idx={seq_token_idx} len={len(token_labels)}"
        )
    slot_label = token_labels[seq_token_idx][2]
    slot_token = token_labels[seq_token_idx][1]
    if slot_label != "query_predictive_token":
        raise AssertionError(
            f"slot label mismatch at seq_token_idx={seq_token_idx}: {slot_label}"
        )
    if slot_token != ":":
        raise AssertionError(
            f"slot token mismatch at seq_token_idx={seq_token_idx}: {repr(slot_token)}"
        )

    dummy_labels_raw = paper_get_dummy_token_labels(
        n_icl_examples,
        tokenizer=tokenizer,
        model_config={"prepend_bos": tok_add_special},
        prefixes=prompt_data.get("prefixes"),
        separators=prompt_data.get("separators"),
    )
    dummy_labels, slot_index_map = paper_labels_to_slot_map(dummy_labels_raw)
    if dummy_labels_ref is not None and dummy_labels != dummy_labels_ref:
        raise ValueError("dummy_labels mismatch across trials")
    if slot_index_map_ref is not None and slot_index_map != slot_index_map_ref:
        raise ValueError("slot_index_map mismatch across trials")

    idx_map_paper, idx_avg = paper_compute_duplicated_labels(token_labels, dummy_labels_raw)
    if len(idx_map_paper) != len(dummy_labels_raw):
        raise AssertionError(
            f"dummy_labels length mismatch: dummy={len(dummy_labels_raw)} real={len(idx_map_paper)}"
        )

    idx_map = {}
    for token_idx, slot_idx in idx_map_paper.items():
        idx_map.setdefault(slot_idx, []).append(token_idx)
    for (i, j) in idx_avg.values():
        slot_idx = idx_map_paper.get(i)
        if slot_idx is None:
            continue
        idx_map[slot_idx] = list(range(i, j + 1))
    for slot_idx in range(len(dummy_labels)):
        idx_map.setdefault(slot_idx, [])
    empty_slots = [i for i in range(len(dummy_labels)) if not idx_map.get(i)]
    if empty_slots:
        raise AssertionError(
            f"empty idx_map slots detected: count={len(empty_slots)} first={empty_slots[0]}"
        )

    special_index_labels = {}
    if (
        special_index_labels_ref is not None
        and special_index_labels != special_index_labels_ref
    ):
        raise ValueError("Special token alignment mismatch across trials")

    slot_q = slot_index_map.get("QUERY_PRED")

    full_ids = tokenizer(
        full_str, return_tensors=None, add_special_tokens=tok_add_special
    )["input_ids"]
    return idx_map, dummy_labels, slot_index_map, special_index_labels, full_ids


def check_successful_icl(
    model, tokenizer, device, prefix_str: str, target_id: int, tok_add_special: bool
):
    import torch

    inputs = tokenizer(
        prefix_str, return_tensors="pt", add_special_tokens=tok_add_special
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    last_index = inputs["input_ids"].shape[1] - 1
    with torch.inference_mode():
        outputs = model(**inputs)
    logits = outputs.logits[0, last_index]
    pred_id = int(torch.argmax(logits).item())
    p_target = torch.softmax(logits.float(), dim=-1)[target_id].item()
    success = pred_id == target_id
    return success, p_target


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "STEP D AIE head sweep. mean_delta_p = p_patch - p_base on the target "
            "first token (comparable to paper indirect_effect)."
        )
    )
    parser.add_argument("--model", default="gpt2", help="Model name or path (default: gpt2)")
    parser.add_argument(
        "--dataset_path",
        default="datasets/processed/antonym.json",
        help="Path to antonym dataset JSON",
    )
    parser.add_argument(
        "--model_spec",
        default="gpt2",
        help="Model spec name for adapter resolution (default: gpt2)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["fp16", "bf16", "fp32"],
        help="Model dtype (default: fp16 on cuda else fp32)",
    )
    parser.add_argument(
        "--quant",
        default="auto",
        choices=["auto", "none", "4bit", "8bit"],
        help="Quantization mode (default: auto)",
    )
    parser.add_argument(
        "--device_map",
        default=None,
        help="Optional HF device_map (default: None or 'auto')",
    )
    parser.add_argument(
        "--layers",
        default="0",
        help="Layer list (examples: --layers all, --layers 0,1,2,3)",
    )
    parser.add_argument("--heads", default="all", help="Head list or 'all' (default: all)")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials (default: 20)")
    parser.add_argument(
        "--n_icl_examples",
        type=int,
        default=3,
        help="Number of ICL demos per prompt (default: 3)",
    )
    parser.add_argument(
        "--n_mean_trials",
        type=int,
        default=None,
        help="Trials for mean_activations (default: n_trials)",
    )
    parser.add_argument(
        "--successful_icl_only",
        type=int,
        default=0,
        help="Keep only successful ICL trials (default: 1)",
    )
    parser.add_argument(
        "--max_trial_attempts",
        type=int,
        default=None,
        help="Max attempts when filtering successful ICL (default: n_trials*50)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--shuffle_labels",
        type=int,
        default=0,
        help="Shuffle demo labels only (default: 0)",
    )
    parser.add_argument(
        "--fixed_trials_path",
        default=None,
        help="Optional fixed_trials.json path for deterministic trials",
    )
    parser.add_argument(
        "--compute_prob_scores",
        type=int,
        default=1,
        help="If 1, compute probability-based scores (delta_p).",
    )
    parser.add_argument(
        "--fixed_out_dir",
        default=None,
        help="When using fixed_trials, write outputs under this directory",
    )
    parser.add_argument(
        "--relation_csv_path",
        default=None,
        help="Relation CSV path (optional; uses relation trial generator)",
    )
    parser.add_argument(
        "--relation_q_list",
        default=None,
        help="Comma-separated q list for relation trials (default: all)",
    )
    parser.add_argument(
        "--relation_n_trials_per_q",
        type=int,
        default=None,
        help="Trials per q for relation CSV (required if relation_csv_path set)",
    )
    parser.add_argument(
        "--relation_n_demos",
        type=int,
        default=10,
        help="Demos per trial for relation CSV (default: 10)",
    )
    parser.add_argument(
        "--relation_save_trials_json",
        type=int,
        default=0,
        help="Save relation trials to JSON (default: 0)",
    )
    parser.add_argument(
        "--relation_out_path",
        default=None,
        help="Output path for relation trials JSON (required if save enabled)",
    )
    parser.add_argument(
        "--mean_only",
        type=int,
        default=0,
        help="If 1, compute mean_activations and exit",
    )
    parser.add_argument(
        "--debug_prompt_check",
        type=int,
        default=0,
        help="If 1, print fixed_trials prompt/token checks and exit",
    )
    parser.add_argument(
        "--debug_n",
        type=int,
        default=3,
        help="Number of fixed_trials to print in debug check",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Run identifier (default: auto timestamp)",
    )
    parser.add_argument(
        "--out_base_dir",
        default=None,
        help="Canonical base dir; writes artifacts/ and logs/ under this directory",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: runs/<run_id>/artifacts/)",
    )
    parser.add_argument(
        "--save_trials",
        type=int,
        default=0,
        help="Save trial-level CSV (default: 0)",
    )
    parser.add_argument(
        "--dump_trial_metrics_jsonl",
        default=None,
        help="Optional JSONL path for trial metrics dump",
    )
    parser.add_argument(
        "--dump_layer",
        type=int,
        default=None,
        help="Filter dump to a layer (dump only)",
    )
    parser.add_argument(
        "--dump_head",
        type=int,
        default=None,
        help="Filter dump to a head (dump only)",
    )
    parser.add_argument(
        "--dump_max_trials",
        type=int,
        default=-1,
        help="Max trials to dump (default: all)",
    )
    parser.add_argument(
        "--dump_include_prompt",
        type=int,
        default=1,
        help="Include prompt tail repr in dump (1/0)",
    )
    parser.add_argument(
        "--score_key",
        default="mean_delta_p",
        help="Score key for ranking (default: mean_delta_p)",
    )
    parser.add_argument(
        "--baseline_cache_scope",
        choices=["layer_trial", "trial"],
        default="layer_trial",
        help="Baseline cache scope for StepD sweep (default: layer_trial)",
    )
    parser.add_argument(
        "--cache_tokenized_inputs",
        type=int,
        choices=[0, 1],
        default=1,
        help="Cache tokenized trial inputs once per sweep (default: 1)",
    )
    parser.add_argument(
        "--persistent_layer_hook",
        type=int,
        choices=[0, 1],
        default=1,
        help="Keep one hook per layer and update state per head/trial (default: 1)",
    )
    parser.add_argument(
        "--perf_log",
        type=int,
        choices=[0, 1],
        default=1,
        help="Save StepD performance counters to artifacts/perf_stats.json (default: 1)",
    )
    parser.add_argument(
        "--debug_stepd",
        action="store_true",
        help="Run StepD parity debug on local gpt2 (cpu) and exit",
    )
    args = parser.parse_args()

    if args.debug_stepd:
        args.model = "gpt2"
        args.model_spec = "gpt2"
        args.device = "cpu"
        args.dtype = "fp32"
        args.quant = "none"
        args.device_map = None
        args.layers = "0"
        args.heads = "0"
        if args.relation_csv_path:
            print("debug_stepd cannot be used with relation_csv_path")
            return 1

    if args.n_trials < 1 and not args.debug_stepd:
        print("n_trials must be >= 1")
        return 1
    if args.n_icl_examples < 1:
        print("n_icl_examples must be >= 1")
        return 1
    if args.n_mean_trials is None:
        args.n_mean_trials = args.n_trials
    if args.n_mean_trials < 1:
        print("n_mean_trials must be >= 1")
        return 1
    if args.successful_icl_only not in (0, 1):
        print("successful_icl_only must be 0 or 1")
        return 1
    if args.max_trial_attempts is None:
        args.max_trial_attempts = args.n_trials * 50
    if args.fixed_trials_path and args.relation_csv_path:
        print("Use either --fixed_trials_path or --relation_csv_path, not both.")
        return 1
    if args.relation_csv_path and not args.relation_n_trials_per_q:
        print("--relation_n_trials_per_q is required with --relation_csv_path.")
        return 1

    run_info = prepare_run_dirs(args.run_id, base_dir="results/attention_head")
    if args.out_base_dir:
        base_dir = resolve_out_dir(args.out_base_dir)
        artifacts_dir = os.path.join(base_dir, "artifacts")
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(artifacts_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        run_info = {
            "run_id": os.path.basename(base_dir),
            "artifacts_dir": artifacts_dir,
            "logs_dir": logs_dir,
        }
    elif args.fixed_trials_path and args.fixed_out_dir:
        base_dir = resolve_out_dir(args.fixed_out_dir)
        artifacts_dir = os.path.join(base_dir, "artifacts")
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(artifacts_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        run_info = {
            "run_id": os.path.basename(base_dir),
            "artifacts_dir": artifacts_dir,
            "logs_dir": logs_dir,
        }
    else:
        if args.out_dir:
            artifacts_dir = resolve_out_dir(args.out_dir)
        else:
            artifacts_dir = run_info["artifacts_dir"]
        os.makedirs(artifacts_dir, exist_ok=True)

    log_path = os.path.join(run_info["logs_dir"], "stepD_aie.log")
    log, log_file = make_logger(log_path)

    log("stepD AIE head sweep start")
    log(f"run_id: {run_info['run_id']}")
    log(f"artifacts_dir: {artifacts_dir}")
    log(f"log_path: {log_path}")
    log(f"model: {args.model}")
    log(f"dataset_path: {args.dataset_path}")
    log(f"model_spec: {args.model_spec}")
    log(f"layers: {args.layers}")
    log(f"heads: {args.heads}")
    log(f"n_trials: {args.n_trials}")
    log(f"n_icl_examples: {args.n_icl_examples}")
    log(f"n_mean_trials: {args.n_mean_trials}")
    log(f"successful_icl_only: {args.successful_icl_only}")
    log(f"max_trial_attempts: {args.max_trial_attempts}")
    log(f"seed: {args.seed}")
    log(f"score_key: {args.score_key}")
    log(f"compute_prob_scores: {args.compute_prob_scores}")
    log(f"shuffle_labels: {args.shuffle_labels}")
    log(f"fixed_trials_path: {args.fixed_trials_path}")
    log(f"relation_csv_path: {args.relation_csv_path}")
    log(f"relation_q_list: {args.relation_q_list}")
    log(f"relation_n_trials_per_q: {args.relation_n_trials_per_q}")
    log(f"relation_n_demos: {args.relation_n_demos}")
    log(f"relation_save_trials_json: {args.relation_save_trials_json}")
    log(f"relation_out_path: {args.relation_out_path}")
    log(f"out_base_dir: {args.out_base_dir}")
    log(f"mean_only: {args.mean_only}")
    log(f"debug_prompt_check: {args.debug_prompt_check}")
    log(f"debug_n: {args.debug_n}")
    log(f"dump_trial_metrics_jsonl: {args.dump_trial_metrics_jsonl}")
    log(f"baseline_cache_scope: {args.baseline_cache_scope}")
    log(f"cache_tokenized_inputs: {args.cache_tokenized_inputs}")
    log(f"persistent_layer_hook: {args.persistent_layer_hook}")
    log(f"perf_log: {args.perf_log}")

    try:
        import torch
        import transformers
    except Exception as exc:  # pragma: no cover - runtime import check
        log(f"Failed to import required libraries: {exc}")
        log_file.close()
        return 1

    if args.device in (None, "auto"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype is None:
        args.dtype = "fp16" if args.device == "cuda" else "fp32"

    try:
        spec = get_model_spec(args.model_spec)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1
    tok_add_special = bool(spec.prepend_bos)
    baseline_recompute_outproj = True
    log(f"tok_add_special: {tok_add_special}")
    if args.baseline_cache_scope == "trial":
        log(
            "[WARN] baseline_cache_scope=trial can change numeric outputs; "
            "use layer_trial for stable parity."
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.shuffle_labels:
        if args.successful_icl_only:
            log(
                "shuffle_labels=True -> forcing successful_icl_only=0 "
                "(shuffled-label control)"
            )
        args.successful_icl_only = 0

    fixed_trials = None
    fixed_meta = None
    if args.fixed_trials_path:
        try:
            fixed_trials = load_fixed_trials(args.fixed_trials_path)
        except Exception as exc:
            log(f"Failed to load fixed_trials: {exc}")
            log_file.close()
            return 1
        fixed_meta = fixed_trials.get("meta", {})
        fixed_n_shots = fixed_meta.get("n_shots")
        if fixed_n_shots is None:
            log("fixed_trials meta missing n_shots")
            log_file.close()
            return 1
        if fixed_n_shots != args.n_icl_examples:
            log(
                "fixed_trials n_shots mismatch; overriding "
                f"args.n_icl_examples {args.n_icl_examples} -> {fixed_n_shots}"
            )
            args.n_icl_examples = int(fixed_n_shots)
        fixed_n_trials = fixed_meta.get("n_trials")
        if fixed_n_trials is not None and fixed_n_trials != args.n_trials:
            log(
                "fixed_trials n_trials mismatch; overriding "
                f"args.n_trials {args.n_trials} -> {fixed_n_trials}"
            )
            args.n_trials = int(fixed_n_trials)
        if args.n_mean_trials is None:
            args.n_mean_trials = args.n_trials
        if args.n_mean_trials > args.n_trials:
            log(
                "n_mean_trials exceeds n_trials; overriding "
                f"{args.n_mean_trials} -> {args.n_trials}"
            )
            args.n_mean_trials = args.n_trials
        if args.successful_icl_only:
            log("fixed_trials provided; forcing successful_icl_only=0")
        args.successful_icl_only = 0
        if args.shuffle_labels:
            log("fixed_trials provided; forcing shuffle_labels=0")
        args.shuffle_labels = 0
        fixed_trials_list = fixed_trials.get("trials", [])
        if not fixed_trials_list:
            log("fixed_trials missing trials list")
            log_file.close()
            return 1
        if args.n_trials > len(fixed_trials_list):
            log(
                "fixed_trials shorter than n_trials; "
                "regenerate fixed_trials or lower --n_trials"
            )
            log_file.close()
            return 1
    else:
        log("no fixed_trials: unchanged")

    pairs = None
    if fixed_trials is None and not args.relation_csv_path:
        try:
            pairs = load_pairs_antonym(args.dataset_path, canonical_by_input=True)
        except Exception as exc:
            log(f"Failed to load dataset: {exc}")
            log_file.close()
            return 1

    try:
        loader_device = None if args.device_map else args.device
        model, tokenizer, diagnostics = load_hf_model_and_tokenizer(
            model_name=args.model,
            model_spec=args.model_spec,
            device=loader_device,
            dtype=args.dtype,
            quant=args.quant,
            device_map=args.device_map,
        )
        log(
            "hf_loader diagnostics: "
            + " ".join(
                f"{key}={value}" for key, value in diagnostics.items()
            )
        )
    except Exception as exc:  # pragma: no cover - runtime load check
        log(f"Failed to load model '{args.model}': {exc}")
        log_file.close()
        return 1

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    device = torch.device(args.device)
    resolved_quant = diagnostics.get("resolved_quant") if diagnostics else None
    if args.device_map:
        log("device_map provided; skipping model.to(device)")
        try:
            device = next(model.parameters()).device
        except StopIteration:
            pass
    elif resolved_quant in {"4bit", "8bit"}:
        log("quantized model; skipping model.to(device)")
    else:
        model.to(device)
    model.eval()
    log(
        "patched run uses manual_matmul_override "
        f"(resolved_quant={resolved_quant})"
    )

    relation_stats = None
    if args.relation_csv_path:
        try:
            fixed_trials, relation_stats = generate_relation_trials(
                csv_path=args.relation_csv_path,
                q_list=args.relation_q_list,
                n_trials_per_q=int(args.relation_n_trials_per_q),
                n_demos=int(args.relation_n_demos),
                seed=int(args.seed),
                tokenizer=tokenizer,
                tok_add_special=tok_add_special,
            )
        except Exception as exc:
            log(f"Failed to generate relation trials: {exc}")
            log_file.close()
            return 1
        fixed_meta = fixed_trials.get("meta", {})
        trials_list = fixed_trials.get("trials", [])
        if not trials_list:
            log("relation trials empty")
            log_file.close()
            return 1
        if args.relation_save_trials_json:
            if not args.relation_out_path:
                log("relation_out_path required when relation_save_trials_json=1")
                log_file.close()
                return 1
            save_trials_json(fixed_trials, args.relation_out_path)
            log(f"saved relation trials: {args.relation_out_path}")
        if args.n_trials != len(trials_list):
            log(
                "relation trials override n_trials: "
                f"{args.n_trials} -> {len(trials_list)}"
            )
            args.n_trials = len(trials_list)
        rel_demos = fixed_meta.get("n_demos")
        if rel_demos is not None and rel_demos != args.n_icl_examples:
            log(
                "relation trials override n_icl_examples: "
                f"{args.n_icl_examples} -> {rel_demos}"
            )
            args.n_icl_examples = int(rel_demos)
        if args.n_mean_trials is None:
            args.n_mean_trials = args.n_trials
        if args.n_mean_trials > args.n_trials:
            log(
                "n_mean_trials exceeds n_trials; overriding "
                f"{args.n_mean_trials} -> {args.n_trials}"
            )
            args.n_mean_trials = args.n_trials
        if args.successful_icl_only:
            log("relation trials provided; forcing successful_icl_only=0")
        args.successful_icl_only = 0
        if args.shuffle_labels:
            log("relation trials provided; forcing shuffle_labels=0")
        args.shuffle_labels = 0
        if relation_stats:
            log("relation_trials: per-q counts")
            for q_id in sorted(relation_stats.q_counts.keys()):
                if q_id in relation_stats.skipped_qs:
                    log(
                        f"  q={q_id} skipped n_examples={relation_stats.q_counts[q_id]}"
                    )
                    continue
                overlaps = relation_stats.shuffle_match_counts.get(q_id, [])
                if overlaps:
                    overlap_msg = (
                        f"overlap_mean={sum(overlaps)/len(overlaps):.2f} "
                        f"min={min(overlaps)} max={max(overlaps)}"
                    )
                else:
                    overlap_msg = "overlap=n/a"
                log(
                    "  q={} n_examples={} n_demos={} n_trials={} {}".format(
                        q_id,
                        relation_stats.q_counts[q_id],
                        relation_stats.q_demo_counts[q_id],
                        relation_stats.q_trials[q_id],
                        overlap_msg,
                    )
                )

    prompt_meta = _resolve_prompt_meta(fixed_meta if fixed_trials else None, tok_add_special)
    log(
        "prompt_meta: "
        f"prefixes={prompt_meta['prefixes']} "
        f"separators={prompt_meta['separators']} "
        f"prepend_bos_token_used={prompt_meta['prepend_bos_token_used']}"
    )

    if fixed_trials is None and args.successful_icl_only:
        log("non-fixed run: forcing successful_icl_only=0")
        args.successful_icl_only = 0

    try:
        dims = infer_head_dims(model, spec_name=args.model_spec)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1

    n_heads = int(dims["n_heads"])
    head_dim = int(dims["head_dim"])
    resid_dim = int(dims["hidden_size"])

    try:
        blocks = resolve_blocks(model, spec, logger=log)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1

    layer_count = len(blocks)
    if layer_count == 0:
        log("No layers available in resolved blocks")
        log_file.close()
        return 1
    model_cfg = {
        "n_heads": n_heads,
        "head_dim": head_dim,
        "resid_dim": resid_dim,
        "n_layers": layer_count,
    }
    if args.layers.strip().lower() == "all":
        layers = list(range(layer_count))
        log(f"[StepD] layers=all resolved to 0..{layer_count - 1} (n={layer_count})")
    else:
        try:
            layers = parse_layers(args.layers)
        except ValueError as exc:
            log(str(exc))
            log_file.close()
            return 1
        if not layers:
            log("No layers selected")
            log_file.close()
            return 1
        min_layer = min(layers)
        max_layer = max(layers)
        if min_layer < 0 or max_layer >= layer_count:
            log(
                "Layer index out of range: "
                f"allowed=[0, {layer_count - 1}] got={layers}"
            )
            log_file.close()
            return 1
        log(f"[StepD] sweeping layers: {min_layer}..{max_layer} (n={len(layers)})")

    try:
        heads = parse_heads(args.heads, n_heads)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1
    if not heads:
        log("No heads selected")
        log_file.close()
        return 1
    for head in heads:
        if head < 0 or head >= n_heads:
            log("Head index out of range")
            log_file.close()
            return 1

    args._resume_fingerprint = _build_stepd_resume_fingerprint(
        args,
        fixed_trials,
        layers=layers,
        heads=heads,
    )
    resume_paths = _ensure_resume_root_state(
        artifacts_dir=artifacts_dir,
        resume_fingerprint=args._resume_fingerprint,
        log=log,
    )

    if args.debug_stepd:
        log("[StepD debug] running raw vs self parity check")
        run_stepd_debug(
            model=model,
            tokenizer=tokenizer,
            device=device,
            pairs=pairs,
            model_cfg=model_cfg,
            layer=layers[0],
            head=heads[0],
            tok_add_special=tok_add_special,
            baseline_recompute_outproj=baseline_recompute_outproj,
            resolved_quant=resolved_quant,
            seed=args.seed,
            n_icl_examples=args.n_icl_examples,
            n_trials=args.n_trials,
            model_spec=args.model_spec,
            prompt_meta=prompt_meta,
            log=log,
        )
        log_file.close()
        return 0

    if fixed_trials is None and len(pairs) < args.n_icl_examples + 1:
        log(
            "Not enough dataset pairs for requested demos + query: "
            f"pairs={len(pairs)} n_icl_examples={args.n_icl_examples}"
        )
        log_file.close()
        return 1

    trials = []
    attempts = 0
    kept = 0
    p_targets = []
    dummy_labels_ref = None
    slot_index_map_ref = None
    special_index_labels_ref = None

    if args.shuffle_labels not in (0, 1):
        log("shuffle_labels must be 0 or 1")
        log_file.close()
        return 1

    if fixed_trials is not None:
        for trial_idx, trial in enumerate(fixed_trials["trials"][: args.n_trials]):
            demos_clean = trial.get("demos_clean")
            demos_corrupted = trial.get("demos_corrupted")
            query = trial.get("query")
            if demos_clean is None or demos_corrupted is None or query is None:
                log(
                    "fixed_trials missing demos/query; "
                    "regenerate fixed_trials with demos_clean/demos_corrupted/query"
                )
                log_file.close()
                return 1

            demos_clean_norm = _normalize_demos(demos_clean)
            demos_corrupted_norm = _normalize_demos(demos_corrupted)
            query_norm = _normalize_query(query)

            fixed_clean = trial.get("clean_prompt_str")
            fixed_corrupted = trial.get("corrupted_prompt_str")
            if fixed_clean is None or fixed_corrupted is None:
                log(f"fixed_trials missing prompt strings at trial {trial_idx}")
                log_file.close()
                return 1
            clean_prefix_str, _clean_full_str, _prompt_data = build_prompt_qa_paper(
                demos_clean_norm,
                query_norm,
                prefixes=prompt_meta["prefixes"],
                separators=prompt_meta["separators"],
                prepend_bos_token=prompt_meta["prepend_bos_token_used"],
                prepend_space=True,
            )
            corrupted_prefix_str, _tmp_full_str, _prompt_data = build_prompt_qa_paper(
                demos_corrupted_norm,
                query_norm,
                prefixes=prompt_meta["prefixes"],
                separators=prompt_meta["separators"],
                prepend_bos_token=prompt_meta["prepend_bos_token_used"],
                prepend_space=True,
            )
            if fixed_clean != clean_prefix_str:
                log(f"fixed_trials clean prompt mismatch at trial {trial_idx}")
                log_file.close()
                return 1
            if fixed_corrupted != corrupted_prefix_str:
                log(f"fixed_trials corrupted prompt mismatch at trial {trial_idx}")
                log_file.close()
                return 1
            target_id = trial.get("target_first_token_id")
            if target_id is None:
                target_id = trial.get("target_id")
            if target_id is None:
                log(f"fixed_trials missing target_first_token_id at trial {trial_idx}")
                log_file.close()
                return 1
            target_str = trial.get("target_str")
            if target_str is None:
                log(f"fixed_trials missing target_str at trial {trial_idx}")
                log_file.close()
                return 1
            corrupted_prefix_str = fixed_corrupted
            clean_prefix_str = fixed_clean
            corrupted_full_str = f"{fixed_corrupted}{target_str}"
            try:
                (
                    idx_map,
                    dummy_labels,
                    slot_index_map,
                    special_index_labels,
                    full_input_ids,
                ) = compute_trial_idx_map(
                    demos_corrupted_norm,
                    query_norm,
                    corrupted_prefix_str,
                    corrupted_full_str,
                    tokenizer,
                    tok_add_special,
                    args.n_icl_examples,
                    dummy_labels_ref=dummy_labels_ref,
                    slot_index_map_ref=slot_index_map_ref,
                    special_index_labels_ref=special_index_labels_ref,
                    prompt_data=trial.get("prompt_data_corrupted"),
                )
            except ValueError as exc:
                log(str(exc))
                log_file.close()
                return 1
            if dummy_labels_ref is None:
                dummy_labels_ref = dummy_labels
                slot_index_map_ref = slot_index_map
                special_index_labels_ref = special_index_labels
            slot_q = slot_index_map["QUERY_PRED"]
            try:
                seq_token_idx = resolve_slot_seq_token_idx(
                    idx_map, slot_q, require_single=False
                )
            except ValueError as exc:
                log(str(exc))
                log_file.close()
                return 1
            prefix_ids = tokenizer.encode(
                corrupted_prefix_str, add_special_tokens=tok_add_special
            )
            prefix_last_idx = len(prefix_ids) - 1
            if seq_token_idx != prefix_last_idx:
                bos_token = tokenizer.bos_token or tokenizer.eos_token
                if bos_token and corrupted_prefix_str.startswith(bos_token):
                    log(
                        "slot alignment mismatch with BOS prefix; "
                        f"seq_token_idx={seq_token_idx} "
                        f"prefix_last_idx={prefix_last_idx} "
                        "-> using prefix_last_idx"
                    )
                    seq_token_idx = prefix_last_idx
                else:
                    log(
                        "slot alignment mismatch: "
                        f"seq_token_idx={seq_token_idx} "
                        f"prefix_last_idx={prefix_last_idx}"
                    )
                    log_file.close()
                    return 1
            trials.append(
                {
                    "q_id": trial.get("q_id"),
                    "trial_idx": trial_idx,
                    "clean_prefix_str": fixed_clean,
                    "corrupted_prefix_str": fixed_corrupted,
                    "corrupted_full_str": corrupted_full_str,
                    "demos_clean": trial.get("demos_clean"),
                    "demos_corrupted": trial.get("demos_corrupted"),
                    "query": trial.get("query"),
                    "prompt_data_clean": trial.get("prompt_data_clean"),
                    "prompt_data_corrupted": trial.get("prompt_data_corrupted"),
                    "target_id": target_id,
                    "target_token": tokenizer.convert_ids_to_tokens(target_id),
                    "demo_perm": None,
                    "demo_fixed_points": None,
                    "demo_outputs_before": None,
                    "demo_outputs_after": None,
                    "idx_map": idx_map,
                    "seq_token_idx": seq_token_idx,
                    "full_input_ids": full_input_ids,
                    "target_str": trial.get("target_str"),
                    "answer_ids": trial.get("answer_ids"),
                }
            )
        attempts = len(trials)
        kept = len(trials)
        if args.debug_prompt_check:
            debug_n = max(0, int(args.debug_n))
            debug_lines = []
            debug_lines.append("[CHECK] tokenizer")
            debug_lines.append(
                "  tok_add_special={} bos_token={} bos_id={}".format(
                    tok_add_special,
                    repr(tokenizer.bos_token),
                    tokenizer.bos_token_id,
                )
            )
            for trial in trials[:debug_n]:
                prompt_string = trial["corrupted_prefix_str"]
                target_id = trial["target_id"]
                target_str = trial.get("target_str")
                answer_ids = trial.get("answer_ids") or []
                enc = tokenizer(prompt_string, add_special_tokens=tok_add_special)
                ids = enc["input_ids"]
                ids_len = len(ids)
                ids_first5 = ids[:5]
                ids_last10 = ids[-10:]
                toks_last10 = [repr(tokenizer.decode([i])) for i in ids_last10]
                answer_ids_first = None
                if answer_ids:
                    answer_ids_first = answer_ids[0]
                elif target_str is not None:
                    answer_ids_first = tokenizer(
                        target_str, add_special_tokens=False
                    )["input_ids"][0]
                debug_lines.append(f"[CHECK] trial={trial['trial_idx']}")
                debug_lines.append(f"  prompt_repr={repr(prompt_string)}")
                debug_lines.append(
                    f"  prompt_head80={repr(prompt_string[:80])} prompt_tail80={repr(prompt_string[-80:])}"
                )
                debug_lines.append(
                    "  tok_add_special={} bos_token={} bos_id={}".format(
                        tok_add_special,
                        repr(tokenizer.bos_token),
                        tokenizer.bos_token_id,
                    )
                )
                debug_lines.append(f"  ids_len={ids_len}")
                debug_lines.append(
                    f"  ids_first5={ids_first5} ids_last10={ids_last10}"
                )
                debug_lines.append(f"  toks_last10={toks_last10}")
                debug_lines.append(f"  target_str_repr={repr(target_str)}")
                debug_lines.append(
                    "  target_id={} target_tok={}".format(
                        target_id,
                        repr(tokenizer.convert_ids_to_tokens(target_id)),
                    )
                )
                match = target_id == answer_ids_first if answer_ids_first is not None else None
                debug_lines.append(
                    f"  answer_ids_first={answer_ids_first} match={match}"
                )
            for line in debug_lines:
                print(line, flush=True)
            debug_dir = os.path.join(PROJECT_ROOT, "runs", "stepd_fixed")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "stepd_prompt_check.txt")
            with open(debug_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(debug_lines) + "\n")
            raise SystemExit(0)
    elif args.successful_icl_only:
        while len(trials) < args.n_trials:
            if attempts >= args.max_trial_attempts:
                log(
                    "successful ICL이 너무 적으니 n_trials 줄이거나 "
                    "max_trial_attempts 늘리거나 successful_icl_only=0으로 끄라"
                )
                log_file.close()
                return 1
            attempt_idx = attempts
            attempts += 1
            demos_orig, query = sample_demos_and_query(
                pairs, args.n_icl_examples, seed=args.seed + attempt_idx
            )
            demo_perm = None
            demo_outputs_before = None
            demo_outputs_after = None
            demo_fixed_points = None
            if args.shuffle_labels:
                rng = random.Random(args.seed + attempt_idx)
                demo_perm = list(range(len(demos_orig)))
                rng.shuffle(demo_perm)
                demos, demo_outputs_before, demo_outputs_after, demo_fixed_points = (
                    _make_demo_only_shuffle(demos_orig, demo_perm)
                )
                if args.n_icl_examples == 1 and attempt_idx == 0:
                    log(f"demo_shuffle fixed_points={demo_fixed_points}")
            else:
                demos = demos_orig
            demos_norm = _normalize_demos(demos)
            query_norm = _normalize_query(query)
            clean_prefix_str, clean_full_str, _prompt_data = build_prompt_qa_paper(
                demos_norm,
                query_norm,
                prefixes=prompt_meta["prefixes"],
                separators=prompt_meta["separators"],
                prepend_bos_token=prompt_meta["prepend_bos_token_used"],
                prepend_space=True,
            )

            if attempts == 1:
                log(f"n_pairs_loaded: {len(pairs)}")
                log(f"n_icl_examples: {args.n_icl_examples}")
                log(f"example query: input='{query_norm[0]}' output='{query_norm[1]}'")
                log(f"prefix_endswith_A_space: {clean_prefix_str.endswith('A: ')}")

            try:
                boundary_prefix, boundary_answer = _boundary_prefix_and_answer_from_full(
                    clean_prefix_str, clean_full_str
                )
                target_id = get_target_first_token_id_from_boundary(
                    boundary_prefix,
                    boundary_answer,
                    tokenizer,
                    tokenize_kwargs={"add_special_tokens": tok_add_special},
                )
            except ValueError as exc:
                log(str(exc))
                log_file.close()
                return 1

            success, p_target = check_successful_icl(
                model,
                tokenizer,
                device,
                clean_prefix_str,
                target_id,
                tok_add_special,
            )
            if not success:
                continue

            corrupted_demos = make_corrupted_demos(
                demos_orig, random.Random(args.seed + attempt_idx), ensure_derangement=True
            )
            if args.shuffle_labels:
                corrupted_demos, _, _, _ = _make_demo_only_shuffle(
                    corrupted_demos, demo_perm
                )
            corrupted_demos_norm = _normalize_demos(corrupted_demos)
            corrupted_prefix_str, corrupted_full_str, prompt_data_corrupted = build_prompt_qa_paper(
                corrupted_demos_norm,
                query_norm,
                prefixes=prompt_meta["prefixes"],
                separators=prompt_meta["separators"],
                prepend_bos_token=prompt_meta["prepend_bos_token_used"],
                prepend_space=True,
            )
            try:
                boundary_prefix, boundary_answer = _boundary_prefix_and_answer_from_full(
                    corrupted_prefix_str, corrupted_full_str
                )
                corrupted_target_id = get_target_first_token_id_from_boundary(
                    boundary_prefix,
                    boundary_answer,
                    tokenizer,
                    tokenize_kwargs={"add_special_tokens": tok_add_special},
                )
            except ValueError as exc:
                log(str(exc))
                log_file.close()
                return 1

            if target_id != corrupted_target_id:
                log("target_id mismatch between clean and corrupted")
                log_file.close()
                return 1
            try:
                (
                    idx_map,
                    dummy_labels,
                    slot_index_map,
                    special_index_labels,
                    full_input_ids,
                ) = compute_trial_idx_map(
                    corrupted_demos_norm,
                    query_norm,
                    corrupted_prefix_str,
                    corrupted_full_str,
                    tokenizer,
                    tok_add_special,
                    args.n_icl_examples,
                    dummy_labels_ref=dummy_labels_ref,
                    slot_index_map_ref=slot_index_map_ref,
                    special_index_labels_ref=special_index_labels_ref,
                    prompt_data=prompt_data_corrupted,
                )
            except ValueError as exc:
                log(str(exc))
                log_file.close()
                return 1
            if dummy_labels_ref is None:
                dummy_labels_ref = dummy_labels
                slot_index_map_ref = slot_index_map
                special_index_labels_ref = special_index_labels
            slot_q = slot_index_map["QUERY_PRED"]
            try:
                seq_token_idx = resolve_slot_seq_token_idx(
                    idx_map, slot_q, require_single=False
                )
            except ValueError as exc:
                log(str(exc))
                log_file.close()
                return 1
            prefix_ids = tokenizer.encode(
                corrupted_prefix_str, add_special_tokens=tok_add_special
            )
            if seq_token_idx != len(prefix_ids) - 1:
                bos_token = tokenizer.bos_token or tokenizer.eos_token
                if bos_token and corrupted_prefix_str.startswith(bos_token):
                    log(
                        "slot alignment mismatch with BOS prefix; "
                        f"seq_token_idx={seq_token_idx} "
                        f"prefix_last_idx={len(prefix_ids) - 1} "
                        "-> using prefix_last_idx"
                    )
                    seq_token_idx = len(prefix_ids) - 1
                else:
                    log(
                        "slot alignment mismatch: "
                        f"seq_token_idx={seq_token_idx} "
                        f"prefix_last_idx={len(prefix_ids) - 1}"
                    )
                    log_file.close()
                    return 1
            decoded_pred_token = tokenizer.decode([full_input_ids[seq_token_idx]])
            if decoded_pred_token != ":":
                log(
                    "QUERY_PRED token mismatch: "
                    f"decoded={repr(decoded_pred_token)}"
                )
                log_file.close()
                return 1
            expected_ids = tokenizer(
                f" {query_norm[1]}", add_special_tokens=False
            )["input_ids"]
            expected_first = expected_ids[0] if expected_ids else None
            if expected_first is None or target_id != expected_first:
                log(
                    "target_id mismatch: "
                    f"target_id={target_id} "
                    f"expected_id={expected_first} "
                    f"expected_token={repr(tokenizer.decode([expected_first])) if expected_first is not None else None}"
                )
                log_file.close()
                return 1
            log(
                "[target_debug] "
                f"trial={kept} "
                f"target_id={target_id} "
                f"token={repr(tokenizer.decode([target_id]))} "
                f"prompt_tail={repr(corrupted_prefix_str[-30:])}"
            )

            kept += 1
            p_targets.append(p_target)
            trials.append(
                {
                    "q_id": "__all__",
                    "trial_idx": kept - 1,
                    "clean_prefix_str": clean_prefix_str,
                    "corrupted_prefix_str": corrupted_prefix_str,
                    "corrupted_full_str": corrupted_full_str,
                    "demos_clean": [{"input": x, "output": y} for x, y in demos_norm],
                    "demos_corrupted": [
                        {"input": x, "output": y} for x, y in corrupted_demos_norm
                    ],
                    "query": {"input": query_norm[0], "output": query_norm[1]},
                    "target_id": target_id,
                    "target_token": tokenizer.convert_ids_to_tokens(target_id),
                    "demo_perm": demo_perm,
                    "demo_fixed_points": demo_fixed_points,
                    "demo_outputs_before": demo_outputs_before,
                    "demo_outputs_after": demo_outputs_after,
                    "idx_map": idx_map,
                    "seq_token_idx": seq_token_idx,
                    "full_input_ids": full_input_ids,
                    "target_str": query_norm[1],
                }
            )
    else:
        for trial_idx in range(args.n_trials):
            demos_orig, query = sample_demos_and_query(
                pairs, args.n_icl_examples, seed=args.seed + trial_idx
            )
            demo_perm = None
            demo_outputs_before = None
            demo_outputs_after = None
            demo_fixed_points = None
            if args.shuffle_labels:
                rng = random.Random(args.seed + trial_idx)
                demo_perm = list(range(len(demos_orig)))
                rng.shuffle(demo_perm)
                demos, demo_outputs_before, demo_outputs_after, demo_fixed_points = (
                    _make_demo_only_shuffle(demos_orig, demo_perm)
                )
                if args.n_icl_examples == 1 and trial_idx == 0:
                    log(f"demo_shuffle fixed_points={demo_fixed_points}")
            else:
                demos = demos_orig
            demos_norm = _normalize_demos(demos)
            query_norm = _normalize_query(query)
            clean_prefix_str, clean_full_str, _prompt_data = build_prompt_qa_paper(
                demos_norm,
                query_norm,
                prefixes=prompt_meta["prefixes"],
                separators=prompt_meta["separators"],
                prepend_bos_token=prompt_meta["prepend_bos_token_used"],
                prepend_space=True,
            )
            corrupted_demos = make_corrupted_demos(
                demos_orig, random.Random(args.seed + trial_idx), ensure_derangement=True
            )
            if args.shuffle_labels:
                corrupted_demos, _, _, _ = _make_demo_only_shuffle(
                    corrupted_demos, demo_perm
                )
            corrupted_demos_norm = _normalize_demos(corrupted_demos)
            corrupted_prefix_str, corrupted_full_str, prompt_data_corrupted = build_prompt_qa_paper(
                corrupted_demos_norm,
                query_norm,
                prefixes=prompt_meta["prefixes"],
                separators=prompt_meta["separators"],
                prepend_bos_token=prompt_meta["prepend_bos_token_used"],
                prepend_space=True,
            )

            if trial_idx == 0:
                log(f"n_pairs_loaded: {len(pairs)}")
                log(f"n_icl_examples: {args.n_icl_examples}")
                log(f"example query: input='{query_norm[0]}' output='{query_norm[1]}'")
                log(f"prefix_endswith_A_space: {clean_prefix_str.endswith('A: ')}")

            try:
                boundary_prefix, boundary_answer = _boundary_prefix_and_answer_from_full(
                    clean_prefix_str, clean_full_str
                )
                target_id = get_target_first_token_id_from_boundary(
                    boundary_prefix,
                    boundary_answer,
                    tokenizer,
                    tokenize_kwargs={"add_special_tokens": tok_add_special},
                )
                boundary_prefix, boundary_answer = _boundary_prefix_and_answer_from_full(
                    corrupted_prefix_str, corrupted_full_str
                )
                corrupted_target_id = get_target_first_token_id_from_boundary(
                    boundary_prefix,
                    boundary_answer,
                    tokenizer,
                    tokenize_kwargs={"add_special_tokens": tok_add_special},
                )
            except ValueError as exc:
                log(str(exc))
                log_file.close()
                return 1

            if target_id != corrupted_target_id:
                log("target_id mismatch between clean and corrupted")
                log_file.close()
                return 1

            try:
                (
                    idx_map,
                    dummy_labels,
                    slot_index_map,
                    special_index_labels,
                    full_input_ids,
                ) = compute_trial_idx_map(
                    corrupted_demos_norm,
                    query_norm,
                    corrupted_prefix_str,
                    corrupted_full_str,
                    tokenizer,
                    tok_add_special,
                    args.n_icl_examples,
                    dummy_labels_ref=dummy_labels_ref,
                    slot_index_map_ref=slot_index_map_ref,
                    special_index_labels_ref=special_index_labels_ref,
                    prompt_data=prompt_data_corrupted,
                )
            except ValueError as exc:
                log(str(exc))
                log_file.close()
                return 1
            if dummy_labels_ref is None:
                dummy_labels_ref = dummy_labels
                slot_index_map_ref = slot_index_map
                special_index_labels_ref = special_index_labels
            slot_q = slot_index_map["QUERY_PRED"]
            try:
                seq_token_idx = resolve_slot_seq_token_idx(
                    idx_map, slot_q, require_single=False
                )
            except ValueError as exc:
                log(str(exc))
                log_file.close()
                return 1
            prefix_ids = tokenizer.encode(
                corrupted_prefix_str, add_special_tokens=tok_add_special
            )
            if seq_token_idx != len(prefix_ids) - 1:
                bos_token = tokenizer.bos_token or tokenizer.eos_token
                if bos_token and corrupted_prefix_str.startswith(bos_token):
                    log(
                        "slot alignment mismatch with BOS prefix; "
                        f"seq_token_idx={seq_token_idx} "
                        f"prefix_last_idx={len(prefix_ids) - 1} "
                        "-> using prefix_last_idx"
                    )
                    seq_token_idx = len(prefix_ids) - 1
                else:
                    log(
                        "slot alignment mismatch: "
                        f"seq_token_idx={seq_token_idx} "
                        f"prefix_last_idx={len(prefix_ids) - 1}"
                    )
                    log_file.close()
                    return 1
            decoded_pred_token = tokenizer.decode([full_input_ids[seq_token_idx]])
            if decoded_pred_token != ":":
                log(
                    "QUERY_PRED token mismatch: "
                    f"decoded={repr(decoded_pred_token)}"
                )
                log_file.close()
                return 1
            expected_ids = tokenizer(
                f" {query_norm[1]}", add_special_tokens=False
            )["input_ids"]
            expected_first = expected_ids[0] if expected_ids else None
            if expected_first is None or target_id != expected_first:
                log(
                    "target_id mismatch: "
                    f"target_id={target_id} "
                    f"expected_id={expected_first} "
                    f"expected_token={repr(tokenizer.decode([expected_first])) if expected_first is not None else None}"
                )
                log_file.close()
                return 1
            log(
                "[target_debug] "
                f"trial={trial_idx} "
                f"target_id={target_id} "
                f"token={repr(tokenizer.decode([target_id]))} "
                f"prompt_tail={repr(corrupted_prefix_str[-30:])}"
            )

            trials.append(
                {
                    "q_id": "__all__",
                    "trial_idx": trial_idx,
                    "clean_prefix_str": clean_prefix_str,
                    "corrupted_prefix_str": corrupted_prefix_str,
                    "corrupted_full_str": corrupted_full_str,
                    "demos_clean": [{"input": x, "output": y} for x, y in demos_norm],
                    "demos_corrupted": [
                        {"input": x, "output": y} for x, y in corrupted_demos_norm
                    ],
                    "query": {"input": query_norm[0], "output": query_norm[1]},
                    "target_id": target_id,
                    "target_token": tokenizer.convert_ids_to_tokens(target_id),
                    "demo_perm": demo_perm,
                    "demo_fixed_points": demo_fixed_points,
                    "demo_outputs_before": demo_outputs_before,
                    "demo_outputs_after": demo_outputs_after,
                    "idx_map": idx_map,
                    "seq_token_idx": seq_token_idx,
                    "full_input_ids": full_input_ids,
                    "target_str": query_norm[1],
                }
            )

        attempts = args.n_trials
        kept = args.n_trials

    kept_ratio = kept / attempts if attempts else 0.0
    p_target_mean = mean(p_targets) if p_targets else 0.0
    log(
        f"trial_sampling_done attempts={attempts} kept={kept} kept_ratio={kept_ratio:.4f}"
    )
    if p_targets:
        log(f"p_target_mean={p_target_mean:.6f}")
    else:
        log("p_target_mean=n/a")

    sampled_trials_rows = []
    for trial in trials:
        sampled_trials_rows.append(
            {
                "q_id": trial.get("q_id", "__all__"),
                "trial_idx": int(trial.get("trial_idx", 0)),
                "clean_prompt_str": trial.get("clean_prefix_str"),
                "corrupted_prompt_str": trial.get("corrupted_prefix_str"),
                "corrupted_full_str": trial.get("corrupted_full_str"),
                "target_str": trial.get("target_str"),
                "target_first_token_id": int(trial["target_id"]),
                "target_token": trial.get("target_token"),
                "demos_clean": trial.get("demos_clean"),
                "demos_corrupted": trial.get("demos_corrupted"),
                "query": trial.get("query"),
                "query_source_index": trial.get("query_source_index"),
                "demo_source_indices": trial.get("demo_source_indices"),
                "demo_order": trial.get("demo_order"),
            }
        )
    sampled_trials_payload = {
        "meta": {
            "source": (
                "relation_csv"
                if args.relation_csv_path
                else ("fixed_trials_override" if args.fixed_trials_path else "dataset_sampled")
            ),
            "relation_csv_path": args.relation_csv_path,
            "fixed_trials_path": args.fixed_trials_path,
            "seed": args.seed,
            "n_trials": len(sampled_trials_rows),
            "n_shots": args.n_icl_examples,
            "n_demos": args.n_icl_examples,
            **storage_metadata(canonical_root=artifacts_dir),
        },
        "trials": sampled_trials_rows,
    }
    sampled_trials_path = os.path.join(artifacts_dir, "sampled_trials.json")
    save_json(sampled_trials_path, sampled_trials_payload)
    log(f"saved sampled trials snapshot: {sampled_trials_path}")

    layer_modules = {}
    for layer in layers:
        try:
            target_module, _target_name = get_out_proj_pre_hook_target(
                model, layer, spec_name=args.model_spec, logger=log
            )
        except ValueError as exc:
            log(str(exc))
            log_file.close()
            return 1
        layer_modules[layer] = target_module

    if args.relation_csv_path:
        # q_id-specific mean activations and CIE/AIE aggregation.
        trials_by_qid = {}
        for trial in trials:
            q_id = trial.get("q_id")
            if q_id is None:
                log("relation mode requires q_id in trials")
                log_file.close()
                return 1
            trials_by_qid.setdefault(q_id, []).append(trial)

        trial_metrics_path = resume_paths["trial_metrics"]
        trial_metrics_file = open(trial_metrics_path, "a", encoding="utf-8")

        cie_rows = []
        aie_accumulator = {}
        aie_counts = {}
        mean_acts_by_qid = {}
        slot_q_ref = None
        perf_groups = {}
        log("starting AIE sweep (qid-specific mean activations)")

        for q_id, q_trials in sorted(trials_by_qid.items()):
            log(f"[qid] computing mean_activations for {q_id} n_trials={len(q_trials)}")
            fixed_subset = {"trials": q_trials}
            try:
                mean_acts, _dummy_labels, slot_index_map = (
                    compute_mean_activations_fixed_trials_ns(
                        model=model,
                        tokenizer=tokenizer,
                        layer_modules=layer_modules,
                        fixed_trials=fixed_subset,
                        n_use=len(q_trials),
                        model_cfg=model_cfg,
                        tok_add_special=tok_add_special,
                        device=device,
                        logger=log,
                    )
                )
                mean_acts_by_qid[q_id] = mean_acts.detach().cpu()
            except ValueError as exc:
                log(str(exc))
                log_file.close()
                trial_metrics_file.close()
                return 1

            slot_q = slot_index_map.get("QUERY_PRED")
            if slot_q is None:
                log("QUERY_PRED missing from slot_index_map")
                log_file.close()
                trial_metrics_file.close()
                return 1
            if slot_q_ref is None:
                slot_q_ref = int(slot_q)
            elif int(slot_q) != slot_q_ref:
                log("QUERY_PRED slot index mismatch across qids")
                log_file.close()
                trial_metrics_file.close()
                return 1

            try:
                q_rows, q_perf = _run_aie_sweep_for_trials(
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    tok_add_special=tok_add_special,
                    layer_modules=layer_modules,
                    layers=layers,
                    heads=heads,
                    trials=q_trials,
                    mean_acts=mean_acts,
                    slot_q=int(slot_q),
                    model_cfg=model_cfg,
                    resolved_quant=resolved_quant,
                    baseline_recompute_outproj=baseline_recompute_outproj,
                    log=log,
                    artifacts_dir=artifacts_dir,
                    trial_metrics_file=trial_metrics_file,
                    progress_desc=f"StepD {q_id}",
                    n_trials_for_row=len(q_trials),
                    q_id=q_id,
                    save_trials_rows=None,
                    dump_handle=None,
                    dump_expected_keys=None,
                    log_patch_debug=False,
                )
            except ValueError as exc:
                log(str(exc))
                log_file.close()
                trial_metrics_file.close()
                return 1
            cie_rows.extend(q_rows)
            perf_groups[q_id] = q_perf

            for cie_row in q_rows:
                key = (int(cie_row["layer"]), int(cie_row["head"]))
                for k, v in cie_row.items():
                    if k in ("q_id", "layer", "head", "n_trials"):
                        continue
                    if not isinstance(v, (int, float)):
                        continue
                    aie_accumulator.setdefault(key, {}).setdefault(k, 0.0)
                    aie_accumulator[key][k] += float(v)
                aie_counts[key] = aie_counts.get(key, 0) + 1

        trial_metrics_file.close()
        _materialize_final_outputs(
            artifacts_dir=artifacts_dir,
            score_rows=cie_rows,
            dump_final_path=args.dump_trial_metrics_jsonl,
            save_trials_enabled=bool(args.save_trials),
        )
        log(f"saved trial metrics: {os.path.join(artifacts_dir, 'trial_metrics.jsonl')}")

        cie_path = os.path.join(artifacts_dir, "cie_scores.csv")
        save_csv(cie_path, cie_rows)
        log(f"saved CIE scores: {cie_path}")

        aie_rows = []
        for (layer, head), sums in sorted(aie_accumulator.items()):
            count = aie_counts.get((layer, head), 1)
            row = {"layer": layer, "head": head, "n_qid": count}
            for k, v in sums.items():
                row[k] = v / count
            aie_rows.append(row)

        if aie_rows and args.score_key in aie_rows[0]:
            aie_rows = sorted(aie_rows, key=lambda row: row[args.score_key], reverse=True)
        scores_path = os.path.join(artifacts_dir, "aie_scores.csv")
        save_csv(scores_path, aie_rows)
        log(f"saved AIE scores: {scores_path}")
        mean_acts_dir = os.path.join(artifacts_dir, "stepD_mean_acts")
        os.makedirs(mean_acts_dir, exist_ok=True)
        qid_paths = []
        for q_id, q_mean in sorted(mean_acts_by_qid.items()):
            if hasattr(q_mean, "dim") and q_mean.dim() == 4:
                q_mean_export = q_mean[:, :, slot_q_ref, :]
            else:
                q_mean_export = q_mean
            q_path = os.path.join(mean_acts_dir, f"qid_{q_id}_clean_mean.pt")
            torch.save(
                {
                    "clean_mean": q_mean_export,
                    "slot_q": slot_q_ref,
                    "n_heads": int(model_cfg["n_heads"]),
                    "head_dim": int(model_cfg["head_dim"]),
                    "resid_dim": int(model_cfg["resid_dim"]),
                },
                q_path,
            )
            qid_paths.append(q_path)
        if qid_paths:
            log(f"saved qid mean acts: n={len(qid_paths)} dir={mean_acts_dir}")
            stacked = torch.stack([mean_acts_by_qid[q] for q in sorted(mean_acts_by_qid.keys())], dim=0)
            global_mean = stacked.mean(dim=0)
            if hasattr(global_mean, "dim") and global_mean.dim() == 4:
                global_clean_mean = global_mean[:, :, slot_q_ref, :]
            else:
                global_clean_mean = global_mean
            global_path = os.path.join(mean_acts_dir, "global_clean_mean.pt")
            torch.save(
                {
                    "clean_mean": global_clean_mean,
                    "slot_q": slot_q_ref,
                    "n_heads": int(model_cfg["n_heads"]),
                    "head_dim": int(model_cfg["head_dim"]),
                    "resid_dim": int(model_cfg["resid_dim"]),
                },
                global_path,
            )
            # Backward compatibility for StepE legacy path probes.
            torch.save(global_clean_mean, os.path.join(artifacts_dir, "mean_activations.pt"))
            log(f"saved global mean acts: {global_path}")
        if args.perf_log:
            perf_path = os.path.join(artifacts_dir, "perf_stats.json")
            save_json(
                perf_path,
                {
                    "mode": "relation_qid",
                    "baseline_cache_scope": args.baseline_cache_scope,
                    "cache_tokenized_inputs": bool(args.cache_tokenized_inputs),
                    "persistent_layer_hook": bool(args.persistent_layer_hook),
                    "groups": perf_groups,
                },
            )
            log(f"saved perf stats: {perf_path}")
        log_file.close()
        return 0

    log("computing mean_activations")
    try:
        if fixed_trials is not None:
            mean_acts, dummy_labels, slot_index_map = (
                compute_mean_activations_fixed_trials_ns(
                    model=model,
                    tokenizer=tokenizer,
                    layer_modules=layer_modules,
                    fixed_trials=fixed_trials,
                    n_use=args.n_mean_trials,
                    model_cfg=model_cfg,
                    tok_add_special=tok_add_special,
                    device=device,
                    logger=log,
                )
            )
        else:
            mean_acts, dummy_labels, slot_index_map = compute_mean_activations_ns(
                model=model,
                tokenizer=tokenizer,
                layer_modules=layer_modules,
                pairs=pairs,
                n_icl_examples=args.n_icl_examples,
                n_mean_trials=args.n_mean_trials,
                model_cfg=model_cfg,
                seed=args.seed,
                tok_add_special=tok_add_special,
                device=device,
                shuffle_labels=bool(args.shuffle_labels),
                prefixes=prompt_meta["prefixes"],
                separators=prompt_meta["separators"],
                logger=log,
            )
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1

    if not args.mean_only:
        if dummy_labels_ref is not None and dummy_labels_ref != dummy_labels:
            log("dummy_labels mismatch between trials and mean_activations")
            log_file.close()
            return 1
        if slot_index_map_ref is not None and slot_index_map_ref != slot_index_map:
            log("slot_index_map mismatch between trials and mean_activations")
            log_file.close()
            return 1

    slot_q = slot_index_map.get("QUERY_PRED")
    if slot_q is None:
        log("QUERY_PRED missing from slot_index_map")
        log_file.close()
        return 1
    log(
        "patch config: "
        f"slot_idx={slot_q} (QUERY_PRED) -> seq_token_idx per trial"
    )
    log(
        "mean_activations: "
        f"shape={tuple(mean_acts.shape)} n_slots={len(dummy_labels)} "
        f"slot_q={slot_q} label={dummy_labels[slot_q]}"
    )

    mean_acts_path = os.path.join(artifacts_dir, "mean_activations.pt")
    torch.save(mean_acts.cpu(), mean_acts_path)
    log(f"saved mean_activations: {mean_acts_path}")

    mean_meta_path = os.path.join(artifacts_dir, "mean_activations_meta.json")
    save_json(
        mean_meta_path,
        {
            "n_mean_trials": args.n_mean_trials,
            "n_icl_examples": args.n_icl_examples,
            "seed": args.seed,
            "layers": layers,
            "n_layers": layer_count,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "resid_dim": resid_dim,
            "dummy_labels": dummy_labels,
            "slot_index_map": slot_index_map,
            "slot_query_pred": slot_q,
            **storage_metadata(canonical_root=artifacts_dir),
        },
    )
    log(f"saved mean_activations meta: {mean_meta_path}")

    if args.mean_only:
        log("mean_only=1; exiting after mean_activations")
        log_file.close()
        return 0

    stepd_filter_path = os.path.join(artifacts_dir, "stepD_success_filter.json")
    save_json(
        stepd_filter_path,
        {
            "successful_icl_only": args.successful_icl_only,
            "max_trial_attempts": args.max_trial_attempts,
            "attempts": attempts,
            "kept": kept,
            "kept_ratio": kept_ratio,
            "seed": args.seed,
            "n_icl_examples": args.n_icl_examples,
        },
    )
    log(f"saved success filter: {stepd_filter_path}")

    run_meta_path = os.path.join(artifacts_dir, "run_meta.json")
    save_json(
        run_meta_path,
        {
            "shuffle_labels": bool(args.shuffle_labels),
            "shuffle_derangement": False,
            "successful_icl_only_effective": args.successful_icl_only,
            "seed_global": args.seed,
            "compute_prob_scores": bool(args.compute_prob_scores),
            "stepd_fingerprint": args._resume_fingerprint,
            **storage_metadata(canonical_root=artifacts_dir),
        },
    )
    log(f"saved run meta: {run_meta_path}")

    scores = []
    trials_rows = []
    trial_metrics_path = resume_paths["trial_metrics"]
    trial_metrics_file = open(trial_metrics_path, "a", encoding="utf-8")
    dump_handle = None
    dump_expected_keys = [
        "trial_idx",
        "layer",
        "head",
        "target_id",
        "target_token",
        "seed_global",
        "n_icl_examples",
        "p_base",
        "p_patch",
        "delta_p",
        "p_base_target",
        "p_patch_target",
        "delta_p_target",
        "logit_base",
        "logit_patch",
        "delta_logit",
        "logprob_base",
        "logprob_patch",
        "delta_logprob",
        "prompt_tail_repr",
        "prompt_ends_with_space",
        "seq_token_idx",
        "ids_len",
        "ids_last10",
        "slot_idx",
    ]
    if args.dump_trial_metrics_jsonl:
        dump_handle = open(resume_paths["dump"], "a", encoding="utf-8")

    log("starting AIE sweep")
    try:
        scores, perf_stats = _run_aie_sweep_for_trials(
            args=args,
            model=model,
            tokenizer=tokenizer,
            device=device,
            tok_add_special=tok_add_special,
            layer_modules=layer_modules,
            layers=layers,
            heads=heads,
            trials=trials,
            mean_acts=mean_acts,
            slot_q=int(slot_q),
            model_cfg=model_cfg,
            resolved_quant=resolved_quant,
            baseline_recompute_outproj=baseline_recompute_outproj,
            log=log,
            artifacts_dir=artifacts_dir,
            trial_metrics_file=trial_metrics_file,
            progress_desc="StepD layer×head",
            n_trials_for_row=args.n_trials,
            q_id=None,
            save_trials_rows=None,
            dump_handle=dump_handle,
            dump_expected_keys=dump_expected_keys,
            log_patch_debug=True,
        )
    except ValueError as exc:
        log(str(exc))
        trial_metrics_file.close()
        if dump_handle is not None:
            dump_handle.close()
        log_file.close()
        return 1
    if args.perf_log:
        perf_path = os.path.join(artifacts_dir, "perf_stats.json")
        save_json(
            perf_path,
            {
                "mode": "single",
                "baseline_cache_scope": args.baseline_cache_scope,
                "cache_tokenized_inputs": bool(args.cache_tokenized_inputs),
                "persistent_layer_hook": bool(args.persistent_layer_hook),
                "stats": perf_stats,
            },
        )
        log(f"saved perf stats: {perf_path}")

    if scores and args.score_key not in scores[0]:
        log(f"score_key not found: {args.score_key}")
        log_file.close()
        trial_metrics_file.close()
        return 1
    scores_sorted = sorted(scores, key=lambda row: row[args.score_key], reverse=True)
    log(f"top-10 by {args.score_key}:")
    for row in scores_sorted[:10]:
        log(
            "top: "
            f"layer={row['layer']} "
            f"head={row['head']} "
            f"{args.score_key}={row[args.score_key]:.6f}"
        )

    scores_path = os.path.join(artifacts_dir, "aie_scores.csv")
    save_csv(scores_path, scores_sorted)

    scores_json_path = os.path.join(artifacts_dir, "aie_scores.json")
    save_json(
        scores_json_path,
        {
            "meta": {
                "model": args.model,
                "model_spec": args.model_spec,
                "layers": layers,
                "heads": heads,
                "n_trials": args.n_trials,
                "n_icl_examples": args.n_icl_examples,
                "n_mean_trials": args.n_mean_trials,
                "seed": args.seed,
                "successful_icl_only": args.successful_icl_only,
                "attempts": attempts,
                "kept": kept,
                "kept_ratio": kept_ratio,
                "seq_token_idx": "per_trial",
                "slot_query_pred": slot_q,
                "slot_label": dummy_labels[slot_q],
                "score_key": args.score_key,
                **storage_metadata(canonical_root=artifacts_dir),
            },
            "scores": scores_sorted,
        },
    )

    log(f"saved scores: {scores_path}")
    log(f"saved scores json: {scores_json_path}")

    trial_metrics_file.close()
    _materialize_final_outputs(
        artifacts_dir=artifacts_dir,
        score_rows=scores,
        dump_final_path=args.dump_trial_metrics_jsonl,
        save_trials_enabled=bool(args.save_trials),
    )
    if args.save_trials:
        log(f"saved trials: {os.path.join(artifacts_dir, 'aie_trials.csv')}")
    log(f"saved trial metrics: {os.path.join(artifacts_dir, 'trial_metrics.jsonl')}")
    if dump_handle is not None:
        dump_handle.close()

    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
