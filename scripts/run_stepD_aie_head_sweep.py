#!/usr/bin/env python3
"""STEP D: AIE head sweep using mean_activations replacement on corrupted prompts."""

import argparse
import json
import os
import random
import statistics
import sys
import time
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
from fv.prompting import build_prompt_qa
from fv.slots import (
    compute_query_predictive_slot,
    resolve_slot_seq_token_idx,
    get_target_first_token_id_from_boundary,
)
from fv.mean_activations import paper_labels_to_slot_map
from src.utils.prompt_utils import (
    compute_duplicated_labels as paper_compute_duplicated_labels,
    get_dummy_token_labels as paper_get_dummy_token_labels,
    get_token_meta_labels as paper_get_token_meta_labels,
    word_pairs_to_prompt_data as paper_word_pairs_to_prompt_data,
)
from fv.relation_trials import generate_relation_trials, save_trials_json


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
        prefix_str, full_str = build_prompt_qa(
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


def compute_trial_metrics(logits_base, logits_patch, target_id):
    import torch.nn.functional as F

    p_base = F.softmax(logits_base, dim=-1)[target_id].item()
    p_patch = F.softmax(logits_patch, dim=-1)[target_id].item()
    delta_p = p_patch - p_base

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
        word_pairs = {"input": [x for x, _y in demos], "output": [y for _x, y in demos]}
        query_target_pair = {"input": [query[0]], "output": [query[1]]}
        prompt_data = paper_word_pairs_to_prompt_data(
            word_pairs,
            query_target_pair=query_target_pair,
            prepend_bos_token=not tok_add_special,
            shuffle_labels=False,
        )

    token_labels, prompt_string = paper_get_token_meta_labels(
        prompt_data, tokenizer, prepend_bos=tok_add_special
    )
    if prompt_string != prefix_str:
        raise ValueError("Prompt builder mismatch with paper prompt_string")

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

    special_index_labels = {}
    if (
        special_index_labels_ref is not None
        and special_index_labels != special_index_labels_ref
    ):
        raise ValueError("Special token alignment mismatch across trials")

    slot_q = slot_index_map.get("QUERY_PRED")
    if slot_q is not None and not idx_map.get(slot_q):
        fallback_idx = None
        try:
            encoded = tokenizer(
                full_str,
                return_offsets_mapping=True,
                add_special_tokens=tok_add_special,
            )
            offsets = encoded.get("offset_mapping") or []
            prefix_end = len(prefix_str)
            for idx, (start, end) in enumerate(offsets):
                if start == end:
                    continue
                if end <= prefix_end:
                    fallback_idx = idx
                else:
                    break
        except Exception:
            fallback_idx = None
        if fallback_idx is None:
            prefix_ids = tokenizer(
                prefix_str, return_tensors=None, add_special_tokens=tok_add_special
            )["input_ids"]
            if not prefix_ids:
                raise ValueError("Empty prefix tokenization for QUERY_PRED fallback")
            fallback_idx = len(prefix_ids) - 1
        idx_map[slot_q] = [fallback_idx]

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
    parser = argparse.ArgumentParser(description="STEP D AIE head sweep.")
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

    run_info = prepare_run_dirs(args.run_id)
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
    log(f"shuffle_labels: {args.shuffle_labels}")
    log(f"fixed_trials_path: {args.fixed_trials_path}")
    log(f"relation_csv_path: {args.relation_csv_path}")
    log(f"relation_q_list: {args.relation_q_list}")
    log(f"relation_n_trials_per_q: {args.relation_n_trials_per_q}")
    log(f"relation_n_demos: {args.relation_n_demos}")
    log(f"relation_save_trials_json: {args.relation_save_trials_json}")
    log(f"relation_out_path: {args.relation_out_path}")
    log(f"mean_only: {args.mean_only}")
    log(f"debug_prompt_check: {args.debug_prompt_check}")
    log(f"debug_n: {args.debug_n}")
    log(f"dump_trial_metrics_jsonl: {args.dump_trial_metrics_jsonl}")

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
            clean_prefix_str, _clean_full_str = build_prompt_qa(
                demos_clean_norm,
                query_norm,
                prefixes=prompt_meta["prefixes"],
                separators=prompt_meta["separators"],
                prepend_bos_token=prompt_meta["prepend_bos_token_used"],
                prepend_space=True,
            )
            corrupted_prefix_str, _tmp_full_str = build_prompt_qa(
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
                    "trial_idx": trial_idx,
                    "clean_prefix_str": fixed_clean,
                    "corrupted_prefix_str": fixed_corrupted,
                    "corrupted_full_str": corrupted_full_str,
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
            clean_prefix_str, clean_full_str = build_prompt_qa(
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
            corrupted_prefix_str, corrupted_full_str = build_prompt_qa(
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
            prompt_data_corrupted = paper_word_pairs_to_prompt_data(
                {
                    "input": [x for x, _y in corrupted_demos_norm],
                    "output": [y for _x, y in corrupted_demos_norm],
                },
                query_target_pair={"input": [query_norm[0]], "output": [query_norm[1]]},
                prepend_bos_token=prompt_meta["prepend_bos_token_used"],
                prefixes=prompt_meta["prefixes"],
                separators=prompt_meta["separators"],
                shuffle_labels=False,
                prepend_space=True,
            )
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
                    "trial_idx": kept - 1,
                    "clean_prefix_str": clean_prefix_str,
                    "corrupted_prefix_str": corrupted_prefix_str,
                    "corrupted_full_str": corrupted_full_str,
                    "target_id": target_id,
                    "target_token": tokenizer.convert_ids_to_tokens(target_id),
                    "demo_perm": demo_perm,
                    "demo_fixed_points": demo_fixed_points,
                    "demo_outputs_before": demo_outputs_before,
                    "demo_outputs_after": demo_outputs_after,
                    "idx_map": idx_map,
                    "seq_token_idx": seq_token_idx,
                    "full_input_ids": full_input_ids,
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
            clean_prefix_str, clean_full_str = build_prompt_qa(
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
            corrupted_prefix_str, corrupted_full_str = build_prompt_qa(
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

            prompt_data_corrupted = paper_word_pairs_to_prompt_data(
                {
                    "input": [x for x, _y in corrupted_demos_norm],
                    "output": [y for _x, y in corrupted_demos_norm],
                },
                query_target_pair={"input": [query_norm[0]], "output": [query_norm[1]]},
                prepend_bos_token=prompt_meta["prepend_bos_token_used"],
                prefixes=prompt_meta["prefixes"],
                separators=prompt_meta["separators"],
                shuffle_labels=False,
                prepend_space=True,
            )
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
                    "trial_idx": trial_idx,
                    "clean_prefix_str": clean_prefix_str,
                    "corrupted_prefix_str": corrupted_prefix_str,
                    "corrupted_full_str": corrupted_full_str,
                    "target_id": target_id,
                    "target_token": tokenizer.convert_ids_to_tokens(target_id),
                    "demo_perm": demo_perm,
                    "demo_fixed_points": demo_fixed_points,
                    "demo_outputs_before": demo_outputs_before,
                    "demo_outputs_after": demo_outputs_after,
                    "idx_map": idx_map,
                    "seq_token_idx": seq_token_idx,
                    "full_input_ids": full_input_ids,
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
        },
    )
    log(f"saved run meta: {run_meta_path}")

    scores = []
    trials_rows = []
    trial_metrics_path = os.path.join(artifacts_dir, "trial_metrics.jsonl")
    trial_metrics_file = open(trial_metrics_path, "w", encoding="utf-8")
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
        dump_dir = os.path.dirname(args.dump_trial_metrics_jsonl)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        dump_handle = open(args.dump_trial_metrics_jsonl, "w", encoding="utf-8")

    log("starting AIE sweep")
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    pairs = [(layer, head) for layer in layers for head in heads]
    total = len(pairs)
    log_every = max(1, total // 100)
    start_time = time.time()
    patch_token_logged = False
    pbar = (
        tqdm(pairs, total=total, desc="StepD layer×head", unit="head")
        if tqdm is not None
        else None
    )

    for idx_pair, (layer, head) in enumerate(pairs, start=1):
        if pbar is not None:
            pbar.set_postfix({"layer": layer, "head": head, "trials": args.n_trials})
        if idx_pair == 1 or idx_pair % log_every == 0 or idx_pair == total:
            elapsed = time.time() - start_time
            rate = idx_pair / elapsed if elapsed > 0 else 0.0
            remaining = total - idx_pair
            eta = remaining / rate if rate > 0 else 0.0
            percent = (idx_pair / total) * 100 if total else 100.0
            log(
                "[StepD] "
                f"{percent:.1f}% ({idx_pair}/{total}) "
                f"layer={layer} head={head} "
                f"elapsed={format_duration(elapsed)} "
                f"ETA={format_duration(eta)}"
            )

        metric_lists = {
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
        nonzero = False
        replace_vec = mean_acts[layer, head, slot_q]
        hook_state = {"mode": "self", "replace_vec": None, "seq_token_idx": None}
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
        for trial in trials:
            prefix_str = trial["corrupted_prefix_str"]
            target_id = trial["target_id"]
            seq_token_idx = trial["seq_token_idx"]
            hook_state["seq_token_idx"] = seq_token_idx

            inputs = tokenizer(
                prefix_str, return_tensors="pt", add_special_tokens=tok_add_special
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hook_state["mode"] = "self"
            hook_state["replace_vec"] = None
            handle_base = layer_modules[layer].register_forward_hook(hook)
            with torch.inference_mode():
                outputs = model(**inputs)
            handle_base.remove()
            baseline_logits = outputs.logits[0, -1]
            if idx_pair == 1 and trial["trial_idx"] == 0:
                target_token = tokenizer.convert_ids_to_tokens(target_id)
                prompt_tail = repr(prefix_str[-20:])
                log(
                    "target_debug: "
                    f"prompt_tail={prompt_tail} "
                    f"target_token={target_token} target_id={target_id}"
                )
            if not patch_token_logged:
                prefix_ids = inputs["input_ids"][0].tolist()
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

            hook_state["mode"] = "replace"
            hook_state["replace_vec"] = replace_vec
            handle = layer_modules[layer].register_forward_hook(hook)
            with torch.inference_mode():
                outputs_patched = model(**inputs)
            handle.remove()

            patched_logits = outputs_patched.logits[0, -1]
            trial_metrics = compute_trial_metrics(
                baseline_logits, patched_logits, target_id
            )
            for key, value in trial_metrics.items():
                metric_lists[key].append(value)
            if abs(trial_metrics["delta_p"]) > 1e-12:
                nonzero = True

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
            trial_metrics_file.write(
                json.dumps(trial_row, ensure_ascii=True, default=_json_safe) + "\n"
            )

            if dump_handle is not None:
                if (args.dump_layer is None or args.dump_layer == layer) and (
                    args.dump_head is None or args.dump_head == head
                ):
                    if args.dump_max_trials < 0 or trial["trial_idx"] < args.dump_max_trials:
                        ids_list = inputs["input_ids"][0].tolist()
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
                            json.dumps(dump_row, ensure_ascii=True, default=_json_safe) + "\n"
                        )

            if args.save_trials:
                trials_rows.append(
                    {
                        "trial_idx": trial["trial_idx"],
                        "layer": layer,
                        "head": head,
                        "target_id": target_id,
                        "p_base": trial_metrics["p_base"],
                        "p_patch": trial_metrics["p_patch"],
                        "delta_p": trial_metrics["delta_p"],
                        "logit_base": trial_metrics["logit_base"],
                        "logit_patch": trial_metrics["logit_patch"],
                        "delta_logit": trial_metrics["delta_logit"],
                        "logprob_base": trial_metrics["logprob_base"],
                        "logprob_patch": trial_metrics["logprob_patch"],
                        "delta_logprob": trial_metrics["delta_logprob"],
                    }
                )

        mean_act_norm = mean_acts[layer, head, slot_q].norm().item()
        mean_delta_p = mean(metric_lists["delta_p"])
        std_delta_p = std(metric_lists["delta_p"])
        mean_abs_delta_p = mean_abs(metric_lists["delta_p"])
        mean_p_base = mean(metric_lists["p_base"])
        std_p_base = std(metric_lists["p_base"])
        mean_p_patch = mean(metric_lists["p_patch"])
        std_p_patch = std(metric_lists["p_patch"])

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
        scores.append(
            {
                "layer": layer,
                "head": head,
                "n_trials": args.n_trials,
                "mean_delta_p": mean_delta_p,
                "std_delta_p": std_delta_p,
                "mean_abs_delta_p": mean_abs_delta_p,
                "mean_p_base": mean_p_base,
                "std_p_base": std_p_base,
                "mean_p_patch": mean_p_patch,
                "std_p_patch": std_p_patch,
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
        )
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()

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
            },
            "scores": scores_sorted,
        },
    )

    log(f"saved scores: {scores_path}")
    log(f"saved scores json: {scores_json_path}")

    if args.save_trials:
        trials_path = os.path.join(artifacts_dir, "aie_trials.csv")
        save_csv(trials_path, trials_rows)
        log(f"saved trials: {trials_path}")

    trial_metrics_file.close()
    log(f"saved trial metrics: {trial_metrics_path}")
    if dump_handle is not None:
        dump_handle.close()

    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
