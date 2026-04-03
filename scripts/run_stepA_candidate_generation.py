#!/usr/bin/env python3
"""Step A candidate generation on sampled trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PROMPT_TEMPLATE = """You are given an input word.
Generate {k} candidates that are semantically related to the input word, while avoiding trivial lexical variants.
Input word: {query_input}
{gold_line}
Rules:
- Output only candidates, one per line.
- No numbering, no bullets, no explanations.
- Single-word only.
- Prefer nouns.
- Do not output synonyms, inflections, or spelling variants of the input word.
"""

BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*]+|\d+[.)])\s*")
MULTISPACE_RE = re.compile(r"\s+")
NUMERIC_ONLY_RE = re.compile(r"^\d+$")


def _now_utc() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _mean(values: Sequence[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _safe_json_dump(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _canonical_query_uid(q_id: str, query_input: str) -> str:
    blob = json.dumps(
        {"q_id": q_id, "query_input": query_input},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    stripped = MULTISPACE_RE.sub(" ", value.strip())
    return stripped.lower()


def _strip_list_prefix(line: str) -> str:
    return BULLET_PREFIX_RE.sub("", line).strip()


def _is_punctuation_only(value: str) -> bool:
    if not value:
        return False
    return all(not ch.isalnum() for ch in value)


def _compute_first_token_id(text: str, tokenizer) -> Optional[int]:
    encoded = tokenizer.encode(" " + text, add_special_tokens=False)
    if not encoded:
        return None
    return int(encoded[0])


def _build_prompt(
    template: str,
    q_id: str,
    query_input: str,
    gold_target: str,
    k: int,
    include_gold: bool,
) -> str:
    gold_line = ""
    if include_gold:
        gold_line = f"Known target word (do not output): {gold_target}"
    try:
        return template.format(
            q_id=q_id,
            query_input=query_input,
            gold_target=gold_target,
            k=k,
            gold_line=gold_line,
        )
    except KeyError as exc:
        raise ValueError(f"Prompt template is missing placeholder: {exc}") from exc


def _parse_generation_text(raw_text: str) -> List[str]:
    candidates: List[str] = []
    for line in raw_text.splitlines():
        cleaned = _strip_list_prefix(line)
        if not cleaned:
            continue
        candidates.append(cleaned)
    return candidates


def _trivial_query_variant(candidate: str, query_input: str) -> bool:
    if candidate == query_input:
        return True
    for suffix in ("s", "es", "ed", "ing"):
        if candidate == query_input + suffix:
            return True
    if len(candidate) >= 3 and candidate in query_input:
        return True
    if len(query_input) >= 3 and query_input in candidate:
        return True
    return False


def _classify_exception(exc: Exception) -> str:
    msg = str(exc).lower()
    if "out of memory" in msg:
        return "oom"
    if "timeout" in msg:
        return "timeout"
    if "parse" in msg:
        return "parse_error"
    return "other"


@dataclass
class TrialInputRow:
    q_id: str
    trial_local_idx: int
    query_input: str
    gold_target: str
    query_source_index: Optional[int]
    query_uid: str


@dataclass
class QueryWorkItem:
    q_id: str
    query_input: str
    gold_target: str
    query_uid: str
    source_trial_indices: List[int] = field(default_factory=list)
    query_source_indices: List[int] = field(default_factory=list)
    prompt_used: Optional[str] = None
    raw_generations: List[str] = field(default_factory=list)
    normalized_candidates: List[str] = field(default_factory=list)
    accepted_candidates: List[str] = field(default_factory=list)
    accepted_candidate_first_token_ids: List[int] = field(default_factory=list)
    accepted_candidate_first_token_logprobs: List[float] = field(default_factory=list)
    accepted_candidate_first_token_probs: List[float] = field(default_factory=list)
    rejected: List[Dict[str, str]] = field(default_factory=list)
    n_attempt_rounds: int = 0
    complete_10: bool = False
    fail_reason: Optional[str] = None


def _load_trials(path: str) -> List[TrialInputRow]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("trials", payload)
    if not isinstance(rows, list):
        raise ValueError("sampled_trials file must include a 'trials' list")
    parsed: List[TrialInputRow] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        q_id = str(row.get("q_id", "")).strip()
        query = row.get("query")
        if not q_id or not isinstance(query, dict):
            continue
        x_val = query.get("input")
        y_val = query.get("output")
        if not isinstance(x_val, str) or not isinstance(y_val, str):
            continue
        query_input = x_val.strip()
        if not query_input:
            continue
        raw_target = row.get("target_str")
        if isinstance(raw_target, str) and raw_target.strip():
            gold_target = raw_target.strip()
        else:
            gold_target = y_val.strip()
        if not gold_target:
            continue
        source_idx = row.get("query_source_index")
        if source_idx is None:
            query_source_index = None
        else:
            try:
                query_source_index = int(source_idx)
            except (TypeError, ValueError):
                query_source_index = None
        query_uid = _canonical_query_uid(q_id, query_input)
        parsed.append(
            TrialInputRow(
                q_id=q_id,
                trial_local_idx=idx,
                query_input=query_input,
                gold_target=gold_target,
                query_source_index=query_source_index,
                query_uid=query_uid,
            )
        )
    if not parsed:
        raise ValueError("No valid trial rows found in sampled_trials input")
    return parsed


def _prepare_query_items(trials: List[TrialInputRow]) -> Dict[str, QueryWorkItem]:
    items: Dict[str, QueryWorkItem] = {}
    for row in trials:
        item = items.get(row.query_uid)
        if item is None:
            item = QueryWorkItem(
                q_id=row.q_id,
                query_input=row.query_input,
                gold_target=row.gold_target,
                query_uid=row.query_uid,
            )
            items[row.query_uid] = item
        else:
            if item.q_id != row.q_id or item.query_input != row.query_input:
                raise ValueError("query_uid collision with non-identical tuple")
            if item.gold_target != row.gold_target:
                raise ValueError(
                    "Same (q_id, query_input) has conflicting gold_target values"
                )
        item.source_trial_indices.append(row.trial_local_idx)
        if row.query_source_index is not None:
            item.query_source_indices.append(row.query_source_index)
    return items


def _generate_text_batch(
    *,
    model,
    tokenizer,
    prompt: str,
    tok_add_special: bool,
    input_device,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> List[str]:
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=tok_add_special,
    )
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": batch_size,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    texts: List[str] = []
    for seq_idx in range(output_ids.shape[0]):
        gen_part = output_ids[seq_idx, input_len:]
        texts.append(tokenizer.decode(gen_part, skip_special_tokens=True))
    return texts


def _compute_prompt_next_logprobs(
    *,
    model,
    tokenizer,
    prompt: str,
    tok_add_special: bool,
    input_device,
):
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=tok_add_special,
    )
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    next_logits = outputs.logits[0, -1, :]
    return torch.log_softmax(next_logits, dim=-1)


def _load_prompt_template(path: Optional[str]) -> str:
    if path is None:
        return DEFAULT_PROMPT_TEMPLATE
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if not text.strip():
        raise ValueError("Prompt template file is empty")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Step A candidate generation.")
    parser.add_argument("--sampled_trials_path", required=True)
    parser.add_argument("--model_path", default="/scratch/sunsik/models/Llama-3.1-70B")
    parser.add_argument("--model_spec", default="llama3")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default=None, choices=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--quant",
        default="auto",
        choices=["auto", "none", "4bit", "8bit"],
    )
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--n_candidates", type=int, default=10)
    parser.add_argument("--max_attempt_rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--do_sample", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--prompt_template_path", default=None)
    parser.add_argument("--include_gold_in_prompt", action="store_true")
    parser.add_argument("--require_complete_10", action="store_true")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    if args.n_candidates < 1:
        print("n_candidates must be >= 1")
        return 1
    if args.max_attempt_rounds < 1:
        print("max_attempt_rounds must be >= 1")
        return 1
    if args.batch_size < 1:
        print("batch_size must be >= 1")
        return 1
    if args.max_new_tokens < 1:
        print("max_new_tokens must be >= 1")
        return 1
    do_sample = bool(args.do_sample)
    if not do_sample and args.batch_size > 1:
        print("batch_size > 1 requires do_sample=1")
        return 1

    try:
        from fv.hf_loader import load_hf_model_and_tokenizer
        from fv.io import resolve_out_dir
        from fv.tokenization import resolve_prompt_add_special_tokens
    except Exception as exc:
        print(f"Failed to import fv helpers: {exc}")
        return 1

    out_dir = resolve_out_dir(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed)

    trials_path = resolve_out_dir(args.sampled_trials_path)
    try:
        trials = _load_trials(trials_path)
    except Exception as exc:
        print(f"Failed to parse sampled trials: {exc}")
        return 1

    try:
        template = _load_prompt_template(args.prompt_template_path)
    except Exception as exc:
        print(f"Failed to load prompt template: {exc}")
        return 1

    query_items = _prepare_query_items(trials)
    query_list = sorted(query_items.values(), key=lambda row: row.query_uid)

    try:
        import torch
        import transformers
    except Exception as exc:
        print(f"Failed to import torch/transformers: {exc}")
        return 1

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype is None:
        args.dtype = "fp16" if args.device == "cuda" else "fp32"
    if args.device == "cpu" and args.dtype in {"fp16", "bf16"}:
        print(f"cpu does not support {args.dtype}; forcing fp32")
        args.dtype = "fp32"
    tok_add_special = resolve_prompt_add_special_tokens(args.model_path, args.model_spec)

    try:
        loader_device = None if args.device_map else args.device
        model, tokenizer, diagnostics = load_hf_model_and_tokenizer(
            model_name=args.model_path,
            model_spec=args.model_spec,
            device=loader_device,
            dtype=args.dtype,
            quant=args.quant,
            device_map=args.device_map,
        )
    except Exception as exc:
        print(f"Failed to load model/tokenizer: {exc}")
        return 1

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    resolved_quant = diagnostics.get("resolved_quant")
    if args.device_map:
        try:
            input_device = next(model.parameters()).device
        except StopIteration:
            input_device = torch.device(args.device)
    elif resolved_quant in {"4bit", "8bit"}:
        try:
            input_device = next(model.parameters()).device
        except StopIteration:
            input_device = torch.device(args.device)
    else:
        input_device = torch.device(args.device)
        model.to(input_device)
    model.eval()

    print(f"loaded queries: {len(query_list)}")
    print(f"loaded trials: {len(trials)}")
    print(
        "loader diagnostics: "
        + " ".join(f"{k}={v}" for k, v in diagnostics.items())
    )

    fail_reason_counts = {
        "oom": 0,
        "timeout": 0,
        "empty": 0,
        "parse_error": 0,
        "too_many_duplicates": 0,
        "other": 0,
    }
    round_candidate_counts: List[float] = []
    round_accept_rates: List[float] = []
    strict_mode_failed = False

    for idx, item in enumerate(query_list, start=1):
        query_seed = args.seed + int(item.query_uid[:8], 16)
        random.seed(query_seed)
        torch.manual_seed(query_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(query_seed)
        text_seen = set()
        token_seen = set()
        total_parsed = 0
        item.prompt_used = None
        print(
            f"[{idx}/{len(query_list)}] q_id={item.q_id} query='{item.query_input}'"
        )
        for round_idx in range(1, args.max_attempt_rounds + 1):
            missing = args.n_candidates - len(item.accepted_candidates)
            if missing <= 0:
                break
            k_request = 20 if round_idx == 1 else missing * 3
            prompt = _build_prompt(
                template=template,
                q_id=item.q_id,
                query_input=item.query_input,
                gold_target=item.gold_target,
                k=k_request,
                include_gold=args.include_gold_in_prompt,
            )
            if item.prompt_used is None:
                item.prompt_used = prompt
            try:
                prompt_next_logprobs = _compute_prompt_next_logprobs(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    tok_add_special=tok_add_special,
                    input_device=input_device,
                )
                generated_texts = _generate_text_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    tok_add_special=tok_add_special,
                    input_device=input_device,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            except Exception as exc:
                item.fail_reason = _classify_exception(exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break

            parsed_this_round = 0
            accepted_this_round = 0
            for raw in generated_texts:
                item.raw_generations.append(raw)
                parsed_candidates = _parse_generation_text(raw)
                parsed_this_round += len(parsed_candidates)
                for candidate in parsed_candidates:
                    normalized = _normalize_text(candidate)
                    if normalized and normalized not in item.normalized_candidates:
                        item.normalized_candidates.append(normalized)
                    if not normalized:
                        item.rejected.append({"candidate": candidate, "reason": "empty"})
                        continue
                    if _is_punctuation_only(normalized):
                        item.rejected.append(
                            {"candidate": normalized, "reason": "punctuation_only"}
                        )
                        continue
                    if NUMERIC_ONLY_RE.fullmatch(normalized):
                        item.rejected.append({"candidate": normalized, "reason": "numeric_only"})
                        continue
                    if len(normalized) <= 2:
                        item.rejected.append({"candidate": normalized, "reason": "too_short"})
                        continue
                    if " " in normalized:
                        item.rejected.append({"candidate": normalized, "reason": "multiword"})
                        continue
                    query_norm = _normalize_text(item.query_input)
                    if _trivial_query_variant(normalized, query_norm):
                        item.rejected.append(
                            {"candidate": normalized, "reason": "query_variant"}
                        )
                        continue
                    gold_norm = _normalize_text(item.gold_target)
                    if normalized == gold_norm:
                        item.rejected.append({"candidate": normalized, "reason": "gold_target"})
                        continue
                    if normalized in text_seen:
                        item.rejected.append(
                            {"candidate": normalized, "reason": "text_duplicate"}
                        )
                        continue
                    token_id = _compute_first_token_id(normalized, tokenizer)
                    if token_id is None:
                        item.rejected.append(
                            {"candidate": normalized, "reason": "tokenization_empty"}
                        )
                        continue
                    if token_id in token_seen:
                        item.rejected.append(
                            {"candidate": normalized, "reason": "first_token_duplicate"}
                        )
                        continue
                    text_seen.add(normalized)
                    token_seen.add(token_id)
                    item.accepted_candidates.append(normalized)
                    item.accepted_candidate_first_token_ids.append(token_id)
                    token_logprob = float(prompt_next_logprobs[token_id].item())
                    item.accepted_candidate_first_token_logprobs.append(token_logprob)
                    item.accepted_candidate_first_token_probs.append(float(math.exp(token_logprob)))
                    accepted_this_round += 1
                    if len(item.accepted_candidates) >= args.n_candidates:
                        break
                if len(item.accepted_candidates) >= args.n_candidates:
                    break
            item.n_attempt_rounds = round_idx
            total_parsed += parsed_this_round
            round_candidate_counts.append(float(parsed_this_round))
            if parsed_this_round > 0:
                round_accept_rates.append(accepted_this_round / parsed_this_round)
            if len(item.accepted_candidates) >= args.n_candidates:
                break

        item.complete_10 = len(item.accepted_candidates) >= args.n_candidates
        if not item.complete_10:
            if item.fail_reason is None:
                item.fail_reason = "empty" if total_parsed == 0 else "too_many_duplicates"
            fail_reason_counts[item.fail_reason] = fail_reason_counts.get(item.fail_reason, 0) + 1
            if args.require_complete_10:
                strict_mode_failed = True

    trial_rows: List[Dict[str, object]] = []
    for row in trials:
        item = query_items[row.query_uid]
        gold_token_id = _compute_first_token_id(_normalize_text(item.gold_target), tokenizer)
        trial_rows.append(
            {
                "q_id": item.q_id,
                "trial_local_idx": row.trial_local_idx,
                "query_input": item.query_input,
                "gold_target": item.gold_target,
                "query_source_index": row.query_source_index,
                "query_uid": item.query_uid,
                "gold_target_first_token_id": gold_token_id,
                "prompt_used": item.prompt_used,
                "raw_generations": item.raw_generations,
                "normalized_candidates": item.normalized_candidates,
                "accepted_candidates": item.accepted_candidates,
                "accepted_candidate_first_token_ids": item.accepted_candidate_first_token_ids,
                "accepted_candidate_first_token_logprobs": item.accepted_candidate_first_token_logprobs,
                "accepted_candidate_first_token_probs": item.accepted_candidate_first_token_probs,
                "rejected": item.rejected,
                "n_attempt_rounds": item.n_attempt_rounds,
                "complete_10": item.complete_10,
            }
        )

    query_cache_records: List[Dict[str, object]] = []
    for item in sorted(query_list, key=lambda x: x.query_uid):
        gold_token_id = _compute_first_token_id(_normalize_text(item.gold_target), tokenizer)
        query_cache_records.append(
            {
                "query_uid": item.query_uid,
                "q_id": item.q_id,
                "query_input": item.query_input,
                "gold_target": item.gold_target,
                "gold_target_first_token_id": gold_token_id,
                "candidates": item.accepted_candidates,
                "candidate_first_token_ids": item.accepted_candidate_first_token_ids,
                "candidate_first_token_logprobs": item.accepted_candidate_first_token_logprobs,
                "candidate_first_token_probs": item.accepted_candidate_first_token_probs,
                "source_trial_indices": sorted(item.source_trial_indices),
            }
        )

    complete_count = sum(1 for item in query_list if item.complete_10)
    stats_payload = {
        "n_queries_total": len(query_list),
        "n_queries_complete_10": complete_count,
        "complete_10_rate": (complete_count / len(query_list)) if query_list else 0.0,
        "avg_attempt_rounds": _mean([item.n_attempt_rounds for item in query_list]),
        "avg_candidates_per_round_before_filter": _mean(round_candidate_counts),
        "avg_accept_rate_after_filter": _mean(round_accept_rates),
        "fail_reason_counts": fail_reason_counts,
        "strict_mode_failed": bool(strict_mode_failed),
        "created_at": _now_utc(),
    }

    run_meta_payload = {
        "created_at": _now_utc(),
        "sampled_trials_path": trials_path,
        "out_dir": out_dir,
        "model_path": args.model_path,
        "model_spec": args.model_spec,
        "seed": args.seed,
        "n_candidates": args.n_candidates,
        "max_attempt_rounds": args.max_attempt_rounds,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(do_sample),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "include_gold_in_prompt": bool(args.include_gold_in_prompt),
        "require_complete_10": bool(args.require_complete_10),
        "device": args.device,
        "dtype": args.dtype,
        "quant": args.quant,
        "device_map": args.device_map,
        "loader_diagnostics": diagnostics,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tok_add_special_prompt": bool(tok_add_special),
        "tokenization_policy": {
            "first_token_prefix_space": True,
            "first_token_add_special_tokens": False,
            "normalization": "lowercase_trim_collapse_whitespace",
            "candidate_first_token_logprob_scope": "next_token_after_generation_prompt",
        },
        "environment": {
            "python": sys.version,
            "torch": getattr(torch, "__version__", None),
            "transformers": getattr(transformers, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
        },
    }

    jsonl_path = os.path.join(out_dir, "stepA_candidates.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for row in trial_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    by_query_path = os.path.join(out_dir, "stepA_candidates_by_query.json")
    run_meta_path = os.path.join(out_dir, "stepA_run_meta.json")
    stats_path = os.path.join(out_dir, "stepA_stats.json")
    _safe_json_dump(by_query_path, query_cache_records)
    _safe_json_dump(run_meta_path, run_meta_payload)
    _safe_json_dump(stats_path, stats_payload)

    print(f"saved: {jsonl_path}")
    print(f"saved: {by_query_path}")
    print(f"saved: {run_meta_path}")
    print(f"saved: {stats_path}")

    if strict_mode_failed:
        print("strict mode failure: at least one query did not reach n_candidates")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
