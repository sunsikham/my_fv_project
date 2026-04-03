#!/usr/bin/env python3
"""StepA shot-sweep candidate generation with relationA/relationB demo pools."""

from __future__ import annotations

import argparse
import csv
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PROMPT_TEMPLATE = """You are given an input word and relation examples.
Relation source: {relation_source}
Shot count: {shot}
Examples:
{demo_block}
Input word: {query_input}
{gold_line}
Generate {k} candidate output words for the input under the same relation.
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


def _stable_seed(*parts: object) -> int:
    joined = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


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


def _resolve_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    col_set = {c.strip(): c for c in fieldnames}
    for cand in candidates:
        if cand in col_set:
            return col_set[cand]
    lower_map = {c.strip().lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _read_relation_csv(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row in {path}")
        col_q = _resolve_column(reader.fieldnames, ["q_id", "q", "id", "qid"])
        col_in = _resolve_column(reader.fieldnames, ["input", "exA", "ex_A", "ex_a"])
        col_out = _resolve_column(reader.fieldnames, ["output", "exB", "ex_B", "ex_b"])
        if col_q is None or col_in is None or col_out is None:
            raise ValueError(
                "CSV missing required columns (q_id/q/id + input/exA + output/exB) "
                f"in {path}"
            )
        rows = []
        for row_idx, row in enumerate(reader):
            q_id = str(row.get(col_q, "")).strip()
            inp = str(row.get(col_in, "")).strip()
            out = str(row.get(col_out, "")).strip()
            if not q_id or not inp or not out:
                continue
            rows.append({"row_id": row_idx, "q_id": q_id, "input": inp, "output": out})
        return rows


def _group_by_qid(rows: Iterable[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["q_id"]), []).append(row)
    return grouped


def _parse_csv_list(raw: str) -> List[str]:
    vals = [part.strip() for part in (raw or "").split(",") if part.strip()]
    out: List[str] = []
    seen = set()
    for val in vals:
        key = val.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _parse_demo_relations(raw: str) -> List[str]:
    rels = _parse_csv_list(raw)
    if not rels:
        raise ValueError("--demo_relations is empty")
    for rel in rels:
        if rel not in {"A", "B"}:
            raise ValueError("--demo_relations supports only A and B")
    return rels


def _parse_shot_list(raw: str) -> List[int]:
    parts = [part.strip() for part in (raw or "").split(",") if part.strip()]
    shots: List[int] = []
    seen = set()
    for part in parts:
        shot = int(part)
        if shot < 1:
            raise ValueError("--shot_list values must be >= 1")
        if shot in seen:
            continue
        seen.add(shot)
        shots.append(shot)
    if not shots:
        raise ValueError("--shot_list is empty")
    return shots


def _jaccard(lhs: Sequence[object], rhs: Sequence[object]) -> float:
    left = set(lhs)
    right = set(rhs)
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _pairwise_shot_pairs(shots: Sequence[int], mode: str) -> List[Tuple[int, int]]:
    ordered = sorted(set(int(x) for x in shots))
    if len(ordered) < 2:
        return []
    if mode == "adjacent":
        return list(zip(ordered[:-1], ordered[1:]))
    pairs: List[Tuple[int, int]] = []
    for i, shot_a in enumerate(ordered):
        for shot_b in ordered[i + 1 :]:
            pairs.append((shot_a, shot_b))
    return pairs


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
        query_input = str(query.get("input", "")).strip()
        if not query_input:
            continue
        raw_target = row.get("target_str")
        if isinstance(raw_target, str) and raw_target.strip():
            gold_target = raw_target.strip()
        else:
            gold_target = str(query.get("output", "")).strip()
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
                raise ValueError("Same (q_id, query_input) has conflicting gold_target values")
        item.source_trial_indices.append(row.trial_local_idx)
        if row.query_source_index is not None:
            item.query_source_indices.append(row.query_source_index)
    return items


def _format_demo_block(demos: Sequence[Tuple[str, str]]) -> str:
    if not demos:
        return "(no examples)"
    return "\n".join(f"- {inp} -> {out}" for inp, out in demos)


def _build_prompt(
    *,
    template: str,
    q_id: str,
    query_input: str,
    gold_target: str,
    k: int,
    include_gold: bool,
    demos: Sequence[Tuple[str, str]],
    shot: int,
    relation_source: str,
) -> str:
    gold_line = ""
    if include_gold:
        gold_line = f"Known target word (do not output): {gold_target}"
    demo_block = _format_demo_block(demos)
    try:
        return template.format(
            q_id=q_id,
            query_input=query_input,
            gold_target=gold_target,
            k=k,
            gold_line=gold_line,
            demo_block=demo_block,
            shot=shot,
            relation_source=relation_source,
        )
    except KeyError as exc:
        raise ValueError(f"Prompt template is missing placeholder: {exc}") from exc


def _parse_generation_text(raw_text: str) -> List[str]:
    candidates: List[str] = []
    for line in raw_text.splitlines():
        cleaned = BULLET_PREFIX_RE.sub("", line).strip()
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


def _run_single_shot(
    *,
    model,
    tokenizer,
    input_device,
    tok_add_special: bool,
    template: str,
    q_id: str,
    query_uid: str,
    query_input: str,
    gold_target: str,
    relation_source: str,
    shot: int,
    demos: Sequence[Tuple[str, str]],
    n_candidates: int,
    max_attempt_rounds: int,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    include_gold_in_prompt: bool,
    seed: int,
):
    import torch

    prompt_used: Optional[str] = None
    raw_generations: List[str] = []
    normalized_candidates: List[str] = []
    accepted_candidates: List[str] = []
    accepted_candidate_first_token_ids: List[int] = []
    accepted_candidate_first_token_logprobs: List[float] = []
    accepted_candidate_first_token_probs: List[float] = []
    rejected: List[Dict[str, str]] = []
    n_attempt_rounds = 0
    fail_reason: Optional[str] = None

    text_seen = set()
    token_seen = set()
    total_parsed = 0
    query_norm = _normalize_text(query_input)
    gold_norm = _normalize_text(gold_target)

    for round_idx in range(1, max_attempt_rounds + 1):
        missing = n_candidates - len(accepted_candidates)
        if missing <= 0:
            break
        k_request = 20 if round_idx == 1 else missing * 3
        prompt = _build_prompt(
            template=template,
            q_id=q_id,
            query_input=query_input,
            gold_target=gold_target,
            k=k_request,
            include_gold=include_gold_in_prompt,
            demos=demos,
            shot=shot,
            relation_source=relation_source,
        )
        if prompt_used is None:
            prompt_used = prompt

        round_seed = _stable_seed(seed, query_uid, relation_source, shot, round_idx)
        random.seed(round_seed)
        torch.manual_seed(round_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(round_seed)

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
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as exc:
            fail_reason = _classify_exception(exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break

        parsed_this_round = 0
        for raw in generated_texts:
            raw_generations.append(raw)
            parsed_candidates = _parse_generation_text(raw)
            parsed_this_round += len(parsed_candidates)
            for candidate in parsed_candidates:
                normalized = _normalize_text(candidate)
                if normalized and normalized not in normalized_candidates:
                    normalized_candidates.append(normalized)
                if not normalized:
                    rejected.append({"candidate": candidate, "reason": "empty"})
                    continue
                if _is_punctuation_only(normalized):
                    rejected.append({"candidate": normalized, "reason": "punctuation_only"})
                    continue
                if NUMERIC_ONLY_RE.fullmatch(normalized):
                    rejected.append({"candidate": normalized, "reason": "numeric_only"})
                    continue
                if len(normalized) <= 2:
                    rejected.append({"candidate": normalized, "reason": "too_short"})
                    continue
                if " " in normalized:
                    rejected.append({"candidate": normalized, "reason": "multiword"})
                    continue
                if _trivial_query_variant(normalized, query_norm):
                    rejected.append({"candidate": normalized, "reason": "query_variant"})
                    continue
                if normalized == gold_norm:
                    rejected.append({"candidate": normalized, "reason": "gold_target"})
                    continue
                if normalized in text_seen:
                    rejected.append({"candidate": normalized, "reason": "text_duplicate"})
                    continue
                token_id = _compute_first_token_id(normalized, tokenizer)
                if token_id is None:
                    rejected.append({"candidate": normalized, "reason": "tokenization_empty"})
                    continue
                if token_id in token_seen:
                    rejected.append({"candidate": normalized, "reason": "first_token_duplicate"})
                    continue
                token_logprob = float(prompt_next_logprobs[token_id].item())
                text_seen.add(normalized)
                token_seen.add(token_id)
                accepted_candidates.append(normalized)
                accepted_candidate_first_token_ids.append(token_id)
                accepted_candidate_first_token_logprobs.append(token_logprob)
                accepted_candidate_first_token_probs.append(float(math.exp(token_logprob)))
                if len(accepted_candidates) >= n_candidates:
                    break
            if len(accepted_candidates) >= n_candidates:
                break

        n_attempt_rounds = round_idx
        total_parsed += parsed_this_round
        if len(accepted_candidates) >= n_candidates:
            break

    complete_n = len(accepted_candidates) >= n_candidates
    if not complete_n and fail_reason is None:
        fail_reason = "empty" if total_parsed == 0 else "too_many_duplicates"

    return {
        "relation_source": relation_source,
        "shot": int(shot),
        "prompt_used": prompt_used,
        "raw_generations": raw_generations,
        "normalized_candidates": normalized_candidates,
        "candidates": accepted_candidates,
        "candidate_first_token_ids": accepted_candidate_first_token_ids,
        "candidate_first_token_logprobs": accepted_candidate_first_token_logprobs,
        "candidate_first_token_probs": accepted_candidate_first_token_probs,
        "rejected": rejected,
        "n_attempt_rounds": int(n_attempt_rounds),
        "complete_n": bool(complete_n),
        "fail_reason": fail_reason,
        "n_candidates_found": int(len(accepted_candidates)),
        "demos_used": [{"input": x, "output": y} for x, y in demos],
    }


def _write_csv(path: str, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_prompt_template(path: Optional[str]) -> str:
    if path is None:
        return DEFAULT_PROMPT_TEMPLATE
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if not text.strip():
        raise ValueError("Prompt template file is empty")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="StepA shot-sweep candidate generation for relationA_ex/relationB_ex."
    )
    parser.add_argument("--sampled_trials_path", required=True)
    parser.add_argument("--relationA_ex_path", required=True)
    parser.add_argument("--relationB_ex_path", required=True)
    parser.add_argument("--demo_relations", default="A,B")
    parser.add_argument("--shot_list", default="1,3,5,7,10")
    parser.add_argument("--shot_eligibility", default="strict", choices=["strict", "adaptive"])
    parser.add_argument("--demo_pool_size", type=int, default=None)
    parser.add_argument("--compare_mode", default="adjacent", choices=["adjacent", "all_pairs"])
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
        demo_relations = _parse_demo_relations(args.demo_relations)
        shots = _parse_shot_list(args.shot_list)
    except Exception as exc:
        print(f"Invalid list argument: {exc}")
        return 1
    max_shot = max(shots)
    if args.demo_pool_size is None:
        args.demo_pool_size = max_shot
    if args.demo_pool_size < max_shot:
        print("demo_pool_size must be >= max(shot_list)")
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

    sampled_trials_path = resolve_out_dir(args.sampled_trials_path)
    relationA_ex_path = resolve_out_dir(args.relationA_ex_path)
    relationB_ex_path = resolve_out_dir(args.relationB_ex_path)

    try:
        trials = _load_trials(sampled_trials_path)
    except Exception as exc:
        print(f"Failed to parse sampled trials: {exc}")
        return 1

    try:
        rel_rows_a = _read_relation_csv(relationA_ex_path)
        rel_rows_b = _read_relation_csv(relationB_ex_path)
    except Exception as exc:
        print(f"Failed to parse relation CSV: {exc}")
        return 1
    rel_by_q = {"A": _group_by_qid(rel_rows_a), "B": _group_by_qid(rel_rows_b)}

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

    query_records: List[Dict[str, object]] = []
    shot_rows: List[Dict[str, object]] = []
    skip_records: List[Dict[str, object]] = []

    for idx, item in enumerate(query_list, start=1):
        print(f"[{idx}/{len(query_list)}] q_id={item.q_id} query='{item.query_input}'")
        query_record: Dict[str, object] = {
            "query_uid": item.query_uid,
            "q_id": item.q_id,
            "query_input": item.query_input,
            "gold_target": item.gold_target,
            "source_trial_indices": sorted(item.source_trial_indices),
            "query_source_indices": sorted(set(item.query_source_indices)),
            "relations": {},
        }

        for relation_source in demo_relations:
            relation_rows = rel_by_q[relation_source].get(item.q_id, [])
            if not relation_rows:
                skip_records.append(
                    {
                        "query_uid": item.query_uid,
                        "q_id": item.q_id,
                        "query_input": item.query_input,
                        "relation_source": relation_source,
                        "reason": "missing_relation_qid",
                    }
                )
                continue

            filtered_rows = [
                row
                for row in relation_rows
                if not (
                    str(row["input"]).strip() == item.query_input
                    and str(row["output"]).strip() == item.gold_target
                )
            ]
            if not filtered_rows:
                skip_records.append(
                    {
                        "query_uid": item.query_uid,
                        "q_id": item.q_id,
                        "query_input": item.query_input,
                        "relation_source": relation_source,
                        "reason": "empty_demo_pool_after_filter",
                    }
                )
                continue

            if args.shot_eligibility == "strict" and len(filtered_rows) < max_shot:
                skip_records.append(
                    {
                        "query_uid": item.query_uid,
                        "q_id": item.q_id,
                        "query_input": item.query_input,
                        "relation_source": relation_source,
                        "reason": "insufficient_pool_for_strict",
                        "pool_size": len(filtered_rows),
                        "required_shot": max_shot,
                    }
                )
                continue

            # Deterministic demo order per query/relation.
            ordered_rows = sorted(
                filtered_rows,
                key=lambda row: (str(row["input"]), str(row["output"]), int(row["row_id"])),
            )
            rng_demo = random.Random(_stable_seed(args.seed, item.query_uid, relation_source, "demo_pool"))
            if len(ordered_rows) > args.demo_pool_size:
                selected_rows = rng_demo.sample(ordered_rows, args.demo_pool_size)
            else:
                selected_rows = list(ordered_rows)
            rng_demo.shuffle(selected_rows)

            if args.shot_eligibility == "strict":
                eligible_shots = list(shots)
            else:
                eligible_shots = [shot for shot in shots if shot <= len(selected_rows)]
            if not eligible_shots:
                skip_records.append(
                    {
                        "query_uid": item.query_uid,
                        "q_id": item.q_id,
                        "query_input": item.query_input,
                        "relation_source": relation_source,
                        "reason": "no_eligible_shots",
                        "pool_size": len(selected_rows),
                    }
                )
                continue

            relation_result: Dict[str, object] = {
                "relation_source": relation_source,
                "pool_size_before_filter": len(relation_rows),
                "pool_size_after_filter": len(filtered_rows),
                "selected_demo_pool_size": len(selected_rows),
                "shots": {},
            }

            for shot in eligible_shots:
                demos_rows = selected_rows[:shot]
                demos_pairs = [
                    (str(row["input"]).strip(), str(row["output"]).strip()) for row in demos_rows
                ]
                shot_result = _run_single_shot(
                    model=model,
                    tokenizer=tokenizer,
                    input_device=input_device,
                    tok_add_special=bool(tok_add_special),
                    template=template,
                    q_id=item.q_id,
                    query_uid=item.query_uid,
                    query_input=item.query_input,
                    gold_target=item.gold_target,
                    relation_source=relation_source,
                    shot=shot,
                    demos=demos_pairs,
                    n_candidates=args.n_candidates,
                    max_attempt_rounds=args.max_attempt_rounds,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    include_gold_in_prompt=bool(args.include_gold_in_prompt),
                    seed=args.seed,
                )
                relation_result["shots"][str(shot)] = shot_result

                fail_reason = shot_result.get("fail_reason")
                if fail_reason:
                    fail_reason_counts[fail_reason] = fail_reason_counts.get(fail_reason, 0) + 1
                if args.require_complete_10 and not bool(shot_result.get("complete_n", False)):
                    strict_mode_failed = True

                round_candidate_counts.append(float(len(shot_result.get("normalized_candidates", []))))
                n_attempt_rounds = int(shot_result.get("n_attempt_rounds", 0))
                if n_attempt_rounds > 0:
                    round_accept_rates.append(
                        float(len(shot_result.get("candidates", []))) / max(1, n_attempt_rounds)
                    )

                shot_rows.append(
                    {
                        "query_uid": item.query_uid,
                        "q_id": item.q_id,
                        "query_input": item.query_input,
                        "gold_target": item.gold_target,
                        "relation_source": relation_source,
                        "shot": int(shot),
                        "source_trial_indices": sorted(item.source_trial_indices),
                        "query_source_indices": sorted(set(item.query_source_indices)),
                        "pool_size_before_filter": len(relation_rows),
                        "pool_size_after_filter": len(filtered_rows),
                        "selected_demo_pool_size": len(selected_rows),
                        "candidate_count": int(shot_result.get("n_candidates_found", 0)),
                        "complete_n": bool(shot_result.get("complete_n", False)),
                        "fail_reason": shot_result.get("fail_reason"),
                        "prompt_used": shot_result.get("prompt_used"),
                        "demos_used": shot_result.get("demos_used", []),
                        "raw_generations": shot_result.get("raw_generations", []),
                        "normalized_candidates": shot_result.get("normalized_candidates", []),
                        "candidates": shot_result.get("candidates", []),
                        "candidate_first_token_ids": shot_result.get("candidate_first_token_ids", []),
                        "candidate_first_token_logprobs": shot_result.get(
                            "candidate_first_token_logprobs", []
                        ),
                        "candidate_first_token_probs": shot_result.get(
                            "candidate_first_token_probs", []
                        ),
                        "rejected": shot_result.get("rejected", []),
                        "n_attempt_rounds": int(shot_result.get("n_attempt_rounds", 0)),
                    }
                )

            query_record["relations"][relation_source] = relation_result

        query_records.append(query_record)

    change_summary_rows: List[Dict[str, object]] = []
    for query_record in query_records:
        for relation_source, relation_payload in query_record.get("relations", {}).items():
            shots_payload = relation_payload.get("shots", {})
            if not isinstance(shots_payload, dict):
                continue
            shot_values = sorted(int(shot) for shot in shots_payload.keys())
            for shot_from, shot_to in _pairwise_shot_pairs(shot_values, args.compare_mode):
                from_payload = shots_payload[str(shot_from)]
                to_payload = shots_payload[str(shot_to)]
                cand_from = list(from_payload.get("candidates", []))
                cand_to = list(to_payload.get("candidates", []))
                ids_from = [int(x) for x in from_payload.get("candidate_first_token_ids", [])]
                ids_to = [int(x) for x in to_payload.get("candidate_first_token_ids", [])]
                logp_from = [float(x) for x in from_payload.get("candidate_first_token_logprobs", [])]
                logp_to = [float(x) for x in to_payload.get("candidate_first_token_logprobs", [])]
                set_cand_from = set(cand_from)
                set_cand_to = set(cand_to)
                set_ids_from = set(ids_from)
                set_ids_to = set(ids_to)
                mean_logprob_from = _mean(logp_from)
                mean_logprob_to = _mean(logp_to)
                change_summary_rows.append(
                    {
                        "query_uid": query_record["query_uid"],
                        "q_id": query_record["q_id"],
                        "query_input": query_record["query_input"],
                        "relation_source": relation_source,
                        "shot_from": shot_from,
                        "shot_to": shot_to,
                        "n_candidates_from": len(cand_from),
                        "n_candidates_to": len(cand_to),
                        "complete_from": bool(from_payload.get("complete_n", False)),
                        "complete_to": bool(to_payload.get("complete_n", False)),
                        "text_jaccard": _jaccard(cand_from, cand_to),
                        "token_id_jaccard": _jaccard(ids_from, ids_to),
                        "overlap_count_text": len(set_cand_from & set_cand_to),
                        "overlap_count_token_id": len(set_ids_from & set_ids_to),
                        "mean_logprob_from": mean_logprob_from,
                        "mean_logprob_to": mean_logprob_to,
                        "mean_logprob_delta": mean_logprob_to - mean_logprob_from,
                    }
                )

    relation_shot_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for relation_source in demo_relations:
        rel_rows = [row for row in shot_rows if row["relation_source"] == relation_source]
        per_shot: Dict[str, Dict[str, float]] = {}
        for shot in sorted(set(int(row["shot"]) for row in rel_rows)):
            shot_rows_sel = [row for row in rel_rows if int(row["shot"]) == shot]
            cand_counts = [int(row["candidate_count"]) for row in shot_rows_sel]
            complete_vals = [1.0 if bool(row["complete_n"]) else 0.0 for row in shot_rows_sel]
            per_shot[str(shot)] = {
                "n_rows": len(shot_rows_sel),
                "avg_candidate_count": _mean(cand_counts),
                "complete_rate": _mean(complete_vals),
            }
        relation_shot_stats[relation_source] = per_shot

    stats_payload = {
        "n_queries_total": len(query_list),
        "n_query_relation_total": len(query_list) * len(demo_relations),
        "n_query_relation_with_results": len(
            [1 for q in query_records for _r in q.get("relations", {}).keys()]
        ),
        "n_shot_rows": len(shot_rows),
        "n_change_rows": len(change_summary_rows),
        "n_skip_records": len(skip_records),
        "avg_candidates_per_round_before_filter": _mean(round_candidate_counts),
        "avg_accept_rate_after_filter": _mean(round_accept_rates),
        "fail_reason_counts": fail_reason_counts,
        "relation_shot_stats": relation_shot_stats,
        "strict_mode_failed": bool(strict_mode_failed),
        "created_at": _now_utc(),
    }

    run_meta_payload = {
        "created_at": _now_utc(),
        "sampled_trials_path": sampled_trials_path,
        "relationA_ex_path": relationA_ex_path,
        "relationB_ex_path": relationB_ex_path,
        "out_dir": out_dir,
        "demo_relations": demo_relations,
        "shot_list": shots,
        "shot_eligibility": args.shot_eligibility,
        "demo_pool_size": int(args.demo_pool_size),
        "compare_mode": args.compare_mode,
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
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }

    jsonl_path = os.path.join(out_dir, "stepA_shot_candidates.jsonl")
    by_query_path = os.path.join(out_dir, "stepA_shot_candidates_by_query.json")
    summary_csv_path = os.path.join(out_dir, "stepA_shot_change_summary.csv")
    stats_path = os.path.join(out_dir, "stepA_shot_stats.json")
    run_meta_path = os.path.join(out_dir, "stepA_shot_run_meta.json")
    skips_path = os.path.join(out_dir, "stepA_shot_skips.json")

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for row in shot_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    _safe_json_dump(by_query_path, query_records)
    _safe_json_dump(stats_path, stats_payload)
    _safe_json_dump(run_meta_path, run_meta_payload)
    _safe_json_dump(skips_path, skip_records)

    summary_fields = [
        "query_uid",
        "q_id",
        "query_input",
        "relation_source",
        "shot_from",
        "shot_to",
        "n_candidates_from",
        "n_candidates_to",
        "complete_from",
        "complete_to",
        "text_jaccard",
        "token_id_jaccard",
        "overlap_count_text",
        "overlap_count_token_id",
        "mean_logprob_from",
        "mean_logprob_to",
        "mean_logprob_delta",
    ]
    _write_csv(summary_csv_path, change_summary_rows, summary_fields)

    print(f"saved: {jsonl_path}")
    print(f"saved: {by_query_path}")
    print(f"saved: {summary_csv_path}")
    print(f"saved: {stats_path}")
    print(f"saved: {run_meta_path}")
    print(f"saved: {skips_path}")

    if strict_mode_failed:
        print("strict mode failure: at least one (query, relation, shot) did not reach n_candidates")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
