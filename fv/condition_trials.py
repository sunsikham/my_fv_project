"""Condition-wise fixed trial generation for AAA/BBB/BABA experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from fv.prompting import build_prompt_qa
from fv.relation_trials import load_relation_csv
from fv.tokenization import resolve_prompt_add_special_tokens

from .prompting import word_pairs_to_prompt_data as paper_word_pairs_to_prompt_data


DEFAULT_PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
DEFAULT_SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}
SUPPORTED_CONDITIONS = ("AAA", "BBB", "BABA")


def parse_conditions(text: str) -> List[str]:
    raw = [part.strip().upper() for part in (text or "").split(",") if part.strip()]
    if not raw:
        raise ValueError("conditions must include at least one value")
    for cond in raw:
        if cond not in SUPPORTED_CONDITIONS:
            raise ValueError(
                f"Unsupported condition '{cond}'. Supported={SUPPORTED_CONDITIONS}"
            )
    seen = set()
    uniq: List[str] = []
    for cond in raw:
        if cond in seen:
            continue
        seen.add(cond)
        uniq.append(cond)
    return uniq


def _stable_seed(*parts: object) -> int:
    joined = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


def _normalize_pair(pair: Tuple[str, str]) -> Tuple[str, str]:
    left = pair[0].lstrip() if isinstance(pair[0], str) else str(pair[0])
    right = pair[1].lstrip() if isinstance(pair[1], str) else str(pair[1])
    return (left, right)


def _normalize_pairs(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    return [_normalize_pair(pair) for pair in pairs]


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
            rotated = shuffled[1:] + shuffled[:1]
            shuffled = rotated
    corrupted = [(inp, out) for (inp, _old), out in zip(demos, shuffled)]
    return corrupted


def _alternating_demos(
    pairs_a: Sequence[Tuple[str, str]],
    pairs_b: Sequence[Tuple[str, str]],
    demo_indices: Sequence[int],
    start_with: str,
) -> List[Tuple[str, str]]:
    demos: List[Tuple[str, str]] = []
    for pos, idx in enumerate(demo_indices):
        use_b = (pos % 2 == 0 and start_with == "B") or (
            pos % 2 == 1 and start_with == "A"
        )
        source = pairs_b if use_b else pairs_a
        demos.append(source[idx])
    return demos


def _build_prompt_data(
    demos: Sequence[Tuple[str, str]],
    query_pair: Tuple[str, str],
    prefixes: Dict[str, str],
    separators: Dict[str, str],
    prepend_bos_token: bool,
) -> Dict[str, object]:
    return paper_word_pairs_to_prompt_data(
        {
            "input": [inp for inp, _out in demos],
            "output": [out for _inp, out in demos],
        },
        query_target_pair={"input": [query_pair[0]], "output": [query_pair[1]]},
        prepend_bos_token=prepend_bos_token,
        prefixes=prefixes,
        separators=separators,
        shuffle_labels=False,
        prepend_space=True,
    )


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


def _condition_payload_meta(
    *,
    condition: str,
    relation_a_csv: str,
    relation_b_csv: str,
    q_id: str,
    n_trials: int,
    n_demos: int,
    seed: int,
    prefixes: Dict[str, str],
    separators: Dict[str, str],
    prepend_bos_token: bool,
) -> Dict[str, object]:
    return {
        "source": "condition_trials_v1",
        "condition": condition,
        "relation_a_csv": str(Path(relation_a_csv)),
        "relation_b_csv": str(Path(relation_b_csv)),
        "q_id": q_id,
        "n_trials": int(n_trials),
        "n_shots": int(n_demos),
        "n_demos": int(n_demos),
        "seed": int(seed),
        "prefixes": prefixes,
        "separators": separators,
        "prepend_bos_token_used": bool(prepend_bos_token),
        "trial_id_format": "t%06d",
    }


def generate_condition_trials_for_q(
    *,
    relation_a_pairs: Sequence[Tuple[str, str]],
    relation_b_pairs: Sequence[Tuple[str, str]],
    relation_a_csv: str,
    relation_b_csv: str,
    q_id: str,
    conditions: Sequence[str],
    n_trials_per_q: int,
    n_demos: int,
    seed: int,
    tokenizer=None,
    tok_add_special: Optional[bool] = None,
    prefixes: Optional[Dict[str, str]] = None,
    separators: Optional[Dict[str, str]] = None,
    prepend_bos_token: bool = False,
    enforce_shared_query_target: bool = False,
) -> Dict[str, Dict[str, object]]:
    if n_trials_per_q < 1:
        raise ValueError("n_trials_per_q must be >= 1")
    if n_demos < 1:
        raise ValueError("n_demos must be >= 1")

    conds = [cond.upper() for cond in conditions]
    for cond in conds:
        if cond not in SUPPORTED_CONDITIONS:
            raise ValueError(f"Unsupported condition '{cond}'")

    pairs_a = _normalize_pairs(relation_a_pairs)
    pairs_b = _normalize_pairs(relation_b_pairs)
    common_count = min(len(pairs_a), len(pairs_b))
    if common_count < (n_demos + 1):
        raise ValueError(
            f"q_id={q_id}: insufficient common rows for demos+query "
            f"(common={common_count}, required={n_demos + 1})"
        )

    use_prefixes = prefixes or DEFAULT_PREFIXES
    use_separators = separators or DEFAULT_SEPARATORS

    trials_by_condition: Dict[str, List[Dict[str, object]]] = {cond: [] for cond in conds}

    for trial_idx in range(n_trials_per_q):
        trial_id = f"t{trial_idx:06d}"
        base_rng = random.Random(_stable_seed(seed, q_id, trial_id, "base"))
        query_source_index = base_rng.randrange(common_count)
        demo_candidates = [idx for idx in range(common_count) if idx != query_source_index]
        base_rng.shuffle(demo_candidates)
        demo_indices = demo_candidates[:n_demos]

        query_anchor = pairs_a[query_source_index]
        query_by_condition: Dict[str, Tuple[str, str]] = {}

        for cond in conds:
            cond_rng = random.Random(_stable_seed(seed, q_id, trial_id, cond, "corr"))
            if cond == "AAA":
                demos_clean = [pairs_a[idx] for idx in demo_indices]
                query_pair = pairs_a[query_source_index]
            elif cond == "BBB":
                demos_clean = [pairs_b[idx] for idx in demo_indices]
                query_pair = pairs_b[query_source_index]
            elif cond == "BABA":
                demos_clean = _alternating_demos(
                    pairs_a,
                    pairs_b,
                    demo_indices,
                    start_with="B",
                )
                query_pair = pairs_a[query_source_index]
            else:  # pragma: no cover - guarded above
                raise ValueError(f"Unsupported condition '{cond}'")

            demos_corrupted = _shuffle_outputs(demos_clean, cond_rng)
            clean_prefix_str, _clean_full = build_prompt_qa(
                demos_clean,
                query_pair,
                prefixes=use_prefixes,
                separators=use_separators,
                prepend_bos_token=prepend_bos_token,
                prepend_space=True,
            )
            corrupted_prefix_str, _corr_full = build_prompt_qa(
                demos_corrupted,
                query_pair,
                prefixes=use_prefixes,
                separators=use_separators,
                prepend_bos_token=prepend_bos_token,
                prepend_space=True,
            )
            prompt_data_clean = _build_prompt_data(
                demos_clean,
                query_pair,
                prefixes=use_prefixes,
                separators=use_separators,
                prepend_bos_token=prepend_bos_token,
            )
            prompt_data_corrupted = _build_prompt_data(
                demos_corrupted,
                query_pair,
                prefixes=use_prefixes,
                separators=use_separators,
                prepend_bos_token=prepend_bos_token,
            )
            target_str = str(prompt_data_clean["query_target"]["output"])
            row: Dict[str, object] = {
                "q_id": q_id,
                "condition": cond,
                "trial_idx": int(trial_idx),
                "trial_id": trial_id,
                "query_source_index": int(query_source_index),
                "demo_source_indices": [int(x) for x in demo_indices],
                "demo_order": [int(i) for i in range(n_demos)],
                "demos_clean": [{"input": inp, "output": out} for inp, out in demos_clean],
                "demos_corrupted": [
                    {"input": inp, "output": out} for inp, out in demos_corrupted
                ],
                "query": {"input": query_pair[0], "output": query_pair[1]},
                "query_anchor": {"input": query_anchor[0], "output": query_anchor[1]},
                "clean_prompt_str": clean_prefix_str,
                "corrupted_prompt_str": corrupted_prefix_str,
                "prompt_data_clean": prompt_data_clean,
                "prompt_data_corrupted": prompt_data_corrupted,
                "target_str": target_str,
            }
            if tokenizer is not None:
                if tok_add_special is None:
                    raise ValueError(
                        "tok_add_special must be set when tokenizer is provided"
                    )
                row.update(
                    _compute_target_token_fields(
                        tokenizer=tokenizer,
                        tok_add_special=bool(tok_add_special),
                        prefix_str=corrupted_prefix_str,
                        target_str=target_str,
                    )
                )
            trials_by_condition[cond].append(row)
            query_by_condition[cond] = query_pair

        if enforce_shared_query_target and len(set(query_by_condition.values())) != 1:
            raise ValueError(
                f"q_id={q_id} trial_id={trial_id}: query/target mismatch across conditions"
            )

    payloads: Dict[str, Dict[str, object]] = {}
    for cond in conds:
        payloads[cond] = {
            "meta": _condition_payload_meta(
                condition=cond,
                relation_a_csv=relation_a_csv,
                relation_b_csv=relation_b_csv,
                q_id=q_id,
                n_trials=n_trials_per_q,
                n_demos=n_demos,
                seed=seed,
                prefixes=use_prefixes,
                separators=use_separators,
                prepend_bos_token=prepend_bos_token,
            ),
            "trials": trials_by_condition[cond],
        }
    return payloads


def generate_condition_trials_all_q(
    *,
    relation_a_csv: str,
    relation_b_csv: str,
    q_ids: Sequence[str],
    conditions: Sequence[str],
    n_trials_per_q: int,
    n_demos: int,
    seed: int,
    model_name_for_tokenizer: Optional[str] = None,
    model_spec: Optional[str] = None,
    enforce_shared_query_target: bool = False,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    by_q_a = load_relation_csv(relation_a_csv)
    by_q_b = load_relation_csv(relation_b_csv)

    tok = None
    tok_add_special = None
    if model_name_for_tokenizer:
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to import transformers tokenizer: {exc}") from exc
        tok = AutoTokenizer.from_pretrained(model_name_for_tokenizer)
        tok_add_special = resolve_prompt_add_special_tokens(
            model_name_for_tokenizer,
            model_spec,
        )

    all_payloads: Dict[str, Dict[str, Dict[str, object]]] = {}
    for q_id in q_ids:
        if q_id not in by_q_a:
            raise ValueError(f"q_id={q_id} not found in relation A CSV")
        if q_id not in by_q_b:
            raise ValueError(f"q_id={q_id} not found in relation B CSV")
        all_payloads[q_id] = generate_condition_trials_for_q(
            relation_a_pairs=by_q_a[q_id],
            relation_b_pairs=by_q_b[q_id],
            relation_a_csv=relation_a_csv,
            relation_b_csv=relation_b_csv,
            q_id=q_id,
            conditions=conditions,
            n_trials_per_q=n_trials_per_q,
            n_demos=n_demos,
            seed=seed,
            tokenizer=tok,
            tok_add_special=tok_add_special,
            prepend_bos_token=bool(tok_add_special) if tok_add_special is not None else False,
            enforce_shared_query_target=enforce_shared_query_target,
        )
    return all_payloads


def save_trials_json(payload: Dict[str, object], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def load_trials_json(path: str) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("trial JSON must be an object")
    if "trials" not in payload or not isinstance(payload["trials"], list):
        raise ValueError("trial JSON missing list field 'trials'")
    return payload


def list_qids(csv_path: str) -> List[str]:
    by_q = load_relation_csv(csv_path)
    return sorted(by_q.keys())


def parse_q_list(q_list_text: Optional[str], available_qids: Iterable[str]) -> List[str]:
    if not q_list_text:
        return sorted(set(str(q_id) for q_id in available_qids))
    parsed = [part.strip() for part in q_list_text.split(",") if part.strip()]
    return parsed


def count_rows_by_q(csv_path: str) -> Dict[str, int]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"relation CSV not found: {csv_path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("relation CSV missing header row")
        col_q = None
        for cand in ("q", "id", "qid"):
            if cand in reader.fieldnames:
                col_q = cand
                break
        if col_q is None:
            lower = {name.lower(): name for name in reader.fieldnames}
            for cand in ("q", "id", "qid"):
                if cand in lower:
                    col_q = lower[cand]
                    break
        if col_q is None:
            raise ValueError("relation CSV missing q/id/qid column")
        counts: Dict[str, int] = {}
        for row in reader:
            q_val = str(row.get(col_q, "")).strip()
            if not q_val:
                continue
            counts[q_val] = counts.get(q_val, 0) + 1
    return counts
