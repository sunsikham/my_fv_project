"""Relation CSV trial generation utilities for StepD."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from fv.prompting import build_prompt_qa
from src.utils.prompt_utils import word_pairs_to_prompt_data as paper_word_pairs_to_prompt_data


DEFAULT_PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
DEFAULT_SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}


@dataclass
class RelationTrialsStats:
    q_counts: Dict[str, int]
    q_demo_counts: Dict[str, int]
    q_trials: Dict[str, int]
    shuffle_match_counts: Dict[str, List[int]]
    skipped_qs: Dict[str, int]
    n_demos_effective: int


def _strip_leading_space(text: str) -> str:
    return text[1:] if isinstance(text, str) and text.startswith(" ") else text


def _normalize_pair(pair: Tuple[str, str]) -> Tuple[str, str]:
    return (_strip_leading_space(pair[0]), _strip_leading_space(pair[1]))


def _normalize_pairs(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    return [_normalize_pair(pair) for pair in pairs]


def _resolve_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    col_set = {c.strip(): c for c in columns}
    for cand in candidates:
        if cand in col_set:
            return col_set[cand]
    lower_map = {c.strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_relation_csv(csv_path: str) -> Dict[str, List[Tuple[str, str]]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"relation CSV not found: {csv_path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("relation CSV missing header row")
        col_q = _resolve_column(reader.fieldnames, ["q", "id", "qid"])
        col_a = _resolve_column(reader.fieldnames, ["exA", "ex_A", "ex_a", "input"])
        col_b = _resolve_column(reader.fieldnames, ["exB", "ex_B", "ex_b", "output"])
        if col_q is None or col_a is None or col_b is None:
            raise ValueError(
                "relation CSV must include q/id and exA/exB columns "
                f"(found={reader.fieldnames})"
            )
        by_q: Dict[str, List[Tuple[str, str]]] = {}
        for row in reader:
            q_id = (row.get(col_q) or "").strip()
            ex_a = (row.get(col_a) or "").strip()
            ex_b = (row.get(col_b) or "").strip()
            if not q_id or not ex_a or not ex_b:
                continue
            by_q.setdefault(q_id, []).append((ex_a, ex_b))
    return by_q


def _parse_q_list(q_list: Optional[str], available: Sequence[str]) -> List[str]:
    if not q_list:
        return sorted(set(available))
    parts = [part.strip() for part in q_list.split(",")]
    return [part for part in parts if part]


def _shuffle_outputs(
    demos: Sequence[Tuple[str, str]], rng: random.Random
) -> Tuple[List[Tuple[str, str]], int]:
    outputs = [y for _x, y in demos]
    shuffled = outputs[:]
    rng.shuffle(shuffled)
    if len(shuffled) > 1 and shuffled == outputs:
        rng.shuffle(shuffled)
    overlap = sum(1 for a, b in zip(outputs, shuffled) if a == b)
    corrupted = [(x, y) for (x, _y), y in zip(demos, shuffled)]
    return corrupted, overlap


def generate_relation_trials(
    csv_path: str,
    q_list: Optional[str],
    n_trials_per_q: int,
    n_demos: int,
    seed: int,
    prefixes: Optional[Dict[str, str]] = None,
    separators: Optional[Dict[str, str]] = None,
    prepend_bos_token: bool = False,
    prepend_space: bool = True,
    tokenizer=None,
    tok_add_special: Optional[bool] = None,
) -> Tuple[Dict[str, object], RelationTrialsStats]:
    by_q = load_relation_csv(csv_path)
    selected_qs = _parse_q_list(q_list, list(by_q.keys()))
    rng = random.Random(seed)

    q_counts: Dict[str, int] = {}
    skipped_qs: Dict[str, int] = {}
    usable_qs: List[str] = []
    for q_id in selected_qs:
        pairs = by_q.get(q_id, [])
        q_counts[q_id] = len(pairs)
        if len(pairs) < 2:
            skipped_qs[q_id] = len(pairs)
            continue
        usable_qs.append(q_id)

    if not usable_qs:
        raise ValueError("No q has at least 2 examples (query + demo).")

    min_avail = min(len(by_q[q_id]) - 1 for q_id in usable_qs)
    n_demos_effective = min(n_demos, min_avail)

    use_prefixes = prefixes or DEFAULT_PREFIXES
    use_separators = separators or DEFAULT_SEPARATORS

    trials: List[Dict[str, object]] = []
    q_demo_counts: Dict[str, int] = {}
    q_trials: Dict[str, int] = {}
    shuffle_match_counts: Dict[str, List[int]] = {}

    for q_id in usable_qs:
        pairs = _normalize_pairs(by_q[q_id])
        q_demo_counts[q_id] = n_demos_effective
        shuffle_match_counts[q_id] = []
        for trial_idx in range(n_trials_per_q):
            query_idx = rng.randrange(len(pairs))
            query = pairs[query_idx]
            demo_candidates = [p for i, p in enumerate(pairs) if i != query_idx]
            rng.shuffle(demo_candidates)
            demos_clean = demo_candidates[:n_demos_effective]
            demos_corrupted, overlap = _shuffle_outputs(demos_clean, rng)
            shuffle_match_counts[q_id].append(overlap)

            clean_prefix_str, _clean_full_str = build_prompt_qa(
                demos_clean,
                query,
                prefixes=use_prefixes,
                separators=use_separators,
                prepend_bos_token=prepend_bos_token,
                prepend_space=prepend_space,
            )
            corrupted_prefix_str, _tmp_full_str = build_prompt_qa(
                demos_corrupted,
                query,
                prefixes=use_prefixes,
                separators=use_separators,
                prepend_bos_token=prepend_bos_token,
                prepend_space=prepend_space,
            )

            prompt_data_clean = paper_word_pairs_to_prompt_data(
                {"input": [x for x, _y in demos_clean], "output": [y for _x, y in demos_clean]},
                query_target_pair={"input": [query[0]], "output": [query[1]]},
                prepend_bos_token=prepend_bos_token,
                prefixes=use_prefixes,
                separators=use_separators,
                shuffle_labels=False,
                prepend_space=prepend_space,
            )
            prompt_data_corrupted = paper_word_pairs_to_prompt_data(
                {"input": [x for x, _y in demos_corrupted], "output": [y for _x, y in demos_corrupted]},
                query_target_pair={"input": [query[0]], "output": [query[1]]},
                prepend_bos_token=prepend_bos_token,
                prefixes=use_prefixes,
                separators=use_separators,
                shuffle_labels=False,
                prepend_space=prepend_space,
            )
            target_str = prompt_data_clean["query_target"]["output"]
            trial = {
                "q_id": q_id,
                "demos_clean": [{"input": x, "output": y} for x, y in demos_clean],
                "demos_corrupted": [{"input": x, "output": y} for x, y in demos_corrupted],
                "query": {"input": query[0], "output": query[1]},
                "clean_prompt_str": clean_prefix_str,
                "corrupted_prompt_str": corrupted_prefix_str,
                "prompt_data_clean": prompt_data_clean,
                "prompt_data_corrupted": prompt_data_corrupted,
                "target_str": target_str,
            }
            if tokenizer is not None and tok_add_special is not None:
                boundary_prefix = corrupted_prefix_str
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
                    raise ValueError("Tokenization does not extend prefix for target.")
                target_id = full_ids[len(prefix_ids)]
                trial["target_first_token_id"] = int(target_id)
                trial["answer_ids"] = full_ids[len(prefix_ids) :]
            trials.append(trial)
        q_trials[q_id] = n_trials_per_q

    rng.shuffle(trials)

    meta = {
        "csv_path": str(Path(csv_path)),
        "q_list": usable_qs,
        "n_trials_per_q": n_trials_per_q,
        "n_demos_requested": n_demos,
        "n_demos": n_demos_effective,
        "seed": seed,
        "prefixes": use_prefixes,
        "separators": use_separators,
        "prepend_bos_token_used": prepend_bos_token,
    }
    fixed_trials = {"meta": meta, "trials": trials}
    stats = RelationTrialsStats(
        q_counts=q_counts,
        q_demo_counts=q_demo_counts,
        q_trials=q_trials,
        shuffle_match_counts=shuffle_match_counts,
        skipped_qs=skipped_qs,
        n_demos_effective=n_demos_effective,
    )
    return fixed_trials, stats


def save_trials_json(trials_data: Dict[str, object], out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(trials_data, handle, indent=2, ensure_ascii=True)
