"""Slot computation utilities."""

from typing import Dict, List, Optional, Sequence, Tuple


def get_target_first_token_id_from_boundary(
    prefix_str: str,
    answer_str: str,
    tokenizer,
    tokenize_kwargs: Optional[Dict[str, object]] = None,
) -> int:
    if tokenize_kwargs is None:
        tokenize_kwargs = {}
    prefix_ids = tokenizer(prefix_str, **tokenize_kwargs)["input_ids"]
    full_ids = tokenizer(prefix_str + answer_str, **tokenize_kwargs)["input_ids"]
    s = len(prefix_ids)
    if len(full_ids) <= s:
        raise ValueError("Full tokenization does not extend prefix tokens")
    if full_ids[:s] != prefix_ids:
        raise ValueError("Prefix tokens mismatch full tokens at boundary")
    return full_ids[s]


def build_prompt_segments(
    demos: Sequence[Tuple[str, str]],
    query: Tuple[str, str],
) -> List[Tuple[str, str]]:
    segments: List[Tuple[str, str]] = []
    for i, (demo_in, demo_out) in enumerate(demos):
        segments.append(("STATIC_Q", "Q:"))
        segments.append((f"DEMO_{i}_IN", f" {demo_in}"))
        segments.append(("STATIC_A", "\nA:"))
        segments.append((f"DEMO_{i}_OUT", f" {demo_out}"))
        segments.append(("STATIC_SEP", "\n\n"))
    segments.append(("STATIC_Q", "Q:"))
    segments.append(("QUERY_IN", f" {query[0]}"))
    segments.append(("QUERY_PRED", "\nA:"))
    segments.append(("QUERY_OUT", f" {query[1]}"))
    return segments


def segments_to_text_and_spans(
    segments: Sequence[Tuple[str, str]],
) -> Tuple[str, List[Tuple[str, int, int]]]:
    parts: List[str] = []
    spans: List[Tuple[str, int, int]] = []
    cursor = 0
    for label, text in segments:
        start = cursor
        end = start + len(text)
        parts.append(text)
        spans.append((label, start, end))
        cursor = end
    return "".join(parts), spans


def infer_special_token_labels(tokenizer, input_ids, offsets) -> Dict[int, str]:
    special_labels: Dict[int, str] = {}
    for idx, (start, end) in enumerate(offsets):
        if start != end:
            continue
        token_id = input_ids[idx]
        if tokenizer.bos_token_id is not None and token_id == tokenizer.bos_token_id:
            special_labels[idx] = "SPECIAL_BOS"
        elif tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            special_labels[idx] = "SPECIAL_EOS"
        else:
            special_labels[idx] = "SPECIAL"
    return special_labels


def get_token_meta_labels(
    tokenizer,
    full_str: str,
    segments: Sequence[Tuple[str, str]],
    tok_add_special: bool,
    special_index_labels: Dict[int, str],
) -> Tuple[List[str], List[int], List[Tuple[int, int]]]:
    encoded = tokenizer(
        full_str,
        return_offsets_mapping=True,
        add_special_tokens=tok_add_special,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    _, spans = segments_to_text_and_spans(segments)
    token_labels: List[str] = []
    span_idx = 0
    for idx, (start, end) in enumerate(offsets):
        if idx in special_index_labels:
            token_labels.append(special_index_labels[idx])
            continue
        while span_idx < len(spans) and start >= spans[span_idx][2]:
            span_idx += 1
        if span_idx >= len(spans):
            raise ValueError("Token offset exceeds prompt span length")
        label, span_start, span_end = spans[span_idx]
        if start < span_start or start >= span_end:
            raise ValueError("Token offset not aligned to prompt spans")
        token_labels.append(label)

    return token_labels, input_ids, offsets


def compute_duplicated_labels(
    token_meta_labels: Sequence[str],
    dummy_labels: Sequence[str],
    allow_empty_labels: Optional[Sequence[str]] = None,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    idx_map: Dict[int, List[int]] = {}
    idx_avg: Dict[int, List[int]] = {}
    cursor = 0
    n_tokens = len(token_meta_labels)
    allow_empty = set(allow_empty_labels or [])
    for slot_idx, slot_label in enumerate(dummy_labels):
        if cursor >= n_tokens or token_meta_labels[cursor] != slot_label:
            if slot_label in allow_empty and slot_label not in token_meta_labels[cursor:]:
                idx_map[slot_idx] = []
                continue
            while cursor < n_tokens and token_meta_labels[cursor] != slot_label:
                cursor += 1
            if cursor >= n_tokens:
                raise ValueError(f"Failed to align slot label: {slot_label}")
        start = cursor
        while cursor < n_tokens and token_meta_labels[cursor] == slot_label:
            cursor += 1
        span = list(range(start, cursor))
        idx_map[slot_idx] = span
        if len(span) > 1:
            idx_avg[slot_idx] = span
    return idx_map, idx_avg


def validate_idx_map(idx_map: Dict[int, List[int]], seq_len: int) -> None:
    prev_end = -1
    for slot_idx in sorted(idx_map.keys()):
        span = idx_map[slot_idx]
        if not span:
            continue
        if span != sorted(span):
            raise ValueError(f"idx_map span not sorted for slot_idx={slot_idx}")
        if span[0] <= prev_end:
            raise ValueError(f"idx_map not monotonic at slot_idx={slot_idx}")
        if span[0] < 0 or span[-1] >= seq_len:
            raise ValueError(f"idx_map span out of range for slot_idx={slot_idx}")
        prev_end = span[-1]


def resolve_slot_seq_token_idx(
    idx_map: Dict[int, List[int]],
    slot_idx: int,
    require_single: bool = True,
) -> int:
    if slot_idx not in idx_map:
        raise ValueError(f"slot_idx missing from idx_map: {slot_idx}")
    span = idx_map[slot_idx]
    if not span:
        raise ValueError(f"Empty idx_map span for slot_idx={slot_idx}")
    if require_single and len(span) != 1:
        raise ValueError(f"slot_idx={slot_idx} spans multiple tokens: {span}")
    return span[-1]


def get_dummy_token_labels_and_slot_map(
    n_icl_examples: int,
    special_prefix: Optional[List[str]] = None,
    special_suffix: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, int]]:
    if n_icl_examples < 0:
        raise ValueError("n_icl_examples must be >= 0")
    dummy_labels: List[str] = []
    slot_index_map: Dict[str, int] = {}

    if special_prefix:
        for label in special_prefix:
            slot_index_map[label] = len(dummy_labels)
            dummy_labels.append(label)

    for i in range(n_icl_examples):
        dummy_labels.append("STATIC_Q")
        slot_index_map[f"DEMO_{i}_IN"] = len(dummy_labels)
        dummy_labels.append(f"DEMO_{i}_IN")
        dummy_labels.append("STATIC_A")
        slot_index_map[f"DEMO_{i}_OUT"] = len(dummy_labels)
        dummy_labels.append(f"DEMO_{i}_OUT")
        if i < n_icl_examples - 1:
            dummy_labels.append("STATIC_SEP")

    dummy_labels.append("STATIC_Q")
    slot_index_map["QUERY_IN"] = len(dummy_labels)
    dummy_labels.append("QUERY_IN")

    # QUERY_PRED aligns with the last input token (end of "A: " prefix).
    slot_index_map["QUERY_PRED"] = len(dummy_labels)
    dummy_labels.append("QUERY_PRED")

    slot_index_map["QUERY_OUT"] = len(dummy_labels)
    dummy_labels.append("QUERY_OUT")

    if special_suffix:
        for label in special_suffix:
            slot_index_map[label] = len(dummy_labels)
            dummy_labels.append(label)

    return dummy_labels, slot_index_map


def compute_query_predictive_slot(
    prefix_str: str,
    full_str: str,
    tokenizer,
    add_special_tokens: bool = False,
) -> Dict[str, object]:
    prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=add_special_tokens)
    full_ids = tokenizer.encode(full_str, add_special_tokens=add_special_tokens)

    s = len(prefix_ids)
    if s >= len(full_ids):
        trimmed_prefix = prefix_str.rstrip(" ")
        if trimmed_prefix != prefix_str:
            prefix_ids = tokenizer.encode(
                trimmed_prefix, add_special_tokens=add_special_tokens
            )
            s = len(prefix_ids)
    if s <= 0 or s >= len(full_ids):
        raise ValueError(
            f"Invalid prefix length for slot computation (s={s}, seq_len={len(full_ids)})"
        )

    slot_index = s - 1
    seq_len = len(full_ids)
    if not (0 <= slot_index < seq_len):
        raise ValueError("slot_index out of range")

    answer = full_str[len(prefix_str) :]
    boundary_prefix = prefix_str
    boundary_answer = answer
    if boundary_prefix.endswith(" ") and not boundary_answer.startswith(" "):
        boundary_prefix = boundary_prefix[:-1]
        boundary_answer = f" {boundary_answer}"
    target_id = get_target_first_token_id_from_boundary(
        boundary_prefix,
        boundary_answer,
        tokenizer,
        tokenize_kwargs={"add_special_tokens": add_special_tokens},
    )
    target_token = tokenizer.convert_ids_to_tokens(target_id)
    full_target_id = full_ids[s]
    full_target_token = tokenizer.convert_ids_to_tokens(full_target_id)

    return {
        "s": s,
        "slot_index": slot_index,
        "seq_len": seq_len,
        "target_id": target_id,
        "full_target_id": full_target_id,
        "full_target_token": full_target_token,
        "target_token": target_token,
        "input_ids": full_ids,
    }
