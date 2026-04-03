#!/usr/bin/env python3
"""
Product-test scorer: 5 edges x shot sweep with q_id-level normalization.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import hashlib
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fv.hf_loader import load_hf_model_and_tokenizer
from fv.model_spec import get_model_spec
from fv.prompting import build_prompt_qa


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score 5-edge ICL prompts with shot sweeps and q_id-level normalization."
        )
    )
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--model_spec", required=True, help="Model spec (e.g. llama3)")
    parser.add_argument("--device", required=True, help="Device (cpu/cuda)")
    parser.add_argument("--dtype", required=False, default=None, help="fp32/fp16/bf16")
    parser.add_argument(
        "--quant",
        required=False,
        default="none",
        help="Quantization mode: none/4bit/8bit/auto",
    )
    parser.add_argument("--relationA_ex_path", required=True, help="A demos CSV")
    parser.add_argument("--relationB_ex_path", required=True, help="B demos CSV")
    parser.add_argument("--icl_B_path", required=True, help="B query CSV")
    parser.add_argument("--icl_C_path", required=True, help="C query CSV")
    parser.add_argument("--icl_D_path", required=True, help="D query CSV")
    parser.add_argument(
        "--shot_list",
        required=False,
        default="1,3,5,7,10",
        help="Comma-separated shot list (default: 1,3,5,7,10)",
    )
    parser.add_argument("--n_trials", type=int, required=True, help="Trials per q_id")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--qid",
        required=False,
        default=None,
        help="Optional q_id or comma-separated q_id list",
    )
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument(
        "--save_edge_topk",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, save lexical top-k trace for selected edges.",
    )
    parser.add_argument(
        "--edge_topk_k",
        type=int,
        default=10,
        help="Lexical top-k size for edge trace (default: 10).",
    )
    parser.add_argument(
        "--edge_topk_edges",
        default="AB,AC,AD,BC,BD",
        help="Comma-separated edges for lexical top-k trace (default: AB,AC,AD,BC,BD).",
    )
    parser.add_argument(
        "--edge_topk_jsonl",
        default=None,
        help="Optional output path for raw edge lexical top-k JSONL.",
    )
    parser.add_argument(
        "--edge_topk_change_csv",
        default=None,
        help="Optional output path for adjacent-shot edge change summary CSV.",
    )
    return parser.parse_args()


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


def _read_relation_csv(path: str) -> List[Dict[str, str]]:
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
            q_id = (row.get(col_q) or "").strip()
            inp = (row.get(col_in) or "").strip()
            out = (row.get(col_out) or "").strip()
            if not q_id or not inp or not out:
                continue
            rows.append(
                {
                    "row_id": row_idx,
                    "q_id": q_id,
                    "input": inp,
                    "output": out,
                }
            )
        return rows


def _group_by_qid(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["q_id"], []).append(row)
    return grouped


def _select_query(rows: List[Dict[str, str]]) -> Dict[str, str]:
    sorted_rows = sorted(rows, key=lambda x: (x["input"], x["output"]))
    return sorted_rows[0]


def _target_first_token_id_with_checks(
    tokenizer,
    prefix_str: str,
    full_str: str,
    tok_add_special: bool,
    prefix_ids_a: List[int],
    q_id: str,
    edge: str,
    shot: int,
    trial_index: int,
    spec_prepend_bos: Optional[bool] = None,
) -> int:
    prefix_ids_b = tokenizer.encode(prefix_str, add_special_tokens=tok_add_special)
    if prefix_ids_a != prefix_ids_b:
        tail = repr(prefix_str[-120:])
        raise ValueError(
            "Prefix token mismatch between model input and target-id calc: "
            f"q_id={q_id} edge={edge} shot={shot} trial={trial_index} "
            f"prefix_tail={tail} tok_add_special={tok_add_special} "
            f"spec_prepend_bos={spec_prepend_bos} "
            f"prefix_ids_a_tail={prefix_ids_a[-20:]} prefix_ids_b_tail={prefix_ids_b[-20:]}"
        )

    full_ids = tokenizer.encode(full_str, add_special_tokens=tok_add_special)
    if len(full_ids) <= len(prefix_ids_b):
        raise ValueError("Tokenization does not extend prefix for target.")
    if full_ids[: len(prefix_ids_b)] != prefix_ids_b:
        window = full_ids[max(0, len(prefix_ids_b) - 10) : len(prefix_ids_b) + 10]
        raise ValueError(
            "Prefix-invariance failed for full_ids prefix: "
            f"q_id={q_id} edge={edge} shot={shot} trial={trial_index} "
            f"prefix_tail={repr(prefix_str[-120:])} "
            f"target_suffix_str={repr(full_str[len(prefix_str):])} "
            f"tok_add_special={tok_add_special} spec_prepend_bos={spec_prepend_bos} "
            f"prefix_ids_tail={prefix_ids_b[-20:]} full_ids_win={window}"
        )
    return int(full_ids[len(prefix_ids_b)])


def _parse_shot_list(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    shots = [int(p) for p in parts]
    if not shots:
        raise ValueError("--shot_list is empty")
    return shots


def _parse_qid_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    out: List[str] = []
    seen = set()
    for part in str(raw).split(","):
        item = part.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _bundle_size_from_shots(shots: Sequence[int]) -> int:
    if not shots:
        raise ValueError("shot_list is empty")
    bundle_size = max(int(shot) for shot in shots)
    if bundle_size > 10:
        raise ValueError("shot_list includes value > 10")
    return bundle_size


def _parse_edge_list(raw: str) -> List[str]:
    parts = [part.strip().upper() for part in (raw or "").split(",") if part.strip()]
    if not parts:
        raise ValueError("--edge_topk_edges is empty")
    allowed = {"AB", "AC", "AD", "BC", "BD"}
    out: List[str] = []
    seen = set()
    for part in parts:
        if part not in allowed:
            raise ValueError(f"Unsupported edge in --edge_topk_edges: {part}")
        if part in seen:
            continue
        seen.add(part)
        out.append(part)
    return out


def _percentile(values: List[float], pct: float) -> float:
    arr = np.array(values, dtype=float)
    return float(np.percentile(arr, pct))


_MULTISPACE_RE = re.compile(r"\s+")
_FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "may",
    "me",
    "might",
    "must",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "should",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
}


def _normalize_candidate_text(text: str) -> str:
    return _MULTISPACE_RE.sub(" ", text.strip()).lower()


def _canonicalize_candidate(text: str) -> str:
    word = _normalize_candidate_text(text)
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if word.endswith("ed") and len(word) > 4:
        base = word[:-2]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if word.endswith("es") and len(word) > 4:
        if word.endswith(("ses", "xes", "zes", "ches", "shes")):
            return word[:-2]
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    return word


def _is_word_start_token(raw_token: str, decoded: str) -> bool:
    if decoded != decoded.lstrip():
        return True
    if raw_token.startswith(("Ġ", "▁")):
        return True
    return False


def _decode_single_token(tokenizer, token_id: int) -> Tuple[str, str]:
    raw_token = tokenizer.convert_ids_to_tokens(int(token_id))
    decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return raw_token, decoded


def _extract_lexical_candidate(tokenizer, token_id: int) -> Optional[Dict[str, object]]:
    raw_token, decoded = _decode_single_token(tokenizer, token_id)
    normalized = _normalize_candidate_text(decoded)
    if not normalized:
        return None
    if not _is_word_start_token(raw_token, decoded):
        return None
    if " " in normalized:
        return None
    if not normalized.isalpha():
        return None
    if normalized in _FUNCTION_WORDS:
        return None
    return {
        "candidate": normalized,
        "canonical": _canonicalize_candidate(normalized),
        "raw_token": raw_token,
        "decoded": decoded,
    }


def _collect_edge_topk(
    *,
    tokenizer,
    next_logits: torch.Tensor,
    next_logprobs: torch.Tensor,
    k: int,
    raw_k: int = 5,
) -> Dict[str, object]:
    vocab_size = int(next_logits.shape[-1])
    raw_top_k = min(raw_k, vocab_size)
    raw_vals, raw_ids = torch.topk(next_logprobs, k=raw_top_k)
    raw_token_strs = []
    for token_id in raw_ids.tolist():
        raw_token, decoded = _decode_single_token(tokenizer, token_id)
        raw_token_strs.append({"raw_token": raw_token, "decoded": decoded})

    accepted: List[Dict[str, object]] = []
    seen_canonical = set()
    seen_ids = set()
    chunk = min(vocab_size, max(256, k * 32))
    while len(accepted) < k:
        vals, ids = torch.topk(next_logprobs, k=chunk)
        for vocab_rank, (token_id, logprob) in enumerate(zip(ids.tolist(), vals.tolist()), start=1):
            if token_id in seen_ids:
                continue
            seen_ids.add(int(token_id))
            extracted = _extract_lexical_candidate(tokenizer, int(token_id))
            if extracted is None:
                continue
            canonical = extracted["canonical"]
            if canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
            accepted.append(
                {
                    "token_id": int(token_id),
                    "candidate": extracted["candidate"],
                    "canonical": canonical,
                    "logit": float(next_logits[int(token_id)].item()),
                    "logprob": float(logprob),
                    "prob": float(torch.exp(torch.tensor(logprob)).item()),
                    "rank": int(vocab_rank),
                }
            )
            if len(accepted) >= k:
                break
        if len(accepted) >= k or chunk >= vocab_size:
            break
        chunk = min(vocab_size, chunk * 2)

    shortfall_reason = None
    if len(accepted) < k:
        shortfall_reason = "filtered_vocabulary_exhausted"

    return {
        "raw_top_token_ids": [int(x) for x in raw_ids.tolist()],
        "raw_top_token_strs": raw_token_strs,
        "raw_top_logprobs": [float(x) for x in raw_vals.tolist()],
        "raw_top_probs": [float(torch.exp(x).item()) for x in raw_vals],
        "lexical_candidates": [row["candidate"] for row in accepted],
        "lexical_candidate_token_ids": [row["token_id"] for row in accepted],
        "lexical_candidate_logits": [row["logit"] for row in accepted],
        "lexical_candidate_logprobs": [row["logprob"] for row in accepted],
        "lexical_candidate_probs": [row["prob"] for row in accepted],
        "lexical_candidate_ranks": [row["rank"] for row in accepted],
        "lexical_candidate_canonical_forms": [row["canonical"] for row in accepted],
        "lexical_candidate_shortfall_reason": shortfall_reason,
    }


def _target_rank_in_vocab(next_logits: torch.Tensor, target_id: int) -> int:
    target_logit = next_logits[target_id]
    return int((next_logits > target_logit).sum().item()) + 1


def _jaccard(lhs: Sequence[object], rhs: Sequence[object]) -> float:
    left = set(lhs)
    right = set(rhs)
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right) / len(union))


def _stable_hash(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resume_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "_resume")


def _resume_state_path(out_dir: str) -> str:
    return os.path.join(_resume_dir(out_dir), "pt_resume_state.json")


def _trial_plan_path(out_dir: str) -> str:
    return os.path.join(_resume_dir(out_dir), "pt_trial_plan.json")


def _qid_raw_rows_path(out_dir: str, q_id: str) -> str:
    return os.path.join(_resume_dir(out_dir), f"raw_rows_{q_id}.jsonl")


def _qid_edge_topk_raw_rows_path(out_dir: str, q_id: str) -> str:
    return os.path.join(_resume_dir(out_dir), f"raw_edge_topk_{q_id}.jsonl")


def _write_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    os.replace(tmp, path)


def _read_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _append_jsonl(path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()


def _read_jsonl(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_config_fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "model": args.model,
        "model_spec": args.model_spec,
        "device": args.device,
        "dtype": args.dtype,
        "quant": args.quant,
        "relationA_ex_path": os.path.abspath(args.relationA_ex_path),
        "relationB_ex_path": os.path.abspath(args.relationB_ex_path),
        "icl_B_path": os.path.abspath(args.icl_B_path),
        "icl_C_path": os.path.abspath(args.icl_C_path),
        "icl_D_path": os.path.abspath(args.icl_D_path),
        "relationA_ex_sha256": _file_sha256(args.relationA_ex_path),
        "relationB_ex_sha256": _file_sha256(args.relationB_ex_path),
        "icl_B_sha256": _file_sha256(args.icl_B_path),
        "icl_C_sha256": _file_sha256(args.icl_C_path),
        "icl_D_sha256": _file_sha256(args.icl_D_path),
        "shot_list": list(_parse_shot_list(args.shot_list)),
        "n_trials": int(args.n_trials),
        "seed": int(args.seed),
        "qid": args.qid,
        "save_edge_topk": int(args.save_edge_topk),
        "edge_topk_k": int(args.edge_topk_k),
        "edge_topk_edges": list(_parse_edge_list(args.edge_topk_edges)) if bool(args.save_edge_topk) else [],
        "scorer_code_sha256": _file_sha256(__file__),
    }
    return _stable_hash(payload)


def _edge_eval_key(row: Dict[str, object]) -> Tuple[int, int, str]:
    return (int(row["trial_index"]), int(row["shot"]), str(row["edge"]))


def _dedupe_by_edge_key(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    latest: Dict[Tuple[int, int, str], Dict[str, object]] = {}
    for row in rows:
        latest[_edge_eval_key(row)] = row
    return [latest[key] for key in sorted(latest.keys())]


def _progress_line(
    *,
    q_id: str,
    q_idx: int,
    q_total: int,
    trial_index: int,
    n_trials: int,
    shot: int,
    edge: str,
    done: int,
    total: int,
    elapsed_sec: float,
) -> str:
    pct = (100.0 * done / total) if total else 100.0
    rate = (done / elapsed_sec) if elapsed_sec > 0 else 0.0
    remaining = max(0, total - done)
    eta_sec = (remaining / rate) if rate > 0 else 0.0
    eta_min, eta_s = divmod(int(eta_sec), 60)
    eta_h, eta_min = divmod(eta_min, 60)
    return (
        f"[progress] q_id={q_id} ({q_idx}/{q_total}) "
        f"trial={trial_index + 1}/{n_trials} shot={shot} edge={edge} "
        f"done={done}/{total} pct={pct:.1f}% "
        f"elapsed={int(elapsed_sec)}s eta={eta_h:02d}:{eta_min:02d}:{eta_s:02d}"
    )


def _bundle_row_ids_from_plan(plan_row: Dict[str, object], side: str) -> List[int]:
    generic_key = f"{side}_bundle_row_ids"
    legacy_key = f"{side}10_row_ids"
    raw_ids = plan_row.get(generic_key)
    if raw_ids is None:
        raw_ids = plan_row.get(legacy_key)
    if raw_ids is None:
        raise KeyError(f"Missing bundle row ids for side={side}")
    return [int(row_id) for row_id in raw_ids]


def _build_trial_plan_rows(
    *,
    qids: Sequence[str],
    A_by: Dict[str, List[Dict[str, str]]],
    B_by: Dict[str, List[Dict[str, str]]],
    icl_B_by: Dict[str, List[Dict[str, str]]],
    icl_C_by: Dict[str, List[Dict[str, str]]],
    icl_D_by: Dict[str, List[Dict[str, str]]],
    n_trials: int,
    seed: int,
    bundle_size: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    plan_rows: List[Dict[str, object]] = []
    for q_id in qids:
        A_q = A_by.get(q_id, [])
        B_q = B_by.get(q_id, [])
        icl_B_q = icl_B_by.get(q_id, [])
        icl_C_q = icl_C_by.get(q_id, [])
        icl_D_q = icl_D_by.get(q_id, [])
        if (
            len(A_q) < bundle_size
            or len(B_q) < bundle_size
            or len(icl_B_q) < 1
            or len(icl_C_q) < 1
            or len(icl_D_q) < 1
        ):
            continue
        B_query = _select_query(icl_B_q)
        C_query = _select_query(icl_C_q)
        D_query = _select_query(icl_D_q)
        forbidden_A = {
            (B_query["input"], B_query["output"]),
            (C_query["input"], C_query["output"]),
            (D_query["input"], D_query["output"]),
        }
        forbidden_B = {
            (C_query["input"], C_query["output"]),
            (D_query["input"], D_query["output"]),
        }
        A_pool = [row for row in A_q if (row["input"], row["output"]) not in forbidden_A]
        B_pool = [row for row in B_q if (row["input"], row["output"]) not in forbidden_B]
        if len(A_pool) < bundle_size or len(B_pool) < bundle_size:
            continue
        for trial_index in range(n_trials):
            A_bundle = rng.sample(A_pool, bundle_size)
            rng.shuffle(A_bundle)
            B_bundle = rng.sample(B_pool, bundle_size)
            rng.shuffle(B_bundle)
            plan_rows.append(
                {
                    "q_id": q_id,
                    "trial_index": int(trial_index),
                    "bundle_size": int(bundle_size),
                    "A_bundle_row_ids": [int(row["row_id"]) for row in A_bundle],
                    "B_bundle_row_ids": [int(row["row_id"]) for row in B_bundle],
                }
            )
    return plan_rows


def main() -> int:
    args = _parse_args()
    edge_topk_enabled = bool(args.save_edge_topk)
    edge_topk_edges = _parse_edge_list(args.edge_topk_edges) if edge_topk_enabled else []
    if args.edge_topk_k < 1:
        raise ValueError("--edge_topk_k must be >= 1")
    if edge_topk_enabled:
        print(
            "[edge-topk] enabled "
            f"edges={','.join(edge_topk_edges)} "
            f"k={args.edge_topk_k}",
            flush=True,
        )

    out_dir = os.path.dirname(args.out_csv) or "."
    if out_dir == "results":
        out_dir = os.path.join(out_dir, "pt_analysis")
        args.out_csv = os.path.join(out_dir, os.path.basename(args.out_csv))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(_resume_dir(out_dir), exist_ok=True)
    edge_topk_jsonl = (
        args.edge_topk_jsonl
        if args.edge_topk_jsonl
        else os.path.join(out_dir, "pt_edge_topk.jsonl")
    )
    edge_topk_change_csv = (
        args.edge_topk_change_csv
        if args.edge_topk_change_csv
        else os.path.join(out_dir, "pt_edge_topk_change_summary.csv")
    )

    A_rows = _read_relation_csv(args.relationA_ex_path)
    B_rows = _read_relation_csv(args.relationB_ex_path)
    icl_B_rows = _read_relation_csv(args.icl_B_path)
    icl_C_rows = _read_relation_csv(args.icl_C_path)
    icl_D_rows = _read_relation_csv(args.icl_D_path)

    A_by = _group_by_qid(A_rows)
    B_by = _group_by_qid(B_rows)
    icl_B_by = _group_by_qid(icl_B_rows)
    icl_C_by = _group_by_qid(icl_C_rows)
    icl_D_by = _group_by_qid(icl_D_rows)

    requested_qids = _parse_qid_list(args.qid)
    if requested_qids:
        qids = requested_qids
    else:
        qids = sorted(set(icl_B_by) & set(icl_C_by) & set(icl_D_by))

    if not qids:
        raise ValueError("No q_id available after intersection")

    shots = _parse_shot_list(args.shot_list)
    bundle_size = _bundle_size_from_shots(shots)

    config_fingerprint = _build_config_fingerprint(args)
    state_path = _resume_state_path(out_dir)
    plan_path = _trial_plan_path(out_dir)
    if os.path.exists(state_path):
        state = _read_json(state_path)
        if state.get("config_fingerprint") != config_fingerprint:
            raise ValueError("Existing PT resume state fingerprint mismatch for this out_dir")
    else:
        inferred_completed_qids: List[str] = []
        if os.path.exists(args.out_csv):
            with open(args.out_csv, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                inferred_completed_qids = sorted({str(row["q_id"]) for row in reader if row.get("q_id")})
        state = {
            "config_fingerprint": config_fingerprint,
            "completed_qids": inferred_completed_qids,
            "created_at": int(time.time()),
        }
        _write_json(state_path, state)
    if os.path.exists(args.out_csv):
        with open(args.out_csv, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_completed_qids = sorted({str(row["q_id"]) for row in reader if row.get("q_id")})
        merged = sorted(set(state.get("completed_qids", [])) | set(existing_completed_qids))
        if merged != state.get("completed_qids", []):
            state["completed_qids"] = merged
            _write_json(state_path, state)

    if os.path.exists(plan_path):
        trial_plan = _read_json(plan_path)
        if trial_plan.get("config_fingerprint") != config_fingerprint:
            raise ValueError("Existing PT trial plan fingerprint mismatch for this out_dir")
    else:
        plan_rows = _build_trial_plan_rows(
            qids=qids,
            A_by=A_by,
            B_by=B_by,
            icl_B_by=icl_B_by,
            icl_C_by=icl_C_by,
            icl_D_by=icl_D_by,
            n_trials=args.n_trials,
            seed=args.seed,
            bundle_size=bundle_size,
        )
        trial_plan = {
            "config_fingerprint": config_fingerprint,
            "bundle_size": int(bundle_size),
            "plan_rows": plan_rows,
        }
        _write_json(plan_path, trial_plan)

    plan_by_q: Dict[str, List[Dict[str, object]]] = {}
    for row in trial_plan["plan_rows"]:
        plan_by_q.setdefault(str(row["q_id"]), []).append(row)
    for q_id in plan_by_q:
        plan_by_q[q_id] = sorted(plan_by_q[q_id], key=lambda row: int(row["trial_index"]))

    spec = get_model_spec(args.model_spec)
    tok_add_special = bool(spec.prepend_bos)

    model, tokenizer, diagnostics = load_hf_model_and_tokenizer(
        model_name=args.model,
        model_spec=args.model_spec,
        device=args.device,
        device_map=None,
        dtype=args.dtype,
        quant=args.quant,
    )
    model.eval()

    skipped_qids = 0
    completed_qids = set(state.get("completed_qids", []))
    eligible_qids = [q_id for q_id in qids if q_id in plan_by_q]
    total_prompt_evals = sum(len(plan_by_q[q_id]) * len(shots) * 5 for q_id in eligible_qids)
    completed_prompt_evals = sum(len(plan_by_q[q_id]) * len(shots) * 5 for q_id in completed_qids if q_id in plan_by_q)
    progress_every = max(1, total_prompt_evals // 200) if total_prompt_evals else 1
    scorer_start = time.time()

    print(
        f"[resume] out_dir={out_dir} completed_qids={len(completed_qids)}/{len(eligible_qids)} "
        f"total_prompt_evals={total_prompt_evals}",
        flush=True,
    )

    for q_idx, q_id in enumerate(qids, start=1):
        A_q = A_by.get(q_id, [])
        B_q = B_by.get(q_id, [])
        icl_B_q = icl_B_by.get(q_id, [])
        icl_C_q = icl_C_by.get(q_id, [])
        icl_D_q = icl_D_by.get(q_id, [])

        if (
            len(A_q) < bundle_size
            or len(B_q) < bundle_size
            or len(icl_B_q) < 1
            or len(icl_C_q) < 1
            or len(icl_D_q) < 1
        ):
            skipped_qids += 1
            continue

        B_query = _select_query(icl_B_q)
        C_query = _select_query(icl_C_q)
        D_query = _select_query(icl_D_q)

        # Prevent query/demo leakage by filtering out any demo rows that match
        # the active query pairs for this q_id.
        forbidden_A = {
            (B_query["input"], B_query["output"]),
            (C_query["input"], C_query["output"]),
            (D_query["input"], D_query["output"]),
        }
        forbidden_B = {
            (C_query["input"], C_query["output"]),
            (D_query["input"], D_query["output"]),
        }
        A_pool = [
            row for row in A_q if (row["input"], row["output"]) not in forbidden_A
        ]
        B_pool = [
            row for row in B_q if (row["input"], row["output"]) not in forbidden_B
        ]
        if len(A_pool) < bundle_size or len(B_pool) < bundle_size:
            skipped_qids += 1
            print(
                f"[warn] skip q_id={q_id}: filtered demo pool too small "
                f"(A_pool={len(A_pool)}, B_pool={len(B_pool)}, bundle_size={bundle_size})",
                flush=True,
            )
            continue

        if q_id not in plan_by_q:
            skipped_qids += 1
            print(f"[warn] skip q_id={q_id}: no trial plan rows", flush=True)
            continue

        if q_id in completed_qids:
            print(f"[qid-skip] q_id={q_id} already completed", flush=True)
            continue

        print(
            f"[qid-start] q_id={q_id} "
            f"({q_idx}/{len(qids)}) "
            f"bundle_size={bundle_size} "
            f"A_pool={len(A_pool)} B_pool={len(B_pool)} "
            f"B_query=({B_query['input']}->{B_query['output']}) "
            f"D_query=({D_query['input']}->{D_query['output']})",
            flush=True,
        )

        row_id_to_A = {int(row["row_id"]): row for row in A_pool}
        row_id_to_B = {int(row["row_id"]): row for row in B_pool}
        q_plan_rows = plan_by_q[q_id]
        q_rows_path = _qid_raw_rows_path(out_dir, q_id)
        q_edge_topk_path = _qid_edge_topk_raw_rows_path(out_dir, q_id)
        q_rows: List[Dict[str, object]] = _dedupe_by_edge_key(_read_jsonl(q_rows_path))
        edge_topk_rows: List[Dict[str, object]] = (
            _dedupe_by_edge_key(_read_jsonl(q_edge_topk_path)) if edge_topk_enabled else []
        )
        completed_edge_keys = {_edge_eval_key(row) for row in q_rows}
        raw_logprobs: List[float] = [float(row["target_logprob_raw"]) for row in q_rows]
        q_edge_topk_count = len(edge_topk_rows)
        completed_prompt_evals += len(completed_edge_keys)

        print(
            f"[qid-resume] q_id={q_id} completed_edges={len(completed_edge_keys)}/{len(q_plan_rows) * len(shots) * 5}",
            flush=True,
        )

        for plan_row in q_plan_rows:
            trial_index = int(plan_row["trial_index"])
            plan_bundle_size = int(plan_row.get("bundle_size", bundle_size))
            A_bundle = [row_id_to_A[int(row_id)] for row_id in _bundle_row_ids_from_plan(plan_row, "A")]
            B_bundle = [row_id_to_B[int(row_id)] for row_id in _bundle_row_ids_from_plan(plan_row, "B")]
            if len(A_bundle) < plan_bundle_size or len(B_bundle) < plan_bundle_size:
                raise ValueError(
                    f"Plan bundle shorter than expected for q_id={q_id} trial={trial_index} "
                    f"(A_bundle={len(A_bundle)} B_bundle={len(B_bundle)} plan_bundle_size={plan_bundle_size})"
                )

            for shot in shots:
                demos_A = A_bundle[:shot]
                demos_B = B_bundle[:shot]

                edges = [
                    ("AB", demos_A, B_query, "A", "B"),
                    ("AC", demos_A, C_query, "A", "C"),
                    ("AD", demos_A, D_query, "A", "D"),
                    ("BC", demos_B, C_query, "B", "C"),
                    ("BD", demos_B, D_query, "B", "D"),
                ]

                for edge, demos, query, demo_src, q_src in edges:
                    edge_key = (trial_index, int(shot), edge)
                    if edge_key in completed_edge_keys:
                        continue
                    for demo in demos:
                        if demo["input"] == query["input"] and demo["output"] == query["output"]:
                            raise ValueError(
                                f"Query overlaps with demo for q_id={q_id} edge={edge} "
                                f"trial={trial_index} shot={shot} query_source={q_src}"
                            )
                    demo_pairs = [(d["input"], d["output"]) for d in demos]
                    query_pair = (query["input"], query["output"])
                    prefix_str, full_str = build_prompt_qa(
                        demo_pairs,
                        query_pair,
                        prepend_bos_token=False,
                        prepend_space=True,
                    )
                    if not full_str.startswith(prefix_str):
                        raise ValueError(
                            "Full prompt does not start with prefix: "
                            f"q_id={q_id} edge={edge} shot={shot} trial={trial_index} "
                            f"prefix_tail={repr(prefix_str[-120:])}"
                        )
                    target_suffix_str = full_str[len(prefix_str) :]
                    inputs = tokenizer(
                        prefix_str, return_tensors="pt", add_special_tokens=tok_add_special
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                    next_logits = outputs.logits[0, -1, :]
                    next_logprobs = torch.log_softmax(next_logits, dim=-1)
                    prefix_ids_a = inputs["input_ids"][0].tolist()
                    target_id = _target_first_token_id_with_checks(
                        tokenizer,
                        prefix_str,
                        full_str,
                        tok_add_special,
                        prefix_ids_a,
                        q_id=q_id,
                        edge=edge,
                        shot=shot,
                        trial_index=trial_index,
                        spec_prepend_bos=spec.prepend_bos,
                    )
                    target_logit = float(next_logits[target_id].item())
                    target_logprob = float(next_logprobs[target_id].item())
                    target_prob = float(torch.exp(next_logprobs[target_id]).item())
                    prompt_len_tokens = int(inputs["input_ids"].shape[1])
                    target_token_str = tokenizer.decode([target_id])

                    row = {
                        "q_id": q_id,
                        "trial_index": trial_index,
                        "shot": shot,
                        "edge": edge,
                        "seed": args.seed,
                        "model": args.model,
                        "model_spec": args.model_spec,
                        "quant": args.quant,
                        "dtype": args.dtype,
                        "device": args.device,
                        "query_source": q_src,
                        "query_input": query["input"],
                        "target_str": query["output"],
                        "target_suffix_str": target_suffix_str,
                        "query_row_id": query["row_id"],
                        "demo_source": demo_src,
                        "demo_bundle_size": plan_bundle_size,
                        "demo_ids_bundle": json.dumps(
                            [d["row_id"] for d in (A_bundle if demo_src == "A" else B_bundle)]
                        ),
                        "demo_ids_used": json.dumps([d["row_id"] for d in demos]),
                        "target_first_token_id": target_id,
                        "target_token_str": target_token_str,
                        "target_logprob_raw": target_logprob,
                        "target_prob_raw": target_prob,
                        "target_logit": target_logit,
                        "prompt_len_tokens": prompt_len_tokens,
                    }
                    q_rows.append(row)
                    _append_jsonl(q_rows_path, row)
                    raw_logprobs.append(target_logprob)
                    if edge_topk_enabled and edge in edge_topk_edges:
                        edge_row = {
                            "q_id": q_id,
                            "trial_index": trial_index,
                            "shot": shot,
                            "edge": edge,
                            "query_source": q_src,
                            "query_input": query["input"],
                            "target_str": query["output"],
                            "target_first_token_id": target_id,
                            "target_token_str": target_token_str,
                            "target_logprob_raw": target_logprob,
                            "target_prob_raw": target_prob,
                            "target_logit": target_logit,
                            "target_rank_in_vocab": _target_rank_in_vocab(next_logits, target_id),
                            **_collect_edge_topk(
                                tokenizer=tokenizer,
                                next_logits=next_logits,
                                next_logprobs=next_logprobs,
                                k=args.edge_topk_k,
                            ),
                        }
                        edge_topk_rows.append(edge_row)
                        _append_jsonl(q_edge_topk_path, edge_row)
                        q_edge_topk_count += 1
                    completed_edge_keys.add(edge_key)
                    completed_prompt_evals += 1
                    if (
                        completed_prompt_evals == 1
                        or completed_prompt_evals % progress_every == 0
                        or completed_prompt_evals == total_prompt_evals
                    ):
                        print(
                            _progress_line(
                                q_id=q_id,
                                q_idx=q_idx,
                                q_total=len(qids),
                                trial_index=trial_index,
                                n_trials=args.n_trials,
                                shot=int(shot),
                                edge=edge,
                                done=completed_prompt_evals,
                                total=total_prompt_evals,
                                elapsed_sec=time.time() - scorer_start,
                            ),
                            flush=True,
                        )

        p_low = _percentile(raw_logprobs, 5)
        p_high = _percentile(raw_logprobs, 95)
        if p_high == p_low:
            for row in q_rows:
                row["target_s_norm"] = 0.5
        else:
            for row in q_rows:
                x = row["target_logprob_raw"]
                s = (x - p_low) / (p_high - p_low)
                if s < 0.0:
                    s = 0.0
                elif s > 1.0:
                    s = 1.0
                row["target_s_norm"] = s

        for row in q_rows:
            row["norm_p_low"] = p_low
            row["norm_p_high"] = p_high
            row["norm_method"] = "robust_minmax_p05_p95"
            row["norm_scope"] = "qid_all_edges_all_shots_this_run"

        if edge_topk_enabled:
            norm_lookup = {
                (row["q_id"], int(row["trial_index"]), int(row["shot"]), row["edge"]): row
                for row in q_rows
            }
            for edge_row in edge_topk_rows:
                if edge_row["q_id"] != q_id:
                    continue
                key = (
                    edge_row["q_id"],
                    int(edge_row["trial_index"]),
                    int(edge_row["shot"]),
                    edge_row["edge"],
                )
                norm_row = norm_lookup[key]
                edge_row["target_s_norm"] = norm_row["target_s_norm"]
                edge_row["norm_p_low"] = norm_row["norm_p_low"]
                edge_row["norm_p_high"] = norm_row["norm_p_high"]
                edge_row["norm_method"] = norm_row["norm_method"]
                edge_row["norm_scope"] = norm_row["norm_scope"]

        write_header = not os.path.exists(args.out_csv) or os.path.getsize(args.out_csv) == 0
        fieldnames = list(q_rows[0].keys())
        with open(args.out_csv, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(q_rows)
            handle.flush()

        if edge_topk_enabled:
            with open(edge_topk_jsonl, "a", encoding="utf-8") as handle:
                for row in edge_topk_rows:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()

            grouped_changes: Dict[Tuple[str, int, str], List[Dict[str, object]]] = {}
            for row in edge_topk_rows:
                key = (str(row["q_id"]), int(row["trial_index"]), str(row["edge"]))
                grouped_changes.setdefault(key, []).append(row)

            change_rows: List[Dict[str, object]] = []
            for (q_edge_id, trial_index, edge), rows_for_key in sorted(grouped_changes.items()):
                rows_sorted = sorted(rows_for_key, key=lambda r: int(r["shot"]))
                for left, right in zip(rows_sorted[:-1], rows_sorted[1:]):
                    change_rows.append(
                        {
                            "q_id": q_edge_id,
                            "trial_index": trial_index,
                            "edge": edge,
                            "shot_from": int(left["shot"]),
                            "shot_to": int(right["shot"]),
                            "target_logprob_from": left["target_logprob_raw"],
                            "target_logprob_to": right["target_logprob_raw"],
                            "target_logprob_delta": float(right["target_logprob_raw"]) - float(left["target_logprob_raw"]),
                            "target_s_norm_from": left["target_s_norm"],
                            "target_s_norm_to": right["target_s_norm"],
                            "target_s_norm_delta": float(right["target_s_norm"]) - float(left["target_s_norm"]),
                            "target_rank_from": int(left["target_rank_in_vocab"]),
                            "target_rank_to": int(right["target_rank_in_vocab"]),
                            "target_rank_delta": int(right["target_rank_in_vocab"]) - int(left["target_rank_in_vocab"]),
                            "top1_candidate_from": (left["lexical_candidates"][0] if left["lexical_candidates"] else ""),
                            "top1_candidate_to": (right["lexical_candidates"][0] if right["lexical_candidates"] else ""),
                            "top1_changed": (
                                (left["lexical_candidates"][0] if left["lexical_candidates"] else "")
                                != (right["lexical_candidates"][0] if right["lexical_candidates"] else "")
                            ),
                            "lexical_token_id_jaccard": _jaccard(left["lexical_candidate_token_ids"], right["lexical_candidate_token_ids"]),
                            "lexical_text_jaccard": _jaccard(left["lexical_candidates"], right["lexical_candidates"]),
                            "lexical_overlap_count": len(set(left["lexical_candidates"]) & set(right["lexical_candidates"])),
                            "canonical_overlap_count": len(
                                set(left["lexical_candidate_canonical_forms"]) & set(right["lexical_candidate_canonical_forms"])
                            ),
                        }
                    )

            write_header = not os.path.exists(edge_topk_change_csv) or os.path.getsize(edge_topk_change_csv) == 0
            with open(edge_topk_change_csv, "a", encoding="utf-8", newline="") as handle:
                fieldnames = list(change_rows[0].keys()) if change_rows else [
                    "q_id",
                    "trial_index",
                    "edge",
                    "shot_from",
                    "shot_to",
                    "target_logprob_from",
                    "target_logprob_to",
                    "target_logprob_delta",
                    "target_s_norm_from",
                    "target_s_norm_to",
                    "target_s_norm_delta",
                    "target_rank_from",
                    "target_rank_to",
                    "target_rank_delta",
                    "top1_candidate_from",
                    "top1_candidate_to",
                    "top1_changed",
                    "lexical_token_id_jaccard",
                    "lexical_text_jaccard",
                    "lexical_overlap_count",
                    "canonical_overlap_count",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                if change_rows:
                    writer.writerows(change_rows)
                handle.flush()

        completed_qids.add(q_id)
        state["completed_qids"] = sorted(completed_qids)
        _write_json(state_path, state)
        print(
            f"[qid-done] q_id={q_id} "
            f"score_rows={len(q_rows)} "
            f"edge_topk_rows={q_edge_topk_count} "
            f"completed_qids={len(completed_qids)}/{len(eligible_qids)}",
            flush=True,
        )

    if not completed_qids and skipped_qids == len(qids):
        raise ValueError("No rows generated; check q_id filters or input files.")

    with open(args.out_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        final_rows = list(reader)

    s_vals = [float(row["target_s_norm"]) for row in final_rows]
    print(
        "summary: "
        f"qids_processed={len(completed_qids)} skipped={skipped_qids} "
        f"s_norm_min={min(s_vals):.4f} s_norm_max={max(s_vals):.4f}",
        flush=True,
    )
    if not all(0.0 <= s <= 1.0 for s in s_vals):
        raise AssertionError("target_s_norm out of [0,1] bounds")

    if edge_topk_enabled:
        print(f"edge_topk_jsonl={edge_topk_jsonl}", flush=True)
        print(f"edge_topk_change_csv={edge_topk_change_csv}", flush=True)
        print(
            f"[edge-topk] saved completed_qids={len(completed_qids)} "
            f"files=({os.path.basename(edge_topk_jsonl)},{os.path.basename(edge_topk_change_csv)})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
