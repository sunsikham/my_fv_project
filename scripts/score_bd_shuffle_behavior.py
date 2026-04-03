#!/usr/bin/env python3
"""Score Q1-style BD alternating vs shuffled behavior with explicit layouts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in os.sys.path:
    os.sys.path.insert(0, _ROOT)

from fv.hf_loader import load_hf_model_and_tokenizer
from fv.model_spec import get_model_spec
from fv.prompting import build_prompt_qa
from scripts.score_cross_relation_target_logit import (
    _append_jsonl,
    _collect_edge_topk,
    _file_sha256,
    _parse_shot_list,
    _read_relation_csv,
    _select_query,
    _stable_hash,
    _target_first_token_id_with_checks,
    _target_rank_in_vocab,
    _write_json,
)


FIELDNAMES = [
    "q_id",
    "query_side",
    "regime_id",
    "baseline_regime_id",
    "case_kind",
    "case_index",
    "layout_pattern",
    "trial_index",
    "shot",
    "seed",
    "model",
    "model_spec",
    "quant",
    "dtype",
    "device",
    "query_input",
    "target_str",
    "query_row_id",
    "full_layout_demo_row_ids",
    "full_layout_demo_sources",
    "prefix_demo_row_ids",
    "prefix_demo_sources",
    "target_first_token_id",
    "target_token_str",
    "target_logprob_raw",
    "target_prob_raw",
    "target_logit",
    "target_rank_in_vocab",
    "prompt_len_tokens",
]


D_QUERY_REGIMES = [
    {
        "regime_id": "BDBDBD_D",
        "query_side": "D",
        "baseline_regime_id": "BDBDBD_D",
        "case_kind": "regular",
        "case_index": 0,
        "layout_pattern": "BDBDBDBDB",
    },
    {
        "regime_id": "BD_SHUF_D1_D",
        "query_side": "D",
        "baseline_regime_id": "BDBDBD_D",
        "case_kind": "shuffled",
        "case_index": 1,
        "layout_pattern": "BDDBBDBDB",
    },
    {
        "regime_id": "BD_SHUF_D2_D",
        "query_side": "D",
        "baseline_regime_id": "BDBDBD_D",
        "case_kind": "shuffled",
        "case_index": 2,
        "layout_pattern": "BBDDBDDBB",
    },
    {
        "regime_id": "BD_SHUF_D3_D",
        "query_side": "D",
        "baseline_regime_id": "BDBDBD_D",
        "case_kind": "shuffled",
        "case_index": 3,
        "layout_pattern": "BBBDDBDDB",
    },
    {
        "regime_id": "BD_SHUF_D4_D",
        "query_side": "D",
        "baseline_regime_id": "BDBDBD_D",
        "case_kind": "shuffled",
        "case_index": 4,
        "layout_pattern": "BBBBDDDDB",
    },
    {
        "regime_id": "BD_SHUF_D5_D",
        "query_side": "D",
        "baseline_regime_id": "BDBDBD_D",
        "case_kind": "shuffled",
        "case_index": 5,
        "layout_pattern": "DBBBDDBBD",
    },
]


B_QUERY_REGIMES = [
    {
        "regime_id": "DBDBDB_B",
        "query_side": "B",
        "baseline_regime_id": "DBDBDB_B",
        "case_kind": "regular",
        "case_index": 0,
        "layout_pattern": "DBDBDBDBD",
    },
    {
        "regime_id": "BD_SHUF_B1_B",
        "query_side": "B",
        "baseline_regime_id": "DBDBDB_B",
        "case_kind": "shuffled",
        "case_index": 1,
        "layout_pattern": "DBBDDBDBD",
    },
    {
        "regime_id": "BD_SHUF_B2_B",
        "query_side": "B",
        "baseline_regime_id": "DBDBDB_B",
        "case_kind": "shuffled",
        "case_index": 2,
        "layout_pattern": "DDBBDBBDD",
    },
    {
        "regime_id": "BD_SHUF_B3_B",
        "query_side": "B",
        "baseline_regime_id": "DBDBDB_B",
        "case_kind": "shuffled",
        "case_index": 3,
        "layout_pattern": "DDDBBDBBD",
    },
    {
        "regime_id": "BD_SHUF_B4_B",
        "query_side": "B",
        "baseline_regime_id": "DBDBDB_B",
        "case_kind": "shuffled",
        "case_index": 4,
        "layout_pattern": "DDDDBBBBD",
    },
    {
        "regime_id": "BD_SHUF_B5_B",
        "query_side": "B",
        "baseline_regime_id": "DBDBDB_B",
        "case_kind": "shuffled",
        "case_index": 5,
        "layout_pattern": "BDDDBBDDB",
    },
]


ALL_REGIMES = D_QUERY_REGIMES + B_QUERY_REGIMES


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _pair_key(row: Dict[str, str]) -> Tuple[str, str]:
    return (str(row["input"]), str(row["output"]))


def _parse_qid_values(*raw_values: Optional[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in raw_values:
        if not raw:
            continue
        for part in str(raw).split(","):
            key = part.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _stable_seed(*parts: object) -> int:
    payload = {"parts": [str(part) for part in parts]}
    digest = _stable_hash(payload)
    return int(digest[:16], 16) % (2**31 - 1)


def _validate_layouts() -> None:
    for regime in ALL_REGIMES:
        pattern = str(regime["layout_pattern"])
        if len(pattern) != 9:
            raise ValueError(f"Layout length must be 9: {regime['regime_id']}={pattern}")
        counts = Counter(pattern)
        side = str(regime["query_side"])
        if side == "D":
            expected = {"B": 5, "D": 4}
        elif side == "B":
            expected = {"B": 4, "D": 5}
        else:
            raise ValueError(f"Unsupported query_side: {side}")
        if counts != expected:
            raise ValueError(
                f"Layout source counts mismatch for {regime['regime_id']}: "
                f"got={dict(counts)} expected={expected}"
            )
        if side == "D" and pattern == "BDBDBDBDB":
            continue
        if side == "B" and pattern == "DBDBDBDBD":
            continue
        alternating = "BDBDBDBDB" if side == "D" else "DBDBDBDBD"
        if pattern == alternating:
            raise ValueError(f"Shuffled layout matches alternating baseline: {regime['regime_id']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score BD alternating vs shuffled behavior.")
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--model_spec", required=True, help="Model spec (e.g. llama3)")
    parser.add_argument("--device", required=True, help="Device (cpu/cuda)")
    parser.add_argument("--dtype", default=None, help="fp32/fp16/bf16")
    parser.add_argument("--quant", default="none", help="Quantization mode: none/4bit/8bit/auto")
    parser.add_argument("--relationB_ex_path", required=True, help="B demos CSV")
    parser.add_argument("--relationD_ex_path", required=True, help="D demos CSV")
    parser.add_argument("--icl_B_path", required=True, help="B query CSV")
    parser.add_argument("--icl_D_path", required=True, help="D query CSV")
    parser.add_argument("--shot_list", default="1,3,5,7,9", help="Comma-separated shot list")
    parser.add_argument("--n_trials", type=int, required=True, help="Trials per q_id")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--qid", default="Q1", help="Single q_id or comma-separated q_id list")
    parser.add_argument("--q_list", default=None, help="Optional comma-separated q_id list")
    parser.add_argument("--out_csv", required=True, help="Raw sweep CSV path")
    parser.add_argument("--edge_topk_jsonl", required=True, help="Lexical top-k JSONL path")
    parser.add_argument("--progress_json", required=True, help="Machine-readable progress JSON path")
    parser.add_argument(
        "--trial_plan_json",
        default=None,
        help="Optional output path for deterministic trial plan JSON",
    )
    parser.add_argument("--edge_topk_k", type=int, default=10, help="Lexical top-k size")
    return parser.parse_args()


def _expected_total_evals(qids: Sequence[str], n_trials: int, shots: Sequence[int]) -> int:
    return len(qids) * n_trials * len(shots) * len(ALL_REGIMES)


def _write_progress(
    *,
    path: str,
    status: str,
    q_ids: Sequence[str],
    done: int,
    total: int,
    current_q_id: Optional[str] = None,
    current_regime_id: Optional[str] = None,
    current_trial_index: Optional[int] = None,
    current_shot: Optional[int] = None,
    elapsed_sec: Optional[float] = None,
    eta_sec: Optional[float] = None,
    last_target_logprob: Optional[float] = None,
    last_target_prob: Optional[float] = None,
    last_target_rank: Optional[int] = None,
    last_top1_candidate: Optional[str] = None,
) -> None:
    pct = (100.0 * done / total) if total else 100.0
    payload = {
        "updated_at": _utc_now(),
        "status": status,
        "q_ids": list(q_ids),
        "done": int(done),
        "total": int(total),
        "pct": pct,
        "current_q_id": current_q_id,
        "current_regime_id": current_regime_id,
        "current_trial_index": current_trial_index,
        "current_shot": current_shot,
        "elapsed_sec": (float(elapsed_sec) if elapsed_sec is not None else None),
        "eta_sec": (float(eta_sec) if eta_sec is not None else None),
        "last_target_logprob": last_target_logprob,
        "last_target_prob": last_target_prob,
        "last_target_rank": last_target_rank,
        "last_top1_candidate": last_top1_candidate,
    }
    _write_json(path, payload)


def _format_progress_line(
    *,
    q_id: str,
    regime_id: str,
    done: int,
    total: int,
    trial_index: int,
    n_trials: int,
    shot: int,
    elapsed_sec: float,
    eta_sec: float,
    target_logprob: float,
    target_prob: float,
    target_rank: int,
    top1_candidate: str,
) -> str:
    pct = (100.0 * done / total) if total else 100.0
    eta_min, eta_s = divmod(int(max(0.0, eta_sec)), 60)
    eta_h, eta_min = divmod(eta_min, 60)
    return (
        f"[progress] q_id={q_id} regime={regime_id} "
        f"trial={trial_index + 1}/{n_trials} shot={shot} "
        f"done={done}/{total} pct={pct:.1f}% "
        f"elapsed={int(elapsed_sec)}s eta={eta_h:02d}:{eta_min:02d}:{eta_s:02d} "
        f"target_lp={target_logprob:.4f} target_p={target_prob:.6f} "
        f"target_rank={target_rank} top1={top1_candidate}"
    )


def _materialize_layout(
    *,
    layout_pattern: str,
    b_rows: Sequence[Dict[str, str]],
    d_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    b_need = layout_pattern.count("B")
    d_need = layout_pattern.count("D")
    if len(b_rows) < b_need or len(d_rows) < d_need:
        raise ValueError(
            f"Insufficient rows for layout={layout_pattern}: "
            f"len(B)={len(b_rows)} len(D)={len(d_rows)}"
        )
    out: List[Dict[str, str]] = []
    b_idx = 0
    d_idx = 0
    for symbol in layout_pattern:
        if symbol == "B":
            out.append(b_rows[b_idx])
            b_idx += 1
        elif symbol == "D":
            out.append(d_rows[d_idx])
            d_idx += 1
        else:
            raise ValueError(f"Unexpected layout symbol: {symbol}")
    return out


def _build_trial_plan(
    *,
    qids: Sequence[str],
    n_trials: int,
    seed: int,
    b_demo_by_q: Dict[str, List[Dict[str, str]]],
    d_demo_by_q: Dict[str, List[Dict[str, str]]],
    b_query_by_q: Dict[str, List[Dict[str, str]]],
    d_query_by_q: Dict[str, List[Dict[str, str]]],
) -> List[Dict[str, object]]:
    plan_rows: List[Dict[str, object]] = []
    for q_id in qids:
        b_query = _select_query(b_query_by_q[q_id])
        d_query = _select_query(d_query_by_q[q_id])
        forbidden = {_pair_key(b_query), _pair_key(d_query)}
        b_pool = [row for row in b_demo_by_q[q_id] if _pair_key(row) not in forbidden]
        d_pool = [row for row in d_demo_by_q[q_id] if _pair_key(row) not in forbidden]
        if len(b_pool) < 5 or len(d_pool) < 5:
            raise ValueError(
                f"Insufficient demo pool after excluding queries for q_id={q_id}: "
                f"B_pool={len(b_pool)} D_pool={len(d_pool)}"
            )
        for trial_index in range(n_trials):
            rng = random.Random(_stable_seed(seed, q_id, trial_index, "bd_shuffle"))
            b_rows = rng.sample(b_pool, 5)
            d_rows = rng.sample(d_pool, 5)
            rng.shuffle(b_rows)
            rng.shuffle(d_rows)
            plan_rows.append(
                {
                    "q_id": q_id,
                    "trial_index": int(trial_index),
                    "B5_row_ids": [int(row["row_id"]) for row in b_rows],
                    "D5_row_ids": [int(row["row_id"]) for row in d_rows],
                    "B_query_row_id": int(b_query["row_id"]),
                    "D_query_row_id": int(d_query["row_id"]),
                }
            )
    return plan_rows


def _rows_by_row_id(rows: Iterable[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    return {int(row["row_id"]): row for row in rows}


def _regimes_for_q_side(side: str) -> List[Dict[str, object]]:
    return D_QUERY_REGIMES if side == "D" else B_QUERY_REGIMES


def main() -> int:
    _validate_layouts()
    args = _parse_args()
    shots = _parse_shot_list(args.shot_list)
    if len(set(shots)) != len(shots):
        raise ValueError("shot_list contains duplicates")
    if any(shot < 1 for shot in shots):
        raise ValueError("shot_list must contain positive shots only")
    if max(shots) > 9:
        raise ValueError("shot_list includes value > 9")
    if args.n_trials < 1:
        raise ValueError("--n_trials must be >= 1")
    if args.edge_topk_k < 1:
        raise ValueError("--edge_topk_k must be >= 1")

    qids = _parse_qid_values(args.qid, args.q_list)
    if not qids:
        qids = ["Q1"]

    out_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.edge_topk_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.progress_json) or ".", exist_ok=True)

    b_demo_rows = _read_relation_csv(args.relationB_ex_path)
    d_demo_rows = _read_relation_csv(args.relationD_ex_path)
    b_query_rows = _read_relation_csv(args.icl_B_path)
    d_query_rows = _read_relation_csv(args.icl_D_path)

    b_demo_by_q: Dict[str, List[Dict[str, str]]] = {}
    d_demo_by_q: Dict[str, List[Dict[str, str]]] = {}
    b_query_by_q: Dict[str, List[Dict[str, str]]] = {}
    d_query_by_q: Dict[str, List[Dict[str, str]]] = {}
    for row in b_demo_rows:
        b_demo_by_q.setdefault(str(row["q_id"]), []).append(row)
    for row in d_demo_rows:
        d_demo_by_q.setdefault(str(row["q_id"]), []).append(row)
    for row in b_query_rows:
        b_query_by_q.setdefault(str(row["q_id"]), []).append(row)
    for row in d_query_rows:
        d_query_by_q.setdefault(str(row["q_id"]), []).append(row)

    for q_id in qids:
        if q_id not in b_demo_by_q:
            raise ValueError(f"q_id={q_id} missing from relation B demos")
        if q_id not in d_demo_by_q:
            raise ValueError(f"q_id={q_id} missing from relation D demos")
        if q_id not in b_query_by_q:
            raise ValueError(f"q_id={q_id} missing from icl_B queries")
        if q_id not in d_query_by_q:
            raise ValueError(f"q_id={q_id} missing from icl_D queries")

    config_fingerprint = _stable_hash(
        {
            "model": args.model,
            "model_spec": args.model_spec,
            "device": args.device,
            "dtype": args.dtype,
            "quant": args.quant,
            "relationB_ex_path": os.path.abspath(args.relationB_ex_path),
            "relationD_ex_path": os.path.abspath(args.relationD_ex_path),
            "icl_B_path": os.path.abspath(args.icl_B_path),
            "icl_D_path": os.path.abspath(args.icl_D_path),
            "relationB_sha256": _file_sha256(args.relationB_ex_path),
            "relationD_sha256": _file_sha256(args.relationD_ex_path),
            "icl_B_sha256": _file_sha256(args.icl_B_path),
            "icl_D_sha256": _file_sha256(args.icl_D_path),
            "shots": list(shots),
            "n_trials": int(args.n_trials),
            "seed": int(args.seed),
            "qids": list(qids),
            "layouts": {row["regime_id"]: row["layout_pattern"] for row in ALL_REGIMES},
            "scorer_code_sha256": _file_sha256(__file__),
        }
    )

    trial_plan_rows = _build_trial_plan(
        qids=qids,
        n_trials=args.n_trials,
        seed=args.seed,
        b_demo_by_q=b_demo_by_q,
        d_demo_by_q=d_demo_by_q,
        b_query_by_q=b_query_by_q,
        d_query_by_q=d_query_by_q,
    )
    if args.trial_plan_json:
        _write_json(
            args.trial_plan_json,
            {
                "config_fingerprint": config_fingerprint,
                "q_ids": list(qids),
                "n_trials": int(args.n_trials),
                "shots": list(shots),
                "plan_rows": trial_plan_rows,
            },
        )

    trial_plan_by_q: Dict[str, List[Dict[str, object]]] = {}
    for row in trial_plan_rows:
        trial_plan_by_q.setdefault(str(row["q_id"]), []).append(row)
    for q_id in trial_plan_by_q:
        trial_plan_by_q[q_id] = sorted(
            trial_plan_by_q[q_id], key=lambda row: int(row["trial_index"])
        )

    spec = get_model_spec(args.model_spec)
    tok_add_special = bool(spec.prepend_bos)
    model, tokenizer, _diagnostics = load_hf_model_and_tokenizer(
        model_name=args.model,
        model_spec=args.model_spec,
        device=args.device,
        device_map=None,
        dtype=args.dtype,
        quant=args.quant,
    )
    model.eval()

    b_demo_maps = {q_id: _rows_by_row_id(rows) for q_id, rows in b_demo_by_q.items()}
    d_demo_maps = {q_id: _rows_by_row_id(rows) for q_id, rows in d_demo_by_q.items()}
    b_query_maps = {q_id: _rows_by_row_id(rows) for q_id, rows in b_query_by_q.items()}
    d_query_maps = {q_id: _rows_by_row_id(rows) for q_id, rows in d_query_by_q.items()}

    total_evals = _expected_total_evals(qids, args.n_trials, shots)
    progress_every = max(1, total_evals // 200) if total_evals else 1
    scorer_start = time.time()
    done = 0

    _write_progress(
        path=args.progress_json,
        status="starting",
        q_ids=qids,
        done=0,
        total=total_evals,
    )
    print(
        f"[start] q_ids={','.join(qids)} n_trials={args.n_trials} "
        f"shots={shots} regimes={len(ALL_REGIMES)} total_evals={total_evals}",
        flush=True,
    )

    with open(args.out_csv, "w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=FIELDNAMES)
        writer.writeheader()

        for q_idx, q_id in enumerate(qids, start=1):
            print(
                f"[q-start] q_id={q_id} ({q_idx}/{len(qids)}) "
                f"expected_evals={args.n_trials * len(shots) * len(ALL_REGIMES)}",
                flush=True,
            )
            trial_rows = trial_plan_by_q[q_id]
            b_query = _select_query(b_query_by_q[q_id])
            d_query = _select_query(d_query_by_q[q_id])

            for regime in ALL_REGIMES:
                regime_id = str(regime["regime_id"])
                query_side = str(regime["query_side"])
                layout_pattern = str(regime["layout_pattern"])
                baseline_regime_id = str(regime["baseline_regime_id"])
                case_kind = str(regime["case_kind"])
                case_index = int(regime["case_index"])
                regime_start = time.time()
                regime_rows_done = 0
                regime_top1_count = 0
                regime_logprob_sum = 0.0

                print(
                    f"[regime-start] q_id={q_id} regime={regime_id} query_side={query_side} "
                    f"layout={layout_pattern} case_kind={case_kind} case_index={case_index}",
                    flush=True,
                )

                for plan_row in trial_rows:
                    trial_index = int(plan_row["trial_index"])
                    b5_rows = [
                        b_demo_maps[q_id][int(row_id)] for row_id in plan_row["B5_row_ids"]
                    ]
                    d5_rows = [
                        d_demo_maps[q_id][int(row_id)] for row_id in plan_row["D5_row_ids"]
                    ]
                    if query_side == "D":
                        query = d_query_maps[q_id][int(plan_row["D_query_row_id"])]
                        selected_b = b5_rows
                        selected_d = d5_rows[:4]
                    elif query_side == "B":
                        query = b_query_maps[q_id][int(plan_row["B_query_row_id"])]
                        selected_b = b5_rows[:4]
                        selected_d = d5_rows
                    else:
                        raise ValueError(f"Unsupported query_side: {query_side}")

                    ordered_rows = _materialize_layout(
                        layout_pattern=layout_pattern,
                        b_rows=selected_b,
                        d_rows=selected_d,
                    )
                    if any(_pair_key(row) == _pair_key(query) for row in ordered_rows):
                        raise ValueError(
                            f"Query overlaps with demos: q_id={q_id} regime={regime_id} "
                            f"trial_index={trial_index}"
                        )

                    for shot in shots:
                        prefix_rows = ordered_rows[:shot]
                        demo_pairs = [(row["input"], row["output"]) for row in prefix_rows]
                        query_pair = (query["input"], query["output"])
                        prefix_str, full_str = build_prompt_qa(
                            demo_pairs,
                            query_pair,
                            prepend_bos_token=False,
                            prepend_space=True,
                        )
                        if not full_str.startswith(prefix_str):
                            raise ValueError(
                                f"Full prompt does not start with prefix for q_id={q_id} regime={regime_id}"
                            )
                        target_suffix_str = full_str[len(prefix_str) :]
                        inputs = tokenizer(
                            prefix_str, return_tensors="pt", add_special_tokens=tok_add_special
                        )
                        inputs = {key: value.to(model.device) for key, value in inputs.items()}
                        with torch.no_grad():
                            outputs = model(**inputs)
                        next_logits = outputs.logits[0, -1, :]
                        next_logprobs = torch.log_softmax(next_logits, dim=-1)
                        prefix_ids = inputs["input_ids"][0].tolist()
                        target_id = _target_first_token_id_with_checks(
                            tokenizer,
                            prefix_str,
                            full_str,
                            tok_add_special,
                            prefix_ids,
                            q_id=q_id,
                            edge=regime_id,
                            shot=int(shot),
                            trial_index=trial_index,
                            spec_prepend_bos=spec.prepend_bos,
                        )
                        target_logit = float(next_logits[target_id].item())
                        target_logprob = float(next_logprobs[target_id].item())
                        target_prob = float(torch.exp(next_logprobs[target_id]).item())
                        target_rank = int(_target_rank_in_vocab(next_logits, target_id))
                        topk_row = _collect_edge_topk(
                            tokenizer=tokenizer,
                            next_logits=next_logits,
                            next_logprobs=next_logprobs,
                            k=args.edge_topk_k,
                        )
                        raw_top1 = ""
                        if topk_row["raw_top_token_strs"]:
                            raw_top1 = str(topk_row["raw_top_token_strs"][0]["decoded"]).strip()

                        row = {
                            "q_id": q_id,
                            "query_side": query_side,
                            "regime_id": regime_id,
                            "baseline_regime_id": baseline_regime_id,
                            "case_kind": case_kind,
                            "case_index": case_index,
                            "layout_pattern": layout_pattern,
                            "trial_index": trial_index,
                            "shot": int(shot),
                            "seed": int(args.seed),
                            "model": args.model,
                            "model_spec": args.model_spec,
                            "quant": args.quant,
                            "dtype": args.dtype,
                            "device": args.device,
                            "query_input": query["input"],
                            "target_str": query["output"],
                            "query_row_id": int(query["row_id"]),
                            "full_layout_demo_row_ids": json.dumps(
                                [int(item["row_id"]) for item in ordered_rows]
                            ),
                            "full_layout_demo_sources": layout_pattern,
                            "prefix_demo_row_ids": json.dumps(
                                [int(item["row_id"]) for item in prefix_rows]
                            ),
                            "prefix_demo_sources": layout_pattern[:shot],
                            "target_first_token_id": int(target_id),
                            "target_token_str": tokenizer.decode([target_id]),
                            "target_logprob_raw": target_logprob,
                            "target_prob_raw": target_prob,
                            "target_logit": target_logit,
                            "target_rank_in_vocab": target_rank,
                            "prompt_len_tokens": int(inputs["input_ids"].shape[1]),
                        }
                        writer.writerow(row)
                        csv_handle.flush()

                        edge_row = {
                            "q_id": q_id,
                            "query_side": query_side,
                            "regime_id": regime_id,
                            "baseline_regime_id": baseline_regime_id,
                            "case_kind": case_kind,
                            "case_index": case_index,
                            "layout_pattern": layout_pattern,
                            "trial_index": trial_index,
                            "shot": int(shot),
                            "query_input": query["input"],
                            "target_str": query["output"],
                            "query_row_id": int(query["row_id"]),
                            "full_layout_demo_row_ids": [int(item["row_id"]) for item in ordered_rows],
                            "full_layout_demo_sources": layout_pattern,
                            "prefix_demo_row_ids": [int(item["row_id"]) for item in prefix_rows],
                            "prefix_demo_sources": layout_pattern[:shot],
                            "target_first_token_id": int(target_id),
                            "target_token_str": tokenizer.decode([target_id]),
                            "target_logprob_raw": target_logprob,
                            "target_prob_raw": target_prob,
                            "target_logit": target_logit,
                            "target_rank_in_vocab": target_rank,
                            "prompt_len_tokens": int(inputs["input_ids"].shape[1]),
                            "target_suffix_str": target_suffix_str,
                            **topk_row,
                        }
                        _append_jsonl(args.edge_topk_jsonl, edge_row)

                        done += 1
                        regime_rows_done += 1
                        regime_logprob_sum += target_logprob
                        if target_rank == 1:
                            regime_top1_count += 1

                        elapsed = time.time() - scorer_start
                        rate = (done / elapsed) if elapsed > 0 else 0.0
                        remaining = max(0, total_evals - done)
                        eta = (remaining / rate) if rate > 0 else 0.0
                        _write_progress(
                            path=args.progress_json,
                            status="running",
                            q_ids=qids,
                            done=done,
                            total=total_evals,
                            current_q_id=q_id,
                            current_regime_id=regime_id,
                            current_trial_index=trial_index,
                            current_shot=int(shot),
                            elapsed_sec=elapsed,
                            eta_sec=eta,
                            last_target_logprob=target_logprob,
                            last_target_prob=target_prob,
                            last_target_rank=target_rank,
                            last_top1_candidate=raw_top1,
                        )
                        if (
                            done == 1
                            or done % progress_every == 0
                            or done == total_evals
                        ):
                            print(
                                _format_progress_line(
                                    q_id=q_id,
                                    regime_id=regime_id,
                                    done=done,
                                    total=total_evals,
                                    trial_index=trial_index,
                                    n_trials=args.n_trials,
                                    shot=int(shot),
                                    elapsed_sec=elapsed,
                                    eta_sec=eta,
                                    target_logprob=target_logprob,
                                    target_prob=target_prob,
                                    target_rank=target_rank,
                                    top1_candidate=raw_top1,
                                ),
                                flush=True,
                            )

                regime_elapsed = time.time() - regime_start
                regime_mean_lp = regime_logprob_sum / max(1, regime_rows_done)
                regime_top1_acc = regime_top1_count / max(1, regime_rows_done)
                print(
                    f"[regime-end] q_id={q_id} regime={regime_id} rows={regime_rows_done} "
                    f"elapsed={int(regime_elapsed)}s mean_target_lp={regime_mean_lp:.4f} "
                    f"provisional_top1_acc={regime_top1_acc:.4f}",
                    flush=True,
                )

    elapsed = time.time() - scorer_start
    _write_progress(
        path=args.progress_json,
        status="completed",
        q_ids=qids,
        done=done,
        total=total_evals,
        elapsed_sec=elapsed,
        eta_sec=0.0,
    )
    print(
        f"[done] q_ids={','.join(qids)} done={done}/{total_evals} elapsed={int(elapsed)}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
