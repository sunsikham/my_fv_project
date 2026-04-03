#!/usr/bin/env python3
"""Validate parity between live selected-target PT rows and offline recompute rows."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


KEY_FIELDS: Tuple[str, ...] = ("family_id", "q_id", "trial_index", "shot", "edge")
DEFAULT_EXACT_FIELDS: Tuple[str, ...] = (
    "gold_target_str",
    "target_str",
    "target_first_token_id",
    "target_token_str",
    "target_rank_in_vocab",
)
DEFAULT_FLOAT_FIELDS: Tuple[str, ...] = (
    "target_logit",
    "target_logprob_raw",
    "target_prob_raw",
    "target_s_norm",
    "norm_p_low",
    "norm_p_high",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate parity between live selected-target Unified PT output and offline recompute output."
    )
    parser.add_argument("--live_csv", required=True, help="Live selected-target PT sweep CSV")
    parser.add_argument("--recompute_csv", required=True, help="Offline recompute PT sweep CSV")
    parser.add_argument(
        "--abs_tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for floating-point comparisons",
    )
    parser.add_argument(
        "--rel_tol",
        type=float,
        default=1e-6,
        help="Relative tolerance for floating-point comparisons",
    )
    return parser.parse_args()


def _load_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row: {path}")
        rows = list(reader)
    return list(reader.fieldnames), rows


def _key_for_row(row: Dict[str, str]) -> Tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def _index_rows(rows: Iterable[Dict[str, str]], label: str) -> Dict[Tuple[str, ...], Dict[str, str]]:
    indexed: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in rows:
        key = _key_for_row(row)
        if key in indexed:
            raise ValueError(f"Duplicate {label} row key: {key}")
        indexed[key] = row
    return indexed


def _parse_float(row: Dict[str, str], field: str, key: Tuple[str, ...], label: str) -> float:
    raw = str(row[field]).strip()
    if not raw:
        raise ValueError(f"Missing float field={field} for {label} key={key}")
    return float(raw)


def _compare_exact(
    live_row: Dict[str, str],
    recompute_row: Dict[str, str],
    fields: Sequence[str],
    key: Tuple[str, ...],
) -> None:
    for field in fields:
        live_val = str(live_row[field])
        recomp_val = str(recompute_row[field])
        if live_val != recomp_val:
            raise ValueError(
                f"validation mismatch key={key} field={field} live={live_val!r} recomp={recomp_val!r}"
            )


def _compare_float(
    live_row: Dict[str, str],
    recompute_row: Dict[str, str],
    fields: Sequence[str],
    key: Tuple[str, ...],
    *,
    abs_tol: float,
    rel_tol: float,
) -> None:
    for field in fields:
        live_val = _parse_float(live_row, field, key, "live")
        recomp_val = _parse_float(recompute_row, field, key, "recompute")
        if not math.isclose(live_val, recomp_val, rel_tol=rel_tol, abs_tol=abs_tol):
            raise ValueError(
                "validation mismatch "
                f"key={key} field={field} live={live_val} recomp={recomp_val} "
                f"abs_diff={abs(live_val - recomp_val)} abs_tol={abs_tol} rel_tol={rel_tol}"
            )


def main() -> int:
    args = _parse_args()
    live_path = Path(args.live_csv).resolve()
    recompute_path = Path(args.recompute_csv).resolve()
    live_fields, live_rows = _load_csv(live_path)
    recompute_fields, recompute_rows = _load_csv(recompute_path)

    exact_fields = [field for field in DEFAULT_EXACT_FIELDS if field in live_fields and field in recompute_fields]
    float_fields = [field for field in DEFAULT_FLOAT_FIELDS if field in live_fields and field in recompute_fields]
    if not exact_fields:
        raise ValueError("No exact-compare fields found in both CSVs")
    if not float_fields:
        raise ValueError("No float-compare fields found in both CSVs")

    live_index = _index_rows(live_rows, "live")
    recompute_index = _index_rows(recompute_rows, "recompute")

    live_keys = set(live_index)
    recompute_keys = set(recompute_index)
    missing_in_recompute = sorted(live_keys - recompute_keys)
    missing_in_live = sorted(recompute_keys - live_keys)
    if missing_in_recompute:
        raise ValueError(f"Missing recompute rows: first={missing_in_recompute[0]} count={len(missing_in_recompute)}")
    if missing_in_live:
        raise ValueError(f"Unexpected recompute rows: first={missing_in_live[0]} count={len(missing_in_live)}")

    for key in sorted(live_keys):
        live_row = live_index[key]
        recompute_row = recompute_index[key]
        _compare_exact(live_row, recompute_row, exact_fields, key)
        _compare_float(
            live_row,
            recompute_row,
            float_fields,
            key,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
        )

    print(
        "validation passed "
        f"rows={len(live_rows)} abs_tol={args.abs_tol} rel_tol={args.rel_tol} "
        f"exact_fields={','.join(exact_fields)} float_fields={','.join(float_fields)} "
        f"live_csv={live_path} recompute_csv={recompute_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
