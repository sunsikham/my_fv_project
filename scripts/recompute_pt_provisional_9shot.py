#!/usr/bin/env python3
"""Build a provisional 9-shot 5-edge PT sweep from existing run outputs.

Policy:
- main 11-Q 5-edge run contributes Q1,Q3,Q4,Q5,Q7,Q8,Q9,Q10,Q16
  and relabels shot 10 -> 9.
- Q6 comes from the dedicated 9-shot recovery run.
- Q11,Q18 come from the dedicated 9-shot override run.

After row selection, the script recomputes q-wise robust min-max normalization
from target_logprob_raw over all selected rows for that q_id and writes a new
5-edge sweep CSV suitable for compute_product_test_bootstrap.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


MAIN_QIDS = ("Q1", "Q3", "Q4", "Q5", "Q7", "Q8", "Q9", "Q10", "Q16")
Q6_QIDS = ("Q6",)
OVERRIDE_QIDS = ("Q11", "Q18")
TARGET_SHOTS = (1, 3, 5, 7, 9)
ALL_QIDS = MAIN_QIDS + Q6_QIDS + OVERRIDE_QIDS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a provisional 9-shot 5-edge PT sweep from existing runs."
    )
    parser.add_argument("--main_run_dir", required=True, help="11-Q 5-edge run dir")
    parser.add_argument("--q6_run_dir", required=True, help="Q6 9-shot recovery run dir")
    parser.add_argument(
        "--override_run_dir",
        required=True,
        help="Q11/Q18 9-shot override run dir",
    )
    parser.add_argument("--out_csv", required=True, help="Output merged sweep CSV")
    parser.add_argument(
        "--out_manifest_json",
        required=True,
        help="Output manifest JSON with source policy details",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row: {path}")
        return list(reader)


def _normalize_q_rows(rows: List[Dict[str, object]]) -> None:
    raw = np.array([float(row["target_logprob_raw"]) for row in rows], dtype=float)
    p_low = float(np.percentile(raw, 5))
    p_high = float(np.percentile(raw, 95))
    if p_high == p_low:
        for row in rows:
            row["target_s_norm"] = 0.5
    else:
        denom = p_high - p_low
        for row in rows:
            x = float(row["target_logprob_raw"])
            s = (x - p_low) / denom
            if s < 0.0:
                s = 0.0
            elif s > 1.0:
                s = 1.0
            row["target_s_norm"] = s
    for row in rows:
        row["norm_p_low"] = p_low
        row["norm_p_high"] = p_high
        row["norm_method"] = "robust_minmax_p05_p95"
        row["norm_scope"] = "qid_all_edges_all_shots_provisional_9shot"


def _select_main_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        q_id = row["q_id"]
        if q_id not in MAIN_QIDS:
            continue
        shot = int(row["shot"])
        if shot not in (1, 3, 5, 7, 10):
            continue
        new_row = dict(row)
        new_row["source_run_label"] = "main_11q_5edge"
        new_row["source_shot"] = shot
        new_row["shot_policy"] = "relabel_10_to_9" if shot == 10 else "as_is"
        new_row["shot"] = 9 if shot == 10 else shot
        out.append(new_row)
    return out


def _select_as_is_rows(
    rows: Iterable[Dict[str, str]],
    allowed_qids: Iterable[str],
    source_run_label: str,
) -> List[Dict[str, object]]:
    allowed = set(allowed_qids)
    out: List[Dict[str, object]] = []
    for row in rows:
        q_id = row["q_id"]
        if q_id not in allowed:
            continue
        shot = int(row["shot"])
        if shot not in TARGET_SHOTS:
            continue
        new_row = dict(row)
        new_row["source_run_label"] = source_run_label
        new_row["source_shot"] = shot
        new_row["shot_policy"] = "as_is"
        new_row["shot"] = shot
        out.append(new_row)
    return out


def _validate_complete_coverage(rows: List[Dict[str, object]]) -> None:
    counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[str(row["q_id"])][int(row["shot"])] += 1
    missing: List[str] = []
    for q_id in ALL_QIDS:
        for shot in TARGET_SHOTS:
            if counts[q_id][shot] != 250:
                missing.append(f"{q_id}@{shot}={counts[q_id][shot]}")
    if missing:
        raise ValueError("Incomplete provisional coverage: " + ", ".join(missing))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    main_csv = Path(args.main_run_dir) / "pt_5edge_shot_sweep.csv"
    q6_csv = Path(args.q6_run_dir) / "pt_5edge_shot_sweep.csv"
    override_csv = Path(args.override_run_dir) / "pt_5edge_shot_sweep.csv"

    main_rows = _load_rows(main_csv)
    q6_rows = _load_rows(q6_csv)
    override_rows = _load_rows(override_csv)

    provisional_rows: List[Dict[str, object]] = []
    provisional_rows.extend(_select_main_rows(main_rows))
    provisional_rows.extend(_select_as_is_rows(q6_rows, Q6_QIDS, "q6_9shot"))
    provisional_rows.extend(
        _select_as_is_rows(override_rows, OVERRIDE_QIDS, "q11_q18_override_9shot")
    )

    _validate_complete_coverage(provisional_rows)

    by_q: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in provisional_rows:
        by_q[str(row["q_id"])].append(row)
    for q_rows in by_q.values():
        _normalize_q_rows(q_rows)

    # Stable output order for downstream diffing and CSV readability.
    provisional_rows.sort(
        key=lambda row: (
            ALL_QIDS.index(str(row["q_id"])),
            int(row["trial_index"]),
            int(row["shot"]),
            str(row["edge"]),
        )
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_csv, provisional_rows)

    manifest = {
        "policy": {
            "target_shots": list(TARGET_SHOTS),
            "main_qids_relabel_10_to_9": list(MAIN_QIDS),
            "actual_9shot_qids": list(Q6_QIDS + OVERRIDE_QIDS),
            "notes": [
                "Use as provisional analysis only.",
                "Main 11-Q run contributes shot=10 rows relabeled to shot=9.",
                "Q6, Q11, and Q18 use actual 9-shot rows.",
                "Normalization rebuilt q-wise from target_logprob_raw across all five edges and selected shots.",
            ],
        },
        "sources": {
            "main_run_dir": args.main_run_dir,
            "q6_run_dir": args.q6_run_dir,
            "override_run_dir": args.override_run_dir,
        },
        "counts": {
            "rows_total": len(provisional_rows),
            "rows_by_qid": {q_id: len(rows) for q_id, rows in by_q.items()},
        },
        "out_csv": str(out_csv),
    }
    out_manifest = Path(args.out_manifest_json)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"[ok] wrote provisional sweep: {out_csv}")
    print(f"[ok] wrote manifest: {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
