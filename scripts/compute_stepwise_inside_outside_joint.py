#!/usr/bin/env python3
"""Compute joint decomposition separately for inside-A and outside-A stepwise components."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REF_NAME = "AAA_ref"
BASIS_SCOPE = "matched"
SLOT_NAMES = ["A_demo_1", "A_demo_2", "A_demo_3", "A_demo_4", "A_query"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inside_outside_npz",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_inside_outside/stepwise_inside_outside_endpoint_arrays.npz",
        help="Inside/outside arrays NPZ",
    )
    p.add_argument(
        "--inside_outside_meta",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_inside_outside/stepwise_inside_outside_endpoint_meta.json",
        help="Inside/outside meta JSON",
    )
    p.add_argument("--q_list", default=None, help="Optional comma-separated q ids")
    p.add_argument(
        "--out_dir",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_inside_outside_joint",
        help="Output directory",
    )
    return p.parse_args()


def _parse_csv_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _qid_sort_key(qid: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)", str(qid))
    return (int(m.group(1)) if m else 1_000_000, str(qid))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    tmp.replace(path)


def _mean_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _joint_decompose(delta: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> Tuple[float, float, float]:
    delta = delta.astype(np.float32, copy=False)
    U = np.stack([u1.astype(np.float32, copy=False), u2.astype(np.float32, copy=False)], axis=1)
    if float(np.linalg.norm(U[:, 0])) == 0.0 and float(np.linalg.norm(U[:, 1])) == 0.0:
        resid = float(np.linalg.norm(delta))
        return float("nan"), float("nan"), resid
    coeffs, _resid, _rank, _svals = np.linalg.lstsq(U, delta, rcond=None)
    recon = U @ coeffs
    resid = float(np.linalg.norm(delta - recon))
    return float(coeffs[0]), float(coeffs[1]), resid


def main() -> int:
    args = _parse_args()
    arrays_npz = Path(args.inside_outside_npz).resolve()
    meta_json = Path(args.inside_outside_meta).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not arrays_npz.exists():
        raise FileNotFoundError(f"Missing inside_outside_npz: {arrays_npz}")
    if not meta_json.exists():
        raise FileNotFoundError(f"Missing inside_outside_meta: {meta_json}")

    arrays = np.load(arrays_npz)
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    qids = _parse_csv_list(args.q_list) or list(meta.get("qids", []))
    qids = sorted(qids, key=_qid_sort_key)

    rows: List[Dict[str, object]] = []
    arrays_out: Dict[str, np.ndarray] = {}
    out_meta: Dict[str, object] = {
        "ref": REF_NAME,
        "basis_scope": BASIS_SCOPE,
        "slot_names": SLOT_NAMES,
        "qids": qids,
    }

    for qid in qids:
        needed = [
            f"{qid}__delta_in_BAB_trials",
            f"{qid}__delta_out_BAB_trials",
            f"{qid}__delta_in_DAD_trials",
            f"{qid}__delta_out_DAD_trials",
            f"{qid}__u_B_in",
            f"{qid}__u_D_in",
            f"{qid}__u_B_out",
            f"{qid}__u_D_out",
        ]
        if any(key not in arrays.files for key in needed):
            continue

        delta_in_BAB = arrays[f"{qid}__delta_in_BAB_trials"].astype(np.float32, copy=False)
        delta_out_BAB = arrays[f"{qid}__delta_out_BAB_trials"].astype(np.float32, copy=False)
        delta_in_DAD = arrays[f"{qid}__delta_in_DAD_trials"].astype(np.float32, copy=False)
        delta_out_DAD = arrays[f"{qid}__delta_out_DAD_trials"].astype(np.float32, copy=False)
        u_B_in = arrays[f"{qid}__u_B_in"].astype(np.float32, copy=False)
        u_D_in = arrays[f"{qid}__u_D_in"].astype(np.float32, copy=False)
        u_B_out = arrays[f"{qid}__u_B_out"].astype(np.float32, copy=False)
        u_D_out = arrays[f"{qid}__u_D_out"].astype(np.float32, copy=False)

        n_bab = int(delta_in_BAB.shape[0])
        n_dad = int(delta_in_DAD.shape[0])
        n_steps = int(delta_in_BAB.shape[1])

        alpha_B_in_BAB = np.zeros((n_bab, n_steps), dtype=np.float32)
        beta_D_in_BAB = np.zeros((n_bab, n_steps), dtype=np.float32)
        resid_in_BAB = np.zeros((n_bab, n_steps), dtype=np.float32)
        alpha_B_out_BAB = np.zeros((n_bab, n_steps), dtype=np.float32)
        beta_D_out_BAB = np.zeros((n_bab, n_steps), dtype=np.float32)
        resid_out_BAB = np.zeros((n_bab, n_steps), dtype=np.float32)

        alpha_B_in_DAD = np.zeros((n_dad, n_steps), dtype=np.float32)
        beta_D_in_DAD = np.zeros((n_dad, n_steps), dtype=np.float32)
        resid_in_DAD = np.zeros((n_dad, n_steps), dtype=np.float32)
        alpha_B_out_DAD = np.zeros((n_dad, n_steps), dtype=np.float32)
        beta_D_out_DAD = np.zeros((n_dad, n_steps), dtype=np.float32)
        resid_out_DAD = np.zeros((n_dad, n_steps), dtype=np.float32)

        for i in range(n_bab):
            for t in range(n_steps):
                a1, b1, r1 = _joint_decompose(delta_in_BAB[i, t], u_B_in, u_D_in)
                a2, b2, r2 = _joint_decompose(delta_out_BAB[i, t], u_B_out, u_D_out)
                alpha_B_in_BAB[i, t] = a1
                beta_D_in_BAB[i, t] = b1
                resid_in_BAB[i, t] = r1
                alpha_B_out_BAB[i, t] = a2
                beta_D_out_BAB[i, t] = b2
                resid_out_BAB[i, t] = r2

        for i in range(n_dad):
            for t in range(n_steps):
                a1, b1, r1 = _joint_decompose(delta_in_DAD[i, t], u_B_in, u_D_in)
                a2, b2, r2 = _joint_decompose(delta_out_DAD[i, t], u_B_out, u_D_out)
                alpha_B_in_DAD[i, t] = a1
                beta_D_in_DAD[i, t] = b1
                resid_in_DAD[i, t] = r1
                alpha_B_out_DAD[i, t] = a2
                beta_D_out_DAD[i, t] = b2
                resid_out_DAD[i, t] = r2

        arrays_out[f"{qid}__alpha_B_in_BAB_trials"] = alpha_B_in_BAB
        arrays_out[f"{qid}__beta_D_in_BAB_trials"] = beta_D_in_BAB
        arrays_out[f"{qid}__resid_in_BAB_trials"] = resid_in_BAB
        arrays_out[f"{qid}__alpha_B_out_BAB_trials"] = alpha_B_out_BAB
        arrays_out[f"{qid}__beta_D_out_BAB_trials"] = beta_D_out_BAB
        arrays_out[f"{qid}__resid_out_BAB_trials"] = resid_out_BAB
        arrays_out[f"{qid}__alpha_B_in_DAD_trials"] = alpha_B_in_DAD
        arrays_out[f"{qid}__beta_D_in_DAD_trials"] = beta_D_in_DAD
        arrays_out[f"{qid}__resid_in_DAD_trials"] = resid_in_DAD
        arrays_out[f"{qid}__alpha_B_out_DAD_trials"] = alpha_B_out_DAD
        arrays_out[f"{qid}__beta_D_out_DAD_trials"] = beta_D_out_DAD
        arrays_out[f"{qid}__resid_out_DAD_trials"] = resid_out_DAD

        for step_idx, slot_name in enumerate(SLOT_NAMES):
            row = {
                "q_id": qid,
                "ref": REF_NAME,
                "basis_scope": BASIS_SCOPE,
                "slot_name": slot_name,
                "slot_index": step_idx,
                "alpha_B_in_BAB": _mean_or_nan(alpha_B_in_BAB[:, step_idx]),
                "beta_D_in_BAB": _mean_or_nan(beta_D_in_BAB[:, step_idx]),
                "resid_in_BAB": _mean_or_nan(resid_in_BAB[:, step_idx]),
                "alpha_B_out_BAB": _mean_or_nan(alpha_B_out_BAB[:, step_idx]),
                "beta_D_out_BAB": _mean_or_nan(beta_D_out_BAB[:, step_idx]),
                "resid_out_BAB": _mean_or_nan(resid_out_BAB[:, step_idx]),
                "alpha_B_in_DAD": _mean_or_nan(alpha_B_in_DAD[:, step_idx]),
                "beta_D_in_DAD": _mean_or_nan(beta_D_in_DAD[:, step_idx]),
                "resid_in_DAD": _mean_or_nan(resid_in_DAD[:, step_idx]),
                "alpha_B_out_DAD": _mean_or_nan(alpha_B_out_DAD[:, step_idx]),
                "beta_D_out_DAD": _mean_or_nan(beta_D_out_DAD[:, step_idx]),
                "resid_out_DAD": _mean_or_nan(resid_out_DAD[:, step_idx]),
                "selectivity_in_BAB": _mean_or_nan(alpha_B_in_BAB[:, step_idx] - beta_D_in_BAB[:, step_idx]),
                "selectivity_out_BAB": _mean_or_nan(alpha_B_out_BAB[:, step_idx] - beta_D_out_BAB[:, step_idx]),
                "selectivity_in_DAD": _mean_or_nan(beta_D_in_DAD[:, step_idx] - alpha_B_in_DAD[:, step_idx]),
                "selectivity_out_DAD": _mean_or_nan(beta_D_out_DAD[:, step_idx] - alpha_B_out_DAD[:, step_idx]),
                "n_trials_BAB": n_bab,
                "n_trials_DAD": n_dad,
            }
            rows.append(row)

    rows = sorted(rows, key=lambda row: (_qid_sort_key(str(row["q_id"])), int(row["slot_index"])))
    _write_csv(out_dir / "stepwise_inside_outside_joint_summary.csv", rows)
    np.savez(out_dir / "stepwise_inside_outside_joint_arrays.npz", **arrays_out)
    _write_json(out_dir / "stepwise_inside_outside_joint_meta.json", out_meta)

    print(f"summary_csv={out_dir / 'stepwise_inside_outside_joint_summary.csv'}")
    print(f"arrays_npz={out_dir / 'stepwise_inside_outside_joint_arrays.npz'}")
    print(f"meta_json={out_dir / 'stepwise_inside_outside_joint_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
