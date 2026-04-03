#!/usr/bin/env python3
"""Compute inside/outside endpoint contribution metrics from stepwise artifacts."""

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
        "--stepwise_root",
        default="/home/sunsik/my_fv_project/results_fv/relation_condition_qwise",
        help="Root containing q dirs with _stepwise_a_states",
    )
    p.add_argument(
        "--reweight_npz",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting/stepwise_reweighting_arrays_AAA_ref.npz",
        help="Stepwise reweighting arrays NPZ",
    )
    p.add_argument(
        "--reweight_meta",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting/stepwise_reweighting_meta.json",
        help="Stepwise reweighting meta JSON",
    )
    p.add_argument("--q_list", default=None, help="Optional comma-separated q ids")
    p.add_argument(
        "--out_dir",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_inside_outside",
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


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return np.zeros_like(vec)
    return (vec / norm).astype(np.float32, copy=False)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _mean_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _find_q_dir(root: Path, qid: str) -> Optional[Path]:
    for child in root.rglob(qid):
        if child.is_dir() and child.name == qid:
            return child
    return None


def _load_stepwise_states(stepwise_root: Path, qid: str) -> np.lib.npyio.NpzFile:
    q_dir = _find_q_dir(stepwise_root, qid)
    if q_dir is None:
        raise FileNotFoundError(f"q_dir not found for {qid} under {stepwise_root}")
    path = q_dir / "_stepwise_a_states" / f"stepwise_a_states_{REF_NAME}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing stepwise state npz: {path}")
    return np.load(path)


def _projector(G: np.ndarray) -> np.ndarray:
    if G.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return (G @ G.T).astype(np.float32, copy=False)


def _safe_diff(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    return float(a - b)


def main() -> int:
    args = _parse_args()
    stepwise_root = Path(args.stepwise_root).resolve()
    reweight_npz = Path(args.reweight_npz).resolve()
    reweight_meta = Path(args.reweight_meta).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not reweight_npz.exists():
        raise FileNotFoundError(f"Missing reweight_npz: {reweight_npz}")
    if not reweight_meta.exists():
        raise FileNotFoundError(f"Missing reweight_meta: {reweight_meta}")

    rw = np.load(reweight_npz)
    meta = json.loads(reweight_meta.read_text(encoding="utf-8"))
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
        G_key = f"{qid}__{BASIS_SCOPE}__G_A_full"
        a_key = f"{qid}__anchor_a_query"
        b_key = f"{qid}__anchor_b_query"
        d_key = f"{qid}__anchor_d_query"
        if any(key not in rw.files for key in [G_key, a_key, b_key, d_key]):
            continue

        G_A = rw[G_key].astype(np.float32, copy=False)
        P_A = _projector(G_A)
        feat_dim = int(G_A.shape[0])
        I = np.eye(feat_dim, dtype=np.float32)
        P_out = I - P_A

        a = rw[a_key].astype(np.float32, copy=False)
        b = rw[b_key].astype(np.float32, copy=False)
        d = rw[d_key].astype(np.float32, copy=False)
        b_minus_a = (b - a).astype(np.float32, copy=False)
        d_minus_a = (d - a).astype(np.float32, copy=False)

        u_B_in = _normalize(P_A @ b_minus_a)
        u_D_in = _normalize(P_A @ d_minus_a)
        u_B_out = _normalize(P_out @ b_minus_a)
        u_D_out = _normalize(P_out @ d_minus_a)

        arrays_out[f"{qid}__u_B_in"] = u_B_in
        arrays_out[f"{qid}__u_D_in"] = u_D_in
        arrays_out[f"{qid}__u_B_out"] = u_B_out
        arrays_out[f"{qid}__u_D_out"] = u_D_out

        step_npz = _load_stepwise_states(stepwise_root, qid)
        AAA = step_npz[f"{qid}__AAA__matched__sum"].astype(np.float32, copy=False)
        BABA = step_npz[f"{qid}__BABA__matched__sum"].astype(np.float32, copy=False)
        DADA = step_npz[f"{qid}__DADA__matched__sum"].astype(np.float32, copy=False)

        delta_BAB = (BABA - AAA).astype(np.float32, copy=False)
        delta_DAD = (DADA - AAA).astype(np.float32, copy=False)
        delta_in_BAB = np.einsum("df,tsf->tsd", P_A, delta_BAB, optimize=True).astype(np.float32, copy=False)
        delta_out_BAB = (delta_BAB - delta_in_BAB).astype(np.float32, copy=False)
        delta_in_DAD = np.einsum("df,tsf->tsd", P_A, delta_DAD, optimize=True).astype(np.float32, copy=False)
        delta_out_DAD = (delta_DAD - delta_in_DAD).astype(np.float32, copy=False)

        arrays_out[f"{qid}__delta_in_BAB_trials"] = delta_in_BAB
        arrays_out[f"{qid}__delta_out_BAB_trials"] = delta_out_BAB
        arrays_out[f"{qid}__delta_in_DAD_trials"] = delta_in_DAD
        arrays_out[f"{qid}__delta_out_DAD_trials"] = delta_out_DAD

        for step_idx, slot_name in enumerate(SLOT_NAMES):
            in_bab = delta_in_BAB[:, step_idx, :]
            out_bab = delta_out_BAB[:, step_idx, :]
            in_dad = delta_in_DAD[:, step_idx, :]
            out_dad = delta_out_DAD[:, step_idx, :]

            in_norm_BAB = [float(np.linalg.norm(vec)) for vec in in_bab]
            out_norm_BAB = [float(np.linalg.norm(vec)) for vec in out_bab]
            in_norm_DAD = [float(np.linalg.norm(vec)) for vec in in_dad]
            out_norm_DAD = [float(np.linalg.norm(vec)) for vec in out_dad]

            align_in_BAB_to_B = [_cosine(vec, u_B_in) for vec in in_bab]
            align_in_BAB_to_D = [_cosine(vec, u_D_in) for vec in in_bab]
            align_out_BAB_to_B = [_cosine(vec, u_B_out) for vec in out_bab]
            align_out_BAB_to_D = [_cosine(vec, u_D_out) for vec in out_bab]

            align_in_DAD_to_B = [_cosine(vec, u_B_in) for vec in in_dad]
            align_in_DAD_to_D = [_cosine(vec, u_D_in) for vec in in_dad]
            align_out_DAD_to_B = [_cosine(vec, u_B_out) for vec in out_dad]
            align_out_DAD_to_D = [_cosine(vec, u_D_out) for vec in out_dad]

            row = {
                "q_id": qid,
                "ref": REF_NAME,
                "basis_scope": BASIS_SCOPE,
                "slot_name": slot_name,
                "slot_index": step_idx,
                "in_norm_BAB": _mean_or_nan(in_norm_BAB),
                "out_norm_BAB": _mean_or_nan(out_norm_BAB),
                "in_norm_DAD": _mean_or_nan(in_norm_DAD),
                "out_norm_DAD": _mean_or_nan(out_norm_DAD),
                "align_in_BAB_to_B": _mean_or_nan(align_in_BAB_to_B),
                "align_in_BAB_to_D": _mean_or_nan(align_in_BAB_to_D),
                "align_out_BAB_to_B": _mean_or_nan(align_out_BAB_to_B),
                "align_out_BAB_to_D": _mean_or_nan(align_out_BAB_to_D),
                "align_in_DAD_to_B": _mean_or_nan(align_in_DAD_to_B),
                "align_in_DAD_to_D": _mean_or_nan(align_in_DAD_to_D),
                "align_out_DAD_to_B": _mean_or_nan(align_out_DAD_to_B),
                "align_out_DAD_to_D": _mean_or_nan(align_out_DAD_to_D),
                "selectivity_in_BAB": _safe_diff(_mean_or_nan(align_in_BAB_to_B), _mean_or_nan(align_in_BAB_to_D)),
                "selectivity_out_BAB": _safe_diff(_mean_or_nan(align_out_BAB_to_B), _mean_or_nan(align_out_BAB_to_D)),
                "selectivity_in_DAD": _safe_diff(_mean_or_nan(align_in_DAD_to_D), _mean_or_nan(align_in_DAD_to_B)),
                "selectivity_out_DAD": _safe_diff(_mean_or_nan(align_out_DAD_to_D), _mean_or_nan(align_out_DAD_to_B)),
                "n_trials_BAB": int(delta_BAB.shape[0]),
                "n_trials_DAD": int(delta_DAD.shape[0]),
            }
            rows.append(row)

    rows = sorted(rows, key=lambda row: (_qid_sort_key(str(row["q_id"])), int(row["slot_index"])))
    _write_csv(out_dir / "stepwise_inside_outside_endpoint_summary.csv", rows)
    np.savez(out_dir / "stepwise_inside_outside_endpoint_arrays.npz", **arrays_out)
    _write_json(out_dir / "stepwise_inside_outside_endpoint_meta.json", out_meta)

    print(f"summary_csv={out_dir / 'stepwise_inside_outside_endpoint_summary.csv'}")
    print(f"arrays_npz={out_dir / 'stepwise_inside_outside_endpoint_arrays.npz'}")
    print(f"meta_json={out_dir / 'stepwise_inside_outside_endpoint_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
