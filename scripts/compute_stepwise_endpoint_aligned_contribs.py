#!/usr/bin/env python3
"""Compute endpoint-aligned signed contribution metrics from stepwise reweighting arrays."""

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
SLOT_NAMES = ["A_demo_1", "A_demo_2", "A_demo_3", "A_demo_4", "A_query"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arrays_npz",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting/stepwise_reweighting_arrays_AAA_ref.npz",
        help="Input stepwise reweighting arrays NPZ",
    )
    p.add_argument(
        "--meta_json",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting/stepwise_reweighting_meta.json",
        help="Input stepwise reweighting meta JSON",
    )
    p.add_argument("--q_list", default=None, help="Optional comma-separated q ids")
    p.add_argument(
        "--out_dir",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_endpoint_alignment",
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


def _contrib_pr(contrib: np.ndarray) -> float:
    if contrib.size == 0:
        return float("nan")
    abs_sum = float(np.sum(np.abs(contrib)))
    sq_sum = float(np.sum(np.square(contrib)))
    if sq_sum == 0.0:
        return 0.0
    return float((abs_sum**2) / sq_sum)


def _projector_from_basis(G: np.ndarray) -> np.ndarray:
    if G.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return (G @ G.T).astype(np.float32, copy=False)


def _soft_energy(delta_c: np.ndarray, align_weights: np.ndarray) -> float:
    if delta_c.size == 0 or align_weights.size == 0:
        return float("nan")
    return float(np.sum(np.square(align_weights) * np.square(delta_c)))


def _descending_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int32)
    return ranks


def main() -> int:
    args = _parse_args()
    arrays_npz = Path(args.arrays_npz).resolve()
    meta_json = Path(args.meta_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not arrays_npz.exists():
        raise FileNotFoundError(f"Missing arrays_npz: {arrays_npz}")
    if not meta_json.exists():
        raise FileNotFoundError(f"Missing meta_json: {meta_json}")

    arrays = np.load(arrays_npz)
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    qids = _parse_csv_list(args.q_list) or list(meta.get("qids", []))
    qids = sorted(qids, key=_qid_sort_key)

    summary_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    feature_step_rows: List[Dict[str, object]] = []
    arrays_out: Dict[str, np.ndarray] = {}
    out_meta: Dict[str, object] = {
        "ref": REF_NAME,
        "basis_scopes": list(meta.get("basis_scopes", [])),
        "slot_names": list(meta.get("matched_slot_names", SLOT_NAMES)),
        "qids": qids,
    }

    for qid in qids:
        anchor_a_key = f"{qid}__anchor_a_query"
        anchor_b_key = f"{qid}__anchor_b_query"
        anchor_d_key = f"{qid}__anchor_d_query"
        if anchor_a_key not in arrays.files or anchor_b_key not in arrays.files or anchor_d_key not in arrays.files:
            continue
        a = arrays[anchor_a_key].astype(np.float32, copy=False)
        b = arrays[anchor_b_key].astype(np.float32, copy=False)
        d = arrays[anchor_d_key].astype(np.float32, copy=False)
        b_minus_a = (b - a).astype(np.float32, copy=False)
        d_minus_a = (d - a).astype(np.float32, copy=False)

        for basis_scope in meta.get("basis_scopes", []):
            G_key = f"{qid}__{basis_scope}__G_A_topk"
            dcb_key = f"{qid}__{basis_scope}__delta_c_BAB_trials_topk"
            dcd_key = f"{qid}__{basis_scope}__delta_c_DAD_trials_topk"
            if G_key not in arrays.files or dcb_key not in arrays.files or dcd_key not in arrays.files:
                continue

            G = arrays[G_key].astype(np.float32, copy=False)
            P_A = _projector_from_basis(G)
            uB_A = _normalize(P_A @ b_minus_a) if P_A.size else np.zeros_like(b_minus_a)
            uD_A = _normalize(P_A @ d_minus_a) if P_A.size else np.zeros_like(d_minus_a)
            aB = (G.T @ uB_A).astype(np.float32, copy=False) if G.size else np.zeros((0,), dtype=np.float32)
            aD = (G.T @ uD_A).astype(np.float32, copy=False) if G.size else np.zeros((0,), dtype=np.float32)

            arrays_out[f"{qid}__{basis_scope}__u_B_A"] = uB_A
            arrays_out[f"{qid}__{basis_scope}__u_D_A"] = uD_A
            arrays_out[f"{qid}__{basis_scope}__a_align_B"] = aB
            arrays_out[f"{qid}__{basis_scope}__a_align_D"] = aD

            delta_c_BAB = arrays[dcb_key].astype(np.float32, copy=False)  # [trial, step, k]
            delta_c_DAD = arrays[dcd_key].astype(np.float32, copy=False)

            # Build delta in A-space.
            deltaA_BAB = np.einsum("dk,tsk->tsd", G, delta_c_BAB, optimize=True).astype(np.float32, copy=False)
            deltaA_DAD = np.einsum("dk,tsk->tsd", G, delta_c_DAD, optimize=True).astype(np.float32, copy=False)
            arrays_out[f"{qid}__{basis_scope}__deltaA_BAB_trials"] = deltaA_BAB
            arrays_out[f"{qid}__{basis_scope}__deltaA_DAD_trials"] = deltaA_DAD

            contrib_BAB_B = (delta_c_BAB * aB[None, None, :]).astype(np.float32, copy=False)
            contrib_BAB_D = (delta_c_BAB * aD[None, None, :]).astype(np.float32, copy=False)
            contrib_DAD_B = (delta_c_DAD * aB[None, None, :]).astype(np.float32, copy=False)
            contrib_DAD_D = (delta_c_DAD * aD[None, None, :]).astype(np.float32, copy=False)
            arrays_out[f"{qid}__{basis_scope}__contrib_BAB_B_trials"] = contrib_BAB_B
            arrays_out[f"{qid}__{basis_scope}__contrib_BAB_D_trials"] = contrib_BAB_D
            arrays_out[f"{qid}__{basis_scope}__contrib_DAD_B_trials"] = contrib_DAD_B
            arrays_out[f"{qid}__{basis_scope}__contrib_DAD_D_trials"] = contrib_DAD_D

            for k_idx in range(aB.shape[0]):
                feature_rows.append(
                    {
                        "q_id": qid,
                        "basis_scope": basis_scope,
                        "feature_index": k_idx,
                        "align_B": float(aB[k_idx]),
                        "align_D": float(aD[k_idx]),
                        "abs_align_B": abs(float(aB[k_idx])),
                        "abs_align_D": abs(float(aD[k_idx])),
                    }
                )

            for step_idx, slot_name in enumerate(SLOT_NAMES):
                babB = contrib_BAB_B[:, step_idx, :]
                babD = contrib_BAB_D[:, step_idx, :]
                dadB = contrib_DAD_B[:, step_idx, :]
                dadD = contrib_DAD_D[:, step_idx, :]
                deltaA_bab_step = deltaA_BAB[:, step_idx, :]
                deltaA_dad_step = deltaA_DAD[:, step_idx, :]
                delta_c_bab_step = delta_c_BAB[:, step_idx, :]
                delta_c_dad_step = delta_c_DAD[:, step_idx, :]

                T_B_BAB_trials = np.sum(babB, axis=1)
                T_D_BAB_trials = np.sum(babD, axis=1)
                T_B_DAD_trials = np.sum(dadB, axis=1)
                T_D_DAD_trials = np.sum(dadD, axis=1)

                T_B_pos_BAB_trials = np.sum(np.maximum(0.0, babB), axis=1)
                T_B_neg_BAB_trials = np.sum(np.maximum(0.0, -babB), axis=1)
                T_D_pos_DAD_trials = np.sum(np.maximum(0.0, dadD), axis=1)
                T_D_neg_DAD_trials = np.sum(np.maximum(0.0, -dadD), axis=1)

                cos_BAB_to_B = [_cosine(vec, uB_A) for vec in deltaA_bab_step]
                cos_BAB_to_D = [_cosine(vec, uD_A) for vec in deltaA_bab_step]
                cos_DAD_to_B = [_cosine(vec, uB_A) for vec in deltaA_dad_step]
                cos_DAD_to_D = [_cosine(vec, uD_A) for vec in deltaA_dad_step]

                mean_delta_c_bab = np.mean(delta_c_bab_step, axis=0).astype(np.float32, copy=False)
                mean_delta_c_dad = np.mean(delta_c_dad_step, axis=0).astype(np.float32, copy=False)
                mean_abs_delta_c_bab = np.mean(np.abs(delta_c_bab_step), axis=0).astype(np.float32, copy=False)
                mean_abs_delta_c_dad = np.mean(np.abs(delta_c_dad_step), axis=0).astype(np.float32, copy=False)
                mean_contrib_bab_b = np.mean(babB, axis=0).astype(np.float32, copy=False)
                mean_contrib_bab_d = np.mean(babD, axis=0).astype(np.float32, copy=False)
                mean_contrib_dad_b = np.mean(dadB, axis=0).astype(np.float32, copy=False)
                mean_contrib_dad_d = np.mean(dadD, axis=0).astype(np.float32, copy=False)
                mean_abs_contrib_bab_b = np.mean(np.abs(babB), axis=0).astype(np.float32, copy=False)
                mean_abs_contrib_dad_d = np.mean(np.abs(dadD), axis=0).astype(np.float32, copy=False)
                rank_bab = _descending_ranks(mean_abs_contrib_bab_b)
                rank_dad = _descending_ranks(mean_abs_contrib_dad_d)

                for k_idx in range(aB.shape[0]):
                    feature_step_rows.append(
                        {
                            "q_id": qid,
                            "ref": REF_NAME,
                            "basis_scope": basis_scope,
                            "slot_name": slot_name,
                            "slot_index": step_idx,
                            "feature_index": k_idx,
                            "feature_name": f"g{k_idx}",
                            "align_B": float(aB[k_idx]),
                            "align_D": float(aD[k_idx]),
                            "mean_delta_c_BAB": float(mean_delta_c_bab[k_idx]),
                            "mean_delta_c_DAD": float(mean_delta_c_dad[k_idx]),
                            "mean_abs_delta_c_BAB": float(mean_abs_delta_c_bab[k_idx]),
                            "mean_abs_delta_c_DAD": float(mean_abs_delta_c_dad[k_idx]),
                            "mean_contrib_BAB_B": float(mean_contrib_bab_b[k_idx]),
                            "mean_contrib_BAB_D": float(mean_contrib_bab_d[k_idx]),
                            "mean_contrib_DAD_B": float(mean_contrib_dad_b[k_idx]),
                            "mean_contrib_DAD_D": float(mean_contrib_dad_d[k_idx]),
                            "mean_abs_contrib_BAB_B": float(mean_abs_contrib_bab_b[k_idx]),
                            "mean_abs_contrib_DAD_D": float(mean_abs_contrib_dad_d[k_idx]),
                            "intended_contrib_rank_BAB": int(rank_bab[k_idx]),
                            "intended_contrib_rank_DAD": int(rank_dad[k_idx]),
                        }
                    )

                row = {
                    "q_id": qid,
                    "ref": REF_NAME,
                    "basis_scope": basis_scope,
                    "slot_name": slot_name,
                    "slot_index": step_idx,
                    "T_B_BAB": _mean_or_nan(T_B_BAB_trials),
                    "T_D_BAB": _mean_or_nan(T_D_BAB_trials),
                    "T_B_DAD": _mean_or_nan(T_B_DAD_trials),
                    "T_D_DAD": _mean_or_nan(T_D_DAD_trials),
                    "T_B_minus_D_BAB": _mean_or_nan(T_B_BAB_trials - T_D_BAB_trials),
                    "T_D_minus_B_DAD": _mean_or_nan(T_D_DAD_trials - T_B_DAD_trials),
                    "T_B_pos_BAB": _mean_or_nan(T_B_pos_BAB_trials),
                    "T_B_neg_BAB": _mean_or_nan(T_B_neg_BAB_trials),
                    "T_D_pos_DAD": _mean_or_nan(T_D_pos_DAD_trials),
                    "T_D_neg_DAD": _mean_or_nan(T_D_neg_DAD_trials),
                    "align_deltaA_BAB_to_B": _mean_or_nan(cos_BAB_to_B),
                    "align_deltaA_BAB_to_D": _mean_or_nan(cos_BAB_to_D),
                    "align_deltaA_DAD_to_B": _mean_or_nan(cos_DAD_to_B),
                    "align_deltaA_DAD_to_D": _mean_or_nan(cos_DAD_to_D),
                    "PR_contrib_BAB_B": _mean_or_nan(_contrib_pr(vec) for vec in babB),
                    "PR_contrib_BAB_D": _mean_or_nan(_contrib_pr(vec) for vec in babD),
                    "PR_contrib_DAD_B": _mean_or_nan(_contrib_pr(vec) for vec in dadB),
                    "PR_contrib_DAD_D": _mean_or_nan(_contrib_pr(vec) for vec in dadD),
                    "E_B_BAB": _mean_or_nan(_soft_energy(vec, aB) for vec in delta_c_bab_step),
                    "E_D_BAB": _mean_or_nan(_soft_energy(vec, aD) for vec in delta_c_bab_step),
                    "E_B_DAD": _mean_or_nan(_soft_energy(vec, aB) for vec in delta_c_dad_step),
                    "E_D_DAD": _mean_or_nan(_soft_energy(vec, aD) for vec in delta_c_dad_step),
                    "n_trials_BAB": int(delta_c_BAB.shape[0]),
                    "n_trials_DAD": int(delta_c_DAD.shape[0]),
                }
                summary_rows.append(row)

    summary_rows = sorted(summary_rows, key=lambda row: (_qid_sort_key(str(row["q_id"])), str(row["basis_scope"]), int(row["slot_index"])))
    feature_rows = sorted(feature_rows, key=lambda row: (_qid_sort_key(str(row["q_id"])), str(row["basis_scope"]), int(row["feature_index"])))
    feature_step_rows = sorted(
        feature_step_rows,
        key=lambda row: (
            _qid_sort_key(str(row["q_id"])),
            str(row["basis_scope"]),
            int(row["slot_index"]),
            int(row["feature_index"]),
        ),
    )

    _write_csv(out_dir / "stepwise_endpoint_alignment_summary.csv", summary_rows)
    _write_csv(out_dir / "stepwise_endpoint_alignment_features.csv", feature_rows)
    _write_csv(out_dir / "stepwise_endpoint_alignment_feature_steps.csv", feature_step_rows)
    np.savez(out_dir / "stepwise_endpoint_alignment_arrays.npz", **arrays_out)
    _write_json(out_dir / "stepwise_endpoint_alignment_meta.json", out_meta)

    print(f"summary_csv={out_dir / 'stepwise_endpoint_alignment_summary.csv'}")
    print(f"features_csv={out_dir / 'stepwise_endpoint_alignment_features.csv'}")
    print(f"feature_steps_csv={out_dir / 'stepwise_endpoint_alignment_feature_steps.csv'}")
    print(f"arrays_npz={out_dir / 'stepwise_endpoint_alignment_arrays.npz'}")
    print(f"meta_json={out_dir / 'stepwise_endpoint_alignment_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
