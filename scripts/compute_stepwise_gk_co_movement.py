#!/usr/bin/env python3
"""Compute top-k g_k coefficient co-movement from stepwise reweighting arrays."""

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
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting_top30/stepwise_reweighting_arrays_AAA_ref.npz",
        help="Input top-k reweighting arrays NPZ",
    )
    p.add_argument(
        "--meta_json",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting_top30/stepwise_reweighting_meta.json",
        help="Input top-k reweighting meta JSON",
    )
    p.add_argument("--q_list", default=None, help="Optional comma-separated q ids")
    p.add_argument(
        "--basis_scope",
        default="matched",
        choices=["matched", "all"],
        help="Which top-k basis scope to analyze for co-movement",
    )
    p.add_argument(
        "--out_dir",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30",
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


def _corr_matrix(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {mat.shape}")
    n_obs, n_feat = mat.shape
    corr = np.full((n_feat, n_feat), np.nan, dtype=np.float32)
    if n_obs < 2 or n_feat == 0:
        return corr
    centered = mat - np.mean(mat, axis=0, keepdims=True)
    denom = n_obs - 1
    cov = (centered.T @ centered) / denom
    var = np.diag(cov)
    std = np.sqrt(np.maximum(var, 0.0))
    nz = std > 0.0
    if np.any(nz):
        scale = np.outer(std, std)
        valid = scale > 0.0
        corr64 = np.full((n_feat, n_feat), np.nan, dtype=np.float64)
        corr64[valid] = cov[valid] / scale[valid]
        corr = corr64.astype(np.float32, copy=False)
        diag_idx = np.where(nz)[0]
        corr[diag_idx, diag_idx] = 1.0
    return corr


def _partner_stats(corr: np.ndarray, feat_idx: int) -> Tuple[float, str, float, str, float]:
    row = corr[feat_idx].astype(np.float64, copy=False)
    mask = np.ones_like(row, dtype=bool)
    mask[feat_idx] = False
    vals = row[mask]
    idxs = np.arange(len(row))[mask]
    finite = np.isfinite(vals)
    if not np.any(finite):
        return float("nan"), "", float("nan"), "", float("nan")
    vals_f = vals[finite]
    idxs_f = idxs[finite]
    mean_abs = float(np.mean(np.abs(vals_f)))
    pos_idx = int(idxs_f[np.argmax(vals_f)])
    neg_idx = int(idxs_f[np.argmin(vals_f)])
    return mean_abs, f"g{pos_idx}", float(np.max(vals_f)), f"g{neg_idx}", float(np.min(vals_f))


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
    if len(qids) != 1:
        raise ValueError(f"Expected exactly one q_id for this run, got {qids}")
    qid = qids[0]
    basis_scope = str(args.basis_scope)

    dcb_key = f"{qid}__{basis_scope}__delta_c_BAB_trials_topk"
    dcd_key = f"{qid}__{basis_scope}__delta_c_DAD_trials_topk"
    if dcb_key not in arrays.files or dcd_key not in arrays.files:
        raise KeyError(f"Missing top-k delta_c arrays for {qid}/{basis_scope}")

    delta_c_bab = arrays[dcb_key].astype(np.float32, copy=False)
    delta_c_dad = arrays[dcd_key].astype(np.float32, copy=False)
    if delta_c_bab.ndim != 3 or delta_c_dad.ndim != 3:
        raise ValueError("delta_c arrays must be [trial, step, feature]")

    n_trials_bab, n_steps, k_feat = delta_c_bab.shape
    n_trials_dad = int(delta_c_dad.shape[0])
    if delta_c_dad.shape[1] != n_steps or delta_c_dad.shape[2] != k_feat:
        raise ValueError("BAB and DAD delta_c arrays must agree on step and feature dimensions")

    flat_bab = delta_c_bab.reshape(n_trials_bab * n_steps, k_feat).astype(np.float32, copy=False)
    flat_dad = delta_c_dad.reshape(n_trials_dad * n_steps, k_feat).astype(np.float32, copy=False)
    corr_flat_bab = _corr_matrix(flat_bab)
    corr_flat_dad = _corr_matrix(flat_dad)
    corr_step_bab = np.stack([_corr_matrix(delta_c_bab[:, step_idx, :]) for step_idx in range(n_steps)], axis=0)
    corr_step_dad = np.stack([_corr_matrix(delta_c_dad[:, step_idx, :]) for step_idx in range(n_steps)], axis=0)

    summary_rows: List[Dict[str, object]] = []
    for feat_idx in range(k_feat):
        mean_abs_bab, top_pos_bab, top_pos_corr_bab, top_neg_bab, top_neg_corr_bab = _partner_stats(corr_flat_bab, feat_idx)
        mean_abs_dad, top_pos_dad, top_pos_corr_dad, top_neg_dad, top_neg_corr_dad = _partner_stats(corr_flat_dad, feat_idx)
        step_abs_bab = [_mean_or_nan(np.abs(corr_step_bab[step_idx, feat_idx, :][np.arange(k_feat) != feat_idx])) for step_idx in range(n_steps)]
        step_abs_dad = [_mean_or_nan(np.abs(corr_step_dad[step_idx, feat_idx, :][np.arange(k_feat) != feat_idx])) for step_idx in range(n_steps)]
        summary_rows.append(
            {
                "q_id": qid,
                "ref": REF_NAME,
                "basis_scope": basis_scope,
                "feature_index": feat_idx,
                "feature_name": f"g{feat_idx}",
                "mean_abs_corr_flat_BAB": mean_abs_bab,
                "mean_abs_corr_flat_DAD": mean_abs_dad,
                "mean_abs_corr_step_BAB": _mean_or_nan(step_abs_bab),
                "mean_abs_corr_step_DAD": _mean_or_nan(step_abs_dad),
                "top_pos_partner_BAB": top_pos_bab,
                "top_pos_partner_corr_BAB": top_pos_corr_bab,
                "top_neg_partner_BAB": top_neg_bab,
                "top_neg_partner_corr_BAB": top_neg_corr_bab,
                "top_pos_partner_DAD": top_pos_dad,
                "top_pos_partner_corr_DAD": top_pos_corr_dad,
                "top_neg_partner_DAD": top_neg_dad,
                "top_neg_partner_corr_DAD": top_neg_corr_dad,
            }
        )

    np.save(out_dir / "stepwise_gk_co_movement_flat_corr_BAB.npy", corr_flat_bab)
    np.save(out_dir / "stepwise_gk_co_movement_flat_corr_DAD.npy", corr_flat_dad)
    np.save(out_dir / "stepwise_gk_co_movement_step_corr_BAB.npy", corr_step_bab)
    np.save(out_dir / "stepwise_gk_co_movement_step_corr_DAD.npy", corr_step_dad)
    _write_csv(out_dir / "stepwise_gk_co_movement_feature_summary.csv", summary_rows)
    _write_json(
        out_dir / "stepwise_gk_co_movement_meta.json",
        {
            "ref": REF_NAME,
            "q_id": qid,
            "basis_scope": basis_scope,
            "slot_names": SLOT_NAMES[:n_steps],
            "k_features": int(k_feat),
            "n_trials_BAB": int(n_trials_bab),
            "n_trials_DAD": int(n_trials_dad),
            "n_obs_flat_BAB": int(flat_bab.shape[0]),
            "n_obs_flat_DAD": int(flat_dad.shape[0]),
            "co_movement_object": "delta_c",
            "flat_definition": "pearson correlation over flattened observations (trial, step)",
            "step_definition": "pearson correlation over trials within each matched step",
        },
    )

    print(f"flat_corr_bab={out_dir / 'stepwise_gk_co_movement_flat_corr_BAB.npy'}")
    print(f"flat_corr_dad={out_dir / 'stepwise_gk_co_movement_flat_corr_DAD.npy'}")
    print(f"step_corr_bab={out_dir / 'stepwise_gk_co_movement_step_corr_BAB.npy'}")
    print(f"step_corr_dad={out_dir / 'stepwise_gk_co_movement_step_corr_DAD.npy'}")
    print(f"feature_summary_csv={out_dir / 'stepwise_gk_co_movement_feature_summary.csv'}")
    print(f"meta_json={out_dir / 'stepwise_gk_co_movement_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
