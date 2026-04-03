#!/usr/bin/env python3
"""Compute q-wise A-basis reweighting metrics for AAA_ref / union_ref."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REF_CHOICES = ("AAA_ref", "union_ref")
COND_ORDER = ("AAA", "BBB", "DDD", "BABA", "DADA")
SLOT_NAME = "A_query"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute q-wise A-basis reweighting metrics")
    p.add_argument("--base_root", required=True, help="Root or comma-separated roots containing per-q directories")
    p.add_argument("--refs", default="AAA_ref,union_ref", help="Comma-separated refs")
    p.add_argument("--q_list", default=None, help="Optional comma-separated q_ids")
    p.add_argument("--k_a", type=int, default=5, help="Top-K A-basis for exploratory retention")
    p.add_argument("--k_b", type=int, default=3, help="Bundle dimension for U_B")
    p.add_argument("--k_d", type=int, default=3, help="Bundle dimension for U_D")
    p.add_argument("--out_dir", default=None, help="Output directory")
    return p.parse_args()


def _parse_csv_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    seen = set()
    out: List[str] = []
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        out.append(part)
    return out


def _parse_roots(raw: str) -> List[Path]:
    roots = [Path(part.strip()).resolve() for part in str(raw).split(",") if part.strip()]
    if not roots:
        raise ValueError("base_root is empty")
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"base_root not found: {root}")
    return roots


def _qid_sort_key(qid: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)", str(qid))
    return (int(m.group(1)) if m else 1_000_000, str(qid))


def _candidate_qids(roots: Sequence[Path]) -> List[str]:
    seen = set()
    out: List[str] = []
    for root in roots:
        for child in sorted(root.iterdir()):
            if child.is_dir() and re.fullmatch(r"Q\d+", child.name) and child.name not in seen:
                seen.add(child.name)
                out.append(child.name)
    return sorted(out, key=_qid_sort_key)


def _find_q_dir(roots: Sequence[Path], qid: str) -> Optional[Path]:
    for root in roots:
        q_dir = root / qid
        if q_dir.exists():
            return q_dir
    return None


def _vector_path(q_dir: Path, ref: str, cond: str) -> Path:
    return q_dir / "_vectors" / f"trial_vectors_{ref}_{cond}.npy"


def _meta_path(q_dir: Path) -> Path:
    return q_dir / "_vectors" / "vector_extraction_meta.json"


def _load_meta(q_dir: Path) -> Dict[str, object]:
    path = _meta_path(q_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing vector_extraction_meta.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trial_ids(meta: Dict[str, object], cond: str, n_rows: int) -> List[str]:
    payload = meta.get("trial_ids", {})
    ids = payload.get(cond, []) if isinstance(payload, dict) else []
    if isinstance(ids, list) and len(ids) == n_rows:
        return [str(x) for x in ids]
    return [f"t{i:06d}" for i in range(n_rows)]


def _center_and_svd(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = arr.mean(axis=0).astype(np.float32, copy=False)
    centered = arr.astype(np.float32, copy=False) - mu
    if centered.shape[0] < 2:
        return mu, centered, np.empty((0, centered.shape[1]), dtype=np.float32)
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    tol = max(centered.shape) * np.finfo(np.float32).eps * (s[0] if len(s) else 1.0)
    rank = int(np.sum(s > tol))
    return mu, centered, vt[:rank].astype(np.float32, copy=False)


def _build_a_basis(
    aaa_arr: np.ndarray,
    *,
    k_a: int,
    contrast_mean: Optional[np.ndarray],
) -> Dict[str, object]:
    mu_A, centered, vt_rank = _center_and_svd(aaa_arr)
    rank_A = int(vt_rank.shape[0])
    if rank_A == 0:
        return {
            "mu_A": mu_A,
            "G_A_full": np.zeros((aaa_arr.shape[1], 0), dtype=np.float32),
            "G_A_topk": np.zeros((aaa_arr.shape[1], 0), dtype=np.float32),
            "rank_A": 0,
            "K_A_eff": 0,
            "explained_variance_ratio": np.zeros((0,), dtype=np.float32),
        }

    # sign stabilization against mean clean contrast if available
    basis = vt_rank.copy()
    if contrast_mean is not None:
        contrast = contrast_mean.astype(np.float32, copy=False)
        for idx in range(basis.shape[0]):
            if float(np.dot(basis[idx], contrast)) < 0.0:
                basis[idx] *= -1.0

    n = centered.shape[0]
    if n > 1:
        svals = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
        explained = (svals**2) / float(n - 1)
        total = float(np.sum(explained))
        evr = (explained[:rank_A] / total).astype(np.float32, copy=False) if total > 0 else np.zeros((rank_A,), dtype=np.float32)
    else:
        evr = np.zeros((rank_A,), dtype=np.float32)

    K_A_eff = min(int(k_a), rank_A, aaa_arr.shape[0] - 1)
    G_A_full = basis.T.astype(np.float32, copy=False)
    G_A_topk = basis[:K_A_eff].T.astype(np.float32, copy=False)
    return {
        "mu_A": mu_A.astype(np.float32, copy=False),
        "G_A_full": G_A_full,
        "G_A_topk": G_A_topk,
        "rank_A": rank_A,
        "K_A_eff": int(K_A_eff),
        "explained_variance_ratio": evr,
    }


def _project_coeffs(v: np.ndarray, mu_A: np.ndarray, G: np.ndarray) -> np.ndarray:
    if G.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return (G.T @ (v.astype(np.float32, copy=False) - mu_A)).astype(np.float32, copy=False)


def _project_energy(v: np.ndarray, mu_A: np.ndarray, G: np.ndarray) -> float:
    centered = v.astype(np.float32, copy=False) - mu_A
    denom = float(np.dot(centered, centered))
    if denom == 0.0:
        return float("nan")
    if G.size == 0:
        return 0.0
    proj = G @ (G.T @ centered)
    return float(np.dot(proj, proj) / denom)


def _change_inside_outside(delta: np.ndarray, G: np.ndarray) -> Tuple[float, float]:
    denom = float(np.dot(delta, delta))
    if denom == 0.0:
        return float("nan"), float("nan")
    if G.size == 0:
        return 0.0, 1.0
    proj = G @ (G.T @ delta.astype(np.float32, copy=False))
    inside = float(np.dot(proj, proj) / denom)
    outside = float((denom - np.dot(proj, proj)) / denom)
    return inside, outside


def _participation_ratio(vec: np.ndarray) -> float:
    if vec.size == 0:
        return float("nan")
    powers2 = np.square(vec.astype(np.float64, copy=False))
    s2 = float(np.sum(powers2))
    s4 = float(np.sum(np.square(powers2)))
    if s4 == 0.0:
        return 0.0
    return float((s2**2) / s4)


def _active_count(vec: np.ndarray, tau: np.ndarray) -> int:
    if vec.size == 0:
        return 0
    return int(np.sum(np.abs(vec) > tau))


def _mean_or_nan(values: List[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _build_bundle_basis(
    deltas: List[np.ndarray],
    *,
    k_bundle: int,
) -> Dict[str, object]:
    if not deltas:
        return {
            "U": np.zeros((0, 0), dtype=np.float32),
            "K_eff": 0,
            "clean_alpha_rms": np.zeros((0,), dtype=np.float32),
        }
    X = np.stack([d.astype(np.float32, copy=False) for d in deltas], axis=0)
    mean_delta = X.mean(axis=0)
    norm_mean = float(np.linalg.norm(mean_delta))
    if norm_mean == 0.0:
        return {
            "U": np.zeros((X.shape[1], 0), dtype=np.float32),
            "K_eff": 0,
            "clean_alpha_rms": np.zeros((0,), dtype=np.float32),
        }
    u1 = (mean_delta / norm_mean).astype(np.float32, copy=False)
    basis_cols = [u1]
    if k_bundle > 1 and X.shape[0] > 1:
        residuals = []
        for row in X:
            coeff1 = float(np.dot(row, u1))
            residuals.append(row - coeff1 * u1)
        R = np.stack(residuals, axis=0)
        _u, s, vt = np.linalg.svd(R, full_matrices=False)
        tol = max(R.shape) * np.finfo(np.float32).eps * (s[0] if len(s) else 1.0)
        rank = int(np.sum(s > tol))
        extra = min(max(0, k_bundle - 1), rank)
        for idx in range(extra):
            vec = vt[idx].astype(np.float32, copy=False)
            # deterministic sign against mean_delta
            if float(np.dot(vec, mean_delta)) < 0.0:
                vec *= -1.0
            basis_cols.append(vec)
    U = np.stack(basis_cols, axis=1).astype(np.float32, copy=False)
    clean_alpha = (X @ U).astype(np.float32, copy=False)
    clean_alpha_rms = np.sqrt(np.mean(np.square(clean_alpha), axis=0)).astype(np.float32, copy=False)
    return {
        "U": U,
        "K_eff": int(U.shape[1]),
        "clean_alpha_rms": clean_alpha_rms,
    }


def _mean_stats(vecs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not vecs:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    X = np.stack(vecs, axis=0).astype(np.float32, copy=False)
    return X.mean(axis=0), X


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    roots = _parse_roots(args.base_root)
    refs = [ref for ref in _parse_csv_list(args.refs) if ref in REF_CHOICES]
    if not refs:
        raise ValueError("No valid refs selected")
    qids = _parse_csv_list(args.q_list) or _candidate_qids(roots)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path("/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_reweighting")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all available arrays first.
    data: Dict[Tuple[str, str], Dict[str, object]] = {}
    manifest_rows: List[Dict[str, object]] = []
    for ref in refs:
        for qid in qids:
            q_dir = _find_q_dir(roots, qid)
            if q_dir is None:
                manifest_rows.append({
                    "q_id": qid, "ref": ref, "slot_name": SLOT_NAME,
                    "eligible": 0, "reason": "q_dir not found",
                    "eligible_BAB": 0, "eligible_DAD": 0,
                    "n_trials_AAA": "", "n_trials_BBB": "", "n_trials_DDD": "", "n_trials_BABA": "", "n_trials_DADA": "",
                    "n_trials_BAB_intersection": "", "n_trials_DAD_intersection": "",
                    "rank_A": "", "K_A_eff": "", "K_A_full": "", "K_B_eff": "", "K_D_eff": "",
                })
                continue
            try:
                meta = _load_meta(q_dir)
            except FileNotFoundError:
                manifest_rows.append({
                    "q_id": qid, "ref": ref, "slot_name": SLOT_NAME,
                    "eligible": 0, "reason": "missing vector_extraction_meta.json",
                    "eligible_BAB": 0, "eligible_DAD": 0,
                    "n_trials_AAA": "", "n_trials_BBB": "", "n_trials_DDD": "", "n_trials_BABA": "", "n_trials_DADA": "",
                    "n_trials_BAB_intersection": "", "n_trials_DAD_intersection": "",
                    "rank_A": "", "K_A_eff": "", "K_A_full": "", "K_B_eff": "", "K_D_eff": "",
                })
                continue
            cond_payload = {}
            missing = []
            feat_dim = None
            for cond in COND_ORDER:
                path = _vector_path(q_dir, ref, cond)
                if not path.exists():
                    missing.append(cond)
                    continue
                arr = np.load(path).astype(np.float32, copy=False)
                if arr.ndim != 2:
                    missing.append(cond)
                    continue
                feat_dim = int(arr.shape[1]) if feat_dim is None else feat_dim
                trial_ids = _load_trial_ids(meta, cond, arr.shape[0])
                cond_payload[cond] = {"arr": arr, "trial_ids": trial_ids}
            data[(ref, qid)] = {"q_dir": q_dir, "meta": meta, "conds": cond_payload, "missing": missing}

    # Build pooled clean contrast bundles per ref.
    bundle_by_ref: Dict[str, Dict[str, object]] = {}
    for ref in refs:
        clean_B_deltas: List[np.ndarray] = []
        clean_D_deltas: List[np.ndarray] = []
        for qid in qids:
            payload = data.get((ref, qid), {})
            conds = payload.get("conds", {})
            if {"AAA", "BBB"} <= set(conds.keys()):
                aaa = {tid: vec for tid, vec in zip(conds["AAA"]["trial_ids"], conds["AAA"]["arr"])}
                bbb = {tid: vec for tid, vec in zip(conds["BBB"]["trial_ids"], conds["BBB"]["arr"])}
                for tid in sorted(set(aaa) & set(bbb)):
                    clean_B_deltas.append(bbb[tid] - aaa[tid])
            if {"AAA", "DDD"} <= set(conds.keys()):
                aaa = {tid: vec for tid, vec in zip(conds["AAA"]["trial_ids"], conds["AAA"]["arr"])}
                ddd = {tid: vec for tid, vec in zip(conds["DDD"]["trial_ids"], conds["DDD"]["arr"])}
                for tid in sorted(set(aaa) & set(ddd)):
                    clean_D_deltas.append(ddd[tid] - aaa[tid])
        bundle_by_ref[ref] = {
            "B": _build_bundle_basis(clean_B_deltas, k_bundle=args.k_b),
            "D": _build_bundle_basis(clean_D_deltas, k_bundle=args.k_d),
        }

    arrays_by_ref: Dict[str, Dict[str, np.ndarray]] = {ref: {} for ref in refs}
    qwise_rows_by_ref: Dict[str, List[Dict[str, object]]] = {ref: [] for ref in refs}

    for ref in refs:
        U_B = bundle_by_ref[ref]["B"]["U"]
        U_D = bundle_by_ref[ref]["D"]["U"]
        tau_bundle_B = 0.1 * bundle_by_ref[ref]["B"]["clean_alpha_rms"]
        tau_bundle_D = 0.1 * bundle_by_ref[ref]["D"]["clean_alpha_rms"]

        arrays_by_ref[ref][f"{SLOT_NAME}__U_B"] = U_B
        arrays_by_ref[ref][f"{SLOT_NAME}__U_D"] = U_D
        arrays_by_ref[ref][f"{SLOT_NAME}__tau_bundle_B"] = tau_bundle_B
        arrays_by_ref[ref][f"{SLOT_NAME}__tau_bundle_D"] = tau_bundle_D

        for qid in qids:
            payload = data.get((ref, qid))
            if payload is None:
                continue
            conds = payload["conds"]
            available = set(conds.keys())
            have_A_basis = "AAA" in available
            if not have_A_basis:
                manifest_rows.append({
                    "q_id": qid, "ref": ref, "slot_name": SLOT_NAME,
                    "eligible": 0, "reason": "missing AAA",
                    "eligible_BAB": 0, "eligible_DAD": 0,
                    **{f"n_trials_{cond}": (conds[cond]["arr"].shape[0] if cond in conds else "") for cond in COND_ORDER},
                    "n_trials_BAB_intersection": "", "n_trials_DAD_intersection": "",
                    "rank_A": "", "K_A_eff": "", "K_A_full": "", "K_B_eff": bundle_by_ref[ref]["B"]["K_eff"], "K_D_eff": bundle_by_ref[ref]["D"]["K_eff"],
                })
                continue

            aaa_arr = conds["AAA"]["arr"]
            aaa_ids = conds["AAA"]["trial_ids"]
            aaa_map = {tid: vec for tid, vec in zip(aaa_ids, aaa_arr)}

            contrast_mean = None
            if "BBB" in conds:
                bbb_map = {tid: vec for tid, vec in zip(conds["BBB"]["trial_ids"], conds["BBB"]["arr"])}
                common = sorted(set(aaa_map) & set(bbb_map))
                if common:
                    contrast_mean = np.mean([bbb_map[tid] - aaa_map[tid] for tid in common], axis=0).astype(np.float32, copy=False)

            basis = _build_a_basis(aaa_arr, k_a=args.k_a, contrast_mean=contrast_mean)
            mu_A = basis["mu_A"]
            G_A_topk = basis["G_A_topk"]
            G_A_full = basis["G_A_full"]
            rank_A = basis["rank_A"]
            K_A_eff = basis["K_A_eff"]
            K_A_full = G_A_full.shape[1]

            c_AAA_topk = np.stack([_project_coeffs(v, mu_A, G_A_topk) for v in aaa_arr], axis=0) if K_A_eff > 0 else np.zeros((aaa_arr.shape[0], 0), dtype=np.float32)
            tau_deltac = 0.1 * np.std(c_AAA_topk, axis=0) if c_AAA_topk.size else np.zeros((0,), dtype=np.float32)

            arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__mu_A"] = mu_A
            arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__G_A_topk"] = G_A_topk
            arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__G_A_full"] = G_A_full
            arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__explained_variance_ratio"] = basis["explained_variance_ratio"]
            arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__c_AAA_trials_topk"] = c_AAA_topk
            arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__tau_deltac"] = tau_deltac

            # branch-wise matching
            bab_ids = sorted(set(aaa_ids) & set(conds["BBB"]["trial_ids"]) & set(conds["BABA"]["trial_ids"])) if {"BBB", "BABA"} <= available else []
            dad_ids = sorted(set(aaa_ids) & set(conds["DDD"]["trial_ids"]) & set(conds["DADA"]["trial_ids"])) if {"DDD", "DADA"} <= available else []
            bab_eligible = len(bab_ids) > 0
            dad_eligible = len(dad_ids) > 0

            # trial maps
            maps = {cond: {tid: vec for tid, vec in zip(conds[cond]["trial_ids"], conds[cond]["arr"])} for cond in available}

            row: Dict[str, object] = {
                "q_id": qid,
                "slot_name": SLOT_NAME,
                "R_BAB_topk": float("nan"),
                "R_DAD_topk": float("nan"),
                "R_BAB_full": float("nan"),
                "R_DAD_full": float("nan"),
                "inside_change_frac_BAB": float("nan"),
                "outside_change_frac_BAB": float("nan"),
                "inside_change_frac_DAD": float("nan"),
                "outside_change_frac_DAD": float("nan"),
                "active_count_deltac_BAB": float("nan"),
                "active_count_deltac_DAD": float("nan"),
                "PR_deltac_BAB": float("nan"),
                "PR_deltac_DAD": float("nan"),
                "F_B_BAB": float("nan"),
                "F_D_BAB": float("nan"),
                "F_B_DAD": float("nan"),
                "F_D_DAD": float("nan"),
                "M_BAB": float("nan"),
                "M_DAD": float("nan"),
                "active_count_bundle_BAB": float("nan"),
                "active_count_bundle_DAD": float("nan"),
                "PR_bundle_BAB": float("nan"),
                "PR_bundle_DAD": float("nan"),
                "n_trials_AAA": aaa_arr.shape[0],
                "n_trials_BBB": (conds["BBB"]["arr"].shape[0] if "BBB" in conds else np.nan),
                "n_trials_DDD": (conds["DDD"]["arr"].shape[0] if "DDD" in conds else np.nan),
                "n_trials_BABA": (conds["BABA"]["arr"].shape[0] if "BABA" in conds else np.nan),
                "n_trials_DADA": (conds["DADA"]["arr"].shape[0] if "DADA" in conds else np.nan),
                "n_trials_BAB_intersection": len(bab_ids),
                "n_trials_DAD_intersection": len(dad_ids),
            }

            if bab_eligible:
                c_A_trials = []
                c_BAB_trials = []
                delta_c_trials = []
                alpha_BAB_B_trials = []
                alpha_BAB_D_trials = []
                R_topk_trials = []
                R_full_trials = []
                inside_trials = []
                outside_trials = []
                F_B_trials = []
                F_D_trials = []
                M_trials = []
                active_deltac_trials = []
                PR_deltac_trials = []
                active_bundle_trials = []
                PR_bundle_trials = []
                for tid in bab_ids:
                    vA = maps["AAA"][tid]
                    vBAB = maps["BABA"][tid]
                    cA = _project_coeffs(vA, mu_A, G_A_topk)
                    cBAB = _project_coeffs(vBAB, mu_A, G_A_topk)
                    delta_c = cBAB - cA
                    delta = (vBAB - vA).astype(np.float32, copy=False)
                    alpha_B = (U_B.T @ delta).astype(np.float32, copy=False) if U_B.size else np.zeros((0,), dtype=np.float32)
                    alpha_D = (U_D.T @ delta).astype(np.float32, copy=False) if U_D.size else np.zeros((0,), dtype=np.float32)
                    c_A_trials.append(cA)
                    c_BAB_trials.append(cBAB)
                    delta_c_trials.append(delta_c)
                    alpha_BAB_B_trials.append(alpha_B)
                    alpha_BAB_D_trials.append(alpha_D)
                    R_topk_trials.append(_project_energy(vBAB, mu_A, G_A_topk))
                    R_full_trials.append(_project_energy(vBAB, mu_A, G_A_full))
                    inside, outside = _change_inside_outside(delta, G_A_full)
                    inside_trials.append(inside)
                    outside_trials.append(outside)
                    F_B = float(np.sum(np.square(alpha_B)))
                    F_D = float(np.sum(np.square(alpha_D)))
                    F_B_trials.append(F_B)
                    F_D_trials.append(F_D)
                    M_trials.append(F_B - F_D)
                    active_deltac_trials.append(_active_count(delta_c, tau_deltac))
                    PR_deltac_trials.append(_participation_ratio(delta_c))
                    active_bundle_trials.append(_active_count(alpha_B, tau_bundle_B))
                    PR_bundle_trials.append(_participation_ratio(alpha_B))

                row.update({
                    "R_BAB_topk": _mean_or_nan(R_topk_trials),
                    "R_BAB_full": _mean_or_nan(R_full_trials),
                    "inside_change_frac_BAB": _mean_or_nan(inside_trials),
                    "outside_change_frac_BAB": _mean_or_nan(outside_trials),
                    "active_count_deltac_BAB": _mean_or_nan(active_deltac_trials),
                    "PR_deltac_BAB": _mean_or_nan(PR_deltac_trials),
                    "F_B_BAB": _mean_or_nan(F_B_trials),
                    "F_D_BAB": _mean_or_nan(F_D_trials),
                    "M_BAB": _mean_or_nan(M_trials),
                    "active_count_bundle_BAB": _mean_or_nan(active_bundle_trials),
                    "PR_bundle_BAB": _mean_or_nan(PR_bundle_trials),
                })
                arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__c_A_BAB_trials_topk"] = np.stack(c_A_trials, axis=0)
                arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__c_BAB_trials_topk"] = np.stack(c_BAB_trials, axis=0)
                arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__delta_c_BAB_trials_topk"] = np.stack(delta_c_trials, axis=0)
                arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__alpha_BAB_B_trials"] = np.stack(alpha_BAB_B_trials, axis=0) if alpha_BAB_B_trials and alpha_BAB_B_trials[0].size else np.zeros((len(alpha_BAB_B_trials), 0), dtype=np.float32)
                arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__alpha_BAB_D_trials"] = np.stack(alpha_BAB_D_trials, axis=0) if alpha_BAB_D_trials and alpha_BAB_D_trials[0].size else np.zeros((len(alpha_BAB_D_trials), 0), dtype=np.float32)

            if dad_eligible:
                c_A_trials = []
                c_DAD_trials = []
                delta_c_trials = []
                alpha_DAD_B_trials = []
                alpha_DAD_D_trials = []
                R_topk_trials = []
                R_full_trials = []
                inside_trials = []
                outside_trials = []
                F_B_trials = []
                F_D_trials = []
                M_trials = []
                active_deltac_trials = []
                PR_deltac_trials = []
                active_bundle_trials = []
                PR_bundle_trials = []
                for tid in dad_ids:
                    vA = maps["AAA"][tid]
                    vDAD = maps["DADA"][tid]
                    cA = _project_coeffs(vA, mu_A, G_A_topk)
                    cDAD = _project_coeffs(vDAD, mu_A, G_A_topk)
                    delta_c = cDAD - cA
                    delta = (vDAD - vA).astype(np.float32, copy=False)
                    alpha_B = (U_B.T @ delta).astype(np.float32, copy=False) if U_B.size else np.zeros((0,), dtype=np.float32)
                    alpha_D = (U_D.T @ delta).astype(np.float32, copy=False) if U_D.size else np.zeros((0,), dtype=np.float32)
                    c_A_trials.append(cA)
                    c_DAD_trials.append(cDAD)
                    delta_c_trials.append(delta_c)
                    alpha_DAD_B_trials.append(alpha_B)
                    alpha_DAD_D_trials.append(alpha_D)
                    R_topk_trials.append(_project_energy(vDAD, mu_A, G_A_topk))
                    R_full_trials.append(_project_energy(vDAD, mu_A, G_A_full))
                    inside, outside = _change_inside_outside(delta, G_A_full)
                    inside_trials.append(inside)
                    outside_trials.append(outside)
                    F_B = float(np.sum(np.square(alpha_B)))
                    F_D = float(np.sum(np.square(alpha_D)))
                    F_B_trials.append(F_B)
                    F_D_trials.append(F_D)
                    M_trials.append(F_D - F_B)
                    active_deltac_trials.append(_active_count(delta_c, tau_deltac))
                    PR_deltac_trials.append(_participation_ratio(delta_c))
                    active_bundle_trials.append(_active_count(alpha_D, tau_bundle_D))
                    PR_bundle_trials.append(_participation_ratio(alpha_D))

                row.update({
                    "R_DAD_topk": _mean_or_nan(R_topk_trials),
                    "R_DAD_full": _mean_or_nan(R_full_trials),
                    "inside_change_frac_DAD": _mean_or_nan(inside_trials),
                    "outside_change_frac_DAD": _mean_or_nan(outside_trials),
                    "active_count_deltac_DAD": _mean_or_nan(active_deltac_trials),
                    "PR_deltac_DAD": _mean_or_nan(PR_deltac_trials),
                    "F_B_DAD": _mean_or_nan(F_B_trials),
                    "F_D_DAD": _mean_or_nan(F_D_trials),
                    "M_DAD": _mean_or_nan(M_trials),
                    "active_count_bundle_DAD": _mean_or_nan(active_bundle_trials),
                    "PR_bundle_DAD": _mean_or_nan(PR_bundle_trials),
                })
                arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__c_A_DAD_trials_topk"] = np.stack(c_A_trials, axis=0)
                arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__c_DAD_trials_topk"] = np.stack(c_DAD_trials, axis=0)
                arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__delta_c_DAD_trials_topk"] = np.stack(delta_c_trials, axis=0)
                arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__alpha_DAD_B_trials"] = np.stack(alpha_DAD_B_trials, axis=0) if alpha_DAD_B_trials and alpha_DAD_B_trials[0].size else np.zeros((len(alpha_DAD_B_trials), 0), dtype=np.float32)
                arrays_by_ref[ref][f"{qid}__{SLOT_NAME}__alpha_DAD_D_trials"] = np.stack(alpha_DAD_D_trials, axis=0) if alpha_DAD_D_trials and alpha_DAD_D_trials[0].size else np.zeros((len(alpha_DAD_D_trials), 0), dtype=np.float32)

            qwise_rows_by_ref[ref].append({"q_id": qid, "ref": ref, **row})
            manifest_rows.append({
                "q_id": qid,
                "ref": ref,
                "slot_name": SLOT_NAME,
                "eligible": int(bab_eligible or dad_eligible),
                "reason": "ok" if (bab_eligible or dad_eligible) else "missing branch data",
                "eligible_BAB": int(bab_eligible),
                "eligible_DAD": int(dad_eligible),
                **{f"n_trials_{cond}": (conds[cond]["arr"].shape[0] if cond in conds else "") for cond in COND_ORDER},
                "n_trials_BAB_intersection": len(bab_ids),
                "n_trials_DAD_intersection": len(dad_ids),
                "rank_A": rank_A,
                "K_A_eff": K_A_eff,
                "K_A_full": K_A_full,
                "K_B_eff": bundle_by_ref[ref]["B"]["K_eff"],
                "K_D_eff": bundle_by_ref[ref]["D"]["K_eff"],
            })

    # write outputs
    manifest_rows = sorted(manifest_rows, key=lambda r: (r["ref"], _qid_sort_key(str(r["q_id"])), r["slot_name"]))
    _write_csv(out_dir / "reweighting_manifest.csv", manifest_rows)
    for ref in refs:
        rows = sorted(qwise_rows_by_ref[ref], key=lambda r: _qid_sort_key(str(r["q_id"])))
        if rows:
            _write_csv(out_dir / f"reweighting_qwise_{ref}.csv", rows)
        else:
            _write_csv(out_dir / f"reweighting_qwise_{ref}.csv", [])
        np.savez(out_dir / f"reweighting_arrays_{ref}.npz", **arrays_by_ref[ref])
        print(f"qwise_csv_{ref}={out_dir / f'reweighting_qwise_{ref}.csv'}")
        print(f"arrays_npz_{ref}={out_dir / f'reweighting_arrays_{ref}.npz'}")
    print(f"manifest_csv={out_dir / 'reweighting_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
