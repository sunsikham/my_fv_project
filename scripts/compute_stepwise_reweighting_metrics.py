#!/usr/bin/env python3
"""Compute stepwise reweighting metrics from saved AAA_ref stepwise A states."""

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
MATCHED_SLOT_NAMES = ["A_demo_1", "A_demo_2", "A_demo_3", "A_demo_4", "A_query"]
VECTOR_CONDS = ("AAA", "BBB", "DDD")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base_root",
        required=True,
        help="Root or comma-separated roots containing q directories",
    )
    p.add_argument("--q_list", default=None, help="Optional comma-separated q ids")
    p.add_argument("--k_a", type=int, default=5, help="Top-K A basis dimension")
    p.add_argument(
        "--bundle_extra_dims",
        type=int,
        default=2,
        help="Number of residual bundle dimensions after the clean anchor axis",
    )
    p.add_argument(
        "--out_dir",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting",
        help="Output directory",
    )
    return p.parse_args()


def _parse_csv_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


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


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    tmp.replace(path)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _load_query_vectors(q_dir: Path, cond: str) -> np.ndarray:
    path = q_dir / "_vectors" / f"trial_vectors_{REF_NAME}_{cond}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing query vector file: {path}")
    arr = np.load(path).astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D query vectors: {path} shape={arr.shape}")
    return arr


def _load_stepwise_arrays(q_dir: Path) -> Tuple[Dict[str, object], np.lib.npyio.NpzFile]:
    state_dir = q_dir / "_stepwise_a_states"
    meta_path = state_dir / "stepwise_a_states_meta.json"
    npz_path = state_dir / f"stepwise_a_states_{REF_NAME}.npz"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing stepwise meta: {meta_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing stepwise npz: {npz_path}")
    meta = _read_json(meta_path)
    return meta, np.load(npz_path)


def _load_stepwise_condition(npz_obj, qid: str, cond: str, view: str) -> np.ndarray:
    key = f"{qid}__{cond}__{view}__sum"
    if key not in npz_obj.files:
        raise KeyError(f"Missing key in stepwise npz: {key}")
    arr = npz_obj[key].astype(np.float32, copy=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D stepwise array for {key}, got {arr.shape}")
    return arr


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return np.zeros_like(vec)
    return (vec / norm).astype(np.float32, copy=False)


def _center_and_svd(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = arr.mean(axis=0).astype(np.float32, copy=False)
    centered = arr.astype(np.float32, copy=False) - mu
    if centered.shape[0] < 2:
        return mu, centered, np.empty((0,), dtype=np.float32), np.empty((0, centered.shape[1]), dtype=np.float32)
    _u, svals, vt = np.linalg.svd(centered, full_matrices=False)
    tol = max(centered.shape) * np.finfo(np.float32).eps * (svals[0] if len(svals) else 1.0)
    rank = int(np.sum(svals > tol))
    return mu, centered, svals[:rank].astype(np.float32, copy=False), vt[:rank].astype(np.float32, copy=False)


def _build_a_basis(
    arr: np.ndarray,
    *,
    k_a: int,
    align_vec: np.ndarray,
) -> Dict[str, object]:
    mu_A, centered, svals_rank, vt_rank = _center_and_svd(arr)
    rank_A = int(vt_rank.shape[0])
    if rank_A == 0:
        feat_dim = int(arr.shape[1])
        return {
            "mu_A": mu_A,
            "G_A_topk": np.zeros((feat_dim, 0), dtype=np.float32),
            "G_A_full": np.zeros((feat_dim, 0), dtype=np.float32),
            "rank_A": 0,
            "K_A_eff": 0,
            "K_A_full": 0,
            "explained_variance_ratio": np.zeros((0,), dtype=np.float32),
        }

    basis = vt_rank.copy()
    for idx in range(basis.shape[0]):
        if float(np.dot(basis[idx], align_vec)) < 0.0:
            basis[idx] *= -1.0

    n = centered.shape[0]
    if n > 1 and len(svals_rank):
        explained = (svals_rank**2) / float(n - 1)
        total = float(np.sum(explained))
        evr = (explained / total).astype(np.float32, copy=False) if total > 0 else np.zeros_like(explained)
    else:
        evr = np.zeros((rank_A,), dtype=np.float32)

    K_A_eff = min(int(k_a), rank_A, arr.shape[0] - 1)
    G_A_topk = basis[:K_A_eff].T.astype(np.float32, copy=False)
    G_A_full = basis.T.astype(np.float32, copy=False)
    return {
        "mu_A": mu_A,
        "G_A_topk": G_A_topk,
        "G_A_full": G_A_full,
        "rank_A": rank_A,
        "K_A_eff": int(K_A_eff),
        "K_A_full": int(G_A_full.shape[1]),
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
    delta = delta.astype(np.float32, copy=False)
    denom = float(np.dot(delta, delta))
    if denom == 0.0:
        return float("nan"), float("nan")
    if G.size == 0:
        return 0.0, 1.0
    proj = G @ (G.T @ delta)
    inside_num = float(np.dot(proj, proj))
    inside = inside_num / denom
    outside = max(0.0, denom - inside_num) / denom
    return float(inside), float(outside)


def _participation_ratio(vec: np.ndarray) -> float:
    if vec.size == 0:
        return float("nan")
    powers2 = np.square(vec.astype(np.float64, copy=False))
    s2 = float(np.sum(powers2))
    s4 = float(np.sum(np.square(powers2)))
    if s4 == 0.0:
        return 0.0
    return float((s2**2) / s4)


def _mean_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _active_count(vec: np.ndarray, tau: np.ndarray) -> int:
    if vec.size == 0 or tau.size == 0:
        return 0
    return int(np.sum(np.abs(vec) > tau))


def _gram_schmidt_extend(anchor: np.ndarray, candidates: np.ndarray, extra_dims: int) -> np.ndarray:
    cols: List[np.ndarray] = []
    anchor_n = _normalize(anchor)
    if float(np.linalg.norm(anchor_n)) > 0.0:
        cols.append(anchor_n)
    if extra_dims <= 0 or candidates.size == 0:
        return np.stack(cols, axis=1).astype(np.float32, copy=False) if cols else np.zeros((anchor.shape[0], 0), dtype=np.float32)

    for cand in candidates:
        vec = cand.astype(np.float32, copy=False).copy()
        for basis in cols:
            vec = vec - float(np.dot(vec, basis)) * basis
        norm = float(np.linalg.norm(vec))
        if norm <= 0.0:
            continue
        cols.append((vec / norm).astype(np.float32, copy=False))
        if len(cols) >= 1 + extra_dims:
            break
    return np.stack(cols, axis=1).astype(np.float32, copy=False) if cols else np.zeros((anchor.shape[0], 0), dtype=np.float32)


def _build_endpoint_anchored_bundle(
    deltas: np.ndarray,
    anchor: np.ndarray,
    *,
    extra_dims: int,
) -> Dict[str, object]:
    anchor_n = _normalize(anchor)
    feat_dim = int(deltas.shape[1]) if deltas.ndim == 2 else int(anchor.shape[0])
    if deltas.ndim != 2 or deltas.shape[0] == 0 or float(np.linalg.norm(anchor_n)) == 0.0:
        U = anchor_n.reshape(-1, 1) if float(np.linalg.norm(anchor_n)) > 0.0 else np.zeros((feat_dim, 0), dtype=np.float32)
        clean_alpha = deltas @ U if U.size and deltas.ndim == 2 else np.zeros((0, U.shape[1]), dtype=np.float32)
        clean_rms = np.sqrt(np.mean(np.square(clean_alpha), axis=0)).astype(np.float32, copy=False) if clean_alpha.size else np.zeros((U.shape[1],), dtype=np.float32)
        return {"U": U.astype(np.float32, copy=False), "clean_alpha_rms": clean_rms, "K_eff": int(U.shape[1])}

    coeff1 = deltas @ anchor_n
    residuals = deltas - np.outer(coeff1, anchor_n).astype(np.float32, copy=False)
    if residuals.shape[0] > 1 and extra_dims > 0:
        _u, svals, vt = np.linalg.svd(residuals, full_matrices=False)
        tol = max(residuals.shape) * np.finfo(np.float32).eps * (svals[0] if len(svals) else 1.0)
        rank = int(np.sum(svals > tol))
        candidate_rows = vt[:rank].astype(np.float32, copy=False)
    else:
        candidate_rows = np.zeros((0, feat_dim), dtype=np.float32)

    U = _gram_schmidt_extend(anchor_n, candidate_rows, extra_dims)
    clean_alpha = (deltas @ U).astype(np.float32, copy=False) if U.size else np.zeros((deltas.shape[0], 0), dtype=np.float32)
    clean_rms = np.sqrt(np.mean(np.square(clean_alpha), axis=0)).astype(np.float32, copy=False) if clean_alpha.size else np.zeros((U.shape[1],), dtype=np.float32)
    return {"U": U, "clean_alpha_rms": clean_rms, "K_eff": int(U.shape[1])}


def _collect_rows_for_basis(arr3: np.ndarray) -> np.ndarray:
    if arr3.ndim != 3:
        raise ValueError(f"Expected 3D stepwise array, got {arr3.shape}")
    n_trials, n_steps, dim = arr3.shape
    return arr3.reshape(n_trials * n_steps, dim).astype(np.float32, copy=False)


def _row_key(qid: str, basis_scope: str, slot_name: str) -> Tuple[Tuple[int, str], str, int]:
    slot_idx = MATCHED_SLOT_NAMES.index(slot_name) if slot_name in MATCHED_SLOT_NAMES else 999
    return (_qid_sort_key(qid), basis_scope, slot_idx)


def main() -> int:
    args = _parse_args()
    roots = _parse_roots(args.base_root)
    qids = _parse_csv_list(args.q_list) or _candidate_qids(roots)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    arrays_out: Dict[str, np.ndarray] = {}
    analysis_meta: Dict[str, object] = {
        "ref": REF_NAME,
        "k_a": int(args.k_a),
        "bundle_extra_dims": int(args.bundle_extra_dims),
        "basis_scopes": ["matched", "all"],
        "matched_slot_names": MATCHED_SLOT_NAMES,
        "qids": [],
    }

    for qid in qids:
        q_dir = _find_q_dir(roots, qid)
        if q_dir is None:
            manifest_rows.append({"q_id": qid, "ref": REF_NAME, "eligible": 0, "reason": "q_dir not found"})
            continue

        try:
            step_meta, step_npz = _load_stepwise_arrays(q_dir)
            aaa_matched = _load_stepwise_condition(step_npz, qid, "AAA", "matched")
            aaa_all = _load_stepwise_condition(step_npz, qid, "AAA", "all")
            bab_matched = _load_stepwise_condition(step_npz, qid, "BABA", "matched")
            dad_matched = _load_stepwise_condition(step_npz, qid, "DADA", "matched")
            qvec_A = _load_query_vectors(q_dir, "AAA")
            qvec_B = _load_query_vectors(q_dir, "BBB")
            qvec_D = _load_query_vectors(q_dir, "DDD")
        except Exception as exc:
            manifest_rows.append({"q_id": qid, "ref": REF_NAME, "eligible": 0, "reason": str(exc)})
            continue

        trial_ids_meta = step_meta.get("trial_ids_by_condition", {})
        aaa_trial_ids = [str(x) for x in trial_ids_meta.get("AAA", [])]
        bab_trial_ids = [str(x) for x in trial_ids_meta.get("BABA", [])]
        dad_trial_ids = [str(x) for x in trial_ids_meta.get("DADA", [])]
        if not (len(aaa_trial_ids) == aaa_matched.shape[0] and len(bab_trial_ids) == bab_matched.shape[0] and len(dad_trial_ids) == dad_matched.shape[0]):
            manifest_rows.append({"q_id": qid, "ref": REF_NAME, "eligible": 0, "reason": "trial_ids length mismatch"})
            continue

        bab_ids = sorted(set(aaa_trial_ids) & set(bab_trial_ids))
        dad_ids = sorted(set(aaa_trial_ids) & set(dad_trial_ids))
        if not bab_ids or not dad_ids:
            manifest_rows.append({"q_id": qid, "ref": REF_NAME, "eligible": 0, "reason": "missing matched BAB or DAD trials"})
            continue

        aaa_map = {tid: idx for idx, tid in enumerate(aaa_trial_ids)}
        bab_map = {tid: idx for idx, tid in enumerate(bab_trial_ids)}
        dad_map = {tid: idx for idx, tid in enumerate(dad_trial_ids)}

        # Endpoint anchors come from existing query-only vectors.
        a_anchor = qvec_A.mean(axis=0).astype(np.float32, copy=False)
        b_anchor = qvec_B.mean(axis=0).astype(np.float32, copy=False)
        d_anchor = qvec_D.mean(axis=0).astype(np.float32, copy=False)
        b_minus_a = (b_anchor - a_anchor).astype(np.float32, copy=False)
        d_minus_a = (d_anchor - a_anchor).astype(np.float32, copy=False)

        # q-local stepwise deltas for bundle construction.
        X_B = []
        X_D = []
        for tid in bab_ids:
            iA = aaa_map[tid]
            iB = bab_map[tid]
            X_B.append((bab_matched[iB] - aaa_matched[iA]).astype(np.float32, copy=False))
        for tid in dad_ids:
            iA = aaa_map[tid]
            iD = dad_map[tid]
            X_D.append((dad_matched[iD] - aaa_matched[iA]).astype(np.float32, copy=False))
        X_B_mat = np.concatenate(X_B, axis=0) if X_B else np.zeros((0, aaa_matched.shape[-1]), dtype=np.float32)
        X_D_mat = np.concatenate(X_D, axis=0) if X_D else np.zeros((0, aaa_matched.shape[-1]), dtype=np.float32)

        bundle_B = _build_endpoint_anchored_bundle(X_B_mat, b_minus_a, extra_dims=args.bundle_extra_dims)
        bundle_D = _build_endpoint_anchored_bundle(X_D_mat, d_minus_a, extra_dims=args.bundle_extra_dims)
        U_B = bundle_B["U"]
        U_D = bundle_D["U"]
        tau_bundle_B = 0.1 * bundle_B["clean_alpha_rms"]
        tau_bundle_D = 0.1 * bundle_D["clean_alpha_rms"]

        arrays_out[f"{qid}__U_B"] = U_B
        arrays_out[f"{qid}__U_D"] = U_D
        arrays_out[f"{qid}__tau_bundle_B"] = tau_bundle_B
        arrays_out[f"{qid}__tau_bundle_D"] = tau_bundle_D
        arrays_out[f"{qid}__anchor_a_query"] = a_anchor
        arrays_out[f"{qid}__anchor_b_query"] = b_anchor
        arrays_out[f"{qid}__anchor_d_query"] = d_anchor

        analysis_meta["qids"].append(qid)
        analysis_meta[qid] = {
            "n_trials_AAA": int(aaa_matched.shape[0]),
            "n_trials_BAB": int(len(bab_ids)),
            "n_trials_DAD": int(len(dad_ids)),
            "K_B_eff": int(bundle_B["K_eff"]),
            "K_D_eff": int(bundle_D["K_eff"]),
        }

        basis_sources = {
            "matched": aaa_matched,
            "all": aaa_all,
        }
        for basis_scope, aaa_basis_arr in basis_sources.items():
            basis_rows = _collect_rows_for_basis(aaa_basis_arr)
            basis = _build_a_basis(
                basis_rows,
                k_a=args.k_a,
                align_vec=b_minus_a,
            )
            mu_A = basis["mu_A"]
            G_A_topk = basis["G_A_topk"]
            G_A_full = basis["G_A_full"]

            # tau for delta-c comes from AAA baseline coefficients over the basis source rows.
            c_AAA_basis_topk = (
                np.stack([_project_coeffs(v, mu_A, G_A_topk) for v in basis_rows], axis=0)
                if G_A_topk.size
                else np.zeros((basis_rows.shape[0], 0), dtype=np.float32)
            )
            tau_deltac = (
                0.1 * np.std(c_AAA_basis_topk, axis=0).astype(np.float32, copy=False)
                if c_AAA_basis_topk.size
                else np.zeros((0,), dtype=np.float32)
            )

            arrays_out[f"{qid}__{basis_scope}__mu_A"] = mu_A
            arrays_out[f"{qid}__{basis_scope}__G_A_topk"] = G_A_topk
            arrays_out[f"{qid}__{basis_scope}__G_A_full"] = G_A_full
            arrays_out[f"{qid}__{basis_scope}__explained_variance_ratio"] = basis["explained_variance_ratio"]
            arrays_out[f"{qid}__{basis_scope}__c_AAA_basis_topk"] = c_AAA_basis_topk
            arrays_out[f"{qid}__{basis_scope}__tau_deltac"] = tau_deltac

            # Trial-wise containers [trial, step, ...].
            n_bab = len(bab_ids)
            n_dad = len(dad_ids)
            n_steps = len(MATCHED_SLOT_NAMES)
            k_topk = G_A_topk.shape[1]
            kB = U_B.shape[1]
            kD = U_D.shape[1]
            c_A_BAB = np.zeros((n_bab, n_steps, k_topk), dtype=np.float32)
            c_BAB = np.zeros((n_bab, n_steps, k_topk), dtype=np.float32)
            delta_c_BAB = np.zeros((n_bab, n_steps, k_topk), dtype=np.float32)
            alpha_BAB_B = np.zeros((n_bab, n_steps, kB), dtype=np.float32)
            alpha_BAB_D = np.zeros((n_bab, n_steps, kD), dtype=np.float32)
            R_BAB_topk = np.zeros((n_bab, n_steps), dtype=np.float32)
            R_BAB_full = np.zeros((n_bab, n_steps), dtype=np.float32)
            inside_BAB = np.zeros((n_bab, n_steps), dtype=np.float32)
            outside_BAB = np.zeros((n_bab, n_steps), dtype=np.float32)

            c_A_DAD = np.zeros((n_dad, n_steps, k_topk), dtype=np.float32)
            c_DAD = np.zeros((n_dad, n_steps, k_topk), dtype=np.float32)
            delta_c_DAD = np.zeros((n_dad, n_steps, k_topk), dtype=np.float32)
            alpha_DAD_B = np.zeros((n_dad, n_steps, kB), dtype=np.float32)
            alpha_DAD_D = np.zeros((n_dad, n_steps, kD), dtype=np.float32)
            R_DAD_topk = np.zeros((n_dad, n_steps), dtype=np.float32)
            R_DAD_full = np.zeros((n_dad, n_steps), dtype=np.float32)
            inside_DAD = np.zeros((n_dad, n_steps), dtype=np.float32)
            outside_DAD = np.zeros((n_dad, n_steps), dtype=np.float32)

            for row_idx, tid in enumerate(bab_ids):
                iA = aaa_map[tid]
                iB = bab_map[tid]
                for step_idx in range(n_steps):
                    vA = aaa_matched[iA, step_idx]
                    vBAB = bab_matched[iB, step_idx]
                    cA = _project_coeffs(vA, mu_A, G_A_topk)
                    cM = _project_coeffs(vBAB, mu_A, G_A_topk)
                    dc = (cM - cA).astype(np.float32, copy=False)
                    delta = (vBAB - vA).astype(np.float32, copy=False)
                    alphaB = (delta @ U_B).astype(np.float32, copy=False) if U_B.size else np.zeros((0,), dtype=np.float32)
                    alphaD = (delta @ U_D).astype(np.float32, copy=False) if U_D.size else np.zeros((0,), dtype=np.float32)
                    c_A_BAB[row_idx, step_idx] = cA
                    c_BAB[row_idx, step_idx] = cM
                    delta_c_BAB[row_idx, step_idx] = dc
                    alpha_BAB_B[row_idx, step_idx] = alphaB
                    alpha_BAB_D[row_idx, step_idx] = alphaD
                    R_BAB_topk[row_idx, step_idx] = _project_energy(vBAB, mu_A, G_A_topk)
                    R_BAB_full[row_idx, step_idx] = _project_energy(vBAB, mu_A, G_A_full)
                    inside, outside = _change_inside_outside(delta, G_A_full)
                    inside_BAB[row_idx, step_idx] = inside
                    outside_BAB[row_idx, step_idx] = outside

            for row_idx, tid in enumerate(dad_ids):
                iA = aaa_map[tid]
                iD = dad_map[tid]
                for step_idx in range(n_steps):
                    vA = aaa_matched[iA, step_idx]
                    vDAD = dad_matched[iD, step_idx]
                    cA = _project_coeffs(vA, mu_A, G_A_topk)
                    cM = _project_coeffs(vDAD, mu_A, G_A_topk)
                    dc = (cM - cA).astype(np.float32, copy=False)
                    delta = (vDAD - vA).astype(np.float32, copy=False)
                    alphaB = (delta @ U_B).astype(np.float32, copy=False) if U_B.size else np.zeros((0,), dtype=np.float32)
                    alphaD = (delta @ U_D).astype(np.float32, copy=False) if U_D.size else np.zeros((0,), dtype=np.float32)
                    c_A_DAD[row_idx, step_idx] = cA
                    c_DAD[row_idx, step_idx] = cM
                    delta_c_DAD[row_idx, step_idx] = dc
                    alpha_DAD_B[row_idx, step_idx] = alphaB
                    alpha_DAD_D[row_idx, step_idx] = alphaD
                    R_DAD_topk[row_idx, step_idx] = _project_energy(vDAD, mu_A, G_A_topk)
                    R_DAD_full[row_idx, step_idx] = _project_energy(vDAD, mu_A, G_A_full)
                    inside, outside = _change_inside_outside(delta, G_A_full)
                    inside_DAD[row_idx, step_idx] = inside
                    outside_DAD[row_idx, step_idx] = outside

            arrays_out[f"{qid}__{basis_scope}__c_A_BAB_trials_topk"] = c_A_BAB
            arrays_out[f"{qid}__{basis_scope}__c_BAB_trials_topk"] = c_BAB
            arrays_out[f"{qid}__{basis_scope}__delta_c_BAB_trials_topk"] = delta_c_BAB
            arrays_out[f"{qid}__{basis_scope}__alpha_BAB_B_trials"] = alpha_BAB_B
            arrays_out[f"{qid}__{basis_scope}__alpha_BAB_D_trials"] = alpha_BAB_D
            arrays_out[f"{qid}__{basis_scope}__R_BAB_topk_trials"] = R_BAB_topk
            arrays_out[f"{qid}__{basis_scope}__R_BAB_full_trials"] = R_BAB_full
            arrays_out[f"{qid}__{basis_scope}__inside_BAB_trials"] = inside_BAB
            arrays_out[f"{qid}__{basis_scope}__outside_BAB_trials"] = outside_BAB
            arrays_out[f"{qid}__{basis_scope}__c_A_DAD_trials_topk"] = c_A_DAD
            arrays_out[f"{qid}__{basis_scope}__c_DAD_trials_topk"] = c_DAD
            arrays_out[f"{qid}__{basis_scope}__delta_c_DAD_trials_topk"] = delta_c_DAD
            arrays_out[f"{qid}__{basis_scope}__alpha_DAD_B_trials"] = alpha_DAD_B
            arrays_out[f"{qid}__{basis_scope}__alpha_DAD_D_trials"] = alpha_DAD_D
            arrays_out[f"{qid}__{basis_scope}__R_DAD_topk_trials"] = R_DAD_topk
            arrays_out[f"{qid}__{basis_scope}__R_DAD_full_trials"] = R_DAD_full
            arrays_out[f"{qid}__{basis_scope}__inside_DAD_trials"] = inside_DAD
            arrays_out[f"{qid}__{basis_scope}__outside_DAD_trials"] = outside_DAD

            # Step summary rows.
            for step_idx, slot_name in enumerate(MATCHED_SLOT_NAMES):
                dc_bab_step = delta_c_BAB[:, step_idx, :]
                dc_dad_step = delta_c_DAD[:, step_idx, :]
                alpha_bab_B_step = alpha_BAB_B[:, step_idx, :]
                alpha_bab_D_step = alpha_BAB_D[:, step_idx, :]
                alpha_dad_B_step = alpha_DAD_B[:, step_idx, :]
                alpha_dad_D_step = alpha_DAD_D[:, step_idx, :]

                F_B_BAB_trials = np.sum(np.square(alpha_bab_B_step), axis=1) if alpha_bab_B_step.size else np.zeros((n_bab,), dtype=np.float32)
                F_D_BAB_trials = np.sum(np.square(alpha_bab_D_step), axis=1) if alpha_bab_D_step.size else np.zeros((n_bab,), dtype=np.float32)
                F_B_DAD_trials = np.sum(np.square(alpha_dad_B_step), axis=1) if alpha_dad_B_step.size else np.zeros((n_dad,), dtype=np.float32)
                F_D_DAD_trials = np.sum(np.square(alpha_dad_D_step), axis=1) if alpha_dad_D_step.size else np.zeros((n_dad,), dtype=np.float32)

                row = {
                    "q_id": qid,
                    "ref": REF_NAME,
                    "basis_scope": basis_scope,
                    "slot_name": slot_name,
                    "slot_index": step_idx,
                    "R_BAB_topk": _mean_or_nan(R_BAB_topk[:, step_idx]),
                    "R_DAD_topk": _mean_or_nan(R_DAD_topk[:, step_idx]),
                    "R_BAB_full": _mean_or_nan(R_BAB_full[:, step_idx]),
                    "R_DAD_full": _mean_or_nan(R_DAD_full[:, step_idx]),
                    "inside_change_frac_BAB": _mean_or_nan(inside_BAB[:, step_idx]),
                    "outside_change_frac_BAB": _mean_or_nan(outside_BAB[:, step_idx]),
                    "inside_change_frac_DAD": _mean_or_nan(inside_DAD[:, step_idx]),
                    "outside_change_frac_DAD": _mean_or_nan(outside_DAD[:, step_idx]),
                    "active_count_deltac_BAB": _mean_or_nan(_active_count(vec, tau_deltac) for vec in dc_bab_step),
                    "active_count_deltac_DAD": _mean_or_nan(_active_count(vec, tau_deltac) for vec in dc_dad_step),
                    "PR_deltac_BAB": _mean_or_nan(_participation_ratio(vec) for vec in dc_bab_step),
                    "PR_deltac_DAD": _mean_or_nan(_participation_ratio(vec) for vec in dc_dad_step),
                    "F_B_BAB": _mean_or_nan(F_B_BAB_trials),
                    "F_D_BAB": _mean_or_nan(F_D_BAB_trials),
                    "F_B_DAD": _mean_or_nan(F_B_DAD_trials),
                    "F_D_DAD": _mean_or_nan(F_D_DAD_trials),
                    "M_BAB": _mean_or_nan(F_B_BAB_trials - F_D_BAB_trials),
                    "M_DAD": _mean_or_nan(F_D_DAD_trials - F_B_DAD_trials),
                    "active_count_bundle_BAB": _mean_or_nan(_active_count(vec, tau_bundle_B) for vec in alpha_bab_B_step),
                    "active_count_bundle_DAD": _mean_or_nan(_active_count(vec, tau_bundle_D) for vec in alpha_dad_D_step),
                    "PR_bundle_BAB": _mean_or_nan(_participation_ratio(vec) for vec in alpha_bab_B_step),
                    "PR_bundle_DAD": _mean_or_nan(_participation_ratio(vec) for vec in alpha_dad_D_step),
                    "n_trials_BAB": int(n_bab),
                    "n_trials_DAD": int(n_dad),
                    "rank_A": int(basis["rank_A"]),
                    "K_A_eff": int(basis["K_A_eff"]),
                    "K_A_full": int(basis["K_A_full"]),
                    "K_B_eff": int(bundle_B["K_eff"]),
                    "K_D_eff": int(bundle_D["K_eff"]),
                }
                summary_rows.append(row)

        manifest_rows.append(
            {
                "q_id": qid,
                "ref": REF_NAME,
                "eligible": 1,
                "reason": "ok",
                "n_trials_AAA": int(aaa_matched.shape[0]),
                "n_trials_BAB": int(len(bab_ids)),
                "n_trials_DAD": int(len(dad_ids)),
                "K_B_eff": int(bundle_B["K_eff"]),
                "K_D_eff": int(bundle_D["K_eff"]),
            }
        )

    summary_rows = sorted(summary_rows, key=lambda row: _row_key(str(row["q_id"]), str(row["basis_scope"]), str(row["slot_name"])))
    manifest_rows = sorted(manifest_rows, key=lambda row: _qid_sort_key(str(row["q_id"])))

    _write_csv(out_dir / "stepwise_reweighting_manifest.csv", manifest_rows)
    _write_csv(out_dir / f"stepwise_reweighting_{REF_NAME}.csv", summary_rows)
    np.savez(out_dir / f"stepwise_reweighting_arrays_{REF_NAME}.npz", **arrays_out)
    _write_json(out_dir / "stepwise_reweighting_meta.json", analysis_meta)

    print(f"manifest_csv={out_dir / 'stepwise_reweighting_manifest.csv'}")
    print(f"summary_csv={out_dir / f'stepwise_reweighting_{REF_NAME}.csv'}")
    print(f"arrays_npz={out_dir / f'stepwise_reweighting_arrays_{REF_NAME}.npz'}")
    print(f"meta_json={out_dir / 'stepwise_reweighting_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
