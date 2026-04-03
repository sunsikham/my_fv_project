#!/usr/bin/env python3
"""Compute q-wise movement scores from condition mean vectors."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


COND_TO_SYMBOL = {
    "AAA": "a",
    "BBB": "b",
    "DDD": "d",
    "BABA": "x_bab",
    "DADA": "x_dad",
}
REQUIRED_CONDS = ("AAA", "BBB", "DDD", "BABA", "DADA")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute q-wise movement scores from AAA_ref condition vectors")
    p.add_argument("--base_root", required=True, help="Root or comma-separated roots containing per-q directories")
    p.add_argument("--q_list", default=None, help="Optional comma-separated q_ids")
    p.add_argument("--vector_ref", default="AAA_ref", help="Vector reference prefix (default: AAA_ref)")
    p.add_argument("--out_csv", default=None, help="Output q-wise CSV path")
    p.add_argument("--out_npz", default=None, help="Output npz path for condition mean vectors")
    p.add_argument("--manifest_csv", default=None, help="Output eligibility manifest CSV path")
    return p.parse_args()


def _parse_q_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    seen = set()
    out: List[str] = []
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        out.append(part)
    return out


def _parse_root_list(raw: str) -> List[Path]:
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise ValueError("base_root is empty")
    return [Path(part).resolve() for part in parts]


def _qid_sort_key(qid: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)", str(qid))
    return (int(m.group(1)) if m else 1_000_000, str(qid))


def _vector_path(q_dir: Path, vector_ref: str, cond: str) -> Path:
    return q_dir / "_vectors" / f"trial_vectors_{vector_ref}_{cond}.npy"


def _mean_vector(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("Need at least one trial vector to compute mean")
    return arr.astype(np.float32, copy=False).mean(axis=0)


def _progress(x: np.ndarray, a: np.ndarray, y: np.ndarray) -> float:
    direction = y - a
    denom = float(np.dot(direction, direction))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(x - a, direction) / denom)


def _off_axis_residual(x: np.ndarray, a: np.ndarray, y: np.ndarray) -> float:
    direction = y - a
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return float("nan")
    progress = _progress(x, a, y)
    if np.isnan(progress):
        return float("nan")
    residual = (x - a) - (progress * direction)
    return float(np.linalg.norm(residual) / norm)


def _safe_cosine(u: np.ndarray, v: np.ndarray) -> float:
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(u, v) / denom)


def _safe_angle_deg(cosine: float) -> float:
    if np.isnan(cosine):
        return float("nan")
    clipped = max(-1.0, min(1.0, float(cosine)))
    return float(np.degrees(np.arccos(clipped)))


def _joint_decomposition(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    d: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    u_b = (b - a).astype(np.float32, copy=False)
    u_d = (d - a).astype(np.float32, copy=False)
    delta = (x - a).astype(np.float32, copy=False)
    U = np.stack([u_b, u_d], axis=1)
    coeffs, _residuals, _rank, _svals = np.linalg.lstsq(U, delta, rcond=None)
    alpha = float(coeffs[0])
    beta = float(coeffs[1])
    x_hat = (alpha * u_b) + (beta * u_d)
    eps = delta - x_hat
    norm_b = float(np.linalg.norm(u_b))
    norm_d = float(np.linalg.norm(u_d))
    resid_b = float(np.linalg.norm(eps) / norm_b) if norm_b != 0.0 else float("nan")
    resid_d = float(np.linalg.norm(eps) / norm_d) if norm_d != 0.0 else float("nan")
    cosine = _safe_cosine(u_b, u_d)
    try:
        cond_num = float(np.linalg.cond(U))
    except Exception:
        cond_num = float("nan")
    return {
        "alpha": alpha,
        "beta": beta,
        "eps": eps.astype(np.float32, copy=False),
        "x_hat": x_hat.astype(np.float32, copy=False),
        "u_b": u_b.astype(np.float32, copy=False),
        "u_d": u_d.astype(np.float32, copy=False),
        "resid_b": resid_b,
        "resid_d": resid_d,
        "axis_cosine": cosine,
        "axis_angle_deg": _safe_angle_deg(cosine),
        "axis_cond_number": cond_num,
    }


def _candidate_qids(base_roots: Sequence[Path]) -> List[str]:
    seen = set()
    out: List[str] = []
    for base_root in base_roots:
        for child in sorted(base_root.iterdir()):
            if child.is_dir() and re.fullmatch(r"Q\d+", child.name) and child.name not in seen:
                seen.add(child.name)
                out.append(child.name)
    return out


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
    base_roots = _parse_root_list(args.base_root)
    for base_root in base_roots:
        if not base_root.exists():
            raise FileNotFoundError(f"base_root not found: {base_root}")

    q_list = _parse_q_list(args.q_list)
    qids = q_list if q_list else _candidate_qids(base_roots)
    qids = sorted(qids, key=_qid_sort_key)
    if not qids:
        raise ValueError("No q_ids found")

    default_analysis_root = base_roots[0] if len(base_roots) == 1 else Path("/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_multi_root")
    out_csv = Path(args.out_csv) if args.out_csv else default_analysis_root / "movement_qwise.csv"
    out_npz = Path(args.out_npz) if args.out_npz else default_analysis_root / "movement_condition_means.npz"
    manifest_csv = Path(args.manifest_csv) if args.manifest_csv else default_analysis_root / "movement_qwise_manifest.csv"

    manifest_rows: List[Dict[str, object]] = []
    score_rows: List[Dict[str, object]] = []
    npz_payload: Dict[str, np.ndarray] = {}

    for qid in qids:
        q_dir = None
        for root in base_roots:
            candidate = root / qid
            if candidate.exists():
                q_dir = candidate
                break
        if q_dir is None:
            manifest_rows.append(
                {
                    "q_id": qid,
                    "eligible": 0,
                    "reason": "q_dir not found in provided roots",
                    **{f"n_trials_{cond}": "" for cond in REQUIRED_CONDS},
                }
            )
            continue
        cond_arrays: Dict[str, np.ndarray] = {}
        missing: List[str] = []
        bad_shape: List[str] = []
        n_trials: Dict[str, int] = {}
        feat_dim: Optional[int] = None

        for cond in REQUIRED_CONDS:
            path = _vector_path(q_dir, args.vector_ref, cond)
            if not path.exists():
                missing.append(cond)
                continue
            arr = np.load(path)
            if arr.ndim != 2:
                bad_shape.append(f"{cond}:{tuple(arr.shape)}")
                continue
            if feat_dim is None:
                feat_dim = int(arr.shape[1])
            elif int(arr.shape[1]) != feat_dim:
                bad_shape.append(f"{cond}:feat{arr.shape[1]}")
                continue
            cond_arrays[cond] = arr
            n_trials[cond] = int(arr.shape[0])

        if missing or bad_shape:
            reason_parts = []
            if missing:
                reason_parts.append(f"missing={','.join(missing)}")
            if bad_shape:
                reason_parts.append(f"bad_shape={';'.join(bad_shape)}")
            manifest_rows.append(
                {
                    "q_id": qid,
                    "eligible": 0,
                    "reason": " | ".join(reason_parts),
                    **{f"n_trials_{cond}": n_trials.get(cond, "") for cond in REQUIRED_CONDS},
                }
            )
            continue

        means = {cond: _mean_vector(arr) for cond, arr in cond_arrays.items()}
        a = means["AAA"]
        b = means["BBB"]
        d = means["DDD"]
        x_bab = means["BABA"]
        x_dad = means["DADA"]

        p_b_x_bab = _progress(x_bab, a, b)
        p_d_x_bab = _progress(x_bab, a, d)
        p_d_x_dad = _progress(x_dad, a, d)
        p_b_x_dad = _progress(x_dad, a, b)
        r_b_x_bab = _off_axis_residual(x_bab, a, b)
        r_d_x_dad = _off_axis_residual(x_dad, a, d)
        joint_bab = _joint_decomposition(x_bab, a, b, d)
        joint_dad = _joint_decomposition(x_dad, a, b, d)

        row = {
            "q_id": qid,
            "p_b_x_bab": p_b_x_bab,
            "p_d_x_bab": p_d_x_bab,
            "p_d_x_dad": p_d_x_dad,
            "p_b_x_dad": p_b_x_dad,
            "s_bab": (p_b_x_bab - p_d_x_bab) if not (np.isnan(p_b_x_bab) or np.isnan(p_d_x_bab)) else float("nan"),
            "s_dad": (p_d_x_dad - p_b_x_dad) if not (np.isnan(p_d_x_dad) or np.isnan(p_b_x_dad)) else float("nan"),
            "r_b_x_bab": r_b_x_bab,
            "r_d_x_dad": r_d_x_dad,
            "alpha_bab": joint_bab["alpha"],
            "beta_bab": joint_bab["beta"],
            "alpha_dad": joint_dad["alpha"],
            "beta_dad": joint_dad["beta"],
            "joint_selectivity_bab": float(joint_bab["alpha"] - joint_bab["beta"]),
            "joint_selectivity_dad": float(joint_dad["beta"] - joint_dad["alpha"]),
            "joint_resid_bab": joint_bab["resid_b"],
            "joint_resid_dad": joint_dad["resid_d"],
            "axis_cosine": joint_bab["axis_cosine"],
            "axis_angle_deg": joint_bab["axis_angle_deg"],
            "axis_cond_number": joint_bab["axis_cond_number"],
            "norm_ab": float(np.linalg.norm(b - a)),
            "norm_ad": float(np.linalg.norm(d - a)),
            **{f"n_trials_{cond}": n_trials[cond] for cond in REQUIRED_CONDS},
        }
        score_rows.append(row)
        manifest_rows.append(
            {
                "q_id": qid,
                "eligible": 1,
                "reason": "ok",
                **{f"n_trials_{cond}": n_trials[cond] for cond in REQUIRED_CONDS},
            }
        )

        for cond, symbol in COND_TO_SYMBOL.items():
            npz_payload[f"{qid}__{symbol}"] = means[cond].astype(np.float32, copy=False)
        npz_payload[f"{qid}__u_B"] = joint_bab["u_b"]
        npz_payload[f"{qid}__u_D"] = joint_bab["u_d"]
        npz_payload[f"{qid}__xhat_bab"] = joint_bab["x_hat"]
        npz_payload[f"{qid}__xhat_dad"] = joint_dad["x_hat"]
        npz_payload[f"{qid}__eps_bab"] = joint_bab["eps"]
        npz_payload[f"{qid}__eps_dad"] = joint_dad["eps"]

    if not manifest_rows:
        raise ValueError("No manifest rows produced")

    if score_rows:
        score_rows = sorted(score_rows, key=lambda r: _qid_sort_key(str(r["q_id"])))
        _write_csv(out_csv, score_rows)
        out_csv_written = str(out_csv)
    else:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "q_id",
                "p_b_x_bab",
                "p_d_x_bab",
                "p_d_x_dad",
                "p_b_x_dad",
                "s_bab",
                "s_dad",
                "r_b_x_bab",
                "r_d_x_dad",
                "alpha_bab",
                "beta_bab",
                "alpha_dad",
                "beta_dad",
                "joint_selectivity_bab",
                "joint_selectivity_dad",
                "joint_resid_bab",
                "joint_resid_dad",
                "axis_cosine",
                "axis_angle_deg",
                "axis_cond_number",
                "norm_ab",
                "norm_ad",
                *[f"n_trials_{cond}" for cond in REQUIRED_CONDS],
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
        out_csv_written = str(out_csv)

    manifest_rows = sorted(manifest_rows, key=lambda r: _qid_sort_key(str(r["q_id"])))
    _write_csv(manifest_csv, manifest_rows)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **npz_payload)

    print(f"out_csv={out_csv_written}")
    print(f"out_npz={out_npz}")
    print(f"manifest_csv={manifest_csv}")
    print(f"eligible_q={sum(int(r['eligible']) for r in manifest_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
