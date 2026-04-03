#!/usr/bin/env python3
"""Build a four-module interpretation from Q1 top30 g_k outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


MODULES = [
    "module_B_push",
    "module_D_push",
    "module_shared_scaffold",
    "module_suppression_disambiguation",
]
STEP_NAMES = ["A_demo_1", "A_demo_2", "A_demo_3", "A_demo_4", "A_query"]
REF_NAME = "AAA_ref"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qid", default="Q1")
    p.add_argument("--basis_scope", default="matched", choices=["matched", "all"])
    p.add_argument(
        "--feature_steps_csv",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_endpoint_alignment_top30/stepwise_endpoint_alignment_feature_steps.csv",
    )
    p.add_argument(
        "--feature_summary_csv",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/stepwise_gk_co_movement_feature_summary.csv",
    )
    p.add_argument(
        "--flat_corr_bab",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/stepwise_gk_co_movement_flat_corr_BAB.npy",
    )
    p.add_argument(
        "--flat_corr_dad",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30/stepwise_gk_co_movement_flat_corr_DAD.npy",
    )
    p.add_argument(
        "--out_dir",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_modules_top30",
    )
    return p.parse_args()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mu = float(np.mean(values))
    sd = float(np.std(values))
    if sd == 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mu) / sd


def _positive(x: float) -> float:
    return max(0.0, float(x))


def _negative_magnitude(x: float) -> float:
    return max(0.0, -float(x))


def _weighted_center(weights: Sequence[float]) -> float:
    arr = np.asarray(list(weights), dtype=np.float64)
    pos = np.clip(arr, 0.0, None)
    denom = float(np.sum(pos))
    if denom == 0.0:
        return float("nan")
    idx = np.arange(len(pos), dtype=np.float64)
    return float(np.sum(idx * pos) / denom)


def _late_minus_early(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size < 4:
        return float("nan")
    early = float(np.mean(arr[:2]))
    late = float(np.mean(arr[-2:]))
    return late - early


def _descending_rank(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _ensure_all_modules(rows: List[Dict[str, object]]) -> None:
    present = {str(row["module_label"]) for row in rows}
    missing = [module for module in MODULES if module not in present]
    if not missing:
        return
    by_margin = sorted(rows, key=lambda row: float(row["module_score_margin"]))
    for module in missing:
        candidate = max(rows, key=lambda row: float(row[f"score_{module}"]))
        if str(candidate["module_label"]) == module:
            continue
        candidate["module_label"] = module
        ranked = _descending_rank({m: float(candidate[f"score_{m}"]) for m in MODULES})
        candidate["module_score_margin"] = float(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else float("nan")


def main() -> int:
    args = _parse_args()
    feature_steps_path = Path(args.feature_steps_csv).resolve()
    feature_summary_path = Path(args.feature_summary_csv).resolve()
    flat_corr_bab_path = Path(args.flat_corr_bab).resolve()
    flat_corr_dad_path = Path(args.flat_corr_dad).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_steps = [
        row
        for row in _read_csv(feature_steps_path)
        if row["q_id"] == args.qid and row["basis_scope"] == args.basis_scope
    ]
    feature_summary_rows = [
        row
        for row in _read_csv(feature_summary_path)
        if row["q_id"] == args.qid and row["basis_scope"] == args.basis_scope
    ]
    if not feature_steps:
        raise ValueError(f"No feature-step rows for qid={args.qid} basis_scope={args.basis_scope}")
    if not feature_summary_rows:
        raise ValueError(f"No feature-summary rows for qid={args.qid} basis_scope={args.basis_scope}")

    by_feature: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in feature_steps:
        by_feature[row["feature_name"]].append(row)
    for feat_rows in by_feature.values():
        feat_rows.sort(key=lambda row: int(row["slot_index"]))

    summary_by_feature = {row["feature_name"]: row for row in feature_summary_rows}
    flat_corr_bab = np.load(flat_corr_bab_path).astype(np.float32, copy=False)
    flat_corr_dad = np.load(flat_corr_dad_path).astype(np.float32, copy=False)

    feature_names = sorted(by_feature.keys(), key=lambda name: int(name[1:]))
    if len(feature_names) != 30:
        raise ValueError(f"Expected 30 top-k features, got {len(feature_names)}")

    signature_rows: List[Dict[str, object]] = []
    raw = defaultdict(list)

    per_feature_payload: Dict[str, Dict[str, object]] = {}
    for feat in feature_names:
        rows = by_feature[feat]
        row0 = rows[0]
        align_b = float(row0["align_B"])
        align_d = float(row0["align_D"])
        bab_contrib = [float(row["mean_contrib_BAB_B"]) for row in rows]
        dad_contrib = [float(row["mean_contrib_DAD_D"]) for row in rows]
        bab_dc = [float(row["mean_delta_c_BAB"]) for row in rows]
        dad_dc = [float(row["mean_delta_c_DAD"]) for row in rows]
        bab_abs_dc = [float(row["mean_abs_delta_c_BAB"]) for row in rows]
        dad_abs_dc = [float(row["mean_abs_delta_c_DAD"]) for row in rows]

        bab_pos_mean = _mean(_positive(v) for v in bab_contrib)
        dad_pos_mean = _mean(_positive(v) for v in dad_contrib)
        bab_abs_mean = _mean(abs(v) for v in bab_contrib)
        dad_abs_mean = _mean(abs(v) for v in dad_contrib)
        bab_dc_mean = _mean(bab_dc)
        dad_dc_mean = _mean(dad_dc)
        shared_support = min(bab_pos_mean, dad_pos_mean)
        branch_selectivity = abs(bab_pos_mean - dad_pos_mean) / (bab_pos_mean + dad_pos_mean + 1e-8)
        balance_score = 1.0 - branch_selectivity
        direct_b_raw = bab_pos_mean if align_b > 0 else 0.0
        direct_d_raw = dad_pos_mean if align_d > 0 else 0.0
        suppress_b_raw = bab_pos_mean if align_b < 0 else 0.0
        suppress_d_raw = dad_pos_mean if align_d < 0 else 0.0
        suppression_total = suppress_b_raw + suppress_d_raw
        support_total = bab_pos_mean + dad_pos_mean
        timing_b = _weighted_center(bab_contrib)
        timing_d = _weighted_center(dad_contrib)
        late_minus_early_b = _late_minus_early(bab_contrib)
        late_minus_early_d = _late_minus_early(dad_contrib)

        summary = summary_by_feature[feat]
        mean_abs_corr_flat_bab = float(summary["mean_abs_corr_flat_BAB"])
        mean_abs_corr_flat_dad = float(summary["mean_abs_corr_flat_DAD"])

        feat_idx = int(feat[1:])
        row_payload = {
            "q_id": args.qid,
            "ref": REF_NAME,
            "basis_scope": args.basis_scope,
            "feature_index": feat_idx,
            "feature_name": feat,
            "align_B": align_b,
            "align_D": align_d,
            "bab_pos_mean": bab_pos_mean,
            "dad_pos_mean": dad_pos_mean,
            "bab_abs_mean": bab_abs_mean,
            "dad_abs_mean": dad_abs_mean,
            "bab_dc_mean": bab_dc_mean,
            "dad_dc_mean": dad_dc_mean,
            "shared_support_raw": shared_support,
            "support_total_raw": support_total,
            "branch_selectivity_raw": branch_selectivity,
            "balance_score_raw": balance_score,
            "direct_B_raw": direct_b_raw,
            "direct_D_raw": direct_d_raw,
            "suppression_B_raw": suppress_b_raw,
            "suppression_D_raw": suppress_d_raw,
            "suppression_total_raw": suppression_total,
            "timing_center_B": timing_b,
            "timing_center_D": timing_d,
            "late_minus_early_B": late_minus_early_b,
            "late_minus_early_D": late_minus_early_d,
            "mean_abs_corr_flat_BAB": mean_abs_corr_flat_bab,
            "mean_abs_corr_flat_DAD": mean_abs_corr_flat_dad,
            "top_pos_partner_BAB": summary["top_pos_partner_BAB"],
            "top_pos_partner_corr_BAB": float(summary["top_pos_partner_corr_BAB"]),
            "top_neg_partner_BAB": summary["top_neg_partner_BAB"],
            "top_neg_partner_corr_BAB": float(summary["top_neg_partner_corr_BAB"]),
            "top_pos_partner_DAD": summary["top_pos_partner_DAD"],
            "top_pos_partner_corr_DAD": float(summary["top_pos_partner_corr_DAD"]),
            "top_neg_partner_DAD": summary["top_neg_partner_DAD"],
            "top_neg_partner_corr_DAD": float(summary["top_neg_partner_corr_DAD"]),
        }
        per_feature_payload[feat] = row_payload
        for key, value in row_payload.items():
            if key in {"q_id", "ref", "basis_scope", "feature_index", "feature_name", "top_pos_partner_BAB", "top_neg_partner_BAB", "top_pos_partner_DAD", "top_neg_partner_DAD"}:
                continue
            raw[key].append(float(value))

    z = {key: _zscore(np.asarray(vals, dtype=np.float64)) for key, vals in raw.items()}

    score_rows: List[Dict[str, object]] = []
    assignment_rows: List[Dict[str, object]] = []
    feature_order = {feat: idx for idx, feat in enumerate(feature_names)}
    for feat in feature_names:
        idx = feature_order[feat]
        payload = per_feature_payload[feat]
        score_b_push = (
            1.25 * z["direct_B_raw"][idx]
            + 0.60 * z["align_B"][idx]
            + 0.45 * z["bab_pos_mean"][idx]
            - 0.40 * z["dad_pos_mean"][idx]
            + 0.20 * z["mean_abs_corr_flat_BAB"][idx]
        )
        score_d_push = (
            1.25 * z["direct_D_raw"][idx]
            + 0.60 * z["align_D"][idx]
            + 0.45 * z["dad_pos_mean"][idx]
            - 0.40 * z["bab_pos_mean"][idx]
            + 0.20 * z["mean_abs_corr_flat_DAD"][idx]
        )
        score_shared = (
            1.10 * z["shared_support_raw"][idx]
            + 0.70 * z["balance_score_raw"][idx]
            + 0.35 * z["support_total_raw"][idx]
            + 0.20 * z["mean_abs_corr_flat_BAB"][idx]
            + 0.20 * z["mean_abs_corr_flat_DAD"][idx]
            - 0.20 * z["suppression_total_raw"][idx]
        )
        score_suppress = (
            1.20 * z["suppression_total_raw"][idx]
            + 0.45 * z["branch_selectivity_raw"][idx]
            + 0.25 * z["late_minus_early_B"][idx]
            + 0.25 * z["late_minus_early_D"][idx]
        )

        scores = {
            "module_B_push": float(score_b_push),
            "module_D_push": float(score_d_push),
            "module_shared_scaffold": float(score_shared),
            "module_suppression_disambiguation": float(score_suppress),
        }
        ranked = _descending_rank(scores)
        top_module, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else float("nan")
        margin = float(top_score - second_score) if np.isfinite(second_score) else float("nan")

        signature_rows.append(payload)
        score_rows.append(
            {
                "q_id": args.qid,
                "ref": REF_NAME,
                "basis_scope": args.basis_scope,
                "feature_index": payload["feature_index"],
                "feature_name": feat,
                **{f"score_{name}": value for name, value in scores.items()},
            }
        )
        assignment_rows.append(
            {
                "q_id": args.qid,
                "ref": REF_NAME,
                "basis_scope": args.basis_scope,
                "feature_index": payload["feature_index"],
                "feature_name": feat,
                "module_label": top_module,
                "module_score_top": float(top_score),
                "module_score_second": float(second_score),
                "module_score_margin": margin,
            }
        )

    _ensure_all_modules(assignment_rows)
    assignment_by_feature = {row["feature_name"]: row["module_label"] for row in assignment_rows}

    module_step_rows: List[Dict[str, object]] = []
    module_coupling_rows: List[Dict[str, object]] = []
    module_members: Dict[str, List[str]] = {module: [] for module in MODULES}
    for feat, module in assignment_by_feature.items():
        module_members[module].append(feat)

    # Per-step module summaries.
    for module in MODULES:
        members = set(module_members[module])
        for step_idx, step_name in enumerate(STEP_NAMES):
            step_rows = [
                row
                for row in feature_steps
                if row["feature_name"] in members and int(row["slot_index"]) == step_idx
            ]
            module_step_rows.append(
                {
                    "q_id": args.qid,
                    "ref": REF_NAME,
                    "basis_scope": args.basis_scope,
                    "module_label": module,
                    "n_members": len(members),
                    "slot_name": step_name,
                    "slot_index": step_idx,
                    "sum_mean_contrib_BAB_B": float(sum(float(row["mean_contrib_BAB_B"]) for row in step_rows)),
                    "sum_mean_contrib_DAD_D": float(sum(float(row["mean_contrib_DAD_D"]) for row in step_rows)),
                    "sum_mean_abs_delta_c_BAB": float(sum(float(row["mean_abs_delta_c_BAB"]) for row in step_rows)),
                    "sum_mean_abs_delta_c_DAD": float(sum(float(row["mean_abs_delta_c_DAD"]) for row in step_rows)),
                    "mean_align_B": _mean(float(row["align_B"]) for row in step_rows) if step_rows else float("nan"),
                    "mean_align_D": _mean(float(row["align_D"]) for row in step_rows) if step_rows else float("nan"),
                }
            )

    # Module coupling summaries using flat correlation matrices.
    for branch_name, corr in [("BAB", flat_corr_bab), ("DAD", flat_corr_dad)]:
        for i, module_a in enumerate(MODULES):
            idx_a = [feature_order[feat] for feat in module_members[module_a]]
            for j, module_b in enumerate(MODULES):
                idx_b = [feature_order[feat] for feat in module_members[module_b]]
                if not idx_a or not idx_b:
                    mean_corr = float("nan")
                    mean_abs_corr = float("nan")
                    n_pairs = 0
                elif module_a == module_b:
                    vals = []
                    for a_pos, a_idx in enumerate(idx_a):
                        for b_idx in idx_a[a_pos + 1 :]:
                            vals.append(float(corr[a_idx, b_idx]))
                    mean_corr = _mean(vals) if vals else float("nan")
                    mean_abs_corr = _mean(abs(v) for v in vals) if vals else float("nan")
                    n_pairs = len(vals)
                else:
                    vals = [float(corr[a_idx, b_idx]) for a_idx in idx_a for b_idx in idx_b]
                    mean_corr = _mean(vals) if vals else float("nan")
                    mean_abs_corr = _mean(abs(v) for v in vals) if vals else float("nan")
                    n_pairs = len(vals)
                module_coupling_rows.append(
                    {
                        "q_id": args.qid,
                        "ref": REF_NAME,
                        "basis_scope": args.basis_scope,
                        "branch": branch_name,
                        "module_a": module_a,
                        "module_b": module_b,
                        "n_features_a": len(idx_a),
                        "n_features_b": len(idx_b),
                        "n_pairs": n_pairs,
                        "mean_corr": mean_corr,
                        "mean_abs_corr": mean_abs_corr,
                    }
                )

    signature_rows = sorted(signature_rows, key=lambda row: int(row["feature_index"]))
    score_rows = sorted(score_rows, key=lambda row: int(row["feature_index"]))
    assignment_rows = sorted(assignment_rows, key=lambda row: int(row["feature_index"]))
    module_step_rows = sorted(module_step_rows, key=lambda row: (row["module_label"], int(row["slot_index"])))
    module_coupling_rows = sorted(module_coupling_rows, key=lambda row: (row["branch"], row["module_a"], row["module_b"]))

    _write_csv(out_dir / "gk_module_signatures.csv", signature_rows)
    _write_csv(out_dir / "gk_module_scores.csv", score_rows)
    _write_csv(out_dir / "gk_module_assignments.csv", assignment_rows)
    _write_csv(out_dir / "module_stepwise_summary.csv", module_step_rows)
    _write_csv(out_dir / "module_coupling_summary.csv", module_coupling_rows)
    _write_json(
        out_dir / "module_meta.json",
        {
            "qid": args.qid,
            "ref": REF_NAME,
            "basis_scope": args.basis_scope,
            "modules": MODULES,
            "module_members": module_members,
            "input_paths": {
                "feature_steps_csv": str(feature_steps_path),
                "feature_summary_csv": str(feature_summary_path),
                "flat_corr_bab": str(flat_corr_bab_path),
                "flat_corr_dad": str(flat_corr_dad_path),
            },
            "assignment_rule": "argmax over interpretable module scores with non-empty-module fallback",
            "score_notes": {
                "module_B_push": "positive B alignment plus positive BAB intended contribution and branch preference toward BAB",
                "module_D_push": "positive D alignment plus positive DAD intended contribution and branch preference toward DAD",
                "module_shared_scaffold": "balanced positive support across both branches with lower branch selectivity",
                "module_suppression_disambiguation": "positive intended contribution achieved through negatively aligned axes decreasing in the supporting branch",
            },
        },
    )

    print(f"signatures_csv={out_dir / 'gk_module_signatures.csv'}")
    print(f"scores_csv={out_dir / 'gk_module_scores.csv'}")
    print(f"assignments_csv={out_dir / 'gk_module_assignments.csv'}")
    print(f"module_stepwise_csv={out_dir / 'module_stepwise_summary.csv'}")
    print(f"module_coupling_csv={out_dir / 'module_coupling_summary.csv'}")
    print(f"meta_json={out_dir / 'module_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
