#!/usr/bin/env python3
"""Run strict same-state inside/outside intervention at Q1 A_query.

This runner is intentionally offline:
  - no model loading
  - no tokenizer use
  - no prompt rebuilding
  - no residual hook

It operates directly in the extracted A_query state space and evaluates
trial-exact / mean-vector interventions against the real AAA/BABA/DADA states.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEPWISE_ROOT = PROJECT_ROOT / "results_fv" / "relation_condition_qwise"
DEFAULT_REWEIGHT_NPZ = Path(
    "/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting/stepwise_reweighting_arrays_AAA_ref.npz"
)
DEFAULT_INSIDE_OUTSIDE_NPZ = Path(
    "/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_inside_outside/stepwise_inside_outside_endpoint_arrays.npz"
)
DEFAULT_OUT_DIR = Path(
    "/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_state_intervention"
)
SLOT_NAMES = ["A_demo_1", "A_demo_2", "A_demo_3", "A_demo_4", "A_query"]
MODE_CHOICES = ("trial_exact", "mean_vector")
CONDITION_ORDER = {"none": 0, "inside": 1, "outside": 2, "full": 3}
BRANCH_ORDER = {"B": 0, "D": 1}
SCENARIO_ORDER = {"sufficiency": 0, "necessity": 1}
MODE_ORDER = {"trial_exact": 0, "mean_vector": 1}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qid", default="Q1")
    p.add_argument("--ref", default="AAA_ref")
    p.add_argument("--basis_scope", default="matched")
    p.add_argument("--slot_name", default="A_query")
    p.add_argument("--mode_list", default="trial_exact,mean_vector")
    p.add_argument("--alpha_list", default="0.0,0.5,1.0,1.5")
    p.add_argument("--n_trials", type=int, default=0, help="Optional cap on number of trials to use.")
    p.add_argument(
        "--stepwise_root",
        default=str(DEFAULT_STEPWISE_ROOT),
        help="Root containing q dirs with _stepwise_a_states.",
    )
    p.add_argument("--stepwise_states_npz", default=None, help="Optional explicit stepwise states NPZ.")
    p.add_argument("--stepwise_meta_json", default=None, help="Optional explicit stepwise meta JSON.")
    p.add_argument("--reweight_npz", default=str(DEFAULT_REWEIGHT_NPZ))
    p.add_argument("--inside_outside_npz", default=str(DEFAULT_INSIDE_OUTSIDE_NPZ))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--out_vectors_npz", default=None)
    p.add_argument("--out_vectors_json", default=None)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--out_summary_csv", default=None)
    p.add_argument("--out_states_npz", default=None)
    p.add_argument("--out_report_md", default=None)

    # Legacy args are accepted so old launch scripts do not crash, but they are ignored.
    p.add_argument("--model", default=None, help=argparse.SUPPRESS)
    p.add_argument("--model_spec", default=None, help=argparse.SUPPRESS)
    p.add_argument("--device", default=None, help=argparse.SUPPRESS)
    p.add_argument("--dtype", default=None, help=argparse.SUPPRESS)
    p.add_argument("--quant", default=None, help=argparse.SUPPRESS)
    p.add_argument("--layers", default=None, help=argparse.SUPPRESS)
    p.add_argument("--shot_list", default=None, help=argparse.SUPPRESS)
    p.add_argument("--seed", default=None, help=argparse.SUPPRESS)
    p.add_argument("--batch_size", default=None, help=argparse.SUPPRESS)
    p.add_argument("--vector_npz", default=None, help=argparse.SUPPRESS)
    p.add_argument("--trial_plan_json", default=None, help=argparse.SUPPRESS)
    p.add_argument("--relationA_ex_path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--relationB_ex_path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--relationD_ex_path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--icl_B_path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--icl_D_path", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def _parse_csv_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


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


def _qid_sort_key(qid: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)", str(qid))
    return (int(m.group(1)) if m else 1_000_000, str(qid))


def _find_q_dir(root: Path, qid: str) -> Optional[Path]:
    for child in root.rglob(qid):
        if child.is_dir() and child.name == qid:
            return child
    return None


def _resolve_stepwise_paths(
    *,
    qid: str,
    ref: str,
    stepwise_root: Path,
    stepwise_states_npz: Optional[str],
    stepwise_meta_json: Optional[str],
) -> Tuple[Path, Path]:
    if stepwise_states_npz:
        npz_path = Path(stepwise_states_npz).resolve()
        meta_path = (
            Path(stepwise_meta_json).resolve()
            if stepwise_meta_json
            else npz_path.parent / "stepwise_a_states_meta.json"
        )
        return npz_path, meta_path

    q_dir = _find_q_dir(stepwise_root.resolve(), qid)
    if q_dir is None:
        raise FileNotFoundError(f"q_dir not found for {qid} under {stepwise_root}")
    stepwise_dir = q_dir / "_stepwise_a_states"
    npz_path = stepwise_dir / f"stepwise_a_states_{ref}.npz"
    meta_path = stepwise_dir / "stepwise_a_states_meta.json"
    return npz_path, meta_path


def _slot_index(slot_name: str, slot_names: Sequence[str]) -> int:
    try:
        return list(slot_names).index(slot_name)
    except ValueError as exc:
        raise ValueError(f"Unknown slot_name={slot_name}; available={list(slot_names)}") from exc


def _alpha_label(alpha: float) -> str:
    text = f"{float(alpha):.3f}".rstrip("0").rstrip(".")
    text = text.replace("-", "m").replace(".", "p")
    if not text:
        text = "0"
    if text == "m0":
        text = "0"
    return f"a{text}"


def _norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec.astype(np.float32, copy=False)))


def _safe_ratio(num: float, den: float) -> float:
    num = float(num)
    den = float(den)
    if den == 0.0:
        return float("nan")
    return float(num / den)


def _mean_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    nrm = float(np.linalg.norm(vec))
    if nrm == 0.0:
        return np.zeros_like(vec)
    return (vec / nrm).astype(np.float32, copy=False)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _project_inside(vec: np.ndarray, G: np.ndarray) -> np.ndarray:
    vec32 = vec.astype(np.float32, copy=False)
    coeffs = vec32 @ G
    return (coeffs @ G.T).astype(np.float32, copy=False)


def _joint_decompose(delta: np.ndarray, intended: np.ndarray, cross: np.ndarray) -> Tuple[float, float, float]:
    delta = delta.astype(np.float32, copy=False)
    intended = intended.astype(np.float32, copy=False)
    cross = cross.astype(np.float32, copy=False)
    if float(np.linalg.norm(intended)) == 0.0 and float(np.linalg.norm(cross)) == 0.0:
        return float("nan"), float("nan"), _norm(delta)
    U = np.stack([intended, cross], axis=1)
    coeffs, _resid, _rank, _svals = np.linalg.lstsq(U, delta, rcond=None)
    recon = U @ coeffs
    resid = float(np.linalg.norm(delta - recon))
    return float(coeffs[0]), float(coeffs[1]), resid


def _default_output_path(out_dir: Path, qid: str, stem: str, suffix: str) -> Path:
    return out_dir / f"{stem}_{qid}{suffix}"


def _legacy_vector_output_path(raw: Optional[str], out_dir: Path, qid: str) -> Path:
    if raw is None:
        return _default_output_path(out_dir, qid, "inside_outside_state_intervention_vectors", ".npz")
    path = Path(raw).expanduser()
    if path.name == f"inside_outside_intervention_vectors_{qid}.npz":
        return path.with_name(f"inside_outside_state_intervention_vectors_{qid}.npz")
    return path


def _resolve_output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_vectors_npz = (
        Path(args.out_vectors_npz).resolve()
        if args.out_vectors_npz
        else _legacy_vector_output_path(args.vector_npz, out_dir, args.qid).resolve()
    )
    out_vectors_json = (
        Path(args.out_vectors_json).resolve()
        if args.out_vectors_json
        else out_vectors_npz.with_suffix(".json")
    )
    out_csv = (
        Path(args.out_csv).resolve()
        if args.out_csv
        else _default_output_path(out_dir, args.qid, "inside_outside_state_intervention_raw_rows", ".csv")
    )
    out_summary_csv = (
        Path(args.out_summary_csv).resolve()
        if args.out_summary_csv
        else _default_output_path(out_dir, args.qid, "inside_outside_state_intervention_summary", ".csv")
    )
    out_states_npz = (
        Path(args.out_states_npz).resolve()
        if args.out_states_npz
        else _default_output_path(out_dir, args.qid, "inside_outside_state_intervention_synthetic_states", ".npz")
    )
    out_report_md = (
        Path(args.out_report_md).resolve()
        if args.out_report_md
        else _default_output_path(out_dir, args.qid, "inside_outside_state_intervention_report", ".md")
    )
    return {
        "out_dir": out_dir,
        "out_vectors_npz": out_vectors_npz,
        "out_vectors_json": out_vectors_json,
        "out_csv": out_csv,
        "out_summary_csv": out_summary_csv,
        "out_states_npz": out_states_npz,
        "out_report_md": out_report_md,
    }


def build_intervention_vector_payload(
    *,
    qid: str,
    ref: str,
    basis_scope: str,
    slot_name: str,
    n_trials_cap: int,
    stepwise_root: Path,
    stepwise_states_npz: Optional[str],
    stepwise_meta_json: Optional[str],
    reweight_npz: Path,
    inside_outside_npz: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    states_npz_path, states_meta_path = _resolve_stepwise_paths(
        qid=qid,
        ref=ref,
        stepwise_root=stepwise_root,
        stepwise_states_npz=stepwise_states_npz,
        stepwise_meta_json=stepwise_meta_json,
    )
    if not states_npz_path.exists():
        raise FileNotFoundError(f"Missing stepwise_states_npz: {states_npz_path}")
    if not states_meta_path.exists():
        raise FileNotFoundError(f"Missing stepwise_meta_json: {states_meta_path}")
    if not reweight_npz.exists():
        raise FileNotFoundError(f"Missing reweight_npz: {reweight_npz}")
    if not inside_outside_npz.exists():
        raise FileNotFoundError(f"Missing inside_outside_npz: {inside_outside_npz}")

    states_npz = np.load(states_npz_path)
    states_meta = _read_json(states_meta_path)
    reweight = np.load(reweight_npz)
    inside_outside = np.load(inside_outside_npz)

    slot_names = list(states_meta.get("matched_slot_names", SLOT_NAMES))
    slot_idx = _slot_index(slot_name, slot_names)
    trial_ids_by_condition = dict(states_meta.get("trial_ids_by_condition", {}))
    trial_ids_A = [str(x) for x in trial_ids_by_condition.get("AAA", [])]
    trial_ids_B = [str(x) for x in trial_ids_by_condition.get("BABA", [])]
    trial_ids_D = [str(x) for x in trial_ids_by_condition.get("DADA", [])]
    if not trial_ids_A or trial_ids_A != trial_ids_B or trial_ids_A != trial_ids_D:
        raise ValueError("Expected identical trial_ids across AAA/BABA/DADA for strict same-state intervention.")

    vA_key = f"{qid}__AAA__{basis_scope}__sum"
    vB_key = f"{qid}__BABA__{basis_scope}__sum"
    vD_key = f"{qid}__DADA__{basis_scope}__sum"
    G_key = f"{qid}__{basis_scope}__G_A_full"
    required_keys = [
        vA_key,
        vB_key,
        vD_key,
        G_key,
        f"{qid}__U_B",
        f"{qid}__U_D",
        f"{qid}__anchor_a_query",
        f"{qid}__anchor_b_query",
        f"{qid}__anchor_d_query",
        f"{qid}__u_B_in",
        f"{qid}__u_D_in",
        f"{qid}__u_B_out",
        f"{qid}__u_D_out",
        f"{qid}__delta_in_BAB_trials",
        f"{qid}__delta_out_BAB_trials",
        f"{qid}__delta_in_DAD_trials",
        f"{qid}__delta_out_DAD_trials",
    ]
    for key in required_keys:
        if key not in states_npz.files and key not in reweight.files and key not in inside_outside.files:
            raise KeyError(f"Missing required key: {key}")

    v_A_all = states_npz[vA_key].astype(np.float32, copy=False)
    v_B_all = states_npz[vB_key].astype(np.float32, copy=False)
    v_D_all = states_npz[vD_key].astype(np.float32, copy=False)
    if v_A_all.shape[:2] != v_B_all.shape[:2] or v_A_all.shape[:2] != v_D_all.shape[:2]:
        raise ValueError("AAA/BABA/DADA state arrays must align in trial and step dimensions.")

    n_available = int(v_A_all.shape[0])
    n_use = n_available if n_trials_cap <= 0 else min(int(n_trials_cap), n_available)

    v_A_trials = v_A_all[:n_use, slot_idx, :].astype(np.float32, copy=False)
    v_B_trials = v_B_all[:n_use, slot_idx, :].astype(np.float32, copy=False)
    v_D_trials = v_D_all[:n_use, slot_idx, :].astype(np.float32, copy=False)

    inside_B_trials = inside_outside[f"{qid}__delta_in_BAB_trials"][:n_use, slot_idx, :].astype(np.float32, copy=False)
    outside_B_trials = inside_outside[f"{qid}__delta_out_BAB_trials"][:n_use, slot_idx, :].astype(np.float32, copy=False)
    inside_D_trials = inside_outside[f"{qid}__delta_in_DAD_trials"][:n_use, slot_idx, :].astype(np.float32, copy=False)
    outside_D_trials = inside_outside[f"{qid}__delta_out_DAD_trials"][:n_use, slot_idx, :].astype(np.float32, copy=False)

    full_B_trials = (v_B_trials - v_A_trials).astype(np.float32, copy=False)
    full_D_trials = (v_D_trials - v_A_trials).astype(np.float32, copy=False)
    if not np.allclose(inside_B_trials + outside_B_trials, full_B_trials, atol=5e-4, rtol=5e-4):
        raise ValueError("B trial vectors do not satisfy inside + outside = full within tolerance.")
    if not np.allclose(inside_D_trials + outside_D_trials, full_D_trials, atol=5e-4, rtol=5e-4):
        raise ValueError("D trial vectors do not satisfy inside + outside = full within tolerance.")

    G_A_full = reweight[G_key].astype(np.float32, copy=False)
    U_B = reweight[f"{qid}__U_B"].astype(np.float32, copy=False)
    U_D = reweight[f"{qid}__U_D"].astype(np.float32, copy=False)
    anchor_a = reweight[f"{qid}__anchor_a_query"].astype(np.float32, copy=False)
    anchor_b = reweight[f"{qid}__anchor_b_query"].astype(np.float32, copy=False)
    anchor_d = reweight[f"{qid}__anchor_d_query"].astype(np.float32, copy=False)
    u_B_in = inside_outside[f"{qid}__u_B_in"].astype(np.float32, copy=False)
    u_D_in = inside_outside[f"{qid}__u_D_in"].astype(np.float32, copy=False)
    u_B_out = inside_outside[f"{qid}__u_B_out"].astype(np.float32, copy=False)
    u_D_out = inside_outside[f"{qid}__u_D_out"].astype(np.float32, copy=False)

    payload: Dict[str, np.ndarray] = {
        "trial_ids": np.asarray(trial_ids_A[:n_use]),
        "v_A_trials": v_A_trials,
        "v_B_trials": v_B_trials,
        "v_D_trials": v_D_trials,
        "inside_B_trials": inside_B_trials,
        "outside_B_trials": outside_B_trials,
        "full_B_trials": full_B_trials,
        "inside_D_trials": inside_D_trials,
        "outside_D_trials": outside_D_trials,
        "full_D_trials": full_D_trials,
        "inside_B_mean": inside_B_trials.mean(axis=0).astype(np.float32, copy=False),
        "outside_B_mean": outside_B_trials.mean(axis=0).astype(np.float32, copy=False),
        "full_B_mean": full_B_trials.mean(axis=0).astype(np.float32, copy=False),
        "inside_D_mean": inside_D_trials.mean(axis=0).astype(np.float32, copy=False),
        "outside_D_mean": outside_D_trials.mean(axis=0).astype(np.float32, copy=False),
        "full_D_mean": full_D_trials.mean(axis=0).astype(np.float32, copy=False),
        "G_A_full": G_A_full,
        "U_B": U_B,
        "U_D": U_D,
        "anchor_a_query": anchor_a,
        "anchor_b_query": anchor_b,
        "anchor_d_query": anchor_d,
        "u_B_in": u_B_in,
        "u_D_in": u_D_in,
        "u_B_out": u_B_out,
        "u_D_out": u_D_out,
    }
    meta: Dict[str, object] = {
        "qid": qid,
        "ref": ref,
        "basis_scope": basis_scope,
        "slot_name": slot_name,
        "slot_index": slot_idx,
        "slot_names": slot_names,
        "n_trials": n_use,
        "n_trials_available": n_available,
        "stepwise_states_npz": str(states_npz_path),
        "stepwise_meta_json": str(states_meta_path),
        "reweight_npz": str(reweight_npz),
        "inside_outside_npz": str(inside_outside_npz),
        "vector_norms": {
            key: float(np.linalg.norm(payload[key]))
            for key in [
                "inside_B_mean",
                "outside_B_mean",
                "full_B_mean",
                "inside_D_mean",
                "outside_D_mean",
                "full_D_mean",
            ]
        },
    }
    return payload, meta


def write_intervention_vector_payload(
    *,
    out_npz: Path,
    out_json: Path,
    payload: Mapping[str, np.ndarray],
    meta: Mapping[str, object],
) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **payload)
    _write_json(out_json, dict(meta))


def _bundle_energies(delta_syn: np.ndarray, U_B: np.ndarray, U_D: np.ndarray) -> Tuple[float, float]:
    coeff_B = delta_syn @ U_B
    coeff_D = delta_syn @ U_D
    return float(np.sum(np.square(coeff_B))), float(np.sum(np.square(coeff_D)))


def _branch_dirs(branch: str, payload: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if branch == "B":
        return payload["u_B_in"], payload["u_D_in"], payload["u_B_out"], payload["u_D_out"]
    if branch == "D":
        return payload["u_D_in"], payload["u_B_in"], payload["u_D_out"], payload["u_B_out"]
    raise ValueError(f"Unsupported branch: {branch}")


def _branch_trial_vector(
    branch: str,
    condition_id: str,
    mode: str,
    trial_index: int,
    payload: Mapping[str, np.ndarray],
) -> np.ndarray:
    key = f"{condition_id}_{branch}"
    if mode == "trial_exact":
        return payload[f"{condition_id}_{branch}_trials"][trial_index]
    if mode == "mean_vector":
        return payload[f"{condition_id}_{branch}_mean"]
    raise ValueError(f"Unsupported mode: {mode}")


def _state_metrics(
    *,
    branch: str,
    scenario: str,
    condition_id: str,
    alpha: float,
    trial_index: int,
    trial_id: str,
    payload: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, object], np.ndarray]:
    v_A = payload["v_A_trials"][trial_index]
    v_branch = payload[f"v_{branch}_trials"][trial_index]

    if scenario == "sufficiency":
        base_state = v_A
        operation = "add"
    elif scenario == "necessity":
        base_state = v_branch
        operation = "remove"
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    if condition_id == "none":
        raw_vector = np.zeros_like(v_A)
        applied_intervention = np.zeros_like(v_A)
        v_syn = base_state.copy()
        operation = "identity"
    else:
        raw_vector = _branch_trial_vector(branch, condition_id, mode=row_mode, trial_index=trial_index, payload=payload)
        applied_intervention = (float(alpha) * raw_vector).astype(np.float32, copy=False)
        if scenario == "sufficiency":
            v_syn = (base_state + applied_intervention).astype(np.float32, copy=False)
        else:
            v_syn = (base_state - applied_intervention).astype(np.float32, copy=False)

    delta_syn = (v_syn - v_A).astype(np.float32, copy=False)
    delta_in = _project_inside(delta_syn, payload["G_A_full"])
    delta_out = (delta_syn - delta_in).astype(np.float32, copy=False)

    comparison_target = v_branch if scenario == "sufficiency" else v_A
    comparison_target_kind = f"{branch}_target" if scenario == "sufficiency" else "A"
    base_comp = (base_state - comparison_target).astype(np.float32, copy=False)
    cur_comp = (v_syn - comparison_target).astype(np.float32, copy=False)

    base_comp_in = _project_inside(base_comp, payload["G_A_full"])
    cur_comp_in = _project_inside(cur_comp, payload["G_A_full"])
    base_comp_out = (base_comp - base_comp_in).astype(np.float32, copy=False)
    cur_comp_out = (cur_comp - cur_comp_in).astype(np.float32, copy=False)

    intended_in_dir, cross_in_dir, intended_out_dir, cross_out_dir = _branch_dirs(branch, payload)
    intended_in = float(np.dot(delta_in, intended_in_dir))
    cross_in = float(np.dot(delta_in, cross_in_dir))
    intended_out = float(np.dot(delta_out, intended_out_dir))
    cross_out = float(np.dot(delta_out, cross_out_dir))
    joint_intended_in, joint_cross_in, joint_resid_in = _joint_decompose(delta_in, intended_in_dir, cross_in_dir)
    joint_intended_out, joint_cross_out, joint_resid_out = _joint_decompose(delta_out, intended_out_dir, cross_out_dir)

    F_B, F_D = _bundle_energies(delta_syn, payload["U_B"], payload["U_D"])
    bundle_dominance = float(F_B - F_D) if branch == "B" else float(F_D - F_B)
    intervention_norm = _norm(applied_intervention)
    vector_norm = _norm(raw_vector)

    base_comp_norm = _norm(base_comp)
    cur_comp_norm = _norm(cur_comp)
    base_comp_in_norm = _norm(base_comp_in)
    cur_comp_in_norm = _norm(cur_comp_in)
    base_comp_out_norm = _norm(base_comp_out)
    cur_comp_out_norm = _norm(cur_comp_out)

    comparison_improvement_abs = base_comp_norm - cur_comp_norm
    comparison_improvement_in_abs = base_comp_in_norm - cur_comp_in_norm
    comparison_improvement_out_abs = base_comp_out_norm - cur_comp_out_norm

    row: Dict[str, object] = {
        "qid": qid_global,
        "ref": ref_global,
        "basis_scope": basis_scope_global,
        "slot_name": slot_name_global,
        "slot_index": int(slot_index_global),
        "trial_index": int(trial_index),
        "trial_id": str(trial_id),
        "mode": row_mode,
        "branch": branch,
        "scenario": scenario,
        "condition_id": condition_id,
        "operation": operation,
        "alpha": float(alpha),
        "comparison_target_kind": comparison_target_kind,
        "vector_norm": vector_norm,
        "intervention_norm": intervention_norm,
        "delta_target_norm": _norm(v_branch - v_A),
        "delta_target_inside_norm": _norm(_project_inside(v_branch - v_A, payload["G_A_full"])),
        "delta_target_outside_norm": _norm((v_branch - v_A) - _project_inside(v_branch - v_A, payload["G_A_full"])),
        "delta_syn_norm": _norm(delta_syn),
        "delta_syn_inside_norm": _norm(delta_in),
        "delta_syn_outside_norm": _norm(delta_out),
        "err_to_branch_target": _norm(v_syn - v_branch),
        "err_to_A": _norm(v_syn - v_A),
        "base_err_to_comparison_target": base_comp_norm,
        "err_to_comparison_target": cur_comp_norm,
        "err_to_comparison_target_in": cur_comp_in_norm,
        "err_to_comparison_target_out": cur_comp_out_norm,
        "comparison_improvement_abs": comparison_improvement_abs,
        "comparison_improvement_frac": _safe_ratio(comparison_improvement_abs, base_comp_norm),
        "comparison_improvement_in_abs": comparison_improvement_in_abs,
        "comparison_improvement_in_frac": _safe_ratio(comparison_improvement_in_abs, base_comp_in_norm),
        "comparison_improvement_out_abs": comparison_improvement_out_abs,
        "comparison_improvement_out_frac": _safe_ratio(comparison_improvement_out_abs, base_comp_out_norm),
        "intended_in": intended_in,
        "cross_in": cross_in,
        "selectivity_in": float(intended_in - cross_in),
        "intended_out": intended_out,
        "cross_out": cross_out,
        "selectivity_out": float(intended_out - cross_out),
        "cos_intended_in": _cosine(delta_in, intended_in_dir),
        "cos_cross_in": _cosine(delta_in, cross_in_dir),
        "cos_intended_out": _cosine(delta_out, intended_out_dir),
        "cos_cross_out": _cosine(delta_out, cross_out_dir),
        "joint_intended_in": joint_intended_in,
        "joint_cross_in": joint_cross_in,
        "joint_resid_in": joint_resid_in,
        "joint_selectivity_in": float(joint_intended_in - joint_cross_in)
        if not math.isnan(joint_intended_in) and not math.isnan(joint_cross_in)
        else float("nan"),
        "joint_intended_out": joint_intended_out,
        "joint_cross_out": joint_cross_out,
        "joint_resid_out": joint_resid_out,
        "joint_selectivity_out": float(joint_intended_out - joint_cross_out)
        if not math.isnan(joint_intended_out) and not math.isnan(joint_cross_out)
        else float("nan"),
        "F_B": F_B,
        "F_D": F_D,
        "bundle_dominance": bundle_dominance,
        "comparison_improvement_per_intervention_norm": _safe_ratio(comparison_improvement_abs, intervention_norm),
        "selectivity_in_per_intervention_norm": _safe_ratio(float(intended_in - cross_in), intervention_norm),
        "selectivity_out_per_intervention_norm": _safe_ratio(float(intended_out - cross_out), intervention_norm),
        "joint_selectivity_in_per_intervention_norm": _safe_ratio(
            row_nan_safe(joint_intended_in, joint_cross_in),
            intervention_norm,
        ),
        "joint_selectivity_out_per_intervention_norm": _safe_ratio(
            row_nan_safe(joint_intended_out, joint_cross_out),
            intervention_norm,
        ),
        "bundle_dominance_per_intervention_norm": _safe_ratio(bundle_dominance, intervention_norm),
    }
    return row, v_syn


def row_nan_safe(a: float, b: float) -> float:
    if math.isnan(float(a)) or math.isnan(float(b)):
        return float("nan")
    return float(a - b)


def _summary_rows(raw_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    metrics = [
        "vector_norm",
        "intervention_norm",
        "delta_target_norm",
        "delta_target_inside_norm",
        "delta_target_outside_norm",
        "delta_syn_norm",
        "delta_syn_inside_norm",
        "delta_syn_outside_norm",
        "err_to_branch_target",
        "err_to_A",
        "base_err_to_comparison_target",
        "err_to_comparison_target",
        "err_to_comparison_target_in",
        "err_to_comparison_target_out",
        "comparison_improvement_abs",
        "comparison_improvement_frac",
        "comparison_improvement_in_abs",
        "comparison_improvement_in_frac",
        "comparison_improvement_out_abs",
        "comparison_improvement_out_frac",
        "intended_in",
        "cross_in",
        "selectivity_in",
        "intended_out",
        "cross_out",
        "selectivity_out",
        "cos_intended_in",
        "cos_cross_in",
        "cos_intended_out",
        "cos_cross_out",
        "joint_intended_in",
        "joint_cross_in",
        "joint_resid_in",
        "joint_selectivity_in",
        "joint_intended_out",
        "joint_cross_out",
        "joint_resid_out",
        "joint_selectivity_out",
        "F_B",
        "F_D",
        "bundle_dominance",
        "comparison_improvement_per_intervention_norm",
        "selectivity_in_per_intervention_norm",
        "selectivity_out_per_intervention_norm",
        "joint_selectivity_in_per_intervention_norm",
        "joint_selectivity_out_per_intervention_norm",
        "bundle_dominance_per_intervention_norm",
    ]
    grouped: Dict[Tuple[str, str, str, str, float], Dict[str, List[float]]] = {}
    for row in raw_rows:
        key = (
            str(row["mode"]),
            str(row["branch"]),
            str(row["scenario"]),
            str(row["condition_id"]),
            float(row["alpha"]),
        )
        bucket = grouped.setdefault(key, {metric: [] for metric in metrics})
        for metric in metrics:
            bucket[metric].append(float(row[metric]))

    out_rows: List[Dict[str, object]] = []
    for key in sorted(
        grouped,
        key=lambda x: (
            MODE_ORDER.get(x[0], 99),
            BRANCH_ORDER.get(x[1], 99),
            SCENARIO_ORDER.get(x[2], 99),
            CONDITION_ORDER.get(x[3], 99),
            float(x[4]),
        ),
    ):
        mode, branch, scenario, condition_id, alpha = key
        bucket = grouped[key]
        row: Dict[str, object] = {
            "qid": qid_global,
            "ref": ref_global,
            "basis_scope": basis_scope_global,
            "slot_name": slot_name_global,
            "slot_index": int(slot_index_global),
            "mode": mode,
            "branch": branch,
            "scenario": scenario,
            "condition_id": condition_id,
            "alpha": float(alpha),
            "n_trials": len(bucket["intervention_norm"]),
        }
        for metric in metrics:
            row[f"mean_{metric}"] = _mean_or_nan(bucket[metric])
        out_rows.append(row)
    return out_rows


def _format_md_table(rows: Sequence[Dict[str, object]], value_keys: Sequence[Tuple[str, str]]) -> List[str]:
    headers = ["condition"] + [label for label, _key in value_keys]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [str(row["condition_id"])]
        for _label, key in value_keys:
            value = row.get(key, float("nan"))
            if isinstance(value, float):
                cells.append("nan" if math.isnan(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _best_condition_line(rows: Sequence[Dict[str, object]], metric_key: str, label: str) -> str:
    usable = [row for row in rows if str(row["condition_id"]) != "none" and not math.isnan(float(row.get(metric_key, float("nan"))))]
    if not usable:
        return f"- {label}: unavailable"
    best = max(usable, key=lambda row: float(row[metric_key]))
    return f"- {label}: `{best['condition_id']}` ({float(best[metric_key]):.4f})"


def _write_report(
    *,
    path: Path,
    summary_rows: Sequence[Dict[str, object]],
    meta: Mapping[str, object],
    alpha_focus: float,
) -> None:
    lines: List[str] = []
    lines.append(f"# Strict Same-State Intervention Report ({meta['qid']})")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- qid: `{meta['qid']}`")
    lines.append(f"- ref: `{meta['ref']}`")
    lines.append(f"- basis_scope: `{meta['basis_scope']}`")
    lines.append(f"- slot: `{meta['slot_name']}`")
    lines.append(f"- n_trials: `{meta['n_trials']}`")
    lines.append(f"- alpha_focus: `{alpha_focus}`")
    lines.append("- modes: `trial_exact`, `mean_vector`")
    lines.append("")

    sections = [
        ("trial_exact", "B", "sufficiency"),
        ("trial_exact", "B", "necessity"),
        ("trial_exact", "D", "sufficiency"),
        ("trial_exact", "D", "necessity"),
        ("mean_vector", "B", "sufficiency"),
        ("mean_vector", "B", "necessity"),
        ("mean_vector", "D", "sufficiency"),
        ("mean_vector", "D", "necessity"),
    ]
    value_keys = [
        ("cmp_gain_frac", "mean_comparison_improvement_frac"),
        ("cmp_gain_in", "mean_comparison_improvement_in_frac"),
        ("cmp_gain_out", "mean_comparison_improvement_out_frac"),
        ("sel_in", "mean_selectivity_in"),
        ("sel_out", "mean_selectivity_out"),
        ("joint_in", "mean_joint_selectivity_in"),
        ("joint_out", "mean_joint_selectivity_out"),
        ("bundle_dom", "mean_bundle_dominance"),
        ("bundle_dom_per_norm", "mean_bundle_dominance_per_intervention_norm"),
    ]

    for mode, branch, scenario in sections:
        rows = [
            row
            for row in summary_rows
            if str(row["mode"]) == mode
            and str(row["branch"]) == branch
            and str(row["scenario"]) == scenario
            and (
                (str(row["condition_id"]) == "none" and abs(float(row["alpha"])) < 1e-8)
                or (str(row["condition_id"]) != "none" and abs(float(row["alpha"]) - float(alpha_focus)) < 1e-8)
            )
        ]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: CONDITION_ORDER.get(str(row["condition_id"]), 99))
        lines.append(f"## {mode} / {branch} / {scenario} / alpha={alpha_focus}")
        lines.append("")
        lines.extend(_format_md_table(rows, value_keys))
        lines.append("")
        lines.append(_best_condition_line(rows, "mean_comparison_improvement_frac", "best comparison gain"))
        lines.append(_best_condition_line(rows, "mean_selectivity_out_per_intervention_norm", "best outside selectivity per norm"))
        lines.append(_best_condition_line(rows, "mean_bundle_dominance", "best bundle dominance"))
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


qid_global = ""
ref_global = ""
basis_scope_global = ""
slot_name_global = ""
slot_index_global = 0
row_mode = ""


def main() -> int:
    global qid_global, ref_global, basis_scope_global, slot_name_global, slot_index_global, row_mode

    args = _parse_args()
    modes = _parse_csv_list(args.mode_list)
    if not modes:
        raise ValueError("mode_list must not be empty")
    unknown_modes = [mode for mode in modes if mode not in MODE_CHOICES]
    if unknown_modes:
        raise ValueError(f"Unsupported modes: {unknown_modes}")
    alpha_list = _parse_float_list(args.alpha_list)
    if not alpha_list:
        raise ValueError("alpha_list must not be empty")

    qid_global = str(args.qid)
    ref_global = str(args.ref)
    basis_scope_global = str(args.basis_scope)
    slot_name_global = str(args.slot_name)

    output_paths = _resolve_output_paths(args)
    payload, meta = build_intervention_vector_payload(
        qid=qid_global,
        ref=ref_global,
        basis_scope=basis_scope_global,
        slot_name=slot_name_global,
        n_trials_cap=int(args.n_trials),
        stepwise_root=Path(args.stepwise_root),
        stepwise_states_npz=args.stepwise_states_npz,
        stepwise_meta_json=args.stepwise_meta_json,
        reweight_npz=Path(args.reweight_npz).resolve(),
        inside_outside_npz=Path(args.inside_outside_npz).resolve(),
    )
    slot_index_global = int(meta["slot_index"])
    write_intervention_vector_payload(
        out_npz=output_paths["out_vectors_npz"],
        out_json=output_paths["out_vectors_json"],
        payload=payload,
        meta=meta,
    )

    trial_ids = [str(x) for x in payload["trial_ids"].tolist()]
    raw_rows: List[Dict[str, object]] = []
    synthetic_states: Dict[str, np.ndarray] = {}

    for mode in modes:
        row_mode = mode
        for branch in ("B", "D"):
            for scenario in ("sufficiency", "necessity"):
                for condition_id in ("none", "inside", "outside", "full"):
                    condition_alphas = [0.0] if condition_id == "none" else alpha_list
                    for alpha in condition_alphas:
                        syn_states = np.zeros_like(payload["v_A_trials"], dtype=np.float32)
                        for trial_index, trial_id in enumerate(trial_ids):
                            row, v_syn = _state_metrics(
                                branch=branch,
                                scenario=scenario,
                                condition_id=condition_id,
                                alpha=float(alpha),
                                trial_index=trial_index,
                                trial_id=trial_id,
                                payload=payload,
                            )
                            raw_rows.append(row)
                            syn_states[trial_index] = v_syn
                        key = "__".join(
                            [
                                qid_global,
                                mode,
                                branch,
                                scenario,
                                condition_id,
                                _alpha_label(float(alpha)),
                                "v_syn",
                            ]
                        )
                        synthetic_states[key] = syn_states

    raw_rows = sorted(
        raw_rows,
        key=lambda row: (
            MODE_ORDER.get(str(row["mode"]), 99),
            BRANCH_ORDER.get(str(row["branch"]), 99),
            SCENARIO_ORDER.get(str(row["scenario"]), 99),
            CONDITION_ORDER.get(str(row["condition_id"]), 99),
            float(row["alpha"]),
            int(row["trial_index"]),
        ),
    )
    _write_csv(output_paths["out_csv"], raw_rows)

    summary_rows = _summary_rows(raw_rows)
    _write_csv(output_paths["out_summary_csv"], summary_rows)

    output_paths["out_states_npz"].parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_paths["out_states_npz"], **synthetic_states)

    alpha_focus = 1.0 if any(abs(alpha - 1.0) < 1e-8 for alpha in alpha_list) else float(alpha_list[0])
    _write_report(
        path=output_paths["out_report_md"],
        summary_rows=summary_rows,
        meta=meta,
        alpha_focus=alpha_focus,
    )

    print(f"vectors_npz={output_paths['out_vectors_npz']}")
    print(f"vectors_json={output_paths['out_vectors_json']}")
    print(f"raw_csv={output_paths['out_csv']}")
    print(f"summary_csv={output_paths['out_summary_csv']}")
    print(f"states_npz={output_paths['out_states_npz']}")
    print(f"report_md={output_paths['out_report_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
