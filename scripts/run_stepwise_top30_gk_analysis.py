#!/usr/bin/env python3
"""Run top30 g_k stepwise analysis and supplementary full-rank coverage checks."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


PROJECT_ROOT = Path("/home/sunsik/my_fv_project")
REF_NAME = "AAA_ref"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base_root",
        default=str(PROJECT_ROOT / "results_fv" / "relation_condition_qwise" / "relA_relationA_ex__relB_relationB_ex__hd011861bcc"),
        help="Root containing q directories",
    )
    p.add_argument("--q_list", default="Q1", help="Comma-separated q ids")
    p.add_argument("--k_a", type=int, default=30, help="Top-k A basis size for the main run")
    p.add_argument(
        "--reweight_out_dir",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_reweighting_top30",
        help="Output root for top30 reweighting",
    )
    p.add_argument(
        "--endpoint_out_dir",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_endpoint_alignment_top30",
        help="Output root for top30 endpoint alignment",
    )
    p.add_argument(
        "--comove_out_dir",
        default="/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/_analysis_stepwise_gk_co_movement_top30",
        help="Output root for top30 coefficient co-movement and coverage summaries",
    )
    p.add_argument(
        "--python_bin",
        default=sys.executable,
        help="Python executable to use for subprocess stages",
    )
    return p.parse_args()


def _run(cmd: Sequence[str]) -> None:
    print("RUN", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], check=True, cwd=str(PROJECT_ROOT))


def _read_reweight_summary(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
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


def _compute_coverage_summary(
    *,
    arrays_npz: Path,
    summary_csv: Path,
    qids: Sequence[str],
    k_a: int,
    out_dir: Path,
) -> None:
    arrays = np.load(arrays_npz)
    summary_rows = _read_reweight_summary(summary_csv)
    coverage_rows: List[Dict[str, object]] = []
    coverage_json: Dict[str, object] = {"ref": REF_NAME, "k_a_main": int(k_a), "qids": list(qids)}

    for qid in qids:
        coverage_json[qid] = {}
        for basis_scope in ("matched", "all"):
            evr_key = f"{qid}__{basis_scope}__explained_variance_ratio"
            if evr_key not in arrays.files:
                continue
            evr = arrays[evr_key].astype(np.float32, copy=False)
            rank_A_full = int(len(evr))
            top_idx = min(max(int(k_a), 1), max(rank_A_full, 1)) - 1
            cumulative = float(np.sum(evr[: top_idx + 1])) if rank_A_full else float("nan")
            tail = float(max(0.0, 1.0 - cumulative)) if rank_A_full else float("nan")
            per_step: List[Dict[str, object]] = []
            for row in summary_rows:
                if row.get("q_id") != qid or row.get("basis_scope") != basis_scope:
                    continue
                r_bab_top = float(row["R_BAB_topk"])
                r_bab_full = float(row["R_BAB_full"])
                r_dad_top = float(row["R_DAD_topk"])
                r_dad_full = float(row["R_DAD_full"])
                per_step_row = {
                    "q_id": qid,
                    "basis_scope": basis_scope,
                    "slot_name": row["slot_name"],
                    "slot_index": int(row["slot_index"]),
                    "rank_A_full": rank_A_full,
                    "k_a_main": int(k_a),
                    "cumulative_evr_top30": cumulative,
                    "tail_evr_after_top30": tail,
                    "R_BAB_top30": r_bab_top,
                    "R_BAB_full": r_bab_full,
                    "R_DAD_top30": r_dad_top,
                    "R_DAD_full": r_dad_full,
                    "R_BAB_top30_over_full": float(r_bab_top / r_bab_full) if r_bab_full > 0 else float("nan"),
                    "R_DAD_top30_over_full": float(r_dad_top / r_dad_full) if r_dad_full > 0 else float("nan"),
                }
                coverage_rows.append(per_step_row)
                per_step.append(per_step_row)
            coverage_json[qid][basis_scope] = {
                "rank_A_full": rank_A_full,
                "k_a_main": int(k_a),
                "cumulative_evr_top30": cumulative,
                "tail_evr_after_top30": tail,
                "by_step": per_step,
            }

    _write_csv(out_dir / "top30_fullrank_coverage_by_step.csv", coverage_rows)
    _write_json(out_dir / "top30_fullrank_coverage_summary.json", coverage_json)
    print(f"coverage_csv={out_dir / 'top30_fullrank_coverage_by_step.csv'}")
    print(f"coverage_json={out_dir / 'top30_fullrank_coverage_summary.json'}")


def main() -> int:
    args = _parse_args()
    qids = [part.strip() for part in str(args.q_list).split(",") if part.strip()]
    reweight_out_dir = Path(args.reweight_out_dir).resolve()
    endpoint_out_dir = Path(args.endpoint_out_dir).resolve()
    comove_out_dir = Path(args.comove_out_dir).resolve()
    reweight_out_dir.mkdir(parents=True, exist_ok=True)
    endpoint_out_dir.mkdir(parents=True, exist_ok=True)
    comove_out_dir.mkdir(parents=True, exist_ok=True)

    reweight_script = PROJECT_ROOT / "scripts" / "compute_stepwise_reweighting_metrics.py"
    endpoint_script = PROJECT_ROOT / "scripts" / "compute_stepwise_endpoint_aligned_contribs.py"
    comove_script = PROJECT_ROOT / "scripts" / "compute_stepwise_gk_co_movement.py"

    _run(
        [
            args.python_bin,
            reweight_script,
            "--base_root",
            args.base_root,
            "--q_list",
            ",".join(qids),
            "--k_a",
            str(args.k_a),
            "--out_dir",
            str(reweight_out_dir),
        ]
    )

    arrays_npz = reweight_out_dir / f"stepwise_reweighting_arrays_{REF_NAME}.npz"
    meta_json = reweight_out_dir / "stepwise_reweighting_meta.json"
    summary_csv = reweight_out_dir / f"stepwise_reweighting_{REF_NAME}.csv"

    _run(
        [
            args.python_bin,
            endpoint_script,
            "--arrays_npz",
            str(arrays_npz),
            "--meta_json",
            str(meta_json),
            "--q_list",
            ",".join(qids),
            "--out_dir",
            str(endpoint_out_dir),
        ]
    )

    _run(
        [
            args.python_bin,
            comove_script,
            "--arrays_npz",
            str(arrays_npz),
            "--meta_json",
            str(meta_json),
            "--q_list",
            ",".join(qids),
            "--basis_scope",
            "matched",
            "--out_dir",
            str(comove_out_dir),
        ]
    )

    _compute_coverage_summary(
        arrays_npz=arrays_npz,
        summary_csv=summary_csv,
        qids=qids,
        k_a=int(args.k_a),
        out_dir=comove_out_dir,
    )

    print(f"reweight_out_dir={reweight_out_dir}")
    print(f"endpoint_out_dir={endpoint_out_dir}")
    print(f"comove_out_dir={comove_out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
