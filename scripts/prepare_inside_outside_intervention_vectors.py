#!/usr/bin/env python3
"""Prepare strict same-state inside/outside intervention vectors at A_query."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_q1_inside_outside_intervention import (
    DEFAULT_INSIDE_OUTSIDE_NPZ,
    DEFAULT_OUT_DIR,
    DEFAULT_REWEIGHT_NPZ,
    DEFAULT_STEPWISE_ROOT,
    build_intervention_vector_payload,
    write_intervention_vector_payload,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qid", default="Q1")
    p.add_argument("--ref", default="AAA_ref")
    p.add_argument("--basis_scope", default="matched")
    p.add_argument("--slot_name", default="A_query")
    p.add_argument("--n_trials", type=int, default=0, help="Optional cap on number of trials to use.")
    p.add_argument("--stepwise_root", default=str(DEFAULT_STEPWISE_ROOT))
    p.add_argument("--stepwise_states_npz", default=None)
    p.add_argument("--stepwise_meta_json", default=None)
    p.add_argument("--reweight_npz", default=str(DEFAULT_REWEIGHT_NPZ))
    p.add_argument("--inside_outside_npz", default=str(DEFAULT_INSIDE_OUTSIDE_NPZ))
    p.add_argument(
        "--out_npz",
        default=str(DEFAULT_OUT_DIR / "inside_outside_state_intervention_vectors_Q1.npz"),
    )
    p.add_argument(
        "--out_json",
        default=str(DEFAULT_OUT_DIR / "inside_outside_state_intervention_vectors_Q1.json"),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    payload, meta = build_intervention_vector_payload(
        qid=str(args.qid),
        ref=str(args.ref),
        basis_scope=str(args.basis_scope),
        slot_name=str(args.slot_name),
        n_trials_cap=int(args.n_trials),
        stepwise_root=Path(args.stepwise_root),
        stepwise_states_npz=args.stepwise_states_npz,
        stepwise_meta_json=args.stepwise_meta_json,
        reweight_npz=Path(args.reweight_npz).resolve(),
        inside_outside_npz=Path(args.inside_outside_npz).resolve(),
    )
    out_npz = Path(args.out_npz).resolve()
    out_json = Path(args.out_json).resolve()
    write_intervention_vector_payload(
        out_npz=out_npz,
        out_json=out_json,
        payload=payload,
        meta=meta,
    )
    print(f"saved_npz={out_npz}")
    print(f"saved_json={out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
