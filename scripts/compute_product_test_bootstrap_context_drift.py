#!/usr/bin/env python3
"""Bootstrap PT for mixed-context drift regimes."""

import argparse
import csv
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


REQUIRED_COLS = {"q_id", "shot", "trial_index", "regime_id", "target_s_norm"}
REGIMES = ("ABABAB_B", "ADADAD_D", "BDBDBD_D")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute product test bootstrap statistics for mixed-context drift runs."
    )
    parser.add_argument("--in_csv", required=True, help="Input drift sweep CSV path")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument("--n_boot", type=int, default=10000, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--qid", required=False, default=None, help="Optional q_id filter")
    parser.add_argument("--shot_list", required=False, default=None, help="Optional comma-separated shot filter")
    parser.add_argument("--baseline_bootstrap_csv", required=False, default=None, help="Optional baseline PT bootstrap CSV to join")
    parser.add_argument("--eps", type=float, default=1e-12, help="Epsilon for PT denominator")
    return parser.parse_args()


def _parse_shot_list(raw: Optional[str]) -> List[int]:
    if raw is None:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def _load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Missing header row in input CSV")
        missing = REQUIRED_COLS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")
        return list(reader)


def _group_rows(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, int], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["q_id"], int(row["shot"])), []).append(row)
    return grouped


def _product_test(x: float, y: float, z: float, eps: float) -> float:
    min_val = max(min(x, y, z), eps)
    return (x * y * z) / (min_val ** 2)


def _percentile(arr: np.ndarray, pct: float) -> float:
    return float(np.percentile(arr, pct))


def _load_baseline_index(path: str) -> Dict[Tuple[str, int], Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {(row["q_id"], int(row["shot"])): row for row in reader}


def main() -> int:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    shot_filter = set(_parse_shot_list(args.shot_list))

    rows = _load_rows(args.in_csv)
    if args.qid:
        rows = [row for row in rows if row["q_id"] == args.qid]
    if shot_filter:
        rows = [row for row in rows if int(row["shot"]) in shot_filter]

    grouped = _group_rows(rows)
    if not grouped:
        raise ValueError("No rows found after filtering")

    baseline_index = _load_baseline_index(args.baseline_bootstrap_csv) if args.baseline_bootstrap_csv else {}

    out_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_fields = [
        "q_id",
        "shot",
        "n_trials_used",
        "n_boot",
        "edge_ab_mean",
        "edge_ad_mean",
        "edge_bd_mean",
        "pt_ctx_abd_mean",
        "pt_ctx_abd_median",
        "pt_ctx_abd_p_gt1",
        "pt_ctx_abd_ci2_low",
        "pt_ctx_abd_ci2_high",
        "pt_ctx_abd_ci1_low",
        "baseline_pt_abd_mean",
        "baseline_pt_abd_p_gt1",
        "delta_vs_baseline_mean",
    ]

    with open(args.out_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()

        for (q_id, shot), group in sorted(grouped.items()):
            regime_map: Dict[int, Dict[str, float]] = {}
            for row in group:
                regime_id = row["regime_id"]
                if regime_id not in REGIMES:
                    continue
                val = float(row["target_s_norm"])
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"target_s_norm out of range for q_id={q_id} shot={shot}")
                trial_idx = int(row["trial_index"])
                regime_map.setdefault(trial_idx, {})[regime_id] = val

            trials = [t for t, regimes in regime_map.items() if all(r in regimes for r in REGIMES)]
            if not trials:
                continue

            s_ab = np.array([regime_map[t]["ABABAB_B"] for t in trials], dtype=float)
            s_ad = np.array([regime_map[t]["ADADAD_D"] for t in trials], dtype=float)
            s_bd = np.array([regime_map[t]["BDBDBD_D"] for t in trials], dtype=float)
            N = len(s_ab)
            if N < 5:
                print(f"[warn] skip q_id={q_id} shot={shot}: N={N} < 5 after drop")
                continue

            pt_ctx_abd = np.zeros(args.n_boot, dtype=float)
            for b in range(args.n_boot):
                idx = rng.integers(0, N, size=N)
                s_ab_bar = float(s_ab[idx].mean())
                s_ad_bar = float(s_ad[idx].mean())
                s_bd_bar = float(s_bd[idx].mean())
                pt_ctx_abd[b] = _product_test(s_ab_bar, s_ad_bar, s_bd_bar, args.eps)

            baseline_row = baseline_index.get((q_id, shot), {})
            baseline_mean = float(baseline_row["pt_abd_mean"]) if baseline_row else None
            baseline_p = float(baseline_row["pt_abd_p_gt1"]) if baseline_row else None
            row_out = {
                "q_id": q_id,
                "shot": shot,
                "n_trials_used": N,
                "n_boot": args.n_boot,
                "edge_ab_mean": float(s_ab.mean()),
                "edge_ad_mean": float(s_ad.mean()),
                "edge_bd_mean": float(s_bd.mean()),
                "pt_ctx_abd_mean": float(pt_ctx_abd.mean()),
                "pt_ctx_abd_median": float(np.median(pt_ctx_abd)),
                "pt_ctx_abd_p_gt1": float((pt_ctx_abd > 1.0).mean()),
                "pt_ctx_abd_ci2_low": _percentile(pt_ctx_abd, 2.5),
                "pt_ctx_abd_ci2_high": _percentile(pt_ctx_abd, 97.5),
                "pt_ctx_abd_ci1_low": _percentile(pt_ctx_abd, 5.0),
                "baseline_pt_abd_mean": baseline_mean,
                "baseline_pt_abd_p_gt1": baseline_p,
                "delta_vs_baseline_mean": (
                    float(pt_ctx_abd.mean()) - baseline_mean if baseline_mean is not None else None
                ),
            }
            writer.writerow(row_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
