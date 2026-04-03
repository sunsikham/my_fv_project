#!/usr/bin/env python3
"""Bootstrap PT for unified baseline/context-drift runs."""

import argparse
import csv
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


REQUIRED_COLS = {"q_id", "shot", "trial_index", "family_id", "regime_id", "target_s_norm"}
BASE_REGIMES = ("BASE_AB", "BASE_AD", "BASE_BD")
CTX_REGIMES = ("CTX_ABABAB_B", "CTX_ADADAD_D", "CTX_BDBDBD_D")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute unified product test bootstrap statistics.")
    parser.add_argument("--in_csv", required=True, help="Input unified sweep CSV path")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument("--n_boot", type=int, default=10000, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--qid", required=False, default=None, help="Optional q_id or comma-separated q_id list filter")
    parser.add_argument("--shot_list", required=False, default=None, help="Optional comma-separated shot filter")
    parser.add_argument("--eps", type=float, default=1e-12, help="Epsilon for PT denominator")
    return parser.parse_args()


def _parse_shot_list(raw: Optional[str]) -> List[int]:
    if raw is None:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_qid_list(raw: Optional[str]) -> List[str]:
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


def _family_bootstrap(
    group: List[Dict[str, str]],
    *,
    family_id: str,
    regimes: Tuple[str, str, str],
    rng: np.random.Generator,
    n_boot: int,
    eps: float,
) -> Optional[Dict[str, object]]:
    regime_map: Dict[int, Dict[str, float]] = {}
    for row in group:
        if row["family_id"] != family_id or row["regime_id"] not in regimes:
            continue
        trial_idx = int(row["trial_index"])
        val = float(row["target_s_norm"])
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"target_s_norm out of range for family_id={family_id}")
        regime_map.setdefault(trial_idx, {})[row["regime_id"]] = val

    trials = [trial_idx for trial_idx, regime_vals in regime_map.items() if all(regime in regime_vals for regime in regimes)]
    if not trials:
        return None

    x = np.array([regime_map[t][regimes[0]] for t in trials], dtype=float)
    y = np.array([regime_map[t][regimes[1]] for t in trials], dtype=float)
    z = np.array([regime_map[t][regimes[2]] for t in trials], dtype=float)
    n = len(x)
    pt = np.zeros(n_boot, dtype=float)
    for idx in range(n_boot):
        boot_idx = rng.integers(0, n, size=n)
        x_bar = float(x[boot_idx].mean())
        y_bar = float(y[boot_idx].mean())
        z_bar = float(z[boot_idx].mean())
        pt[idx] = _product_test(x_bar, y_bar, z_bar, eps)
    return {
        "n_trials": n,
        "edge_1_mean": float(x.mean()),
        "edge_2_mean": float(y.mean()),
        "edge_3_mean": float(z.mean()),
        "pt_mean": float(pt.mean()),
        "pt_median": float(np.median(pt)),
        "pt_p_gt1": float((pt > 1.0).mean()),
        "pt_ci2_low": _percentile(pt, 2.5),
        "pt_ci2_high": _percentile(pt, 97.5),
    }


def main() -> int:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    shot_filter = set(_parse_shot_list(args.shot_list))

    rows = _load_rows(args.in_csv)
    qid_filter = set(_parse_qid_list(args.qid))
    if qid_filter:
        rows = [row for row in rows if row["q_id"] in qid_filter]
    if shot_filter:
        rows = [row for row in rows if int(row["shot"]) in shot_filter]
    rows = [row for row in rows if int(row["shot"]) > 0]

    grouped = _group_rows(rows)
    if not grouped:
        raise ValueError("No rows found after filtering")

    out_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        "q_id",
        "shot",
        "n_boot",
        "n_trials_base",
        "n_trials_ctx",
        "base_ab_mean",
        "base_ad_mean",
        "base_bd_mean",
        "base_abd_mean",
        "base_abd_median",
        "base_abd_p_gt1",
        "base_abd_ci2_low",
        "base_abd_ci2_high",
        "ctx_ab_mean",
        "ctx_ad_mean",
        "ctx_bd_mean",
        "ctx_abd_mean",
        "ctx_abd_median",
        "ctx_abd_p_gt1",
        "ctx_abd_ci2_low",
        "ctx_abd_ci2_high",
        "delta_ctx_minus_base",
    ]

    with open(args.out_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (q_id, shot), group in sorted(grouped.items()):
            base_stats = _family_bootstrap(
                group,
                family_id="BASE_ABD",
                regimes=BASE_REGIMES,
                rng=rng,
                n_boot=args.n_boot,
                eps=args.eps,
            )
            ctx_stats = _family_bootstrap(
                group,
                family_id="CTX_ABD",
                regimes=CTX_REGIMES,
                rng=rng,
                n_boot=args.n_boot,
                eps=args.eps,
            )
            if base_stats is None and ctx_stats is None:
                continue

            row = {
                "q_id": q_id,
                "shot": shot,
                "n_boot": args.n_boot,
                "n_trials_base": (base_stats["n_trials"] if base_stats is not None else None),
                "n_trials_ctx": (ctx_stats["n_trials"] if ctx_stats is not None else None),
                "base_ab_mean": (base_stats["edge_1_mean"] if base_stats is not None else None),
                "base_ad_mean": (base_stats["edge_2_mean"] if base_stats is not None else None),
                "base_bd_mean": (base_stats["edge_3_mean"] if base_stats is not None else None),
                "base_abd_mean": (base_stats["pt_mean"] if base_stats is not None else None),
                "base_abd_median": (base_stats["pt_median"] if base_stats is not None else None),
                "base_abd_p_gt1": (base_stats["pt_p_gt1"] if base_stats is not None else None),
                "base_abd_ci2_low": (base_stats["pt_ci2_low"] if base_stats is not None else None),
                "base_abd_ci2_high": (base_stats["pt_ci2_high"] if base_stats is not None else None),
                "ctx_ab_mean": (ctx_stats["edge_1_mean"] if ctx_stats is not None else None),
                "ctx_ad_mean": (ctx_stats["edge_2_mean"] if ctx_stats is not None else None),
                "ctx_bd_mean": (ctx_stats["edge_3_mean"] if ctx_stats is not None else None),
                "ctx_abd_mean": (ctx_stats["pt_mean"] if ctx_stats is not None else None),
                "ctx_abd_median": (ctx_stats["pt_median"] if ctx_stats is not None else None),
                "ctx_abd_p_gt1": (ctx_stats["pt_p_gt1"] if ctx_stats is not None else None),
                "ctx_abd_ci2_low": (ctx_stats["pt_ci2_low"] if ctx_stats is not None else None),
                "ctx_abd_ci2_high": (ctx_stats["pt_ci2_high"] if ctx_stats is not None else None),
                "delta_ctx_minus_base": (
                    float(ctx_stats["pt_mean"]) - float(base_stats["pt_mean"])
                    if base_stats is not None and ctx_stats is not None
                    else None
                ),
            }
            writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
