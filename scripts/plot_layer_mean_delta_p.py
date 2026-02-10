#!/usr/bin/env python3
"""Plot Step6 layer-wise mean_delta_p curves for single run or relation q-wise runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_LAYER_RE = re.compile(r"^layer_(\d+)$")


@dataclass(frozen=True)
class LayerPoint:
    layer: int
    mean_delta_p: float


@dataclass(frozen=True)
class Step6LoadResult:
    points: List[LayerPoint]
    found_layers: List[int]
    loaded_layers: List[int]
    missing_layers: List[int]
    skipped: List[Dict[str, object]]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_numeric(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def qid_sort_key(qid: str) -> Tuple[int, str]:
    text = (qid or "").strip()
    if text.upper().startswith("Q"):
        suffix = text[1:]
        if suffix.isdigit():
            return (int(suffix), text)
    return (10**9, text)


def parse_q_filter(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    qids: List[str] = []
    seen = set()
    for token in raw.split(","):
        qid = token.strip()
        if not qid:
            continue
        if qid in seen:
            continue
        seen.add(qid)
        qids.append(qid)
    return qids


def discover_layer_dirs(step6_dir: Path) -> List[Tuple[int, Path]]:
    rows: List[Tuple[int, Path]] = []
    for child in step6_dir.iterdir():
        if not child.is_dir():
            continue
        match = _LAYER_RE.match(child.name)
        if not match:
            continue
        layer = int(match.group(1))
        rows.append((layer, child))
    rows.sort(key=lambda item: item[0])
    return rows


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_step6_curve(step6_dir: Path) -> Step6LoadResult:
    if not step6_dir.exists() or not step6_dir.is_dir():
        raise FileNotFoundError(f"missing step6 directory: {step6_dir}")

    layer_dirs = discover_layer_dirs(step6_dir)
    if not layer_dirs:
        raise ValueError(f"no layer_* directories found under: {step6_dir}")

    points: List[LayerPoint] = []
    skipped: List[Dict[str, object]] = []
    found_layers = [layer for layer, _ in layer_dirs]

    for layer, layer_dir in layer_dirs:
        summary_path = layer_dir / "eval_summary.json"
        if not summary_path.exists():
            skipped.append(
                {
                    "layer": layer,
                    "reason": "missing_eval_summary",
                    "path": str(summary_path),
                }
            )
            continue

        try:
            summary = read_json(summary_path)
        except Exception as exc:
            skipped.append(
                {
                    "layer": layer,
                    "reason": "invalid_eval_summary_json",
                    "error": str(exc),
                    "path": str(summary_path),
                }
            )
            continue

        mean_delta_p = summary.get("mean_delta_p")
        if not is_numeric(mean_delta_p):
            skipped.append(
                {
                    "layer": layer,
                    "reason": "invalid_mean_delta_p",
                    "value": mean_delta_p,
                    "path": str(summary_path),
                }
            )
            continue

        points.append(LayerPoint(layer=layer, mean_delta_p=float(mean_delta_p)))

    if not points:
        raise ValueError(
            "no valid eval_summary.json files with numeric mean_delta_p in "
            f"{step6_dir}"
        )

    points.sort(key=lambda item: item.layer)
    loaded_layers = [item.layer for item in points]
    loaded_set = set(loaded_layers)

    if found_layers:
        full_range = range(min(found_layers), max(found_layers) + 1)
        missing_layers = [layer for layer in full_range if layer not in loaded_set]
    else:
        missing_layers = []

    return Step6LoadResult(
        points=points,
        found_layers=found_layers,
        loaded_layers=loaded_layers,
        missing_layers=missing_layers,
        skipped=skipped,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def best_point(points: Sequence[LayerPoint]) -> LayerPoint:
    return max(points, key=lambda item: item.mean_delta_p)


def default_single_title() -> str:
    return "Step6: Layer vs Mean Delta p"


def default_qwise_title() -> str:
    return "Step6 Q-wise: Layer vs Mean Delta p"


def plot_single(
    points: Sequence[LayerPoint],
    out_path: Path,
    title: str,
    dpi: int,
) -> None:
    ensure_dir(out_path.parent)
    x_vals = [item.layer for item in points]
    y_vals = [item.mean_delta_p for item in points]
    best = best_point(points)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    ax.plot(x_vals, y_vals, "-o", linewidth=1.8, markersize=4)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="gray", alpha=0.8)
    ax.scatter([best.layer], [best.mean_delta_p], color="crimson", s=44, zorder=5)
    ax.annotate(
        f"best L={best.layer}",
        (best.layer, best.mean_delta_p),
        textcoords="offset points",
        xytext=(6, 6),
        color="crimson",
        fontsize=8,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Delta p")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_combined(
    qid_to_points: Dict[str, Sequence[LayerPoint]],
    out_path: Path,
    title: str,
    dpi: int,
) -> None:
    ensure_dir(out_path.parent)
    qids = sorted(qid_to_points.keys(), key=qid_sort_key)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    for qid in qids:
        points = sorted(qid_to_points[qid], key=lambda item: item.layer)
        x_vals = [item.layer for item in points]
        y_vals = [item.mean_delta_p for item in points]
        ax.plot(x_vals, y_vals, "-o", linewidth=1.5, markersize=3.5, label=qid)

    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="gray", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Delta p")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)

    if len(qids) >= 8:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
        fig.tight_layout(rect=(0, 0, 0.82, 1))
    else:
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def run_single(args: argparse.Namespace) -> int:
    step6_dir = Path(args.step6_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    try:
        loaded = load_step6_curve(step6_dir)
    except Exception as exc:
        print(f"[single] failed to load step6 summary: {exc}")
        return 1

    points = loaded.points
    best = best_point(points)

    out_plot = out_dir / args.out_name
    out_csv = out_dir / args.csv_name
    meta_name = f"{Path(args.out_name).stem}.meta.json"
    out_meta = out_dir / meta_name

    title = args.title or default_single_title()
    plot_single(points=points, out_path=out_plot, title=title, dpi=args.dpi)
    write_csv(
        out_csv,
        header=["layer", "mean_delta_p"],
        rows=[(item.layer, item.mean_delta_p) for item in points],
    )

    meta_payload: Dict[str, object] = {
        "step6_dir": str(step6_dir),
        "num_layers_found": len(loaded.found_layers),
        "num_layers_loaded": len(loaded.loaded_layers),
        "found_layers": loaded.found_layers,
        "loaded_layers": loaded.loaded_layers,
        "missing_layers": loaded.missing_layers,
        "best_layer": best.layer,
        "best_mean_delta_p": best.mean_delta_p,
        "generated_at": utc_now_iso(),
    }
    if loaded.skipped:
        meta_payload["skipped_layers"] = loaded.skipped
    write_json(out_meta, meta_payload)

    print(f"saved: {out_plot}")
    print(f"saved: {out_csv}")
    print(f"saved: {out_meta}")
    return 0


def discover_qids(qwise_root: Path) -> List[str]:
    qids: List[str] = []
    for child in qwise_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("Q"):
            continue
        qids.append(child.name)
    qids.sort(key=qid_sort_key)
    return qids


def run_qwise(args: argparse.Namespace) -> int:
    qwise_root = Path(args.qwise_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not qwise_root.exists() or not qwise_root.is_dir():
        print(f"[qwise] missing root directory: {qwise_root}")
        return 1

    discovered_qids = discover_qids(qwise_root)
    q_filter = parse_q_filter(args.q_filter)

    if q_filter is None:
        selected_qids = discovered_qids
    else:
        selected_qids = q_filter

    if not selected_qids:
        print(f"[qwise] no target QIDs found in: {qwise_root}")
        return 1

    status_rows: List[Dict[str, object]] = []
    status_meta: Dict[str, Dict[str, object]] = {}
    valid_curves: Dict[str, List[LayerPoint]] = {}

    emit_per_q = args.q_plot_mode in {"per_q", "both"}
    emit_combined = args.q_plot_mode in {"combined", "both"}

    for q_id in selected_qids:
        step6_dir = qwise_root / q_id / "artifacts" / "step6"
        row: Dict[str, object] = {
            "q_id": q_id,
            "num_layers_loaded": 0,
            "best_layer": "",
            "best_mean_delta_p": "",
            "status": "skipped",
        }
        meta_entry: Dict[str, object] = {
            "q_id": q_id,
            "step6_dir": str(step6_dir),
        }

        if not step6_dir.exists():
            meta_entry["status"] = "skipped"
            meta_entry["reason"] = "missing_step6_dir"
            status_rows.append(row)
            status_meta[q_id] = meta_entry
            continue

        try:
            loaded = load_step6_curve(step6_dir)
        except ValueError as exc:
            row["status"] = "failed"
            meta_entry["status"] = "failed"
            meta_entry["reason"] = str(exc)
            status_rows.append(row)
            status_meta[q_id] = meta_entry
            continue
        except Exception as exc:
            row["status"] = "failed"
            meta_entry["status"] = "failed"
            meta_entry["reason"] = f"unexpected_error: {exc}"
            status_rows.append(row)
            status_meta[q_id] = meta_entry
            continue

        points = loaded.points
        best = best_point(points)

        row["status"] = "ok"
        row["num_layers_loaded"] = len(points)
        row["best_layer"] = best.layer
        row["best_mean_delta_p"] = best.mean_delta_p

        meta_entry.update(
            {
                "status": "ok",
                "num_layers_found": len(loaded.found_layers),
                "num_layers_loaded": len(loaded.loaded_layers),
                "found_layers": loaded.found_layers,
                "loaded_layers": loaded.loaded_layers,
                "missing_layers": loaded.missing_layers,
                "best_layer": best.layer,
                "best_mean_delta_p": best.mean_delta_p,
                "skipped_layers": loaded.skipped,
            }
        )

        status_rows.append(row)
        status_meta[q_id] = meta_entry
        valid_curves[q_id] = points

        if emit_per_q:
            base = f"{q_id}.layer_mean_delta_p"
            out_plot = out_dir / f"{base}.png"
            out_csv = out_dir / f"{base}.csv"
            out_meta = out_dir / f"{base}.meta.json"
            title = args.title or f"{q_id}: {default_single_title()}"
            plot_single(points=points, out_path=out_plot, title=title, dpi=args.dpi)
            write_csv(
                out_csv,
                header=["layer", "mean_delta_p"],
                rows=[(item.layer, item.mean_delta_p) for item in points],
            )
            write_json(
                out_meta,
                {
                    "q_id": q_id,
                    "step6_dir": str(step6_dir),
                    "num_layers_found": len(loaded.found_layers),
                    "num_layers_loaded": len(loaded.loaded_layers),
                    "found_layers": loaded.found_layers,
                    "loaded_layers": loaded.loaded_layers,
                    "missing_layers": loaded.missing_layers,
                    "best_layer": best.layer,
                    "best_mean_delta_p": best.mean_delta_p,
                    "generated_at": utc_now_iso(),
                    "skipped_layers": loaded.skipped,
                },
            )
            print(f"saved: {out_plot}")
            print(f"saved: {out_csv}")
            print(f"saved: {out_meta}")

    batch_csv_path = out_dir / args.batch_csv_name
    write_csv(
        batch_csv_path,
        header=["q_id", "num_layers_loaded", "best_layer", "best_mean_delta_p", "status"],
        rows=[
            (
                row["q_id"],
                row["num_layers_loaded"],
                row["best_layer"],
                row["best_mean_delta_p"],
                row["status"],
            )
            for row in status_rows
        ],
    )
    print(f"saved: {batch_csv_path}")

    batch_meta_path = out_dir / "qwise_layer_mean_delta_p_batch.meta.json"
    write_json(
        batch_meta_path,
        {
            "qwise_root": str(qwise_root),
            "q_filter": q_filter,
            "selected_qids": selected_qids,
            "discovered_qids": discovered_qids,
            "num_selected_qids": len(selected_qids),
            "num_ok_qids": sum(1 for row in status_rows if row["status"] == "ok"),
            "num_failed_qids": sum(1 for row in status_rows if row["status"] == "failed"),
            "num_skipped_qids": sum(1 for row in status_rows if row["status"] == "skipped"),
            "q_plot_mode": args.q_plot_mode,
            "q_status": status_meta,
            "generated_at": utc_now_iso(),
        },
    )
    print(f"saved: {batch_meta_path}")

    if not valid_curves:
        print(
            "[qwise] no valid q_id data: all selected/discovered QIDs have no valid "
            "layer summaries"
        )
        return 1

    if emit_combined:
        combined_plot = out_dir / args.combined_out_name
        combined_csv = out_dir / args.combined_csv_name
        combined_meta = out_dir / "qwise_layer_mean_delta_p_combined.meta.json"

        combined_rows: List[Tuple[str, int, float]] = []
        for q_id in sorted(valid_curves.keys(), key=qid_sort_key):
            for point in sorted(valid_curves[q_id], key=lambda item: item.layer):
                combined_rows.append((q_id, point.layer, point.mean_delta_p))

        title = args.title or default_qwise_title()
        plot_combined(
            qid_to_points=valid_curves,
            out_path=combined_plot,
            title=title,
            dpi=args.dpi,
        )
        write_csv(
            combined_csv,
            header=["q_id", "layer", "mean_delta_p"],
            rows=combined_rows,
        )
        write_json(
            combined_meta,
            {
                "qwise_root": str(qwise_root),
                "num_qids_plotted": len(valid_curves),
                "q_ids_plotted": sorted(valid_curves.keys(), key=qid_sort_key),
                "num_points": len(combined_rows),
                "generated_at": utc_now_iso(),
            },
        )
        print(f"saved: {combined_plot}")
        print(f"saved: {combined_csv}")
        print(f"saved: {combined_meta}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Step6 layer-wise mean_delta_p curves for single run or relation q-wise batch."
        )
    )
    parser.add_argument("--mode", required=True, choices=["single", "qwise"])

    parser.add_argument(
        "--step6_dir",
        default=None,
        help="Single mode Step6 root (e.g., results_fv/antonym_t25_llama31/artifacts/step6)",
    )
    parser.add_argument(
        "--qwise_root",
        default=None,
        help="Q-wise root (e.g., results_fv/relation_qwise/relationB_ex)",
    )

    parser.add_argument("--out_dir", required=True, help="Output directory")

    parser.add_argument("--out_name", default="layer_mean_delta_p.png")
    parser.add_argument("--csv_name", default="layer_mean_delta_p.csv")

    parser.add_argument("--q_filter", default=None, help="Comma-separated QIDs")
    parser.add_argument(
        "--q_plot_mode",
        default="combined",
        choices=["combined", "per_q", "both"],
    )
    parser.add_argument(
        "--combined_out_name",
        default="qwise_layer_mean_delta_p_combined.png",
    )
    parser.add_argument(
        "--combined_csv_name",
        default="qwise_layer_mean_delta_p_combined.csv",
    )
    parser.add_argument(
        "--batch_csv_name",
        default="qwise_layer_mean_delta_p_summary.csv",
    )

    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "single":
        if not args.step6_dir:
            parser.error("--step6_dir is required when --mode single")
        return run_single(args)

    if args.mode == "qwise":
        if not args.qwise_root:
            parser.error("--qwise_root is required when --mode qwise")
        return run_qwise(args)

    parser.error(f"unsupported mode: {args.mode}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
