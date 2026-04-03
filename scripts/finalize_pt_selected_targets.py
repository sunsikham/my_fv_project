#!/usr/bin/env python3
"""Finalize a reviewed PT scaffold into a canonical selected-target artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fv.pt_selected_targets import (
    infer_source_model_fields,
    iter_scaffold_units,
    normalize_selected_target,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize a reviewed PT scaffold into selected targets.")
    parser.add_argument("--scaffold_json", required=True, help="Reviewed scaffold JSON path")
    parser.add_argument("--out_json", required=True, help="Output selected-target JSON path")
    parser.add_argument("--out_md", default=None, help="Optional output markdown summary path")
    parser.add_argument(
        "--canonical_root",
        default=None,
        help="Canonical artifact root. Defaults to the output JSON parent directory.",
    )
    parser.add_argument("--sync_root", default=None, help="Optional mirror root")
    parser.add_argument("--sync_mode", default="none", help="Sync mode metadata")
    parser.add_argument("--artifact_profile", default="core", help="Artifact profile metadata")
    return parser.parse_args()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _candidate_allowed_forms(unit: Dict[str, object]) -> Set[str]:
    allowed = {normalize_selected_target(str(unit.get("gold_target", "")))}
    allowed.add(normalize_selected_target(str(unit.get("selected_target", ""))))
    for candidate in unit.get("candidate_suggestions", []):
        allowed.add(normalize_selected_target(str(candidate.get("display_candidate", ""))))
        allowed.add(normalize_selected_target(str(candidate.get("canonical", ""))))
    return {item for item in allowed if item}


def _select_target_from_unit(unit: Dict[str, object]) -> str:
    selected = str(unit.get("selected_target", "")).strip()
    if selected:
        return selected
    valid_answers = [str(item).strip() for item in unit.get("valid_answers", []) if str(item).strip()]
    if len(valid_answers) == 1:
        return valid_answers[0]
    raise ValueError(f"unit_id={unit.get('unit_id', '')} missing selected_target")


def _build_units(scaffold: Dict[str, object]) -> List[Dict[str, object]]:
    units_out: List[Dict[str, object]] = []
    seen = set()
    for q_id, unit in iter_scaffold_units(scaffold):
        unit_id = str(unit.get("unit_id", "")).strip()
        if not unit_id:
            raise ValueError("Encountered scaffold unit without unit_id")
        if unit_id in seen:
            raise ValueError(f"Duplicate unit_id in scaffold: {unit_id}")
        seen.add(unit_id)
        review_status = str(unit.get("review_status", "")).strip().lower()
        if review_status != "approved":
            raise ValueError(f"unit_id={unit_id} review_status must be approved")
        selected_target = _select_target_from_unit(unit)
        selected_canonical = str(unit.get("selected_target_canonical", "")).strip() or normalize_selected_target(selected_target)
        allowed = _candidate_allowed_forms(unit)
        if selected_canonical not in allowed and normalize_selected_target(selected_target) not in allowed:
            raise ValueError(
                f"unit_id={unit_id} selected_target={selected_target!r} not traceable to reviewed candidate pool"
            )
        units_out.append(
            {
                "q_id": q_id,
                "unit_id": unit_id,
                "query_source": str(unit.get("query_source", "")),
                "query_input": str(unit.get("query_input", "")),
                "gold_target": str(unit.get("gold_target", "")),
                "selected_target": selected_target,
                "selected_target_canonical": selected_canonical,
                "review_status": "approved",
                "notes": str(unit.get("notes", "")),
                "families_seen": list(unit.get("families_seen", [])),
                "regimes_seen": list(unit.get("regimes_seen", [])),
                "shots_seen": list(unit.get("shots_seen", [])),
                "source_row_count": int(unit.get("source_row_count", 0)),
            }
        )
    if not units_out:
        raise ValueError("No approved selected-target units found")
    units_out.sort(key=lambda row: (row["q_id"], row["query_source"], row["query_input"], row["unit_id"]))
    return units_out


def _render_markdown(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# PT Selected Targets")
    lines.append("")
    lines.append(f"- `source_run_dir`: `{payload['source_run_dir']}`")
    lines.append(f"- `source_scaffold_json`: `{payload['source_scaffold_json']}`")
    lines.append(f"- `source_model`: `{payload.get('source_model') or ''}`")
    lines.append(f"- `source_model_spec`: `{payload.get('source_model_spec') or ''}`")
    lines.append("")
    current_qid: Optional[str] = None
    for unit in payload["units"]:
        if unit["q_id"] != current_qid:
            current_qid = unit["q_id"]
            lines.append(f"## {current_qid}")
            lines.append("")
        lines.append(
            f"- `{unit['query_source']}` query `{unit['query_input']}`: "
            f"gold=`{unit['gold_target']}` selected=`{unit['selected_target']}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    scaffold_path = Path(args.scaffold_json).resolve()
    scaffold = json.loads(scaffold_path.read_text(encoding="utf-8"))
    source_run_dir = str(scaffold.get("source_run_dir", ""))
    source_topk_jsonl = str(scaffold.get("source_topk_jsonl", ""))
    source_model, source_model_spec = infer_source_model_fields(source_run_dir)

    units = _build_units(scaffold)
    out_json = Path(args.out_json).resolve()
    _ensure_parent(out_json)
    canonical_root = str(Path(args.canonical_root).resolve()) if args.canonical_root else str(out_json.parent.resolve())
    payload = {
        "format_version": 1,
        "artifact_kind": "pt_selected_targets",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "canonical_root": canonical_root,
        "sync_root": str(Path(args.sync_root).resolve()) if args.sync_root else None,
        "sync_mode": str(args.sync_mode),
        "artifact_profile": str(args.artifact_profile),
        "source_run_dir": source_run_dir,
        "source_topk_jsonl": source_topk_jsonl,
        "source_scaffold_json": str(scaffold_path),
        "source_model": source_model,
        "source_model_spec": source_model_spec,
        "selection_policy_version": "human_single_target_v1",
        "families_included": list(scaffold.get("families_included", [])),
        "shots_included": list(scaffold.get("shots_included", [])),
        "qids_included": sorted({unit["q_id"] for unit in units}),
        "units": units,
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"saved_json={out_json}")
    print(f"units={len(units)}")
    if args.out_md:
        out_md = Path(args.out_md).resolve()
        _ensure_parent(out_md)
        out_md.write_text(_render_markdown(payload), encoding="utf-8")
        print(f"saved_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
