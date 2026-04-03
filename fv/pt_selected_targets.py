from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REVIEW_COVERED_FAMILIES = ("BASE_ABD", "CTX_ABD")


def build_unit_id(q_id: str, query_source: str, query_input: str, gold_target: str) -> str:
    return f"{q_id}::{query_source}::{query_input}->{gold_target}"


def normalize_selected_target(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


@dataclass(frozen=True)
class SelectedTargetRecord:
    q_id: str
    unit_id: str
    query_source: str
    query_input: str
    gold_target: str
    selected_target: str
    selected_target_canonical: str
    notes: str


@dataclass(frozen=True)
class SelectedTargetArtifact:
    path: Path
    payload: Dict[str, object]
    records_by_unit: Dict[str, SelectedTargetRecord]

    @property
    def source_model(self) -> Optional[str]:
        value = self.payload.get("source_model")
        return str(value) if value else None

    @property
    def source_model_spec(self) -> Optional[str]:
        value = self.payload.get("source_model_spec")
        return str(value) if value else None


def parse_family_ids(raw: str | None, *, allowed: Sequence[str]) -> List[str]:
    if raw is None:
        return list(allowed)
    out: List[str] = []
    seen = set()
    for part in str(raw).split(","):
        family_id = part.strip()
        if not family_id or family_id in seen:
            continue
        if family_id not in allowed:
            raise ValueError(f"Unsupported family_id: {family_id}")
        seen.add(family_id)
        out.append(family_id)
    if not out:
        raise ValueError("No family_ids selected")
    return out


def preflight_review_scope(*, family_ids: Sequence[str], shots: Sequence[int]) -> None:
    unsupported_families = [family_id for family_id in family_ids if family_id not in REVIEW_COVERED_FAMILIES]
    if unsupported_families:
        raise ValueError(
            "selected-target mode supports only review-covered families "
            f"{','.join(REVIEW_COVERED_FAMILIES)} in the first pass; got {','.join(unsupported_families)}"
        )
    if any(int(shot) <= 0 for shot in shots):
        raise ValueError("selected-target mode supports only positive shots in the first pass")


def load_selected_target_artifact(path: str) -> SelectedTargetArtifact:
    artifact_path = Path(path).resolve()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if str(payload.get("artifact_kind")) != "pt_selected_targets":
        raise ValueError(f"Unsupported artifact_kind in {artifact_path}")
    if int(payload.get("format_version", 0)) < 1:
        raise ValueError(f"Unsupported format_version in {artifact_path}")

    records_by_unit: Dict[str, SelectedTargetRecord] = {}
    for row in payload.get("units", []):
        unit_id = str(row.get("unit_id", "")).strip()
        if not unit_id:
            raise ValueError(f"selected target unit missing unit_id in {artifact_path}")
        if unit_id in records_by_unit:
            raise ValueError(f"duplicate selected target unit_id in {artifact_path}: {unit_id}")
        selected_target = str(row.get("selected_target", "")).strip()
        if not selected_target:
            raise ValueError(f"selected target unit missing selected_target in {artifact_path}: {unit_id}")
        canonical = str(row.get("selected_target_canonical", "")).strip() or normalize_selected_target(selected_target)
        records_by_unit[unit_id] = SelectedTargetRecord(
            q_id=str(row.get("q_id", "")),
            unit_id=unit_id,
            query_source=str(row.get("query_source", "")),
            query_input=str(row.get("query_input", "")),
            gold_target=str(row.get("gold_target", "")),
            selected_target=selected_target,
            selected_target_canonical=canonical,
            notes=str(row.get("notes", "")),
        )
    if not records_by_unit:
        raise ValueError(f"No selected target units found in {artifact_path}")
    return SelectedTargetArtifact(path=artifact_path, payload=payload, records_by_unit=records_by_unit)


def resolve_selected_target(
    artifact: SelectedTargetArtifact,
    *,
    q_id: str,
    query_source: str,
    query_input: str,
    gold_target: str,
) -> SelectedTargetRecord:
    unit_id = build_unit_id(q_id, query_source, query_input, gold_target)
    record = artifact.records_by_unit.get(unit_id)
    if record is None:
        raise ValueError(f"Missing selected target for unit_id={unit_id} in {artifact.path}")
    return record


def infer_source_model_fields(source_run_dir: str | None) -> Tuple[Optional[str], Optional[str]]:
    if not source_run_dir:
        return None, None
    meta_path = Path(source_run_dir) / "run_meta.json"
    if not meta_path.exists():
        return None, None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    model = payload.get("model")
    model_spec = payload.get("model_spec")
    return (str(model) if model else None, str(model_spec) if model_spec else None)


def iter_scaffold_units(scaffold: Dict[str, object]) -> Iterable[Tuple[str, Dict[str, object]]]:
    for question in scaffold.get("questions", []):
        q_id = str(question.get("q_id", ""))
        for unit in question.get("units", []):
            yield q_id, unit
