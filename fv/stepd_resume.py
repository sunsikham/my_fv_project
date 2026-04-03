from __future__ import annotations

import hashlib
import json
from pathlib import Path


STEPD_FINGERPRINT_FILES = (
    "scripts/run_stepD_aie_head_sweep.py",
    "fv/adapters.py",
    "fv/corrupt.py",
    "fv/dataset_loader.py",
    "fv/fixed_trials.py",
    "fv/hf_loader.py",
    "fv/hooks.py",
    "fv/io.py",
    "fv/mean_activations.py",
    "fv/model_spec.py",
    "fv/patch.py",
    "fv/prompting.py",
    "fv/relation_trials.py",
    "fv/slots.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash_hex(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_stepd_code_fingerprint(project_root: Path) -> str:
    payload = {}
    for rel_path in STEPD_FINGERPRINT_FILES:
        abs_path = project_root / rel_path
        payload[rel_path] = file_sha256(abs_path)
    return stable_hash_hex(payload)
