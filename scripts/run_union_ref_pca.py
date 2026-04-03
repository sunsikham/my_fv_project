#!/usr/bin/env python3
"""Build union_ref vectors for one q_dir and run common PCA."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.head_vector_extract import (  # noqa: E402
    build_rank_map,
    extract_condition_trial_vectors,
    load_stepd_scores_csv,
    serialize_heads,
    select_topk,
    unique_heads,
)

Head = Tuple[int, int]


def parse_conditions(text: str) -> List[str]:
    raw = [part.strip().upper() for part in (text or "").split(",") if part.strip()]
    if not raw:
        raise ValueError("conditions must not be empty")
    return raw


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    tmp.replace(path)


def load_pipeline_module():
    script_path = PROJECT_ROOT / "scripts" / "run_condition_qwise_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_condition_qwise_pipeline", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_union_top_heads(
    *,
    q_dir: Path,
    conditions: Sequence[str],
    topk: int,
    score_key: str,
) -> Dict[str, object]:
    stepd_dir = q_dir / "_stepd"
    top_heads_dir = q_dir / "_top_heads"
    sets_dir = top_heads_dir / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)

    stepd_rows: Dict[str, list] = {}
    topk_rows: Dict[str, list] = {}
    rank_maps: Dict[str, Dict[Head, int]] = {}
    for cond in conditions:
        score_path = stepd_dir / f"aie_scores_{cond}.csv"
        if not score_path.exists():
            raise FileNotFoundError(f"Missing StepD score CSV: {score_path}")
        rows = load_stepd_scores_csv(str(score_path), score_key=score_key)
        stepd_rows[cond] = rows
        topk_rows[cond] = select_topk(rows, k=topk)
        rank_maps[cond] = build_rank_map(rows)
        print(f"[union] loaded {cond}: rows={len(rows)} topk={len(topk_rows[cond])}")

    top_sets = {
        cond: set((row.layer, row.head) for row in topk_rows[cond]) for cond in conditions
    }
    union_candidate = set().union(*top_sets.values())
    union_rows: List[Dict[str, object]] = []
    for layer, head in sorted(union_candidate):
        ranks: Dict[str, int] = {}
        for cond in conditions:
            rank = rank_maps[cond].get((layer, head))
            if rank is None:
                raise ValueError(
                    f"Missing rank for union candidate layer={layer} head={head} cond={cond}"
                )
            ranks[cond] = int(rank)
        agg_rank = sum(ranks.values()) / float(len(conditions))
        union_rows.append(
            {
                "layer": int(layer),
                "head": int(head),
                "agg_rank": float(agg_rank),
                "ranks": ranks,
            }
        )
    union_rows.sort(key=lambda row: (row["agg_rank"], row["layer"], row["head"]))
    union_top = union_rows[:topk]

    write_json(
        sets_dir / "top_heads_ref_union.json",
        {
            "meta": {
                "source": "rank_mean_union",
                "score_key": score_key,
                "k": int(topk),
                "conditions": list(conditions),
            },
            "heads": union_top,
        },
    )
    print(
        f"[union] saved top_heads_ref_union.json union_candidate={len(union_candidate)} "
        f"union_top={len(union_top)}"
    )

    return {
        "union_candidate": sorted(union_candidate),
        "union_top": union_top,
    }


def maybe_extract_union_vectors(
    *,
    q_dir: Path,
    conditions: Sequence[str],
    union_candidate: Sequence[Head],
    union_top: Sequence[Dict[str, object]],
    model_name: str,
    model_spec: str,
    device: str,
    dtype: Optional[str],
    quant: str,
    device_map: Optional[str],
    rebuild_vectors: bool,
) -> None:
    vectors_dir = q_dir / "_vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    vector_paths = {cond: vectors_dir / f"trial_vectors_union_ref_{cond}.npy" for cond in conditions}
    headwise_paths = {
        cond: vectors_dir / f"trial_vectors_union_ref_capture_headwise_{cond}.npy" for cond in conditions
    }
    missing = [cond for cond, path in vector_paths.items() if not path.exists()]
    if not rebuild_vectors and not missing:
        print("[union] union_ref vectors already exist; skipping extraction")
        return

    if not model_name:
        raise ValueError("model is required to extract missing union_ref vectors")

    pipeline = load_pipeline_module()
    tok_add_special = pipeline.resolve_prompt_add_special_tokens(model_name, model_spec)
    model, tokenizer, diagnostics = pipeline.load_hf_model_and_tokenizer(
        model_name=model_name,
        model_spec=model_spec,
        device=device,
        dtype=dtype,
        quant=quant,
        device_map=device_map,
    )
    print("[union] model loaded diagnostics=" + json.dumps(diagnostics, ensure_ascii=True))
    model.eval()

    dims = pipeline.infer_head_dims(model, spec_name=model_spec)
    n_heads = int(dims["n_heads"])
    head_dim = int(dims["head_dim"])
    resid_dim = int(dims["hidden_size"])

    capture_heads = unique_heads(union_candidate)
    capture_layers = sorted({layer for layer, _head in capture_heads})
    layer_modules = {}
    for layer in capture_layers:
        module, _path = pipeline.get_out_proj_pre_hook_target(
            model, layer_idx=layer, spec_name=model_spec, logger=None
        )
        layer_modules[layer] = module

    union_ref_heads: Set[Head] = set((int(row["layer"]), int(row["head"])) for row in union_top)
    for cond in conditions:
        trials_path = q_dir / "_trials" / f"condition_{cond}.json"
        if not trials_path.exists():
            raise FileNotFoundError(f"Missing trials JSON: {trials_path}")
        trials = pipeline.load_trials_json(str(trials_path))["trials"]
        print(f"[union] extracting {cond}: n_trials={len(trials)}")
        extracted = extract_condition_trial_vectors(
            model=model,
            tokenizer=tokenizer,
            tok_add_special=bool(tok_add_special),
            layer_modules=layer_modules,
            n_heads=n_heads,
            head_dim=head_dim,
            resid_dim=resid_dim,
            trials=trials,
            capture_heads=capture_heads,
            aaa_ref_heads=set(),
            union_ref_heads=union_ref_heads,
            cond_topk_heads=set(),
            logger=lambda msg, c=cond: print(f"[{c}] {msg}"),
        )
        union_arr = extracted.get("union_ref")
        if union_arr is None:
            raise RuntimeError(f"union_ref output missing for cond={cond}")
        np.save(vector_paths[cond], union_arr.astype(np.float16, copy=False))
        np.save(headwise_paths[cond], extracted["capture_headwise"])
        print(f"[union] saved {vector_paths[cond]} shape={tuple(union_arr.shape)}")

    meta_path = vectors_dir / "vector_extraction_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    headwise_sets = meta.get("headwise_sets", {})
    if not isinstance(headwise_sets, dict):
        headwise_sets = {}
    headwise_sets["union_ref_capture"] = {
        "file_pattern": "trial_vectors_union_ref_capture_headwise_<COND>.npy",
        "heads": serialize_heads(capture_heads),
        "head_count": int(len(capture_heads)),
        "dtype": "float16",
        "shape_note": "(n_trials, n_capture_heads, resid_dim)",
        "conditions": list(conditions),
        "source": "union_ref_capture",
    }
    meta["headwise_sets"] = headwise_sets
    write_json(meta_path, meta)

    del model


def run_union_pca(
    *,
    q_dir: Path,
    conditions: Sequence[str],
    n_components: int,
    balance_trials: int,
    seed: int,
) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_condition_common_pca.py"),
        "--q_dir",
        str(q_dir),
        "--ref_mode",
        "union_ref",
        "--conditions",
        ",".join(conditions),
        "--n_components",
        str(n_components),
        "--balance_trials",
        str(balance_trials),
        "--seed",
        str(seed),
    ]
    print("[pca] exec:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"run_condition_common_pca failed rc={proc.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build union_ref vectors and run union_ref PCA.")
    parser.add_argument("--q_dir", required=True, help="Per-q output directory")
    parser.add_argument("--conditions", default=None, help="CSV list (default: from config)")
    parser.add_argument("--topk", type=int, default=None, help="Top-k heads per condition")
    parser.add_argument("--score_key", default=None, help="StepD score column name")
    parser.add_argument("--model", default=None, help="Model name/path for vector extraction")
    parser.add_argument("--model_spec", default=None, help="Model spec for adapter resolution")
    parser.add_argument("--device", default=None, help="Device for model loading")
    parser.add_argument("--dtype", default=None, help="dtype (fp16|bf16|fp32)")
    parser.add_argument("--quant", default=None, help="Quant mode (auto|none|4bit|8bit)")
    parser.add_argument("--device_map", default=None, help="Optional HF device_map")
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--balance_trials", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rebuild_vectors", type=int, default=0, choices=[0, 1])
    parser.add_argument("--skip_pca", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    q_dir = Path(args.q_dir).expanduser().resolve()
    if not q_dir.exists():
        print(f"Missing q_dir: {q_dir}")
        return 1

    cfg_payload: Dict[str, Any] = {}
    cfg_path = q_dir / "_status" / "config_fingerprint.json"
    if cfg_path.exists():
        cfg = read_json(cfg_path)
        payload = cfg.get("config_payload", {})
        if isinstance(payload, dict):
            cfg_payload = payload
            print(f"[cfg] loaded config from {cfg_path}")
    else:
        print(f"[cfg] config_fingerprint not found: {cfg_path}")

    conditions = parse_conditions(args.conditions or str(cfg_payload.get("conditions", "AAA,BBB,BABA")))
    topk = args.topk if args.topk is not None else int(cfg_payload.get("topk", 20))
    score_key = args.score_key or str(cfg_payload.get("score_key", "mean_delta_p"))
    seed = args.seed if args.seed is not None else int(cfg_payload.get("seed", 0))

    union_state = build_union_top_heads(
        q_dir=q_dir,
        conditions=conditions,
        topk=topk,
        score_key=score_key,
    )

    model_name = args.model or str(cfg_payload.get("model", ""))
    model_spec = args.model_spec or str(cfg_payload.get("model_spec", "llama3"))
    device = args.device or str(cfg_payload.get("device", "cuda"))
    dtype = args.dtype if args.dtype is not None else cfg_payload.get("dtype", None)
    if dtype is not None:
        dtype = str(dtype)
    quant = args.quant or str(cfg_payload.get("quant", "4bit"))
    device_map = args.device_map if args.device_map is not None else cfg_payload.get("device_map", None)
    if device_map is not None:
        device_map = str(device_map)

    maybe_extract_union_vectors(
        q_dir=q_dir,
        conditions=conditions,
        union_candidate=union_state["union_candidate"],
        union_top=union_state["union_top"],
        model_name=model_name,
        model_spec=model_spec,
        device=device,
        dtype=dtype,
        quant=quant,
        device_map=device_map,
        rebuild_vectors=bool(args.rebuild_vectors),
    )

    if not args.skip_pca:
        run_union_pca(
            q_dir=q_dir,
            conditions=conditions,
            n_components=int(args.n_components),
            balance_trials=int(args.balance_trials),
            seed=seed,
        )
        print(f"[done] union_ref PCA saved under: {q_dir / '_pca_common' / 'union_ref'}")
    else:
        print("[done] skip_pca=1, only union_ref vectors/heads updated")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
