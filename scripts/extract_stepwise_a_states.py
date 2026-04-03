#!/usr/bin/env python3
"""Extract stepwise A states for AAA/BABA/DADA using fixed top-head pools.

This script does not rerun StepD ranking. It reuses existing head sets and
re-forwards the fixed trials to save stepwise A-state representations for:
  - common PCA trajectory
  - later A-basis / multi-feature reweighting analysis
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.adapters import infer_head_dims
from fv.head_vector_extract import prepare_weight_slices, unique_heads
from fv.hf_loader import load_hf_model_and_tokenizer
from fv.hooks import get_out_proj_pre_hook_target
from fv.mean_activations import paper_labels_and_maps
from fv.tokenization import resolve_prompt_add_special_tokens


Head = Tuple[int, int]
DEFAULT_CONDITIONS = ("AAA", "BABA", "DADA")
REF_CHOICES = ("AAA_ref", "union_ref")
PROMPT_MODE_CHOICES = ("clean", "corrupted")
MATCHED_SLOT_NAMES = ["A_demo_1", "A_demo_2", "A_demo_3", "A_demo_4", "A_query"]


@dataclass(frozen=True)
class SlotView:
    name: str
    slot_names: List[str]
    physical_demo_positions: List[int]
    token_indices: List[int]


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    tmp.replace(path)


def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        handle.write(message + "\n")
        handle.flush()

    return log, handle


def cleanup_gpu_memory(log=None) -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        if log is not None:
            log(f"[warn] torch.cuda.empty_cache failed: {exc}")
    try:
        torch.cuda.ipc_collect()
    except Exception as exc:
        if log is not None:
            log(f"[warn] torch.cuda.ipc_collect failed: {exc}")


def parse_csv_list(text: str) -> List[str]:
    return [part.strip() for part in (text or "").split(",") if part.strip()]


def _head_entry_to_tuple(entry: object) -> Head:
    if isinstance(entry, dict):
        return (int(entry["layer"]), int(entry["head"]))
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return (int(entry[0]), int(entry[1]))
    raise ValueError(f"Unsupported head entry: {entry!r}")


def load_head_set(path: Path) -> List[Head]:
    payload = read_json(path)
    heads_raw = payload.get("heads")
    if not isinstance(heads_raw, list):
        raise ValueError(f"Invalid head set payload: {path}")
    return unique_heads(_head_entry_to_tuple(entry) for entry in heads_raw)


def _paper_label_name(entry: object) -> str:
    if isinstance(entry, (tuple, list)) and len(entry) >= 2:
        return str(entry[1])
    return str(entry)


def _slot_to_tokens_from_paper_idx_map(idx_map: Mapping[int, int]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for token_idx, slot_idx in idx_map.items():
        out.setdefault(int(slot_idx), []).append(int(token_idx))
    for tokens in out.values():
        tokens.sort()
    return out


def _resolve_single_slot_token_idx(
    slot_to_tokens: Mapping[int, List[int]],
    slot_idx: int,
    *,
    label: str,
) -> int:
    tokens = list(slot_to_tokens.get(int(slot_idx), []))
    if not tokens:
        raise ValueError(f"Missing token span for slot {label} (slot_idx={slot_idx})")
    if len(tokens) != 1:
        raise ValueError(
            f"Expected single-token slot for {label}, got span={tokens} slot_idx={slot_idx}"
        )
    return int(tokens[0])


def _resolve_predictive_slot_indices(dummy_labels: Sequence[object]) -> Tuple[Dict[int, int], int]:
    demo_predictive_slots: Dict[int, int] = {}
    query_predictive_slot: Optional[int] = None
    last_predictive_slot: Optional[int] = None

    for slot_idx, entry in enumerate(dummy_labels):
        label = _paper_label_name(entry)
        if label == "predictive_token":
            last_predictive_slot = int(slot_idx)
            continue
        if label == "query_predictive_token":
            query_predictive_slot = int(slot_idx)
            continue
        match = re.fullmatch(r"demonstration_(\d+)_label_token", label)
        if match is None:
            continue
        demo_num = int(match.group(1))
        if last_predictive_slot is None:
            raise ValueError(
                f"Missing predictive_token before demonstration_{demo_num}_label_token"
            )
        demo_predictive_slots[demo_num] = int(last_predictive_slot)

    if query_predictive_slot is None:
        raise ValueError("Missing query_predictive_token in dummy labels")
    if not demo_predictive_slots:
        raise ValueError("No demonstration predictive slots resolved")
    return demo_predictive_slots, int(query_predictive_slot)


def extract_prompt_slot_views(
    *,
    trial: Mapping[str, object],
    condition: str,
    tokenizer,
    tok_add_special: bool,
    prompt_mode: str,
) -> Dict[str, SlotView]:
    prompt_data_key = "prompt_data_clean" if prompt_mode == "clean" else "prompt_data_corrupted"
    prompt_key = "clean_prompt_str" if prompt_mode == "clean" else "corrupted_prompt_str"

    prompt_data = trial.get(prompt_data_key)
    if not isinstance(prompt_data, dict):
        raise ValueError(f"Trial missing {prompt_data_key}")
    prompt_str = trial.get(prompt_key)
    if not isinstance(prompt_str, str):
        raise ValueError(f"Trial missing {prompt_key}")

    prompt_string, dummy_labels, idx_map, _idx_avg = paper_labels_and_maps(
        prompt_data,
        tokenizer,
        tok_add_special=tok_add_special,
        prefixes=prompt_data.get("prefixes"),
        separators=prompt_data.get("separators"),
    )
    if prompt_string != prompt_str:
        raise ValueError(f"{condition}: prompt string mismatch against trial payload")

    slot_to_tokens = _slot_to_tokens_from_paper_idx_map(idx_map)
    demo_pred_slots, query_pred_slot = _resolve_predictive_slot_indices(dummy_labels)
    demo_token_indices = {
        demo_num: _resolve_single_slot_token_idx(
            slot_to_tokens,
            slot_idx,
            label=f"demo_{demo_num}_predictive",
        )
        for demo_num, slot_idx in demo_pred_slots.items()
    }
    query_token_idx = _resolve_single_slot_token_idx(
        slot_to_tokens,
        query_pred_slot,
        label="query_predictive",
    )

    n_demos = len(prompt_data.get("examples", []))
    all_demo_positions = list(range(1, n_demos + 1))
    matched_demo_positions = [pos for pos in all_demo_positions if pos % 2 == 0]

    if condition in {"BABA", "DADA"} and len(matched_demo_positions) != 4:
        # This script is currently designed for the 9-demo alternating setup.
        raise ValueError(
            f"{condition}: expected 4 matched A demo positions, got {matched_demo_positions}"
        )

    views: Dict[str, SlotView] = {}
    if condition == "AAA":
        views["matched"] = SlotView(
            name="matched",
            slot_names=MATCHED_SLOT_NAMES[: len(matched_demo_positions)] + ["A_query"],
            physical_demo_positions=matched_demo_positions,
            token_indices=[demo_token_indices[pos] for pos in matched_demo_positions] + [query_token_idx],
        )
        views["all"] = SlotView(
            name="all",
            slot_names=[f"A_demo_{pos}" for pos in all_demo_positions] + ["A_query"],
            physical_demo_positions=all_demo_positions,
            token_indices=[demo_token_indices[pos] for pos in all_demo_positions] + [query_token_idx],
        )
    elif condition in {"BABA", "DADA"}:
        views["matched"] = SlotView(
            name="matched",
            slot_names=MATCHED_SLOT_NAMES[: len(matched_demo_positions)] + ["A_query"],
            physical_demo_positions=matched_demo_positions,
            token_indices=[demo_token_indices[pos] for pos in matched_demo_positions] + [query_token_idx],
        )
    else:
        raise ValueError(f"Unsupported condition for stepwise A extraction: {condition}")
    return views


def _capture_layers_from_refs(ref_heads: Mapping[str, Sequence[Head]]) -> List[Head]:
    union: List[Head] = []
    for heads in ref_heads.values():
        union.extend(list(heads))
    return unique_heads(union)


def _build_ref_status_row(*, q_id: str, ref: str, eligible: int, reason: str, n_heads: int) -> Dict[str, object]:
    return {
        "q_id": q_id,
        "ref": ref,
        "eligible": int(eligible),
        "reason": str(reason),
        "head_count": int(n_heads),
    }


def extract_stepwise_states(
    *,
    q_dir: Path,
    refs: Sequence[str],
    model_name: str,
    model_spec: str,
    device: str,
    dtype: Optional[str],
    quant: str,
    device_map: Optional[str],
    prompt_mode: str,
    conditions: Sequence[str],
    log,
) -> Dict[str, object]:
    trials_dir = q_dir / "_trials"
    heads_dir = q_dir / "_top_heads" / "sets"
    output_dir = q_dir / "_stepwise_a_states"
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_heads: Dict[str, List[Head]] = {}
    manifest_rows: List[Dict[str, object]] = []
    for ref in refs:
        if ref == "AAA_ref":
            head_path = heads_dir / "top_heads_ref_AAA.json"
        elif ref == "union_ref":
            head_path = heads_dir / "top_heads_ref_union.json"
        else:
            raise ValueError(f"Unsupported ref: {ref}")
        if not head_path.exists():
            manifest_rows.append(
                _build_ref_status_row(
                    q_id=q_dir.name,
                    ref=ref,
                    eligible=0,
                    reason=f"missing head set: {head_path.name}",
                    n_heads=0,
                )
            )
            continue
        heads = load_head_set(head_path)
        ref_heads[ref] = heads
        manifest_rows.append(
            _build_ref_status_row(
                q_id=q_dir.name,
                ref=ref,
                eligible=1,
                reason="ok",
                n_heads=len(heads),
            )
        )

    if not ref_heads:
        raise FileNotFoundError("No eligible ref head sets found")

    capture_heads = _capture_layers_from_refs(ref_heads)
    capture_layers = sorted({layer for layer, _head in capture_heads})
    tok_add_special = bool(resolve_prompt_add_special_tokens(model_name, model_spec))

    model = None
    tokenizer = None
    layer_modules: Dict[int, object] = {}
    q_id: Optional[str] = None

    ref_arrays: Dict[str, Dict[str, List[np.ndarray]]] = {
        ref: {} for ref in ref_heads
    }
    ref_token_idx_arrays: Dict[str, Dict[str, List[np.ndarray]]] = {
        ref: {} for ref in ref_heads
    }
    trial_ids_by_condition: Dict[str, List[str]] = {cond: [] for cond in conditions}
    physical_positions_meta: Dict[str, Dict[str, List[int]]] = {}
    slot_names_meta: Dict[str, Dict[str, List[str]]] = {}

    cleanup_gpu_memory(log)
    try:
        model, tokenizer, diagnostics = load_hf_model_and_tokenizer(
            model_name=model_name,
            model_spec=model_spec,
            device=device,
            dtype=dtype,
            quant=quant,
            device_map=device_map,
        )
        log(f"[stepwise] model loaded diagnostics={json.dumps(diagnostics, ensure_ascii=True)}")
        model.eval()

        dims = infer_head_dims(model, spec_name=model_spec)
        n_heads = int(dims["n_heads"])
        head_dim = int(dims["head_dim"])
        resid_dim = int(dims["hidden_size"])

        for layer in capture_layers:
            module, _path = get_out_proj_pre_hook_target(
                model, layer_idx=layer, spec_name=model_spec, logger=None
            )
            layer_modules[layer] = module

        weight_slices = prepare_weight_slices(
            layer_modules=layer_modules,
            capture_heads=capture_heads,
            head_dim=head_dim,
            logger=log,
        )
        heads_by_layer: Dict[int, List[int]] = {}
        for layer, head in capture_heads:
            heads_by_layer.setdefault(layer, []).append(head)

        capture_state: Dict[int, object] = {}

        def _make_hook(layer_idx: int):
            def _hook(_module, inputs):
                if not inputs:
                    return
                x = inputs[0]
                if x is None:
                    return
                capture_state[layer_idx] = x.detach()

            return _hook

        handles = []
        for layer in sorted(heads_by_layer.keys()):
            handles.append(layer_modules[layer].register_forward_pre_hook(_make_hook(layer)))

        try:
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                import torch

                model_device = torch.device("cpu")

            for cond in conditions:
                trials_path = trials_dir / f"condition_{cond}.json"
                if not trials_path.exists():
                    log(f"[stepwise] skip condition={cond} missing trials: {trials_path}")
                    continue
                payload = read_json(trials_path)
                trials = payload.get("trials")
                if not isinstance(trials, list):
                    raise ValueError(f"Invalid trials payload: {trials_path}")
                log(f"[stepwise] start condition={cond} n_trials={len(trials)}")

                for trial_idx, trial in enumerate(trials):
                    if q_id is None:
                        q_id = str(trial.get("q_id", q_dir.name))
                    trial_id = str(trial.get("trial_id", f"t{trial_idx:06d}"))
                    trial_ids_by_condition.setdefault(cond, []).append(trial_id)

                    views = extract_prompt_slot_views(
                        trial=trial,
                        condition=cond,
                        tokenizer=tokenizer,
                        tok_add_special=tok_add_special,
                        prompt_mode=prompt_mode,
                    )
                    prompt_key = "clean_prompt_str" if prompt_mode == "clean" else "corrupted_prompt_str"
                    prompt_str = trial.get(prompt_key)
                    if not isinstance(prompt_str, str):
                        raise ValueError(f"{cond} trial missing {prompt_key}")

                    inputs = tokenizer(
                        prompt_str,
                        return_tensors="pt",
                        add_special_tokens=tok_add_special,
                    )
                    inputs = {key: value.to(model_device) for key, value in inputs.items()}
                    capture_state.clear()
                    import torch

                    with torch.inference_mode():
                        _ = model(**inputs)

                    unique_token_indices = sorted(
                        {token_idx for view in views.values() for token_idx in view.token_indices}
                    )
                    contrib_cache: Dict[Tuple[int, Head], np.ndarray] = {}
                    for layer in sorted(heads_by_layer.keys()):
                        x = capture_state.get(layer)
                        if x is None:
                            raise ValueError(f"Missing captured activation for layer={layer}")
                        if x.dim() != 3 or x.shape[0] != 1 or x.shape[-1] != resid_dim:
                            raise ValueError(
                                f"Unexpected activation shape layer={layer} shape={tuple(x.shape)}"
                            )
                        x_heads = x[0].reshape(x.shape[1], n_heads, head_dim).to(
                            dtype=torch.float32,
                            device="cpu",
                        )
                        for token_idx in unique_token_indices:
                            if token_idx < 0 or token_idx >= x_heads.shape[0]:
                                raise ValueError(
                                    f"{cond} trial_id={trial_id} token_idx out of range: "
                                    f"{token_idx} seq_len={x_heads.shape[0]}"
                                )
                            token_vec = x_heads[token_idx]
                            for head in heads_by_layer[layer]:
                                key = (layer, head)
                                w_sub = weight_slices[key]
                                contrib = torch.mv(w_sub, token_vec[head]).numpy().astype(
                                    np.float16, copy=False
                                )
                                contrib_cache[(int(token_idx), key)] = contrib

                    for ref, heads in ref_heads.items():
                        for view_name, view in views.items():
                            arr_headwise = np.zeros(
                                (len(view.slot_names), len(heads), resid_dim), dtype=np.float16
                            )
                            arr_sum = np.zeros((len(view.slot_names), resid_dim), dtype=np.float16)
                            for slot_pos, token_idx in enumerate(view.token_indices):
                                for head_pos, head in enumerate(heads):
                                    contrib = contrib_cache[(int(token_idx), head)]
                                    arr_headwise[slot_pos, head_pos, :] = contrib
                                arr_sum[slot_pos, :] = arr_headwise[slot_pos].sum(
                                    axis=0, dtype=np.float32
                                ).astype(np.float16, copy=False)

                            base_key = f"{q_id}__{cond}__{view_name}"
                            ref_arrays[ref].setdefault(f"{base_key}__headwise", []).append(arr_headwise)
                            ref_arrays[ref].setdefault(f"{base_key}__sum", []).append(arr_sum)
                            ref_token_idx_arrays[ref].setdefault(
                                f"{base_key}__token_indices", []
                            ).append(np.asarray(view.token_indices, dtype=np.int32))

                            if cond not in physical_positions_meta:
                                physical_positions_meta[cond] = {}
                            if cond not in slot_names_meta:
                                slot_names_meta[cond] = {}
                            physical_positions_meta[cond][view_name] = [
                                int(x) for x in view.physical_demo_positions
                            ]
                            slot_names_meta[cond][view_name] = list(view.slot_names)

                    if ((trial_idx + 1) % 10 == 0) or ((trial_idx + 1) == len(trials)):
                        log(f"[stepwise] condition={cond} processed {trial_idx + 1}/{len(trials)}")
        finally:
            for handle in handles:
                handle.remove()

        if q_id is None:
            q_id = q_dir.name

        for ref, arrays in ref_arrays.items():
            save_payload: Dict[str, np.ndarray] = {}
            for key, arr_list in arrays.items():
                save_payload[key] = np.stack(arr_list, axis=0)
            for key, arr_list in ref_token_idx_arrays[ref].items():
                save_payload[key] = np.stack(arr_list, axis=0)
            out_path = output_dir / f"stepwise_a_states_{ref}.npz"
            np.savez_compressed(out_path, **save_payload)
            log(f"[stepwise] saved {out_path} n_arrays={len(save_payload)}")

        manifest_path = output_dir / "stepwise_a_states_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["q_id", "ref", "eligible", "reason", "head_count"],
            )
            writer.writeheader()
            for row in manifest_rows:
                writer.writerow(row)

        meta = {
            "created_at": utc_now(),
            "q_id": q_id,
            "prompt_mode": prompt_mode,
            "tok_add_special": bool(tok_add_special),
            "conditions": list(conditions),
            "refs": list(ref_heads.keys()),
            "available_refs": {
                row["ref"]: {
                    "eligible": int(row["eligible"]),
                    "reason": row["reason"],
                    "head_count": int(row["head_count"]),
                }
                for row in manifest_rows
            },
            "matched_slot_names": MATCHED_SLOT_NAMES,
            "slot_names_by_condition": slot_names_meta,
            "physical_demo_positions_by_condition": physical_positions_meta,
            "trial_ids_by_condition": trial_ids_by_condition,
            "capture_heads_count": len(capture_heads),
            "capture_heads": [{"layer": int(layer), "head": int(head)} for layer, head in capture_heads],
            "head_lists_by_ref": {
                ref: [{"layer": int(layer), "head": int(head)} for layer, head in heads]
                for ref, heads in ref_heads.items()
            },
            "resid_dim": resid_dim,
            "n_heads_model": n_heads,
            "head_dim": head_dim,
            "artifact_keys": {
                ref: sorted(list(ref_arrays[ref].keys()) + list(ref_token_idx_arrays[ref].keys()))
                for ref in ref_heads
            },
        }
        write_json(output_dir / "stepwise_a_states_meta.json", meta)
        log(f"[stepwise] saved {output_dir / 'stepwise_a_states_meta.json'}")
        return meta
    finally:
        layer_modules.clear()
        if tokenizer is not None:
            del tokenizer
        if model is not None:
            del model
        cleanup_gpu_memory(log)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q_dir", required=True, help="Per-q artifact directory")
    parser.add_argument("--model", required=True, help="HF model name/path")
    parser.add_argument("--model_spec", default="llama", help="Model spec key")
    parser.add_argument("--device", default="cuda", help="Device string")
    parser.add_argument("--dtype", default="bf16", help="fp32/fp16/bf16")
    parser.add_argument("--quant", default="4bit", help="none/4bit/8bit/auto")
    parser.add_argument("--device_map", default="auto", help="HF device_map or empty")
    parser.add_argument(
        "--refs",
        default="AAA_ref,union_ref",
        help="Comma-separated refs to extract",
    )
    parser.add_argument(
        "--conditions",
        default="AAA,BABA,DADA",
        help="Comma-separated conditions to extract",
    )
    parser.add_argument(
        "--prompt_mode",
        choices=PROMPT_MODE_CHOICES,
        default="corrupted",
        help="Use clean or corrupted prompt payloads",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    q_dir = Path(args.q_dir).expanduser().resolve()
    if not q_dir.exists():
        raise FileNotFoundError(f"Missing q_dir: {q_dir}")

    refs = parse_csv_list(args.refs)
    if not refs:
        raise ValueError("refs must not be empty")
    for ref in refs:
        if ref not in REF_CHOICES:
            raise ValueError(f"Unsupported ref: {ref}")

    conditions = [cond.upper() for cond in parse_csv_list(args.conditions)]
    if not conditions:
        raise ValueError("conditions must not be empty")
    for cond in conditions:
        if cond not in DEFAULT_CONDITIONS:
            raise ValueError(f"Unsupported condition: {cond}")

    output_dir = q_dir / "_stepwise_a_states"
    log, log_handle = make_logger(output_dir / "extract_stepwise_a_states.log")
    try:
        extract_stepwise_states(
            q_dir=q_dir,
            refs=refs,
            model_name=args.model,
            model_spec=args.model_spec,
            device=args.device,
            dtype=None if str(args.dtype).lower() in {"", "none"} else args.dtype,
            quant=args.quant,
            device_map=None if str(args.device_map).lower() in {"", "none"} else args.device_map,
            prompt_mode=args.prompt_mode,
            conditions=conditions,
            log=log,
        )
    finally:
        log_handle.close()


if __name__ == "__main__":
    main()
