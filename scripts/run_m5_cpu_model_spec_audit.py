#!/usr/bin/env python3
"""Run M5-A model-spec tokenizer/resolver audit and write canonical artifacts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.adapters import resolve_attn, resolve_blocks, resolve_out_proj
from fv.model_spec import get_model_spec


CANONICAL_SPEC_KEYS = ["gpt2", "gpt2_xl", "gptj", "neox", "gemma", "llama", "olmo"]


def _parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _first_token_is_bos(tokenizer, prompt: str, add_special_tokens: bool) -> bool:
    encoded = tokenizer(prompt, add_special_tokens=add_special_tokens)
    input_ids = encoded.get("input_ids", [])
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if not input_ids:
        return False
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        return False
    return int(input_ids[0]) == int(bos_id)


def _decide_bos_policy(
    bos_token_id: Optional[int], eos_token_id: Optional[int], first_true: bool
) -> str:
    if bos_token_id is None:
        return "none"
    if eos_token_id is not None and int(bos_token_id) == int(eos_token_id):
        return "none"
    if first_true:
        return "tokenizer"
    return "manual"


def _decide_pad_policy(pad_token_id: Optional[int], eos_token_id: Optional[int]) -> str:
    if pad_token_id is not None:
        return "tokenizer_provided_pad"
    if eos_token_id is not None:
        return "use_eos_as_pad"
    return "none"


def _decision_prepend_bos(policy: str) -> bool:
    return policy in {"tokenizer", "manual"}


def _requires_manual_bos_injection(policy: str) -> bool:
    return policy == "manual"


def _layer_indices(n_layers: int) -> List[int]:
    if n_layers <= 0:
        return []
    if n_layers == 1:
        return [0]
    if n_layers == 2:
        return [0, 1]
    mid = n_layers // 2
    return sorted(set([0, mid, n_layers - 1]))


def _to_shape_list(tensor) -> Optional[List[int]]:
    if tensor is None or not hasattr(tensor, "shape"):
        return None
    return [int(x) for x in tensor.shape]


def _resolve_for_layers(model, spec) -> Tuple[Dict[str, object], bool]:
    try:
        blocks = resolve_blocks(model, spec, logger=None)
        blocks_path_ok = True
    except Exception as exc:
        return (
            {
                "checked_layer_indices": [],
                "derived_out_proj_paths": [],
                "blocks_path_ok": False,
                "out_proj_path_ok": False,
                "out_proj_module_class": None,
                "out_proj_weight_shape": None,
                "out_proj_bias_shape": None,
                "failed_segment": "blocks_path",
                "exception_type": exc.__class__.__name__,
                "message": str(exc),
            },
            False,
        )

    checked = _layer_indices(len(blocks))
    derived = [
        f"{spec.blocks_path}.{layer}.{spec.attn_path_in_block}.{spec.out_proj_path_in_attn}"
        for layer in checked
    ]
    out_proj_ok = True
    failed_segment = None
    exception_type = None
    message = None
    out_proj_module_class = None
    out_proj_weight_shape = None
    out_proj_bias_shape = None

    for layer in checked:
        try:
            attn = resolve_attn(blocks[layer], spec, logger=None)
            out_proj = resolve_out_proj(attn, spec, logger=None)
            if out_proj_module_class is None:
                out_proj_module_class = out_proj.__class__.__name__
                out_proj_weight_shape = _to_shape_list(getattr(out_proj, "weight", None))
                out_proj_bias_shape = _to_shape_list(getattr(out_proj, "bias", None))
        except Exception as exc:
            out_proj_ok = False
            failed_segment = f"layer_{layer}_out_proj_path"
            exception_type = exc.__class__.__name__
            message = str(exc)
            break

    return (
        {
            "checked_layer_indices": checked,
            "derived_out_proj_paths": derived,
            "blocks_path_ok": blocks_path_ok,
            "out_proj_path_ok": out_proj_ok,
            "out_proj_module_class": out_proj_module_class,
            "out_proj_weight_shape": out_proj_weight_shape,
            "out_proj_bias_shape": out_proj_bias_shape,
            "failed_segment": failed_segment,
            "exception_type": exception_type,
            "message": message,
        },
        out_proj_ok,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="M5-A model spec audit runner")
    parser.add_argument(
        "--spec_keys",
        default=",".join(CANONICAL_SPEC_KEYS),
        help="Comma-separated spec keys",
    )
    parser.add_argument(
        "--l2_spec_keys",
        default="gpt2,neox",
        help="Comma-separated spec keys to attempt L2 model-loaded smoke",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/model_spec_audit",
        help="Output directory",
    )
    parser.add_argument(
        "--prompt",
        default="Hello world",
        help="Probe prompt for tokenizer BOS audit",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loaders",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help="Optional HF cache dir (use workspace-local path in sandboxed runs)",
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        print(f"failed to import torch/transformers: {exc}")
        return 1

    spec_keys = _parse_csv_list(args.spec_keys)
    l2_keys = set(_parse_csv_list(args.l2_spec_keys))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = None
    if args.hf_cache_dir:
        cache_dir = Path(args.hf_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
        os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")

    summary_rows: List[Dict[str, object]] = []
    generated_at = datetime.now(timezone.utc).isoformat()

    for spec_key in spec_keys:
        spec = get_model_spec(spec_key)
        hf_model_id = spec.hf_model_id
        tokenizer_class = None
        tokenizer_name_or_path = None
        bos_token_id = None
        eos_token_id = None
        pad_token_id = None
        first_true = False
        first_false = False
        decision_bos_policy = "none"
        decision_pad_policy = "none"
        decision_prepend_bos = False
        status = "pass"
        tokenizer_error = None

        try:
            tok_kwargs = {"trust_remote_code": args.trust_remote_code}
            if cache_dir is not None:
                tok_kwargs["cache_dir"] = str(cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, **tok_kwargs)
            tokenizer_class = tokenizer.__class__.__name__
            tokenizer_name_or_path = getattr(tokenizer, "name_or_path", None)
            bos_token_id = getattr(tokenizer, "bos_token_id", None)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            first_true = _first_token_is_bos(tokenizer, args.prompt, add_special_tokens=True)
            first_false = _first_token_is_bos(tokenizer, args.prompt, add_special_tokens=False)
            decision_bos_policy = _decide_bos_policy(
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                first_true=first_true,
            )
            decision_pad_policy = _decide_pad_policy(pad_token_id, eos_token_id)
            decision_prepend_bos = _decision_prepend_bos(decision_bos_policy)
        except Exception as exc:
            status = "fail"
            tokenizer_error = {
                "exception_type": exc.__class__.__name__,
                "message": str(exc),
            }

        bos_audit = {
            "spec_key": spec_key,
            "hf_model_id": hf_model_id,
            "tokenizer_class": tokenizer_class,
            "tokenizer_name_or_path": tokenizer_name_or_path,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "first_token_is_bos_add_special_tokens_true": bool(first_true),
            "first_token_is_bos_add_special_tokens_false": bool(first_false),
            "decision_bos_policy": decision_bos_policy,
            "decision_pad_policy": decision_pad_policy,
            "decision_prepend_bos": bool(decision_prepend_bos),
            "requires_manual_bos_injection": _requires_manual_bos_injection(
                decision_bos_policy
            ),
            "tokenizer_error": tokenizer_error,
            "generated_at": generated_at,
        }
        (out_dir / f"bos_audit_{spec_key}.json").write_text(
            json.dumps(bos_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if tokenizer_error is not None:
            resolver_smoke = {
                "spec_key": spec_key,
                "hf_model_id": hf_model_id,
                "smoke_level": "L1_tokenizer_only",
                "blocks_path": spec.blocks_path,
                "checked_layer_indices": [],
                "derived_out_proj_paths": [],
                "blocks_path_ok": None,
                "out_proj_path_ok": None,
                "out_proj_module_class": None,
                "out_proj_weight_shape": None,
                "out_proj_bias_shape": None,
                "failed_segment": "tokenizer_load",
                "exception_type": tokenizer_error["exception_type"],
                "message": tokenizer_error["message"],
                "deferred_to_m5b": True,
                "generated_at": generated_at,
            }
        elif spec_key not in l2_keys:
            resolver_smoke = {
                "spec_key": spec_key,
                "hf_model_id": hf_model_id,
                "smoke_level": "L1_tokenizer_only",
                "blocks_path": spec.blocks_path,
                "checked_layer_indices": [],
                "derived_out_proj_paths": [],
                "blocks_path_ok": None,
                "out_proj_path_ok": None,
                "out_proj_module_class": None,
                "out_proj_weight_shape": None,
                "out_proj_bias_shape": None,
                "failed_segment": "model_load_skipped",
                "exception_type": None,
                "message": "L2 skipped by --l2_spec_keys policy",
                "deferred_to_m5b": True,
                "generated_at": generated_at,
            }
        else:
            try:
                model_kwargs = {
                    "torch_dtype": torch.float32,
                    "trust_remote_code": args.trust_remote_code,
                    "low_cpu_mem_usage": True,
                }
                if cache_dir is not None:
                    model_kwargs["cache_dir"] = str(cache_dir)
                model = AutoModelForCausalLM.from_pretrained(hf_model_id, **model_kwargs)
                resolved, resolved_ok = _resolve_for_layers(model, spec)
                resolver_smoke = {
                    "spec_key": spec_key,
                    "hf_model_id": hf_model_id,
                    "smoke_level": "L2_model_loaded",
                    "blocks_path": spec.blocks_path,
                    **resolved,
                    "deferred_to_m5b": False,
                    "generated_at": generated_at,
                }
                if not resolved_ok:
                    status = "fail"
            except Exception as exc:
                resolver_smoke = {
                    "spec_key": spec_key,
                    "hf_model_id": hf_model_id,
                    "smoke_level": "L1_tokenizer_only",
                    "blocks_path": spec.blocks_path,
                    "checked_layer_indices": [],
                    "derived_out_proj_paths": [],
                    "blocks_path_ok": None,
                    "out_proj_path_ok": None,
                    "out_proj_module_class": None,
                    "out_proj_weight_shape": None,
                    "out_proj_bias_shape": None,
                    "failed_segment": "model_load",
                    "exception_type": exc.__class__.__name__,
                    "message": str(exc),
                    "deferred_to_m5b": True,
                    "generated_at": generated_at,
                }

        (out_dir / f"resolver_smoke_{spec_key}.json").write_text(
            json.dumps(resolver_smoke, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        summary_rows.append(
            {
                "spec_key": spec_key,
                "decision_prepend_bos": bool(decision_prepend_bos),
                "decision_bos_policy": decision_bos_policy,
                "decision_pad_policy": decision_pad_policy,
                "requires_manual_bos_injection": _requires_manual_bos_injection(
                    decision_bos_policy
                ),
                "smoke_level": resolver_smoke["smoke_level"],
                "deferred_to_m5b": bool(resolver_smoke["deferred_to_m5b"]),
                "status": status,
            }
        )

    summary = {"generated_at": generated_at, "rows": summary_rows}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote audit artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
