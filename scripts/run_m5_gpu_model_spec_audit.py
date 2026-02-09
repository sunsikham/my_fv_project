#!/usr/bin/env python3
"""Run M5-B GPU model-spec audit and write canonical artifacts."""

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


CANONICAL_SPEC_KEYS = ["llama", "gemma", "neox", "gptj", "olmo", "gpt2", "gpt2_xl"]


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
    return policy != "none"


def _requires_manual_bos_injection(policy: str) -> bool:
    return policy == "manual"


def _layer_indices(n_layers: int) -> List[int]:
    if n_layers <= 0:
        return []
    if n_layers == 1:
        return [0]
    if n_layers == 2:
        return [0, 1]
    mid = (n_layers - 1) // 2
    return sorted(set([0, mid, n_layers - 1]))


def _to_shape_list(tensor) -> Optional[List[int]]:
    if tensor is None or not hasattr(tensor, "shape"):
        return None
    return [int(x) for x in tensor.shape]


def _resolve_for_layers(model, spec) -> Tuple[Dict[str, object], bool]:
    try:
        blocks = resolve_blocks(model, spec, logger=None)
    except Exception as exc:
        return (
            {
                "checked_layer_indices": [],
                "derived_out_proj_paths": [],
                "blocks_path_ok": False,
                "out_proj_path_ok": False,
                "out_proj_module_class_by_layer": [],
                "out_proj_weight_shape_by_layer": [],
                "out_proj_bias_shape_by_layer": [],
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
    class_by_layer: List[Optional[str]] = []
    weight_shape_by_layer: List[Optional[List[int]]] = []
    bias_shape_by_layer: List[Optional[List[int]]] = []

    for layer in checked:
        try:
            attn = resolve_attn(blocks[layer], spec, logger=None)
            out_proj = resolve_out_proj(attn, spec, logger=None)
            class_by_layer.append(out_proj.__class__.__name__)
            weight_shape_by_layer.append(_to_shape_list(getattr(out_proj, "weight", None)))
            bias_shape_by_layer.append(_to_shape_list(getattr(out_proj, "bias", None)))
        except Exception as exc:
            return (
                {
                    "checked_layer_indices": checked,
                    "derived_out_proj_paths": derived,
                    "blocks_path_ok": True,
                    "out_proj_path_ok": False,
                    "out_proj_module_class_by_layer": class_by_layer,
                    "out_proj_weight_shape_by_layer": weight_shape_by_layer,
                    "out_proj_bias_shape_by_layer": bias_shape_by_layer,
                    "failed_segment": f"layer_{layer}_out_proj_path",
                    "exception_type": exc.__class__.__name__,
                    "message": str(exc),
                },
                False,
            )

    return (
        {
            "checked_layer_indices": checked,
            "derived_out_proj_paths": derived,
            "blocks_path_ok": True,
            "out_proj_path_ok": True,
            "out_proj_module_class_by_layer": class_by_layer,
            "out_proj_weight_shape_by_layer": weight_shape_by_layer,
            "out_proj_bias_shape_by_layer": bias_shape_by_layer,
            "failed_segment": None,
            "exception_type": None,
            "message": None,
        },
        True,
    )


def _build_gpu_meta(torch_mod, transformers_mod, dtype: str, load_mode: str) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "device": "cuda",
        "gpu_name": None,
        "torch_version": getattr(torch_mod, "__version__", None),
        "torch_cuda_version": getattr(torch_mod.version, "cuda", None),
        "transformers_version": getattr(transformers_mod, "__version__", None),
        "tokenizers_version": None,
        "dtype": dtype,
        "load_mode": load_mode,
    }
    try:
        import tokenizers  # type: ignore

        meta["tokenizers_version"] = tokenizers.__version__
    except Exception:
        meta["tokenizers_version"] = None

    try:
        if torch_mod.cuda.is_available() and torch_mod.cuda.device_count() > 0:
            meta["gpu_name"] = torch_mod.cuda.get_device_name(0)
    except Exception:
        meta["gpu_name"] = None
    return meta


def _probe_cuda_runtime_constraints() -> Dict[str, object]:
    probe: Dict[str, object] = {
        "device_rw_ok": True,
        "device_rw_errors": {},
        "no_new_privs": None,
        "seccomp": None,
    }
    device_paths = ["/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm"]
    rw_errors: Dict[str, str] = {}
    for path in device_paths:
        try:
            fd = os.open(path, os.O_RDWR)
            os.close(fd)
        except Exception as exc:
            rw_errors[path] = f"{exc.__class__.__name__}: {exc}"
    if rw_errors:
        probe["device_rw_ok"] = False
        probe["device_rw_errors"] = rw_errors

    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("NoNewPrivs:"):
                probe["no_new_privs"] = line.split(":", 1)[1].strip()
            elif line.startswith("Seccomp:"):
                probe["seccomp"] = line.split(":", 1)[1].strip()
    except Exception:
        pass
    return probe


def main() -> int:
    parser = argparse.ArgumentParser(description="M5-B GPU model spec audit runner")
    parser.add_argument(
        "--spec_keys",
        default="llama",
        help="Comma-separated spec keys",
    )
    parser.add_argument(
        "--l2_spec_keys",
        default="llama",
        help="Comma-separated spec keys to attempt L2 model-loaded smoke",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/model_spec_audit_m5b",
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
        help="Optional HF cache dir",
    )
    parser.add_argument(
        "--hf_model_id_override",
        default=None,
        help="Optional override HF model id for all selected spec keys",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Require local cache only (no network).",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Torch dtype for non-quantized loading",
    )
    parser.add_argument(
        "--load_mode",
        choices=["full", "device_map_auto", "bnb_4bit", "bnb_8bit"],
        default="device_map_auto",
        help="Model loading mode",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Target device (M5-B is intended for cuda)",
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import transformers
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

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    gpu_meta = _build_gpu_meta(torch, transformers, args.dtype, args.load_mode)

    summary_rows: List[Dict[str, object]] = []
    generated_at = datetime.now(timezone.utc).isoformat()

    for spec_key in spec_keys:
        spec = get_model_spec(spec_key)
        hf_model_id = args.hf_model_id_override or spec.hf_model_id
        common_kwargs = {
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
        }
        if cache_dir is not None:
            common_kwargs["cache_dir"] = str(cache_dir)

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
        failure_stage = None
        message = None
        tokenizer_error = None

        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, **common_kwargs)
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
            failure_stage = "tokenizer_audit"
            message = str(exc)
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

        deferred_to_m5b = True
        smoke_level = "L1_tokenizer_only"
        resolver_smoke: Dict[str, object] = {
            "spec_key": spec_key,
            "hf_model_id": hf_model_id,
            "smoke_level": smoke_level,
            "blocks_path": spec.blocks_path,
            "checked_layer_indices": [],
            "derived_out_proj_paths": [],
            "blocks_path_ok": None,
            "out_proj_path_ok": None,
            "out_proj_module_class_by_layer": [],
            "out_proj_weight_shape_by_layer": [],
            "out_proj_bias_shape_by_layer": [],
            "failed_segment": None,
            "exception_type": None,
            "message": None,
            "status": "fail",
            "deferred_to_m5b": deferred_to_m5b,
            "generated_at": generated_at,
            **gpu_meta,
        }

        if tokenizer_error is not None:
            resolver_smoke["failed_segment"] = "tokenizer_load"
            resolver_smoke["exception_type"] = tokenizer_error["exception_type"]
            resolver_smoke["message"] = tokenizer_error["message"]
        elif spec_key not in l2_keys:
            status = "fail"
            resolver_smoke["failed_segment"] = "model_load_skipped"
            resolver_smoke["message"] = "L2 skipped by --l2_spec_keys policy"
            if failure_stage is None:
                failure_stage = "resolver"
                message = "L2 skipped by policy"
        elif args.device == "cuda" and not torch.cuda.is_available():
            status = "fail"
            runtime_probe = _probe_cuda_runtime_constraints()
            runtime_msg = "CUDA is not available in this runtime."
            if not runtime_probe.get("device_rw_ok", True):
                runtime_msg += (
                    " /dev/nvidia* write access failed; likely sandbox/device-cgroup "
                    "restriction."
                )
            resolver_smoke["failed_segment"] = "model_load"
            resolver_smoke["exception_type"] = "CudaUnavailable"
            resolver_smoke["message"] = runtime_msg
            resolver_smoke["cuda_runtime_probe"] = runtime_probe
            if failure_stage is None:
                failure_stage = "model_load"
                message = runtime_msg
        else:
            try:
                model_kwargs: Dict[str, object] = {
                    **common_kwargs,
                    "low_cpu_mem_usage": True,
                }
                if args.load_mode == "full":
                    model_kwargs["torch_dtype"] = dtype_map[args.dtype]
                elif args.load_mode == "device_map_auto":
                    model_kwargs["torch_dtype"] = dtype_map[args.dtype]
                    model_kwargs["device_map"] = "auto"
                elif args.load_mode == "bnb_4bit":
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=dtype_map[args.dtype],
                    )
                elif args.load_mode == "bnb_8bit":
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )

                model = AutoModelForCausalLM.from_pretrained(hf_model_id, **model_kwargs)
                if args.load_mode == "full":
                    model = model.to(args.device)

                resolved, resolved_ok = _resolve_for_layers(model, spec)
                smoke_level = "L2_model_loaded"
                deferred_to_m5b = not resolved_ok
                if resolved_ok and _requires_manual_bos_injection(decision_bos_policy):
                    status = "fail"
                    failure_stage = "manual_bos"
                    message = (
                        "manual BOS detected; runtime prepend realization must be "
                        "implemented/verified before pass."
                    )
                    deferred_to_m5b = False
                elif resolved_ok:
                    status = "pass"
                    failure_stage = None
                    message = None
                else:
                    status = "fail"
                    failure_stage = "resolver"
                    message = str(resolved.get("message"))
                resolver_smoke = {
                    "spec_key": spec_key,
                    "hf_model_id": hf_model_id,
                    "smoke_level": smoke_level,
                    "blocks_path": spec.blocks_path,
                    **resolved,
                    "status": status,
                    "deferred_to_m5b": deferred_to_m5b,
                    "generated_at": generated_at,
                    **gpu_meta,
                }
            except Exception as exc:
                status = "fail"
                failure_stage = "model_load"
                message = str(exc)
                resolver_smoke["failed_segment"] = "model_load"
                resolver_smoke["exception_type"] = exc.__class__.__name__
                resolver_smoke["message"] = str(exc)

        (out_dir / f"resolver_smoke_{spec_key}.json").write_text(
            json.dumps(resolver_smoke, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        summary_rows.append(
            {
                "spec_key": spec_key,
                "hf_model_id": hf_model_id,
                "decision_prepend_bos": bool(decision_prepend_bos),
                "decision_bos_policy": decision_bos_policy,
                "decision_pad_policy": decision_pad_policy,
                "requires_manual_bos_injection": _requires_manual_bos_injection(
                    decision_bos_policy
                ),
                "smoke_level": resolver_smoke["smoke_level"],
                "deferred_to_m5b": bool(resolver_smoke["deferred_to_m5b"]),
                "status": resolver_smoke["status"],
                "failure_stage": failure_stage,
                "message": message,
                **gpu_meta,
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
