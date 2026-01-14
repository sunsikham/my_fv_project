"""Shared quantization config helpers for consistent 4bit loading."""

from __future__ import annotations

from typing import Optional


def _dtype_to_torch(dtype: str, torch_module):
    mapping = {
        "fp32": torch_module.float32,
        "fp16": torch_module.float16,
        "bf16": torch_module.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unknown dtype for 4bit config: {dtype}")
    return mapping[dtype]


def make_bnb4_config(dtype: str, torch_module):
    """Return BitsAndBytesConfig matching paper defaults."""
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:  # pragma: no cover - runtime import check
        raise RuntimeError(f"Failed to import BitsAndBytesConfig: {exc}") from exc

    torch_dtype = _dtype_to_torch(dtype, torch_module)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
    )
