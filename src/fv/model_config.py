#!/usr/bin/env python3
"""Model-specific config lookup for hooks and shapes."""

from typing import Dict, Optional


MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    "gpt2": {
        "n_heads": 12,
        "head_dim": 64,
        "resid_dim": 768,
        "hook_target": "transformer.h.{layer}.attn.c_proj",
        "hook_type": "pre",
        "reshape": "resid_to_heads",
    },
}


def _normalize_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def get_model_config(model_name: str) -> Optional[Dict[str, object]]:
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    normalized = _normalize_name(model_name)
    return MODEL_CONFIGS.get(normalized)
