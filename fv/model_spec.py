"""Model specifications for adapter resolution."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_model_id: str
    prepend_bos: bool
    blocks_path: str
    attn_path_in_block: str
    out_proj_path_in_attn: str
    n_layers: Optional[int]
    n_heads: Optional[int]
    head_dim: Optional[int]
    hidden_size: Optional[int]
    dtype: Optional[str]
    device_map: Optional[str]
    quantization: Optional[Dict[str, object]]


MODEL_SPECS: Dict[str, ModelSpec] = {
    "gpt2": ModelSpec(
        name="gpt2",
        hf_model_id="gpt2",
        prepend_bos=False,
        blocks_path="transformer.h",
        attn_path_in_block="attn",
        out_proj_path_in_attn="c_proj",
        n_layers=12,
        n_heads=12,
        head_dim=64,
        hidden_size=768,
        dtype="fp32",
        device_map=None,
        quantization=None,
    ),
    "gpt2_xl": ModelSpec(
        name="gpt2_xl",
        hf_model_id="gpt2-xl",
        prepend_bos=False,
        blocks_path="transformer.h",
        attn_path_in_block="attn",
        out_proj_path_in_attn="c_proj",
        n_layers=48,
        n_heads=25,
        head_dim=64,
        hidden_size=1600,
        dtype="fp32",
        device_map=None,
        quantization=None,
    ),
    "gptj": ModelSpec(
        name="gptj",
        hf_model_id="EleutherAI/gpt-j-6b",
        prepend_bos=False,
        blocks_path="transformer.h",
        attn_path_in_block="attn",
        out_proj_path_in_attn="out_proj",
        n_layers=28,
        n_heads=16,
        head_dim=256,
        hidden_size=4096,
        dtype="fp32",
        device_map=None,
        quantization=None,
    ),
    "neox": ModelSpec(
        name="neox",
        hf_model_id="EleutherAI/pythia-410m",
        prepend_bos=False,
        blocks_path="gpt_neox.layers",
        attn_path_in_block="attention",
        out_proj_path_in_attn="dense",
        n_layers=None,
        n_heads=None,
        head_dim=None,
        hidden_size=None,
        dtype="fp32",
        device_map=None,
        quantization=None,
    ),
    "gemma": ModelSpec(
        name="gemma",
        hf_model_id="google/gemma-2-2b",
        prepend_bos=True,
        blocks_path="model.layers",
        attn_path_in_block="self_attn",
        out_proj_path_in_attn="o_proj",
        n_layers=None,
        n_heads=None,
        head_dim=None,
        hidden_size=None,
        dtype="fp32",
        device_map=None,
        quantization=None,
    ),
    "llama": ModelSpec(
        name="llama",
        hf_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        prepend_bos=True,
        blocks_path="model.layers",
        attn_path_in_block="self_attn",
        out_proj_path_in_attn="o_proj",
        n_layers=None,
        n_heads=None,
        head_dim=None,
        hidden_size=None,
        dtype="fp32",
        device_map=None,
        quantization=None,
    ),
    "olmo": ModelSpec(
        name="olmo",
        hf_model_id="allenai/OLMo-1B-hf",
        prepend_bos=False,
        blocks_path="model.layers",
        attn_path_in_block="self_attn",
        out_proj_path_in_attn="o_proj",
        n_layers=None,
        n_heads=None,
        head_dim=None,
        hidden_size=None,
        dtype="fp32",
        device_map=None,
        quantization=None,
    ),
    # Backward-compatible aliases used by existing scripts/runs.
    "llama3": ModelSpec(
        name="llama3",
        hf_model_id="llama3",
        prepend_bos=True,
        blocks_path="model.layers",
        attn_path_in_block="self_attn",
        out_proj_path_in_attn="o_proj",
        n_layers=None,
        n_heads=None,
        head_dim=None,
        hidden_size=None,
        dtype=None,
        device_map=None,
        quantization=None,
    ),
    "llama3_wrapped": ModelSpec(
        name="llama3_wrapped",
        hf_model_id="llama3_wrapped",
        prepend_bos=True,
        blocks_path="model.model.layers",
        attn_path_in_block="self_attn",
        out_proj_path_in_attn="o_proj",
        n_layers=None,
        n_heads=None,
        head_dim=None,
        hidden_size=None,
        dtype=None,
        device_map=None,
        quantization=None,
    ),
}

MODEL_SPEC_ALIASES = {
    "gpt2-xl": "gpt2_xl",
    "gpt2_xl": "gpt2_xl",
    "gpt-j": "gptj",
    "gpt-j-6b": "gptj",
    "gptj": "gptj",
    "gpt-neox": "neox",
    "pythia": "neox",
    "neox": "neox",
    "gemma": "gemma",
    "llama": "llama",
    "llama3": "llama3",
    "llama3_wrapped": "llama3_wrapped",
    "olmo": "olmo",
}


def list_supported_spec_keys() -> List[str]:
    return sorted(MODEL_SPECS.keys())


def get_model_spec(name_or_id: str) -> ModelSpec:
    key = name_or_id.strip()
    if key in MODEL_SPECS:
        return MODEL_SPECS[key]
    lowered = key.lower()
    if lowered in MODEL_SPEC_ALIASES:
        return MODEL_SPECS[MODEL_SPEC_ALIASES[lowered]]
    for spec in MODEL_SPECS.values():
        if key == spec.hf_model_id:
            return spec
    if "gpt2-xl" in lowered:
        return MODEL_SPECS["gpt2_xl"]
    if "gpt-j" in lowered:
        return MODEL_SPECS["gptj"]
    if "gpt-neox" in lowered or "pythia" in lowered:
        return MODEL_SPECS["neox"]
    if "gemma" in lowered:
        return MODEL_SPECS["gemma"]
    if "llama" in lowered:
        return MODEL_SPECS["llama"]
    if "olmo" in lowered:
        return MODEL_SPECS["olmo"]
    raise ValueError(f"Unsupported model spec: '{name_or_id}'")
