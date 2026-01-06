"""Hook utilities for capturing standardized activations."""

from typing import Tuple

from .adapters import get_attention_module, get_attn_out_proj, get_blocks


def get_c_proj_pre_hook(model, layer: int) -> Tuple[object, str]:
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise ValueError("Model does not expose transformer.h blocks")
    if layer < 0 or layer >= len(model.transformer.h):
        raise ValueError("Layer index out of range")
    block = model.transformer.h[layer]
    if not hasattr(block, "attn") or not hasattr(block.attn, "c_proj"):
        raise ValueError("Could not find transformer.h.{layer}.attn.c_proj".format(layer=layer))
    return block.attn.c_proj, f"transformer.h.{layer}.attn.c_proj"


def get_out_proj_pre_hook_target(model, layer: int) -> Tuple[object, str]:
    """Find attention output projection module for pre-hook attachment."""
    blocks, blocks_path = get_blocks(model)
    if layer < 0 or layer >= len(blocks):
        raise ValueError("Layer index out of range")
    block = blocks[layer]
    attn_module, attn_name = get_attention_module(block)
    out_proj, out_name = get_attn_out_proj(attn_module)
    return out_proj, f"{blocks_path}.{layer}.{attn_name}.{out_name}"


def reshape_resid_to_heads(tensor, n_heads: int, head_dim: int, resid_dim: int):
    if tensor is None or not hasattr(tensor, "shape"):
        raise ValueError("Hook input is not a tensor")
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D input, got {tuple(tensor.shape)}")
    if tensor.shape[-1] != resid_dim:
        raise ValueError(f"Expected last dim {resid_dim}, got {tensor.shape[-1]}")
    reshaped = tensor.reshape(tensor.shape[0], tensor.shape[1], n_heads, head_dim)
    return reshaped
