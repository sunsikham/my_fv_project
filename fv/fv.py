"""FV synthesis utilities."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def parse_heads(heads_str: str) -> List[Tuple[int, int]]:
    heads = []
    for part in heads_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid head entry: '{part}'")
        layer_str, head_str = part.split(":", 1)
        heads.append((int(layer_str), int(head_str)))
    return heads


def build_fv(mean_activations, selected_heads: List[int], head_dim: int, resid_dim: int):
    fv = mean_activations.new_zeros((resid_dim,))
    for head in selected_heads:
        start = head * head_dim
        end = start + head_dim
        fv[start:end] = mean_activations[head]
    return fv


def _resolve_out_proj_src_compatible(model, model_config: dict, layer_idx: int):
    """Mirror src out_proj resolution rules for FV parity."""
    name_or_path = str(model_config.get("name_or_path", ""))
    lowered = name_or_path.lower()
    if "gpt2-xl" in lowered or lowered == "gpt2":
        return model.transformer.h[layer_idx].attn.c_proj
    if "gpt-j" in lowered:
        return model.transformer.h[layer_idx].attn.out_proj
    if "llama" in lowered or "gemma" in lowered or "olmo" in lowered:
        return model.model.layers[layer_idx].self_attn.o_proj
    if "gpt-neox" in lowered or "pythia" in lowered:
        return model.gpt_neox.layers[layer_idx].attention.dense
    raise ValueError(f"Unsupported model for out_proj resolution: {name_or_path}")


def compute_function_vector(
    mean_activations,
    indirect_effect,
    model,
    model_config,
    n_top_heads: int = 10,
    token_class_idx: int = -1,
):
    """Compute FV with src-compatible semantics for parity."""
    model_resid_dim = int(model_config["resid_dim"])
    model_n_heads = int(model_config["n_heads"])
    model_head_dim = model_resid_dim // model_n_heads
    device = model.device

    li_dims = len(indirect_effect.shape)
    if li_dims == 3 and token_class_idx == -1:
        mean_indirect_effect = indirect_effect.mean(dim=0)
    else:
        if li_dims != 4:
            raise ValueError(
                "indirect_effect must have rank 3 or 4; got "
                f"rank={li_dims} shape={tuple(indirect_effect.shape)}"
            )
        mean_indirect_effect = indirect_effect[:, :, :, token_class_idx].mean(dim=0)

    h_shape = mean_indirect_effect.shape
    topk_vals, topk_inds = torch.topk(
        mean_indirect_effect.view(-1), k=n_top_heads, largest=True
    )
    unraveled = np.unravel_index(topk_inds.detach().cpu().numpy(), h_shape)
    top_lh = list(zip(*unraveled, [round(x.item(), 4) for x in topk_vals]))
    top_heads = top_lh[:n_top_heads]

    function_vector = torch.zeros((1, 1, model_resid_dim)).to(device)
    token_slot = -1
    for layer_idx, head_idx, _score in top_heads:
        out_proj = _resolve_out_proj_src_compatible(
            model, model_config, int(layer_idx)
        )
        x = torch.zeros(model_resid_dim)
        start = int(head_idx) * model_head_dim
        end = (int(head_idx) + 1) * model_head_dim
        x[start:end] = mean_activations[int(layer_idx), int(head_idx), token_slot]
        d_out = out_proj(x.reshape(1, 1, model_resid_dim).to(device).to(model.dtype))
        if isinstance(d_out, tuple):
            d_out = d_out[0]
        function_vector += d_out

    function_vector = function_vector.to(model.dtype)
    function_vector = function_vector.reshape(1, model_resid_dim)
    return function_vector, top_heads
