"""Model adapter helpers for locating blocks and projections."""


def get_blocks(model):
    """Return (blocks, path_str) for the model's transformer block list."""
    candidates = [
        ("transformer.h", "model.transformer.h"),
        ("gpt_neox.layers", "model.gpt_neox.layers"),
        ("model.layers", "model.model.layers"),
        ("layers", "model.layers"),
        ("decoder.layers", "model.decoder.layers"),
    ]
    for attr_path, path_str in candidates:
        parts = attr_path.split(".")
        current = model
        found = True
        for part in parts:
            if not hasattr(current, part):
                found = False
                break
            current = getattr(current, part)
        if found:
            return list(current), path_str
    raise ValueError("Unsupported model: cannot locate transformer blocks")


def get_attention_module(block):
    """Return (attn_module, path_str) for a transformer block."""
    for name in ("attn", "self_attn", "attention"):
        if hasattr(block, name):
            return getattr(block, name), name
    raise ValueError("Unsupported block: cannot locate attention module")


def get_attn_out_proj(attn_module):
    """Return (out_proj_module, name_str) for an attention output projection."""
    for name in ("c_proj", "o_proj", "out_proj", "dense", "proj"):
        if hasattr(attn_module, name):
            return getattr(attn_module, name), name
    raise ValueError("Unsupported attention: cannot locate output projection module")


def infer_head_dims(model):
    """Infer hidden_size, n_heads, and head_dim from model config."""
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model config not found")

    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "n_embd", None)
    n_heads = getattr(config, "num_attention_heads", None)
    if n_heads is None:
        n_heads = getattr(config, "n_head", None)
    if hidden_size is None or n_heads is None:
        raise ValueError("Model config missing hidden_size or n_heads")
    if hidden_size % n_heads != 0:
        raise ValueError("hidden_size must be divisible by n_heads")

    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = hidden_size // n_heads

    return {"hidden_size": hidden_size, "n_heads": n_heads, "head_dim": head_dim}
