"""Tokenization helpers for prompt inputs."""

from typing import Optional

from .model_spec import get_model_spec


def resolve_prompt_add_special_tokens(
    model_name: str, model_spec: Optional[str] = None
) -> bool:
    if model_spec:
        try:
            return bool(get_model_spec(model_spec).prepend_bos)
        except ValueError:
            pass
    try:
        return bool(get_model_spec(model_name).prepend_bos)
    except ValueError:
        pass
    return "llama" in model_name.lower()
