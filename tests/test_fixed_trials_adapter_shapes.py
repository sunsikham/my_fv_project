import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_FIXED_TRIALS_TESTS") != "1",
    reason="Set RUN_FIXED_TRIALS_TESTS=1 to enable fixed trials adapter test.",
)
def test_fixed_trials_adapter_shapes():
    import torch
    from fv.fixed_trials_adapter import (
        get_mean_head_activations_from_fixed_trials_paper_exact,
        load_fixed_trials,
    )
    from fv.hf_loader import load_hf_model_and_tokenizer
    from fv.model_spec import get_model_spec

    fixed_path = Path("datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json")
    if not fixed_path.exists():
        pytest.skip("fixed_trials file not found")

    fixed = load_fixed_trials(fixed_path)

    model, tokenizer, _ = load_hf_model_and_tokenizer(
        model_name="gpt2",
        model_spec="gpt2",
        device="cpu",
        dtype=None,
        quant="none",
        device_map=None,
    )
    model.eval()
    spec = get_model_spec("gpt2")
    model_config = {
        "n_heads": spec.n_heads,
        "n_layers": spec.n_layers,
        "resid_dim": spec.hidden_size,
        "attn_hook_names": [f"transformer.h.{i}.attn.c_proj" for i in range(spec.n_layers)],
        "prepend_bos": spec.prepend_bos,
    }

    mean_acts, dummy_labels = get_mean_head_activations_from_fixed_trials_paper_exact(
        fixed_trials_or_path=fixed,
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        mode="clean",
        n_use=1,
    )

    assert isinstance(mean_acts, torch.Tensor)
    assert mean_acts.shape == (
        model_config["n_layers"],
        model_config["n_heads"],
        len(dummy_labels),
        model_config["resid_dim"] // model_config["n_heads"],
    )
