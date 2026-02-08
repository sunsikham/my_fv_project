"""Intervention utilities for FV injection."""

from __future__ import annotations

from typing import Tuple

from baukit import TraceDict


def make_residual_injection_hook(state):
    def inject_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None

        if state["fv"] is None:
            raise RuntimeError("FV not set in injection state")
        if not hasattr(hidden, "shape") or hidden.dim() != 3:
            return output
        if hidden.shape[-1] != state["fv"].shape[0]:
            raise RuntimeError("Hidden size mismatch during injection")
        if state["batch_indices"] is None or state["last_indices"] is None:
            raise RuntimeError("Injection indices not set")

        injected = hidden.clone()
        injected[state["batch_indices"], state["last_indices"], :] += state[
            "alpha"
        ] * state["fv"].to(hidden.dtype)
        state["calls"] = state.get("calls", 0) + 1
        if rest is None:
            return injected
        return (injected,) + rest

    return inject_hook


def make_out_proj_pre_injection_hook(state):
    def inject_pre_hook(_module, inputs):
        import torch

        hidden = inputs[0] if inputs else None
        rest = inputs[1:] if inputs else None

        if state["fv"] is None:
            raise RuntimeError("FV not set in injection state")
        if hidden is None or not torch.is_tensor(hidden):
            raise RuntimeError("out_proj pre-hook expected Tensor input")
        if hidden.dim() < 2:
            raise RuntimeError(
                f"out_proj pre-hook expected ndim>=2, got {hidden.dim()}"
            )
        if hidden.shape[-1] != state["fv"].shape[0]:
            raise RuntimeError(
                "Hidden size mismatch during injection: "
                f"got {hidden.shape[-1]} expected {state['fv'].shape[0]}"
            )
        if state["batch_indices"] is None or state["last_indices"] is None:
            raise RuntimeError("Injection indices not set")

        injected = hidden.clone()
        injected[state["batch_indices"], state["last_indices"], :] += state[
            "alpha"
        ] * state["fv"].to(hidden.dtype)
        state["calls"] = state.get("calls", 0) + 1

        if rest is None:
            return (injected,)
        return (injected,) + rest

    return inject_pre_hook


def add_function_vector(edit_layer, fv_vector, device, idx: int = -1):
    """Src-compatible edit_output callback for layer-output FV addition."""

    def add_act(output, layer_name):
        current_layer = int(layer_name.split(".")[2])
        if current_layer == edit_layer:
            if isinstance(output, tuple):
                output[0][:, idx] += fv_vector.to(device)
                return output
            return output
        return output

    return add_act


def function_vector_intervention(
    sentence,
    target,
    edit_layer,
    function_vector,
    model,
    model_config,
    tokenizer,
    compute_nll: bool = False,
    generate_str: bool = False,
) -> Tuple:
    """Src-compatible clean/intervention forward for FV injection parity."""
    device = model.device
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    original_pred_idx = len(inputs.input_ids.squeeze()) - 1

    if compute_nll:
        target_completion = "".join(sentence + target)
        nll_inputs = tokenizer(target_completion, return_tensors="pt").to(device)
        nll_targets = nll_inputs.input_ids.clone()
        target_len = len(nll_targets.squeeze()) - len(inputs.input_ids.squeeze())
        nll_targets[:, :-target_len] = -100
        output = model(**nll_inputs, labels=nll_targets)
        clean_nll = output.loss.item()
        clean_output = output.logits[:, original_pred_idx, :]
        intervention_idx = -1 - target_len
    elif generate_str:
        max_new_tokens = 16
        output = model.generate(
            inputs.input_ids, top_p=0.9, temperature=0.1, max_new_tokens=max_new_tokens
        )
        clean_output = tokenizer.decode(output.squeeze()[-max_new_tokens:])
        intervention_idx = -1
    else:
        clean_output = model(**inputs).logits[:, -1, :]
        intervention_idx = -1

    intervention_fn = add_function_vector(
        edit_layer,
        function_vector.reshape(1, model_config["resid_dim"]),
        model.device,
        idx=intervention_idx,
    )
    with TraceDict(
        model, layers=model_config["layer_hook_names"], edit_output=intervention_fn
    ):
        if compute_nll:
            output = model(**nll_inputs, labels=nll_targets)
            intervention_nll = output.loss.item()
            intervention_output = output.logits[:, original_pred_idx, :]
        elif generate_str:
            output = model.generate(
                inputs.input_ids,
                top_p=0.9,
                temperature=0.1,
                max_new_tokens=max_new_tokens,
            )
            intervention_output = tokenizer.decode(output.squeeze()[-max_new_tokens:])
        else:
            intervention_output = model(**inputs).logits[:, -1, :]

    fvi_output = (clean_output, intervention_output)
    if compute_nll:
        fvi_output += (clean_nll, intervention_nll)
    return fvi_output
