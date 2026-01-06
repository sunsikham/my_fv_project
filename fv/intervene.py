"""Intervention utilities for FV injection."""


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
        if rest is None:
            return injected
        return (injected,) + rest

    return inject_hook


def make_out_proj_pre_injection_hook(state):
    def inject_pre_hook(_module, inputs):
        hidden = inputs[0] if inputs else None
        rest = inputs[1:] if inputs else None

        if state["fv"] is None:
            raise RuntimeError("FV not set in injection state")
        if hidden is None or not hasattr(hidden, "shape") or hidden.dim() != 3:
            return None
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
            return (injected,)
        return (injected,) + rest

    return inject_pre_hook
