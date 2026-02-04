"""Mean activation utilities with slot alignment."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .prompting import build_prompt_qa
from .slots import (
    build_prompt_segments,
    compute_duplicated_labels,
    get_dummy_token_labels_and_slot_map,
    get_token_meta_labels,
    infer_special_token_labels,
    segments_to_text_and_spans,
    validate_idx_map,
)
from .prompting import (
    compute_duplicated_labels as paper_compute_duplicated_labels,
    get_dummy_token_labels as paper_get_dummy_token_labels,
    get_token_meta_labels as paper_get_token_meta_labels,
    word_pairs_to_prompt_data as paper_word_pairs_to_prompt_data,
)


def apply_idx_map_to_activations(
    acts,
    idx_map: Dict[int, List[int]],
    n_slots: int,
):
    import torch

    if acts.dim() != 4:
        raise ValueError("Expected activations with shape (L,T,H,D)")
    n_layers, _seq_len, n_heads, head_dim = acts.shape
    slot_acts = torch.zeros(
        (n_layers, n_heads, n_slots, head_dim),
        device=acts.device,
        dtype=acts.dtype,
    )
    for slot_idx, token_indices in idx_map.items():
        if not token_indices:
            continue
        token_act = acts[:, token_indices, :, :]
        if len(token_indices) > 1:
            slot_act = token_act.mean(dim=1)
        else:
            slot_act = token_act[:, 0, :, :]
        slot_acts[:, :, slot_idx, :] = slot_act
    return slot_acts


def apply_paper_idx_map_to_activations(acts, idx_map, idx_avg):
    # Paper idx_map maps prompt token indices -> dummy label indices.
    acts = acts.permute(0, 2, 1, 3)  # (L, H, T, D)
    token_indices = list(idx_map.keys())
    slot_acts = acts[:, :, token_indices, :]
    for (i, j) in idx_avg.values():
        slot_acts[:, :, idx_map[i], :] = acts[:, :, i : j + 1, :].mean(dim=2)
    return slot_acts


def apply_paper_idx_map_to_slots(acts, idx_map, n_slots):
    # Build slot activations directly using idx_map token->slot mapping.
    import torch

    slot_acts = torch.zeros(
        (acts.shape[0], acts.shape[1], n_slots, acts.shape[3]),
        device=acts.device,
        dtype=acts.dtype,
    )
    slot_to_tokens: Dict[int, List[int]] = {i: [] for i in range(n_slots)}
    for token_idx, slot_idx in idx_map.items():
        if 0 <= slot_idx < n_slots:
            slot_to_tokens[slot_idx].append(token_idx)
    for slot_idx, token_indices in slot_to_tokens.items():
        if not token_indices:
            continue
        slot_acts[:, :, slot_idx, :] = acts[:, :, token_indices, :].mean(dim=2)
    return slot_acts


def paper_prompt_data_from_pairs(
    demos,
    query,
    tok_add_special: bool,
    prefixes=None,
    separators=None,
    shuffle_labels: bool = False,
):
    word_pairs = {"input": [x for x, _y in demos], "output": [y for _x, y in demos]}
    query_target_pair = {"input": [query[0]], "output": [query[1]]}
    return paper_word_pairs_to_prompt_data(
        word_pairs,
        query_target_pair=query_target_pair,
        prepend_bos_token=not tok_add_special,
        prefixes=prefixes,
        separators=separators,
        shuffle_labels=shuffle_labels,
    )


def paper_labels_and_maps(
    prompt_data,
    tokenizer,
    tok_add_special: bool,
    prefixes=None,
    separators=None,
):
    token_labels, prompt_string = paper_get_token_meta_labels(
        prompt_data, tokenizer, prepend_bos=tok_add_special
    )
    dummy_labels = paper_get_dummy_token_labels(
        len(prompt_data.get("examples", [])),
        tokenizer=tokenizer,
        model_config={"prepend_bos": tok_add_special},
        prefixes=prefixes,
        separators=separators,
    )
    idx_map, idx_avg = paper_compute_duplicated_labels(token_labels, dummy_labels)
    if len(idx_map) != len(dummy_labels):
        raise AssertionError(
            f"dummy_labels length mismatch: dummy={len(dummy_labels)} real={len(idx_map)}"
        )
    return prompt_string, dummy_labels, idx_map, idx_avg


def paper_labels_to_slot_map(dummy_labels):
    labels = [label for _idx, label in dummy_labels]
    slot_index_map = {}
    if "query_predictive_token" in labels:
        q_idx = labels.index("query_predictive_token")
        slot_index_map["QUERY_PRED"] = q_idx
        slot_index_map["query_predictive_token"] = q_idx
    return labels, slot_index_map


def compute_mean_activations_ns(
    model,
    tokenizer,
    layer_modules: Dict[int, object],
    pairs,
    n_icl_examples: int,
    n_mean_trials: int,
    model_cfg: Dict[str, int],
    seed: int,
    tok_add_special: bool,
    device,
    shuffle_labels: bool = False,
    prefixes=None,
    separators=None,
    allow_alignment_skip: bool = False,
    logger=None,
):
    import torch

    if n_mean_trials < 1:
        raise ValueError("n_mean_trials must be >= 1")

    n_heads = int(model_cfg["n_heads"])
    head_dim = int(model_cfg["head_dim"])
    resid_dim = int(model_cfg["resid_dim"])
    n_layers = int(model_cfg["n_layers"])

    layers = sorted(layer_modules.keys())
    if not layers:
        raise ValueError("No layers provided for mean activation computation")

    def log(message: str) -> None:
        if logger is None:
            return
        if callable(logger):
            logger(message)
        elif hasattr(logger, "info"):
            logger.info(message)

    demo_seed_base = seed

    dummy_labels = None
    slot_index_map = None
    avg_logged = False

    mean = torch.zeros(
        (n_layers, n_heads, 0, head_dim),
        dtype=torch.float32,
        device="cpu",
    )
    count = 0

    for trial_idx in range(n_mean_trials):
        from .dataset_loader import sample_demos_and_query

        demos_orig, query = sample_demos_and_query(
            pairs, n_icl_examples, seed=demo_seed_base + trial_idx
        )
        def _strip_leading_space(text: str) -> str:
            return text[1:] if isinstance(text, str) and text.startswith(" ") else text

        demos_orig = [(_strip_leading_space(x), _strip_leading_space(y)) for x, y in demos_orig]
        query = (_strip_leading_space(query[0]), _strip_leading_space(query[1]))
        if shuffle_labels:
            import random

            rng = random.Random(seed + trial_idx)
            perm = list(range(len(demos_orig)))
            rng.shuffle(perm)
            outputs = [y for _x, y in demos_orig]
            shuffled = [outputs[i] for i in perm]
            demos = [(demos_orig[i][0], shuffled[i]) for i in range(len(demos_orig))]
        else:
            demos = demos_orig
        prompt_data = paper_prompt_data_from_pairs(
            demos,
            query,
            tok_add_special=tok_add_special,
            prefixes=prefixes,
            separators=separators,
            shuffle_labels=shuffle_labels,
        )
        prompt_string, dummy_labels_raw, idx_map, idx_avg = paper_labels_and_maps(
            prompt_data,
            tokenizer,
            tok_add_special=tok_add_special,
            prefixes=prompt_data.get("prefixes"),
            separators=prompt_data.get("separators"),
        )
        if dummy_labels is None:
            n_slots = len(dummy_labels_raw)
            mean = torch.zeros(
                (n_layers, n_heads, n_slots, head_dim),
                dtype=torch.float32,
                device="cpu",
            )
            dummy_labels, slot_index_map = paper_labels_to_slot_map(dummy_labels_raw)
        else:
            labels, slot_map = paper_labels_to_slot_map(dummy_labels_raw)
            if dummy_labels != labels:
                raise ValueError("dummy_labels mismatch across trials")
            if slot_index_map != slot_map:
                raise ValueError("slot_index_map mismatch across trials")

        if idx_avg and not avg_logged:
            log("multi-token slot detected; applying mean over span")
            avg_logged = True

        hook_state: Dict[int, torch.Tensor] = {}
        errors: List[str] = []

        def make_pre_hook(layer_idx: int):
            def pre_hook(_module, inputs):
                tensor = inputs[0] if inputs else None
                if tensor is None:
                    errors.append("Missing out_proj input")
                    return
                hook_state[layer_idx] = tensor.detach()

            return pre_hook

        handles = []
        for layer_idx, module in layer_modules.items():
            handles.append(module.register_forward_pre_hook(make_pre_hook(layer_idx)))

        inputs = tokenizer(
            prompt_string,
            return_tensors="pt",
            add_special_tokens=tok_add_special,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            _ = model(**inputs)

        for handle in handles:
            handle.remove()

        if errors:
            raise ValueError("; ".join(errors))

        layer_acts: List[torch.Tensor] = []
        for layer_idx in layers:
            if layer_idx not in hook_state:
                raise ValueError("Missing activation for layer")
            x = hook_state[layer_idx]
            if x.dim() != 3 or x.shape[-1] != resid_dim:
                raise ValueError("Activation shape mismatch")
            batch_size, seq_len, _hidden = x.shape
            if batch_size != 1:
                raise ValueError("Expected batch size 1 during mean computation")
            x_heads = x.reshape(batch_size, seq_len, n_heads, head_dim)
            layer_acts.append(x_heads.squeeze(0))

        acts = torch.stack(layer_acts, dim=0).permute(0, 2, 1, 3)
        slot_acts = apply_paper_idx_map_to_slots(acts, idx_map, len(dummy_labels_raw))
        slot_acts_cpu = slot_acts.to(dtype=torch.float32, device="cpu")

        mean_layers = mean[layers]
        mean[layers] = mean_layers + (slot_acts_cpu - mean_layers) / (count + 1)
        count += 1

    if count == 0:
        raise ValueError("No trials counted for mean_activations")

    return mean, dummy_labels, slot_index_map


def compute_mean_activations_fixed_trials_ns(
    *,
    model,
    tokenizer,
    layer_modules: Dict[int, object],
    fixed_trials,
    n_use,
    model_cfg: Dict[str, int],
    tok_add_special: bool,
    device,
    logger=None,
):
    import torch

    trials = fixed_trials.get("trials", fixed_trials)
    if not trials:
        raise ValueError("No trials provided for fixed_trials mean activations")

    n_use = len(trials) if n_use is None else min(int(n_use), len(trials))
    if n_use < 1:
        raise ValueError("n_use must be >= 1")

    n_heads = int(model_cfg["n_heads"])
    head_dim = int(model_cfg["head_dim"])
    resid_dim = int(model_cfg["resid_dim"])
    n_layers = int(model_cfg["n_layers"])

    layers = sorted(layer_modules.keys())
    if not layers:
        raise ValueError("No layers provided for mean activation computation")

    def log(message: str) -> None:
        if logger is None:
            return
        if callable(logger):
            logger(message)
        elif hasattr(logger, "info"):
            logger.info(message)

    dummy_labels = None
    slot_index_map = None
    avg_logged = False

    mean = torch.zeros(
        (n_layers, n_heads, 0, head_dim),
        dtype=torch.float32,
        device="cpu",
    )
    count = 0

    for trial_idx, trial in enumerate(trials[:n_use]):
        prompt_data = trial.get("prompt_data_clean")
        if prompt_data is None:
            raise ValueError("fixed_trials missing prompt_data_clean")

        prompt_string, dummy_labels_raw, idx_map, idx_avg = paper_labels_and_maps(
            prompt_data,
            tokenizer,
            tok_add_special=tok_add_special,
            prefixes=prompt_data.get("prefixes"),
            separators=prompt_data.get("separators"),
        )
        clean_prompt = trial.get("clean_prompt_str")
        if clean_prompt is not None and clean_prompt != prompt_string:
            raise ValueError("fixed_trials clean prompt mismatch with paper prompt_string")

        if dummy_labels is None:
            mean = torch.zeros(
                (n_layers, n_heads, len(dummy_labels_raw), head_dim),
                dtype=torch.float32,
                device="cpu",
            )
            dummy_labels, slot_index_map = paper_labels_to_slot_map(dummy_labels_raw)
        else:
            labels, slot_map = paper_labels_to_slot_map(dummy_labels_raw)
            if dummy_labels != labels:
                raise ValueError("dummy_labels mismatch across trials")
            if slot_index_map != slot_map:
                raise ValueError("slot_index_map mismatch across trials")

        if idx_avg and not avg_logged:
            log("multi-token slot detected; applying mean over span")
            avg_logged = True

        hook_state: Dict[int, torch.Tensor] = {}
        errors: List[str] = []

        def make_pre_hook(layer_idx: int):
            def pre_hook(_module, inputs):
                tensor = inputs[0] if inputs else None
                if tensor is None:
                    errors.append("Missing out_proj input")
                    return
                hook_state[layer_idx] = tensor.detach()

            return pre_hook

        handles = []
        for layer_idx, module in layer_modules.items():
            handles.append(module.register_forward_pre_hook(make_pre_hook(layer_idx)))

        inputs = tokenizer(
            prompt_string,
            return_tensors="pt",
            add_special_tokens=tok_add_special,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            _ = model(**inputs)

        for handle in handles:
            handle.remove()

        if errors:
            raise ValueError("; ".join(errors))

        layer_acts: List[torch.Tensor] = []
        for layer_idx in layers:
            if layer_idx not in hook_state:
                raise ValueError("Missing activation for layer")
            x = hook_state[layer_idx]
            if x.dim() != 3 or x.shape[-1] != resid_dim:
                raise ValueError("Activation shape mismatch")
            batch_size, seq_len, _hidden = x.shape
            if batch_size != 1:
                raise ValueError("Expected batch size 1 during mean computation")
            x_heads = x.reshape(batch_size, seq_len, n_heads, head_dim)
            layer_acts.append(x_heads.squeeze(0))

        acts = torch.stack(layer_acts, dim=0).permute(0, 2, 1, 3)
        slot_acts = apply_paper_idx_map_to_slots(acts, idx_map, len(dummy_labels_raw))
        slot_acts_cpu = slot_acts.to(dtype=torch.float32, device="cpu")

        mean_layers = mean[layers]
        mean[layers] = mean_layers + (slot_acts_cpu - mean_layers) / (count + 1)
        count += 1

    if count == 0:
        raise ValueError("No trials counted for mean_activations")

    return mean, dummy_labels, slot_index_map
