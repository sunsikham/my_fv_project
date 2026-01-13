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
    special_prefix = None
    special_suffix = None
    special_index_labels_ref = None
    avg_logged = False
    align_logged = False
    check_limit = 5
    skipped = 0

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
        prefix_str, full_str = build_prompt_qa(demos, query)

        segments = build_prompt_segments(demos, query)
        full_str_segments, _spans = segments_to_text_and_spans(segments)
        if full_str_segments != full_str:
            raise ValueError("Prompt builder mismatch with SSOT")

        encoded = tokenizer(
            full_str,
            return_offsets_mapping=True,
            add_special_tokens=tok_add_special,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if trial_idx < check_limit:
            ids_check = tokenizer(
                full_str,
                add_special_tokens=tok_add_special,
            )["input_ids"]
            if input_ids != ids_check:
                raise ValueError("Tokenizer output mismatch for prompt")

        special_index_labels = infer_special_token_labels(
            tokenizer, input_ids, offsets
        )
        special_positions = sorted(special_index_labels.keys())
        if special_positions:
            if special_positions[0] != 0:
                raise ValueError("Special token in unexpected position")
            if special_positions[-1] != len(input_ids) - 1 and len(special_positions) > 1:
                raise ValueError("Special token not in prefix/suffix position")

        if dummy_labels is None:
            special_prefix = []
            special_suffix = []
            if special_positions and special_positions[0] == 0:
                special_prefix.append(special_index_labels[special_positions[0]])
            if special_positions and special_positions[-1] == len(input_ids) - 1:
                if special_positions[-1] != special_positions[0]:
                    special_suffix.append(special_index_labels[special_positions[-1]])
            dummy_labels, slot_index_map = get_dummy_token_labels_and_slot_map(
                n_icl_examples,
                special_prefix=special_prefix,
                special_suffix=special_suffix,
            )
            n_slots = len(dummy_labels)
            mean = torch.zeros(
                (n_layers, n_heads, n_slots, head_dim),
                dtype=torch.float32,
                device="cpu",
            )
            special_index_labels_ref = special_index_labels
        else:
            if special_index_labels != special_index_labels_ref:
                raise ValueError("Special token alignment mismatch across trials")

        token_meta_labels, _input_ids, _offsets = get_token_meta_labels(
            tokenizer,
            full_str,
            segments,
            tok_add_special,
            special_index_labels,
        )
        idx_map, idx_avg = compute_duplicated_labels(
            token_meta_labels,
            dummy_labels,
            allow_empty_labels=["QUERY_OUT"],
        )
        validate_idx_map(idx_map, len(input_ids))
        if len(idx_map) != len(dummy_labels):
            raise ValueError("idx_map missing slots")
        if sorted(idx_map.keys()) != list(range(len(dummy_labels))):
            raise ValueError("idx_map keys are not contiguous")
        if idx_avg and not avg_logged:
            log("multi-token slot detected; applying mean over span")
            avg_logged = True
        for slot_idx, token_indices in idx_map.items():
            if dummy_labels[slot_idx] == "QUERY_OUT":
                continue
            if not token_indices:
                raise ValueError("Empty token span for slot")
        prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=tok_add_special)
        if not prefix_ids:
            raise ValueError("Prefix tokenization failed")
        prefix_last_idx = len(prefix_ids) - 1
        slot_q = slot_index_map["QUERY_PRED"]
        if prefix_last_idx not in idx_map.get(slot_q, []):
            if allow_alignment_skip:
                skipped += 1
                log(
                    "alignment_skip: "
                    f"trial={trial_idx} prefix_last_idx={prefix_last_idx} "
                    f"slot_q={slot_q} span={idx_map.get(slot_q, [])}"
                )
                continue
            raise ValueError("QUERY_PRED does not align with last token")
        if not align_logged:
            log(
                "QUERY_PRED aligns with last token: True "
                f"(last_idx={prefix_last_idx} span={idx_map.get(slot_q, [])})"
            )
            align_logged = True

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
            full_str,
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

        acts = torch.stack(layer_acts, dim=0)
        slot_acts = apply_idx_map_to_activations(acts, idx_map, len(dummy_labels))
        slot_acts_cpu = slot_acts.to(dtype=torch.float32, device="cpu")

        mean_layers = mean[layers]
        mean[layers] = mean_layers + (slot_acts_cpu - mean_layers) / (count + 1)
        count += 1

    if skipped:
        log(f"alignment_skipped_trials={skipped}")
    if count == 0:
        raise ValueError("No trials counted for mean_activations")

    return mean, dummy_labels, slot_index_map
