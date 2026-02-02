import json
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import torch
from baukit import TraceDict

from fv.prompting import (
    compute_duplicated_labels,
    create_prompt,
    get_dummy_token_labels,
    get_token_meta_labels,
)


def load_fixed_trials(path: Union[str, Path]) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_fixed_trials(
    fixed_trials: dict, mode: str = "clean"
) -> Iterator[Tuple[dict, str, int]]:
    if mode not in {"clean", "corrupted"}:
        raise ValueError(f"mode must be 'clean' or 'corrupted', got: {mode}")
    trials = fixed_trials.get("trials", fixed_trials)
    data_key = "prompt_data_clean" if mode == "clean" else "prompt_data_corrupted"
    prompt_key = "clean_prompt_str" if mode == "clean" else "corrupted_prompt_str"
    for trial in trials:
        if data_key not in trial:
            raise ValueError(f"Missing {data_key} in fixed_trials trial")
        prompt_data = trial[data_key]
        prompt_str = trial.get(prompt_key)
        if prompt_str is None:
            prompt_str = create_prompt(prompt_data=prompt_data)
        target_first_token_id = trial.get("target_first_token_id")
        if target_first_token_id is None:
            raise ValueError("Missing target_first_token_id in fixed_trials trial")
        yield prompt_data, prompt_str, int(target_first_token_id)


def _resolve_prefixes_separators(
    prompt_data: dict,
    prefixes: Optional[dict],
    separators: Optional[dict],
) -> Tuple[Optional[dict], Optional[dict]]:
    use_prefixes = prefixes
    use_separators = separators
    if use_prefixes is None:
        use_prefixes = prompt_data.get("prefixes")
    if use_separators is None:
        use_separators = prompt_data.get("separators")
    return use_prefixes, use_separators


def gather_attn_activations(prompt_data, layers, dummy_labels, model, tokenizer, model_config):
    query = prompt_data["query_target"]["input"]
    token_labels, prompt_string = get_token_meta_labels(
        prompt_data, tokenizer, query, prepend_bos=model_config["prepend_bos"]
    )
    sentence = [prompt_string]

    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)

    with TraceDict(model, layers=layers, retain_input=True, retain_output=False) as td:
        model(**inputs)

    return td, idx_map, idx_avg


def get_mean_head_activations_from_fixed_trials_paper_exact(
    fixed_trials_or_path: Union[dict, str, Path],
    model,
    model_config: dict,
    tokenizer,
    mode: str = "clean",
    n_use: Optional[int] = None,
    prefixes: Optional[dict] = None,
    separators: Optional[dict] = None,
):
    if isinstance(fixed_trials_or_path, (str, Path)):
        fixed_trials = load_fixed_trials(fixed_trials_or_path)
    else:
        fixed_trials = fixed_trials_or_path

    if mode not in {"clean", "corrupted"}:
        raise ValueError(f"mode must be 'clean' or 'corrupted', got: {mode}")

    trials = fixed_trials.get("trials", fixed_trials)
    if not trials:
        raise ValueError("No trials found in fixed_trials.json")

    data_key = "prompt_data_clean" if mode == "clean" else "prompt_data_corrupted"
    if data_key not in trials[0]:
        raise ValueError(f"Missing {data_key} in fixed_trials")
    prompt_data_first = trials[0][data_key]
    n_icl_examples = len(prompt_data_first.get("examples", []))
    if n_icl_examples == 0:
        raise ValueError("Fixed trials contain zero examples in prompt_data")

    use_prefixes, use_separators = _resolve_prefixes_separators(
        prompt_data_first, prefixes, separators
    )
    dummy_labels = get_dummy_token_labels(
        n_icl_examples,
        tokenizer=tokenizer,
        prefixes=use_prefixes,
        separators=use_separators,
        model_config=model_config,
    )

    if n_use is None:
        n_use = len(trials)
    else:
        n_use = min(int(n_use), len(trials))

    activation_storage = torch.zeros(
        n_use,
        model_config["n_layers"],
        model_config["n_heads"],
        len(dummy_labels),
        model_config["resid_dim"] // model_config["n_heads"],
    )

    def split_activations_by_head(activations, model_config):
        new_shape = activations.size()[:-1] + (
            model_config["n_heads"],
            model_config["resid_dim"] // model_config["n_heads"],
        )
        activations = activations.view(*new_shape)
        return activations

    for n, trial in enumerate(trials[:n_use]):
        if data_key not in trial:
            raise ValueError(f"Missing {data_key} in trial index {n}")
        prompt_data = trial[data_key]

        activations_td, idx_map, idx_avg = gather_attn_activations(
            prompt_data=prompt_data,
            layers=model_config["attn_hook_names"],
            dummy_labels=dummy_labels,
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
        )

        stack_initial = torch.vstack(
            [
                split_activations_by_head(activations_td[layer].input, model_config)
                for layer in model_config["attn_hook_names"]
            ]
        ).permute(0, 2, 1, 3)

        stack_filtered = stack_initial[:, :, list(idx_map.keys())]
        for (i, j) in idx_avg.values():
            stack_filtered[:, :, idx_map[i]] = stack_initial[:, :, i : j + 1].mean(axis=2)

        activation_storage[n] = stack_filtered

    mean_activations = activation_storage.mean(dim=0)
    return mean_activations, dummy_labels
