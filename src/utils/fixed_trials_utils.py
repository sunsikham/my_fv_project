import json


def load_fixed_trials(path):
    with open(path, "r") as f:
        return json.load(f)


def iter_fixed_trials(fixed_trials, mode="clean"):
    if mode not in {"clean", "corrupted"}:
        raise ValueError(f"mode must be 'clean' or 'corrupted', got: {mode}")

    trials = fixed_trials.get("trials", fixed_trials)
    data_key = "prompt_data_clean" if mode == "clean" else "prompt_data_corrupted"
    prompt_key = "clean_prompt_str" if mode == "clean" else "corrupted_prompt_str"

    for trial in trials:
        yield trial[data_key], trial[prompt_key], trial["target_first_token_id"]
