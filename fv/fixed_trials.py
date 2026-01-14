"""Fixed trials loader for StepD."""

import json


def load_fixed_trials(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
