"""Prompt builders for antonym-style tasks."""

import random
from typing import List, Tuple


ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("hot", "cold"),
    ("up", "down"),
    ("big", "small"),
    ("happy", "sad"),
    ("fast", "slow"),
    ("young", "old"),
    ("light", "dark"),
    ("hard", "soft"),
    ("early", "late"),
    ("open", "close"),
    ("high", "low"),
    ("strong", "weak"),
    ("clean", "dirty"),
    ("sharp", "dull"),
]


def build_two_shot_prompt(rng: random.Random) -> Tuple[str, str, str]:
    pairs = rng.sample(ANTONYM_PAIRS, 3)
    shot_1, shot_2, query = pairs
    lines = [
        "Antonyms:",
        f"{shot_1[0]} -> {shot_1[1]}",
        f"{shot_2[0]} -> {shot_2[1]}",
        f"{query[0]} ->",
    ]
    prompt = "\n".join(lines)
    prefix_str = f"{prompt} "
    full_str = f"{prefix_str}{query[1]}"
    return prefix_str, full_str, query[1]


def build_zero_shot_prompt(rng: random.Random) -> Tuple[str, str]:
    query, answer = rng.choice(ANTONYM_PAIRS)
    lines = ["Antonyms:", f"{query} ->"]
    return "\n".join(lines), answer
