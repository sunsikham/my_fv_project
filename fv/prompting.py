"""Prompt builders for antonym-style tasks."""

import random
from typing import Dict, List, Optional, Tuple


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


def build_prompt_qa(
    demos: List[Tuple[str, str]],
    query: Tuple[str, str],
    prefixes: Optional[Dict[str, str]] = None,
    separators: Optional[Dict[str, str]] = None,
    prepend_bos_token: bool = False,
    prepend_space: bool = True,
) -> Tuple[str, str]:
    """Build Q/A style prompt using paper-style prefixes/separators."""
    use_prefixes = prefixes or {"input": "Q:", "output": "A:", "instructions": ""}
    use_separators = separators or {"input": "\n", "output": "\n\n", "instructions": ""}
    if prepend_bos_token:
        use_prefixes = {
            k: (v if k != "instructions" else "<|endoftext|>" + v)
            for k, v in use_prefixes.items()
        }

    prompt = ""
    prompt += (
        use_prefixes["instructions"]
        + ""
        + use_separators["instructions"]
    )

    for x, y in demos:
        demo_in = f" {x}" if prepend_space else str(x)
        demo_out = f" {y}" if prepend_space else str(y)
        prompt += use_prefixes["input"] + demo_in + use_separators["input"]
        prompt += use_prefixes["output"] + demo_out + use_separators["output"]

    query_in = f" {query[0]}" if prepend_space else str(query[0])
    prompt += use_prefixes["input"] + query_in + use_separators["input"]
    prompt += use_prefixes["output"]

    prefix_str = prompt
    query_out = f" {query[1]}" if prepend_space else str(query[1])
    full_str = f"{prefix_str}{query_out}"
    return prefix_str, full_str
