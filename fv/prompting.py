"""Prompt builders and prompt-data utilities."""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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


def build_prompt_qa_paper(
    demos: List[Tuple[str, str]],
    query: Tuple[str, str],
    prefixes: Optional[Dict[str, str]] = None,
    separators: Optional[Dict[str, str]] = None,
    prepend_bos_token: bool = False,
    prepend_space: bool = True,
) -> Tuple[str, str, dict]:
    """Build prompt strings using a paper-identical prompt_data path."""
    word_pairs = {"input": [x for x, _y in demos], "output": [y for _x, y in demos]}
    query_target_pair = {"input": [query[0]], "output": [query[1]]}
    prompt_data = word_pairs_to_prompt_data(
        word_pairs,
        query_target_pair=query_target_pair,
        prepend_bos_token=prepend_bos_token,
        prefixes=prefixes,
        separators=separators,
        shuffle_labels=False,
        prepend_space=prepend_space,
    )
    prefix_str = create_prompt(prompt_data)
    query_out = prompt_data["query_target"]["output"]
    if isinstance(query_out, list):
        query_out = query_out[0]
    full_str = f"{prefix_str}{query_out}"
    return prefix_str, full_str, prompt_data


def create_fewshot_primer(prompt_data) -> str:
    prompt = ""
    prompt += (
        prompt_data["prefixes"]["instructions"]
        + prompt_data["instructions"]
        + prompt_data["separators"]["instructions"]
    )
    for example in prompt_data["examples"]:
        prompt += prompt_data["prefixes"]["input"] + example["input"] + prompt_data["separators"]["input"]
        prompt += prompt_data["prefixes"]["output"] + example["output"] + prompt_data["separators"]["output"]
    return prompt


def create_prompt(prompt_data, sentence=None) -> str:
    if sentence is None and prompt_data.get("query_target") is not None:
        sentence = prompt_data["query_target"]["input"]
    if isinstance(sentence, list):
        sentence = sentence[0]
    prompt_init = create_fewshot_primer(prompt_data)
    prompt = prompt_init + prompt_data["prefixes"]["input"] + sentence + prompt_data["separators"]["input"]
    prompt += prompt_data["prefixes"]["output"]
    return prompt


def word_pairs_to_prompt_data(
    word_pairs: dict,
    instructions: str = "",
    prefixes: dict = {"input": "Q:", "output": "A:", "instructions": ""},
    separators: dict = {"input": "\n", "output": "\n\n", "instructions": ""},
    query_target_pair: dict = None,
    prepend_bos_token: bool = False,
    shuffle_labels: bool = False,
    prepend_space: bool = True,
) -> dict:
    prompt_data = {}
    prompt_data["instructions"] = instructions
    prompt_data["separators"] = separators
    if prepend_bos_token:
        prefixes = {
            k: (v if k != "instructions" else "<|endoftext|>" + v)
            for (k, v) in prefixes.items()
        }
    prompt_data["prefixes"] = prefixes

    if query_target_pair is not None:
        query_target_pair = {k: (v[0] if isinstance(v, list) else v) for k, v in query_target_pair.items()}
    prompt_data["query_target"] = query_target_pair

    if shuffle_labels:
        randomized_pairs = [
            np.random.permutation(x).tolist() if i == 1 else x
            for (i, x) in enumerate(list(word_pairs.values()))
        ]
        if prepend_space:
            prompt_data["examples"] = [
                {"input": " " + str(w1), "output": " " + str(w2)}
                for (w1, w2) in list(zip(*randomized_pairs))
            ]
            prompt_data["query_target"] = (
                {k: " " + str(v) for k, v in query_target_pair.items()}
                if query_target_pair is not None
                else None
            )
        else:
            prompt_data["examples"] = [
                {"input": w1, "output": w2} for (w1, w2) in list(zip(*randomized_pairs))
            ]
    else:
        if prepend_space:
            prompt_data["examples"] = [
                {"input": " " + str(w1), "output": " " + str(w2)}
                for (w1, w2) in list(zip(*word_pairs.values()))
            ]
            prompt_data["query_target"] = (
                {k: " " + str(v) for k, v in query_target_pair.items()}
                if query_target_pair is not None
                else None
            )
        else:
            prompt_data["examples"] = [
                {"input": w1, "output": w2} for (w1, w2) in list(zip(*word_pairs.values()))
            ]

    return prompt_data


def get_prompt_parts_and_labels(prompt_data, query_sentence=None):
    if query_sentence is None and prompt_data["query_target"] is not None:
        query_sentence = prompt_data["query_target"]["input"]
    if isinstance(query_sentence, list):
        query_sentence = query_sentence[0]
    n_examples = len(prompt_data["examples"])
    assemble_icl_example = lambda example, prompt_data: [
        prompt_data["prefixes"]["input"],
        example["input"],
        prompt_data["separators"]["input"],
        prompt_data["prefixes"]["output"],
        example["output"],
        prompt_data["separators"]["output"],
    ]
    assemble_icl_query = lambda query, prompt_data: [
        prompt_data["prefixes"]["input"],
        query,
        prompt_data["separators"]["input"],
        prompt_data["prefixes"]["output"],
    ]

    prompt_instructions = [
        prompt_data["prefixes"]["instructions"],
        prompt_data["instructions"],
        prompt_data["separators"]["instructions"],
    ]
    prompt_icl_examples = [
        assemble_icl_example(prompt_data["examples"][i], prompt_data)
        for i in range(n_examples)
    ]
    prompt_icl_query = [assemble_icl_query(query_sentence, prompt_data)]

    prompt_instructions_labels = ["bos_token", "instructions_token", "separator_token"]
    prompt_icl_examples_labels = [
        [
            "structural_token",
            f"demonstration_{i+1}_token",
            "separator_token",
            "structural_token",
            f"demonstration_{i+1}_label_token",
            "separator_token",
        ]
        for i in range(n_examples)
    ]
    prompt_icl_query_labels = [
        [
            "query_structural_token",
            "query_demonstration_token",
            "query_separator_token",
            "query_structural_token",
        ]
    ]

    prompt_parts = prompt_instructions + prompt_icl_examples + prompt_icl_query
    prompt_part_labels = prompt_instructions_labels + prompt_icl_examples_labels + prompt_icl_query_labels

    return prompt_parts, prompt_part_labels


def extend_labels(sentence_parts, text_labels, tokenizer, label_init=None):
    if label_init is None:
        label_init = []
    zipped_up = [
        list(zip(x, y)) if isinstance(x, list) else [(x, y)]
        for x, y in list(zip(sentence_parts, text_labels))
    ]

    prompt_builder = ""
    final_labels = label_init
    for element in zipped_up:
        for j, (word, label) in enumerate(element):
            if len(word) == 0:
                continue
            pre = len(tokenizer.tokenize(prompt_builder))
            prompt_builder += word
            post = len(tokenizer.tokenize(prompt_builder))

            actual_tokens = post - pre

            if actual_tokens == 0:
                final_labels[-1] = label

            final_labels.extend([label] * (actual_tokens))

            if j == 3 or (j == 2 and len(element[3]) == 0):
                final_labels[-1] = (
                    final_labels[-1]
                    .replace("structural", "predictive")
                    .replace("separator", "predictive")
                )
            if j == 5:
                final_labels[-actual_tokens] = final_labels[-actual_tokens].replace(
                    "separator", "end_of_example"
                )

    return final_labels


def tokenize_labels(sentence_parts, text_labels, tokenizer, prepend_bos: bool = False):
    label_init = ["bos_token"] if prepend_bos else []
    return extend_labels(sentence_parts, text_labels, tokenizer, label_init=label_init)


def get_token_meta_labels(prompt_data, tokenizer, query=None, prepend_bos: bool = False):
    if query is None and prompt_data["query_target"] is not None:
        query = prompt_data["query_target"]["input"]
    if isinstance(query, list):
        query = query[0]

    prompt_parts, prompt_part_labels = get_prompt_parts_and_labels(prompt_data, query_sentence=query)
    token_meta_labels = tokenize_labels(prompt_parts, prompt_part_labels, tokenizer, prepend_bos)
    prompt_string = create_prompt(prompt_data=prompt_data, sentence=query)
    tokens = [tokenizer.decode(x) for x in tokenizer(prompt_string).input_ids]
    token_labels = list(zip(np.arange(len(tokens)), tokens, token_meta_labels))
    return token_labels, prompt_string


def get_dummy_token_labels(n_icl_examples, tokenizer, model_config, prefixes=None, separators=None):
    # BOS SSOT: do not inject BOS string in dummy prompts
    prepend_bos = False
    if prefixes is not None and separators is not None:
        dummy_prompt_data = word_pairs_to_prompt_data(
            {"input": ["a"] * n_icl_examples, "output": ["a"] * n_icl_examples},
            query_target_pair={"input": ["a"], "output": ["a"]},
            prepend_bos_token=prepend_bos,
            prefixes=prefixes,
            separators=separators,
        )
    else:
        dummy_prompt_data = word_pairs_to_prompt_data(
            {"input": ["a"] * n_icl_examples, "output": ["a"] * n_icl_examples},
            query_target_pair={"input": ["a"], "output": ["a"]},
            prepend_bos_token=prepend_bos,
        )
    final_token_labels, _ = get_token_meta_labels(
        dummy_prompt_data, tokenizer, prepend_bos=model_config["prepend_bos"]
    )
    final_token_labels = [(x[0], x[-1]) for x in final_token_labels]
    return final_token_labels


def compute_duplicated_labels(token_labels, gt_labels):
    check_inds = list(filter(lambda x: "demo" in x[2], token_labels))
    dup_ranges = (
        pd.DataFrame(check_inds).groupby(2)[0].aggregate(lambda x: (x.min(), x.max()))
    )
    dup_labels = [v for v, x in dup_ranges.items() if (x[1] - x[0]) > 0]

    dup_label_ranges = dup_ranges[dup_labels].to_dict()
    dup_inds = pd.DataFrame(check_inds)[pd.DataFrame(check_inds)[2].duplicated()][0].values

    index_map = {k: v[0] for (k, v) in zip([x[0] for x in token_labels if x[0] not in dup_inds], gt_labels)}

    return index_map, dup_label_ranges


def update_idx_map(idx_map, idx_avg) -> dict:
    update_map = {}
    for (i, j) in idx_avg.values():
        for k in range(i, j + 1):
            if k not in idx_map.keys():
                update_map[k] = idx_map[i]
    update_map = {**idx_map, **update_map}
    return update_map
