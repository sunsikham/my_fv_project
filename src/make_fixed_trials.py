import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

try:
    from src.utils.prompt_utils import (
        ICLDataset,
        create_prompt,
        split_icl_dataset,
        word_pairs_to_prompt_data,
    )
    from src.utils.eval_utils import get_answer_id
except ImportError:
    from utils.prompt_utils import (
        ICLDataset,
        create_prompt,
        split_icl_dataset,
        word_pairs_to_prompt_data,
    )
    from utils.eval_utils import get_answer_id


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def compute_split_indices(raw_df, test_split, seed):
    train_df, valid_df = train_test_split(raw_df, test_size=test_split, random_state=seed)
    test_df, valid_df = train_test_split(valid_df, test_size=test_split, random_state=seed)
    return list(train_df.index), list(valid_df.index), list(test_df.index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", type=str, required=False, help="Path to antonym.json")
    parser.add_argument("--out_path", type=str, required=True, help="Path to save fixed_trials.json")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--n_shots", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_split", type=float, default=0.3)
    parser.add_argument(
        "--prefixes",
        type=json.loads,
        default={"input": "Q:", "output": "A:", "instructions": ""},
    )
    parser.add_argument(
        "--separators",
        type=json.loads,
        default={"input": "\n", "output": "\n\n", "instructions": ""},
    )
    parser.add_argument("--model_name_for_tokenizer", type=str, default="gpt2")
    parser.add_argument("--model_prepend_bos", type=parse_bool, default=False)
    parser.add_argument("--prepend_bos_token_used", type=parse_bool, default=None)
    parser.add_argument("--verify", type=parse_bool, default=False)
    parser.add_argument("--verify_n", type=int, default=3)
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Optional path to write summary output instead of printing.",
    )

    args = parser.parse_args()

    def _emit(line: str):
        if log_handle is None:
            print(line)
        else:
            log_handle.write(line + "\n")

    log_handle = None
    if args.log_path:
        Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(args.log_path, "w", encoding="utf-8")

    if args.verify and args.dataset_json is None and args.out_path:
        fixed_trials = json.load(open(args.out_path, "r", encoding="utf-8"))
        model_name = fixed_trials.get("meta", {}).get("model_name_for_tokenizer")
        if model_name is None:
            raise ValueError("fixed_trials meta missing model_name_for_tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        def _verify_only(trials_data):
            add_special_tokens = bool(trials_data["meta"].get("model_prepend_bos"))
            for trial in trials_data["trials"]:
                assert "demos_clean" in trial and "demos_corrupted" in trial and "query" in trial
                assert len(trial["demos_clean"]) == trials_data["meta"]["n_shots"]
                assert len(trial["demos_corrupted"]) == trials_data["meta"]["n_shots"]
                check_ids = get_answer_id(
                    trial["clean_prompt_str"], trial["target_str"], tokenizer
                )
                if "answer_ids" in trial:
                    assert int(check_ids[0]) == int(trial["answer_ids"][0])
                assert int(check_ids[0]) == int(trial["first_token_id"])
            for i, trial in enumerate(trials_data["trials"][: max(0, int(args.verify_n))]):
                _emit(f"[VERIFY] trial={i}")
                _emit(
                    f"  clean_head80={repr(trial['clean_prompt_str'][:80])} "
                    f"clean_tail80={repr(trial['clean_prompt_str'][-80:])}"
                )
                _emit(
                    f"  corrupted_head80={repr(trial['corrupted_prompt_str'][:80])} "
                    f"corrupted_tail80={repr(trial['corrupted_prompt_str'][-80:])}"
                )
                enc = tokenizer(
                    trial["corrupted_prompt_str"],
                    add_special_tokens=add_special_tokens,
                )
                ids = enc["input_ids"]
                _emit(f"  ids_len={len(ids)} ids_last10={ids[-10:]}")
                answer_ids = trial.get("answer_ids") or []
                answer_ids_first = answer_ids[0] if answer_ids else None
                match = (
                    int(trial["first_token_id"]) == int(answer_ids_first)
                    if answer_ids_first is not None
                    else None
                )
                _emit(f"  target_str_repr={repr(trial['target_str'])}")
                _emit(
                    f"  first_token_id={trial['first_token_id']} "
                    f"answer_ids_first={answer_ids_first} match={match}"
                )
                if len(ids) == 0:
                    raise ValueError("tokenized ids empty for corrupted_prompt_str")
        _verify_only(fixed_trials)
        if log_handle is not None:
            log_handle.close()
        return

    if not args.dataset_json:
        raise ValueError("--dataset_json is required unless --verify is used with existing out_path")

    rng = np.random.default_rng(args.seed)

    dataset = ICLDataset(args.dataset_json)
    splits = split_icl_dataset(dataset, test_size=args.test_split, seed=args.seed)
    train_indices, valid_indices, test_indices = compute_split_indices(
        dataset.raw_data, args.test_split, args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_for_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # SSOT for BOS: do not inject BOS token via prompt strings.
    # Rely on tokenizer add_special_tokens instead.
    if args.prepend_bos_token_used is None:
        prepend_bos_token_used = False
    else:
        prepend_bos_token_used = bool(args.prepend_bos_token_used)

    # Defensive: strip BOS string from instructions prefix if present.
    bos_token_str = tokenizer.bos_token or tokenizer.eos_token or ""
    if bos_token_str and args.prefixes.get("instructions", "").startswith(bos_token_str):
        args.prefixes["instructions"] = args.prefixes["instructions"].replace(bos_token_str, "", 1)

    trials = []
    for trial_id in range(args.n_trials):
        demo_indices = rng.choice(len(splits["train"]), size=args.n_shots, replace=False).tolist()
        query_index = int(rng.choice(len(splits["valid"]), size=1, replace=False)[0])

        demos_clean_raw = splits["train"][demo_indices]
        query_target_raw = splits["valid"][query_index]

        shuffled_outputs = rng.permutation(demos_clean_raw["output"]).tolist()
        demos_corrupted_raw = {
            "input": list(demos_clean_raw["input"]),
            "output": shuffled_outputs,
        }

        def _coerce_item(value):
            if isinstance(value, list):
                return value[0]
            return value

        prompt_data_clean = word_pairs_to_prompt_data(
            word_pairs=demos_clean_raw,
            query_target_pair=query_target_raw,
            shuffle_labels=False,
            prepend_bos_token=prepend_bos_token_used,
            prefixes=args.prefixes,
            separators=args.separators,
            prepend_space=True,
        )
        prompt_data_corrupted = word_pairs_to_prompt_data(
            word_pairs=demos_corrupted_raw,
            query_target_pair=query_target_raw,
            shuffle_labels=False,
            prepend_bos_token=prepend_bos_token_used,
            prefixes=args.prefixes,
            separators=args.separators,
            prepend_space=True,
        )

        demos_clean = [
            {
                "input": _coerce_item(ex["input"]),
                "output": _coerce_item(ex["output"]),
            }
            for ex in prompt_data_clean["examples"]
        ]
        demos_corrupted = [
            {
                "input": _coerce_item(ex["input"]),
                "output": _coerce_item(ex["output"]),
            }
            for ex in prompt_data_corrupted["examples"]
        ]
        query = {
            "input": _coerce_item(prompt_data_clean["query_target"]["input"]),
            "output": _coerce_item(prompt_data_clean["query_target"]["output"]),
        }

        clean_prompt_str = create_prompt(prompt_data_clean)
        corrupted_prompt_str = create_prompt(prompt_data_corrupted)

        target_str = prompt_data_clean["query_target"]["output"]
        answer_ids = get_answer_id(clean_prompt_str, target_str, tokenizer)
        first_token_id = int(answer_ids[0])

        trials.append(
            {
                "trial_id": trial_id,
                "demo_indices_in_train": demo_indices,
                "query_index_in_valid": query_index,
                "demos_clean": demos_clean,
                "demos_corrupted": demos_corrupted,
                "query": query,
                "prompt_data_clean": prompt_data_clean,
                "prompt_data_corrupted": prompt_data_corrupted,
                "clean_prompt_str": clean_prompt_str,
                "corrupted_prompt_str": corrupted_prompt_str,
                "target_str": target_str,
                "answer_ids": answer_ids,
                "first_token_id": first_token_id,
                "target_first_token_id": first_token_id,
            }
        )

    fixed_trials = {
        "meta": {
            "dataset_json": str(Path(args.dataset_json)),
            "seed": args.seed,
            "n_trials": args.n_trials,
            "n_shots": args.n_shots,
            "test_split": args.test_split,
            "prefixes": args.prefixes,
            "separators": args.separators,
            "model_name_for_tokenizer": args.model_name_for_tokenizer,
            "model_prepend_bos": args.model_prepend_bos,
            "prepend_bos_token_used": prepend_bos_token_used,
        },
        "splits": {
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "test_indices": test_indices,
        },
        "trials": trials,
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(fixed_trials, f, indent=2, ensure_ascii=True)

    def _verify_trials(trials_data, verify_n):
        add_special_tokens = bool(trials_data["meta"].get("model_prepend_bos"))
        for trial in trials_data["trials"]:
            assert "demos_clean" in trial and "demos_corrupted" in trial and "query" in trial
            assert len(trial["demos_clean"]) == trials_data["meta"]["n_shots"]
            assert len(trial["demos_corrupted"]) == trials_data["meta"]["n_shots"]
            check_ids = get_answer_id(
                trial["clean_prompt_str"], trial["target_str"], tokenizer
            )
            if "answer_ids" in trial:
                assert int(check_ids[0]) == int(trial["answer_ids"][0])
            assert int(check_ids[0]) == int(trial["first_token_id"])
        for i, trial in enumerate(trials_data["trials"][: max(0, int(verify_n))]):
            print(f"[VERIFY] trial={i}")
            print(
                f"  clean_head80={repr(trial['clean_prompt_str'][:80])} "
                f"clean_tail80={repr(trial['clean_prompt_str'][-80:])}"
            )
            print(
                f"  corrupted_head80={repr(trial['corrupted_prompt_str'][:80])} "
                f"corrupted_tail80={repr(trial['corrupted_prompt_str'][-80:])}"
            )
            enc = tokenizer(
                trial["corrupted_prompt_str"],
                add_special_tokens=add_special_tokens,
            )
            ids = enc["input_ids"]
            print(f"  ids_len={len(ids)} ids_last10={ids[-10:]}")
            answer_ids = trial.get("answer_ids") or []
            answer_ids_first = answer_ids[0] if answer_ids else None
            match = (
                int(trial["first_token_id"]) == int(answer_ids_first)
                if answer_ids_first is not None
                else None
            )
            print(f"  target_str_repr={repr(trial['target_str'])}")
            print(
                f"  first_token_id={trial['first_token_id']} "
                f"answer_ids_first={answer_ids_first} match={match}"
            )
            if len(ids) == 0:
                raise ValueError("tokenized ids empty for corrupted_prompt_str")

    _verify_trials(fixed_trials, verify_n=0)
    if args.verify:
        _verify_trials(fixed_trials, verify_n=args.verify_n)

    if fixed_trials["trials"]:
        first_trial = fixed_trials["trials"][0]
        _emit("Trial 0 clean prompt:")
        _emit(first_trial["clean_prompt_str"])
        _emit("Trial 0 corrupted prompt:")
        _emit(first_trial["corrupted_prompt_str"])
        _emit(f"Trial 0 first_token_id: {first_trial['first_token_id']}")

    if log_handle is not None:
        log_handle.close()


if __name__ == "__main__":
    main()
