import os, re, json
from tqdm import tqdm
import torch, numpy as np
import argparse
from baukit import TraceDict

# Include prompt creation helper functions
try:
    from src.utils.prompt_utils import *
    from src.utils.intervention_utils import *
    from src.utils.model_utils import *
    from src.utils.extract_utils import *
    from src.utils.fixed_trials_utils import load_fixed_trials, iter_fixed_trials
except ImportError:
    from utils.prompt_utils import *
    from utils.intervention_utils import *
    from utils.model_utils import *
    from utils.extract_utils import *
    from utils.fixed_trials_utils import load_fixed_trials, iter_fixed_trials


def activation_replacement_per_class_intervention(
    prompt_data,
    avg_activations,
    dummy_labels,
    model,
    model_config,
    tokenizer,
    last_token_only=True,
    token_id_of_interest=None,
    dump_cfg=None,
):
    """
    Experiment to determine top intervention locations through avg activation replacement. 
    Performs a systematic sweep over attention heads (layer, head) to track their causal influence on probs of key tokens.

    Parameters: 
    prompt_data: dict containing ICL prompt examples, and template information
    avg_activations: avg activation of each attention head in the model taken across n_trials ICL prompts
    dummy_labels: labels and indices for a baseline prompt with the same number of example pairs
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    last_token_only: If True, only computes indirect effect for heads at the final token position. If False, computes indirect_effect for heads for all token classes

    Returns:   
    indirect_effect_storage: torch tensor containing the indirect_effect of each head for each token class.
    """
    device = model.device

    # Get sentence and token labels
    query_target_pair = prompt_data['query_target']

    query = query_target_pair['input']
    token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query=query, prepend_bos=model_config['prepend_bos'])

    idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)
    idx_map = update_idx_map(idx_map, idx_avg)
      
    sentences = [prompt_string]# * model.config.n_head # batch things by head

    # Figure out tokens of interest
    if token_id_of_interest is None:
        target = [query_target_pair['output']]
        token_id_of_interest = get_answer_id(sentences[0], target[0], tokenizer)
    if isinstance(token_id_of_interest, list):
        token_id_of_interest = token_id_of_interest[:1]
    else:
        token_id_of_interest = [int(token_id_of_interest)]
        
    inputs = tokenizer(sentences, return_tensors='pt').to(device)
    input_ids = inputs["input_ids"][0].detach().cpu().tolist()

    # Speed up computation by only computing causal effect at last token
    if last_token_only:
        token_classes = ['query_predictive']
        token_classes_regex = ['query_predictive_token']
    # Compute causal effect for all token classes (instead of just last token)
    else:
        token_classes = ['demonstration', 'label', 'separator', 'predictive', 'structural','end_of_example', 
                        'query_demonstration', 'query_structural', 'query_separator', 'query_predictive']
        token_classes_regex = ['demonstration_[\d]{1,}_token', 'demonstration_[\d]{1,}_label_token', 'separator_token', 'predictive_token', 'structural_token','end_of_example_token', 
                            'query_demonstration_token', 'query_structural_token', 'query_separator_token', 'query_predictive_token']
    

    indirect_effect_storage = torch.zeros(model_config['n_layers'], model_config['n_heads'],len(token_classes))

    # Clean Run of Baseline:
    clean_output = model(**inputs).logits[:,-1,:]
    clean_logits = clean_output[0]
    clean_probs = torch.softmax(clean_logits, dim=-1)
    clean_logprobs = torch.log_softmax(clean_logits, dim=-1)
    dump_target_id = int(token_id_of_interest[0])

    # For every layer, head, token combination perform the replacement & track the change in meaningful tokens
    for layer in range(model_config['n_layers']):
        head_hook_layer = [model_config['attn_hook_names'][layer]]
        
        for head_n in range(model_config['n_heads']):
            for i,(token_class, class_regex) in enumerate(zip(token_classes, token_classes_regex)):
                reg_class_match = re.compile(f"^{class_regex}$")
                class_token_inds = [x[0] for x in token_labels if reg_class_match.match(x[2])]

                intervention_locations = [(layer, head_n, token_n) for token_n in class_token_inds]
                intervention_fn = replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                           model=model, model_config=model_config,
                                                           batched_input=False, idx_map=idx_map, last_token_only=last_token_only)
                with TraceDict(model, layers=head_hook_layer, edit_output=intervention_fn) as td:                
                    output = model(**inputs).logits[:,-1,:] # batch_size x n_tokens x vocab_size, only want last token prediction
                
                # TRACK probs of tokens of interest
                intervention_probs = torch.softmax(output, dim=-1) # convert to probability distribution
                indirect_effect_storage[layer,head_n,i] = (intervention_probs-clean_probs).index_select(1, torch.LongTensor(token_id_of_interest).to(device).squeeze()).squeeze()

                if dump_cfg and last_token_only and i == 0:
                    dump_handle = dump_cfg.get("handle")
                    if dump_handle is not None:
                        trial_idx = dump_cfg.get("trial_idx")
                        max_trials = dump_cfg.get("max_trials")
                        if max_trials is None or trial_idx < max_trials:
                            dump_layer = dump_cfg.get("dump_layer")
                            dump_head = dump_cfg.get("dump_head")
                            if (dump_layer is None or dump_layer == layer) and (
                                dump_head is None or dump_head == head_n
                            ):
                                import json as _json

                                patch_logits = output[0]
                                patch_probs = intervention_probs[0]
                                patch_logprobs = torch.log_softmax(patch_logits, dim=-1)
                                p_base = clean_probs[dump_target_id].item()
                                p_patch = patch_probs[dump_target_id].item()
                                logit_base = clean_logits[dump_target_id].item()
                                logit_patch = patch_logits[dump_target_id].item()
                                logprob_base = clean_logprobs[dump_target_id].item()
                                logprob_patch = patch_logprobs[dump_target_id].item()
                                prompt_tail_repr = ""
                                if dump_cfg.get("include_prompt", True):
                                    prompt_tail_repr = repr(prompt_string[-60:])
                                ids_last10 = input_ids[-10:]
                                row = {
                                    "trial_idx": int(trial_idx),
                                    "layer": int(layer),
                                    "head": int(head_n),
                                    "target_id": int(dump_target_id),
                                    "target_token": tokenizer.convert_ids_to_tokens(dump_target_id),
                                    "seed_global": dump_cfg.get("seed_global"),
                                    "shuffle_labels": dump_cfg.get("shuffle_labels"),
                                    "shuffle_derangement": dump_cfg.get("shuffle_derangement"),
                                    "n_icl_examples": dump_cfg.get("n_icl_examples"),
                                    "demo_perm": None,
                                    "demo_fixed_points": None,
                                    "demo_outputs_before": None,
                                    "demo_outputs_after": None,
                                    "p_base": float(p_base),
                                    "p_patch": float(p_patch),
                                    "delta_p": float(p_patch - p_base),
                                    "logit_base": float(logit_base),
                                    "logit_patch": float(logit_patch),
                                    "delta_logit": float(logit_patch - logit_base),
                                    "logprob_base": float(logprob_base),
                                    "logprob_patch": float(logprob_patch),
                                    "delta_logprob": float(logprob_patch - logprob_base),
                                    "prompt_tail_repr": prompt_tail_repr,
                                    "prompt_ends_with_space": prompt_string.endswith(" "),
                                    "slot_idx": None,
                                    "seq_token_idx": int(len(input_ids) - 1),
                                    "ids_len": int(len(input_ids)),
                                    "ids_last10": ids_last10,
                                }
                                expected_keys = dump_cfg.get("expected_keys")
                                if expected_keys and set(row.keys()) != set(expected_keys):
                                    missing = set(expected_keys) - set(row.keys())
                                    extra = set(row.keys()) - set(expected_keys)
                                    raise ValueError(
                                        f"dump keys mismatch missing={sorted(missing)} extra={sorted(extra)}"
                                    )
                                dump_handle.write(_json.dumps(row, ensure_ascii=True) + "\n")

    return indirect_effect_storage


def compute_indirect_effect(
    dataset,
    mean_activations,
    model,
    model_config,
    tokenizer,
    n_shots=10,
    n_trials=25,
    last_token_only=True,
    prefixes=None,
    separators=None,
    filter_set=None,
    fixed_trials=None,
    dump_cfg=None,
):
    """
    Computes Indirect Effect of each head in the model

    Parameters:
    dataset: ICL dataset
    mean_activations:
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_shots: Number of shots in each in-context prompt
    n_trials: Number of in-context prompts to average over
    last_token_only: If True, only computes Indirect Effect for heads at the final token position. If False, computes Indirect Effect for heads for all token classes


    Returns:
    indirect_effect: torch tensor of the indirect effect for each attention head in the model, size n_trials * n_layers * n_heads
    """
    n_test_examples = 1

    if fixed_trials is not None:
        trials_list = fixed_trials.get("trials", [])
        if len(trials_list) == 0:
            raise ValueError("No trials found in fixed_trials.json")
        prompt_data_first = trials_list[0]["prompt_data_corrupted"]
        n_shots = len(prompt_data_first["examples"])
        prefixes = prompt_data_first["prefixes"]
        separators = prompt_data_first["separators"]

    if prefixes is not None and separators is not None:
        dummy_gt_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer, prefixes=prefixes, separators=separators, model_config=model_config)
    else:
        dummy_gt_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer, model_config=model_config)

    # If the model already prepends a bos token by default, we don't want to add one
    prepend_bos = False if model_config['prepend_bos'] else True

    if last_token_only:
        indirect_effect = torch.zeros(n_trials,model_config['n_layers'], model_config['n_heads'])
    else:
        indirect_effect = torch.zeros(n_trials,model_config['n_layers'], model_config['n_heads'],10) # have 10 classes of tokens

    if fixed_trials is None:
        if filter_set is None:
            filter_set = np.arange(len(dataset['valid']))

        for i in tqdm(range(n_trials), total=n_trials):
            word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_shots, replace=False)]
            word_pairs_test = dataset['valid'][np.random.choice(filter_set,n_test_examples, replace=False)]
            if prefixes is not None and separators is not None:
                prompt_data_random = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, shuffle_labels=True, 
                                                               prepend_bos_token=prepend_bos, prefixes=prefixes, separators=separators)
            else:
                prompt_data_random = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, 
                                                               shuffle_labels=True, prepend_bos_token=prepend_bos)
            
            if dump_cfg is not None:
                dump_cfg["trial_idx"] = i
            ind_effects = activation_replacement_per_class_intervention(prompt_data=prompt_data_random, 
                                                                        avg_activations = mean_activations, 
                                                                        dummy_labels=dummy_gt_labels, 
                                                                        model=model, model_config=model_config, tokenizer=tokenizer, 
                                                                        last_token_only=last_token_only,
                                                                        dump_cfg=dump_cfg)
            indirect_effect[i] = ind_effects.squeeze()
    else:
        n_trials_to_use = min(n_trials, len(trials_list))
        for i, (prompt_data, _, target_first_token_id) in enumerate(iter_fixed_trials(fixed_trials, mode="corrupted")):
            if i >= n_trials_to_use:
                break
            if dump_cfg is not None:
                dump_cfg["trial_idx"] = i
            ind_effects = activation_replacement_per_class_intervention(
                prompt_data=prompt_data,
                avg_activations=mean_activations,
                dummy_labels=dummy_gt_labels,
                model=model,
                model_config=model_config,
                tokenizer=tokenizer,
                last_token_only=last_token_only,
                token_id_of_interest=target_first_token_id,
                dump_cfg=dump_cfg,
            )
            indirect_effect[i] = ind_effects.squeeze()

    return indirect_effect


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=False)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save indirect effect to', type=str, required=False, default='../results')
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type =int, required=False, default=10)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over", type=int, required=False, default=25)
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mean_activations_path', help='Path to mean activations file used for intervention', required=False, type=str, default=None)
    parser.add_argument('--last_token_only', help='Whether to compute indirect effect for heads at only the final token position, or for all token classes', required=False, type=bool, default=True)
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})    
    parser.add_argument('--revision', help='Specify model checkpoints for pythia or olmo models', type=str, required=False, default=None)
    parser.add_argument('--fixed_trials_path', help='Optional fixed_trials.json path', type=str, required=False, default=None)
    parser.add_argument('--dump_trial_metrics_jsonl', help='Optional JSONL path for trial metrics dump', type=str, required=False, default=None)
    parser.add_argument('--dump_max_trials', help='Max trials to dump (default: all)', type=int, required=False, default=-1)
    parser.add_argument('--dump_include_prompt', help='Include prompt tail repr in dump (1/0)', type=int, required=False, default=1)
    parser.add_argument('--dump_layer', help='Filter dump to a layer (dump only)', type=int, required=False, default=None)
    parser.add_argument('--dump_head', help='Filter dump to a head (dump only)', type=int, required=False, default=None)
    parser.add_argument('--debug_prompt_check', help='Print fixed_trials prompt/token debug and exit', type=int, required=False, default=0)
    parser.add_argument('--debug_n', help='Number of fixed_trials to print in debug check', type=int, required=False, default=3)
        
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name
    root_data_dir = args.root_data_dir
    save_path_root = f"{args.save_path_root}/{dataset_name}"
    seed = args.seed
    n_shots = args.n_shots
    n_trials = args.n_trials
    test_split = args.test_split
    device = args.device
    mean_activations_path = args.mean_activations_path
    last_token_only = args.last_token_only
    prefixes = args.prefixes
    separators = args.separators
    fixed_trials_path = args.fixed_trials_path
    dump_trial_metrics_jsonl = args.dump_trial_metrics_jsonl
    dump_max_trials = args.dump_max_trials
    dump_include_prompt = args.dump_include_prompt
    dump_layer = args.dump_layer
    dump_head = args.dump_head
    debug_prompt_check = args.debug_prompt_check
    debug_n = args.debug_n


    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device, revision=args.revision)

    set_seed(seed)

    fixed_trials = None
    if fixed_trials_path is not None:
        print("Loading Fixed Trials")
        fixed_trials = load_fixed_trials(fixed_trials_path)
    if debug_prompt_check:
        if fixed_trials is None:
            raise ValueError("debug_prompt_check requires --fixed_trials_path")
        trials = fixed_trials.get("trials", [])
        if not trials:
            raise ValueError("No trials found in fixed_trials.json")
        add_special_tokens = True
        debug_lines = []
        debug_lines.append("[CHECK] tokenizer")
        debug_lines.append(
            "  add_special_tokens={} prepend_bos={} bos_token={} bos_id={}".format(
                add_special_tokens,
                model_config.get("prepend_bos"),
                repr(tokenizer.bos_token),
                tokenizer.bos_token_id,
            )
        )
        for trial_idx, trial in enumerate(trials[: max(0, int(debug_n))]):
            prompt_string = trial.get("corrupted_prompt_str")
            target_id = trial.get("target_first_token_id")
            target_str = trial.get("target_str")
            answer_ids = trial.get("answer_ids") or []
            if prompt_string is None or target_id is None:
                raise ValueError(f"Missing prompt/target in trial {trial_idx}")
            enc = tokenizer(prompt_string, add_special_tokens=add_special_tokens)
            ids = enc["input_ids"]
            ids_len = len(ids)
            ids_first5 = ids[:5]
            ids_last10 = ids[-10:]
            toks_last10 = [repr(tokenizer.decode([i])) for i in ids_last10]
            if answer_ids:
                answer_ids_first = answer_ids[0]
            else:
                answer_ids_first = tokenizer(
                    target_str, add_special_tokens=False
                )["input_ids"][0]
            debug_lines.append(f"[CHECK] trial={trial_idx}")
            debug_lines.append(f"  prompt_repr={repr(prompt_string)}")
            debug_lines.append(
                f"  prompt_head80={repr(prompt_string[:80])} prompt_tail80={repr(prompt_string[-80:])}"
            )
            debug_lines.append(
                "  add_special_tokens={} prepend_bos={} bos_token={} bos_id={}".format(
                    add_special_tokens,
                    model_config.get("prepend_bos"),
                    repr(tokenizer.bos_token),
                    tokenizer.bos_token_id,
                )
            )
            debug_lines.append(f"  ids_len={ids_len}")
            debug_lines.append(
                f"  ids_first5={ids_first5} ids_last10={ids_last10}"
            )
            debug_lines.append(f"  toks_last10={toks_last10}")
            debug_lines.append(f"  target_str_repr={repr(target_str)}")
            debug_lines.append(
                "  target_id={} target_tok={}".format(
                    target_id,
                    repr(tokenizer.convert_ids_to_tokens(target_id)),
                )
            )
            match = target_id == answer_ids_first
            debug_lines.append(
                f"  answer_ids_first={answer_ids_first} match={match}"
            )
        for line in debug_lines:
            print(line, flush=True)
        os.makedirs(os.path.join("runs", "stepd_fixed"), exist_ok=True)
        debug_path = os.path.join("runs", "stepd_fixed", "paper_prompt_check.txt")
        with open(debug_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(debug_lines) + "\n")
        raise SystemExit(0)

    if dataset_name is None:
        raise ValueError("Missing --dataset_name")

    # Load the dataset
    print("Loading Dataset")
    dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    # Load or Re-Compute Mean Activations
    if mean_activations_path is not None and os.path.exists(mean_activations_path):
        mean_activations = torch.load(mean_activations_path)
    elif mean_activations_path is None and os.path.exists(f'{save_path_root}/{dataset_name}_mean_head_activations.pt'):
        mean_activations_path = f'{save_path_root}/{dataset_name}_mean_head_activations.pt'
        mean_activations = torch.load(mean_activations_path)
    else:
        print("Computing Mean Activations")
        mean_activations = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, 
                                                     n_icl_examples=n_shots, N_TRIALS=n_trials, prefixes=prefixes, separators=separators)
        torch.save(mean_activations, f'{save_path_root}/{dataset_name}_mean_head_activations.pt')

    dump_handle = None
    dump_cfg = None
    if dump_trial_metrics_jsonl:
        dump_dir = os.path.dirname(dump_trial_metrics_jsonl)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        dump_handle = open(dump_trial_metrics_jsonl, "w", encoding="utf-8")
        expected_keys = [
            "trial_idx",
            "layer",
            "head",
            "target_id",
            "target_token",
            "seed_global",
            "shuffle_labels",
            "shuffle_derangement",
            "n_icl_examples",
            "demo_perm",
            "demo_fixed_points",
            "demo_outputs_before",
            "demo_outputs_after",
            "p_base",
            "p_patch",
            "delta_p",
            "logit_base",
            "logit_patch",
            "delta_logit",
            "logprob_base",
            "logprob_patch",
            "delta_logprob",
            "prompt_tail_repr",
            "prompt_ends_with_space",
            "slot_idx",
            "seq_token_idx",
            "ids_len",
            "ids_last10",
        ]
        dump_cfg = {
            "handle": dump_handle,
            "max_trials": None if dump_max_trials is None or dump_max_trials < 0 else int(dump_max_trials),
            "include_prompt": bool(dump_include_prompt),
            "dump_layer": dump_layer,
            "dump_head": dump_head,
            "expected_keys": expected_keys,
            "seed_global": seed,
            "shuffle_labels": None,
            "shuffle_derangement": None,
            "n_icl_examples": n_shots,
        }

    print("Computing Indirect Effect")
    indirect_effect = compute_indirect_effect(dataset, mean_activations, model=model, model_config=model_config, tokenizer=tokenizer, 
                                              n_shots=n_shots, n_trials=n_trials, last_token_only=last_token_only, prefixes=prefixes, separators=separators,
                                              fixed_trials=fixed_trials, dump_cfg=dump_cfg)
    if dump_handle is not None:
        dump_handle.close()

    # Write args to file
    args.save_path_root = save_path_root
    args.mean_activations_path = mean_activations_path
    with open(f'{save_path_root}/indirect_effect_args.txt', 'w') as arg_file:
        json.dump(args.__dict__, arg_file, indent=2)

    torch.save(indirect_effect, f'{save_path_root}/{dataset_name}_indirect_effect.pt')

    
