import os, json
import torch, numpy as np
import argparse

# Include prompt creation helper functions
try:
    from src.utils.prompt_utils import *
    from src.utils.intervention_utils import *
    from src.utils.model_utils import *
    from src.utils.extract_utils import *
except ImportError:
    from utils.prompt_utils import *
    from utils.intervention_utils import *
    from utils.model_utils import *
    from utils.extract_utils import *




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save mean activations to', type=str, required=False, default='../results')
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", required=False, default=10)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over", required=False, default=100)
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)
    parser.add_argument('--device', help='Device to run on', required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})    
    parser.add_argument('--revision', help='Specify model checkpoints for pythia or olmo models', type=str, required=False, default=None)
    parser.add_argument('--fixed_trials_path', help='Optional fixed_trials.json path', type=str, required=False, default=None)
        
    
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
    prefixes = args.prefixes
    separators = args.separators
    fixed_trials_path = args.fixed_trials_path
    

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device, revision=args.revision)

    if fixed_trials_path is None:
        set_seed(seed)
        # Load the dataset
        print("Loading Dataset")
        dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)

        print("Computing Mean Activations")
        mean_activations = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, 
                                                     n_icl_examples=n_shots, N_TRIALS=n_trials, prefixes=prefixes, separators=separators)
        dummy_labels = None
    else:
        print("Loading Fixed Trials")
        with open(fixed_trials_path, "r") as fixed_file:
            fixed_trials = json.load(fixed_file)
        trials_list = fixed_trials.get("trials", [])
        if not trials_list:
            raise ValueError("No trials found in fixed_trials.json")

        if n_trials is not None:
            n_trials_to_use = min(int(n_trials), len(trials_list))
        else:
            n_trials_to_use = len(trials_list)

        print("Computing Mean Activations (Fixed Trials)")
        mean_activations, dummy_labels = get_mean_head_activations_from_fixed_trials(
            fixed_trials=fixed_trials,
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            mode="clean",
            n_use=n_trials_to_use,
            prefixes=prefixes,
            separators=separators,
        )
        query_pred_index = next(
            (i for i, (_, label) in enumerate(dummy_labels) if label == "query_predictive_token"),
            None,
        )
        if query_pred_index is not None:
            print(f"QUERY_PRED(slot='query_predictive_token') index = {query_pred_index}")
        else:
            print("QUERY_PRED(slot='query_predictive_token') index = <not found>")

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    # Write args to file
    args.save_path_root = save_path_root # update for logging
    with open(f'{save_path_root}/mean_head_activation_args.txt', 'w') as arg_file:
        json.dump(args.__dict__, arg_file, indent=2)
    
    if fixed_trials_path is None:
        mean_path = f'{save_path_root}/{dataset_name}_mean_head_activations.pt'
    else:
        mean_path = f'{save_path_root}/{dataset_name}_mean_head_activations_FIXED.pt'
        dummy_labels_path = f'{save_path_root}/{dataset_name}_dummy_labels.json'
        with open(dummy_labels_path, "w") as labels_file:
            def _json_safe_label(value):
                if isinstance(value, np.integer):
                    return int(value)
                return value
            labels_payload = [[_json_safe_label(v) for v in item] for item in dummy_labels]
            json.dump({"labels": labels_payload}, labels_file, indent=2)

    torch.save(mean_activations, mean_path)
    
