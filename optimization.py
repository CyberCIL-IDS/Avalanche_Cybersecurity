from functools import partial
import json
import sys
import logging
import time
import torch
import optuna
import itertools
import argparse
import warnings

# --- 1. GLOBAL FILTER (Try to catch them early) ---
warnings.filterwarnings("ignore", category=DeprecationWarning, module="avalanche.*")
warnings.filterwarnings("ignore", message="Call to deprecated function update")
# --------------------------------------------------

from preprocessing.UNSW_NB15.preprocessing_UNSW_NB15 import prepare_UNSW_NB15
from preprocessing.CICIDS_2017.preprocessing_CICIDS_2017 import preprocessing_CICIDS
from utils.benchmark import create_benchmark
from utils.optuna_custom import objective
from utils.training import train
from utils.plotting import plot_metrics
from utils.config_loader import load_config


def save_results_to_file(data_list, output_path):
    #Write on file
    try:
        with open(output_path, 'w') as output_file:
            json.dump(data_list, output_file, indent=4)
        print(f"\n[Saved] Results ({len(data_list)} entries) saved to: {output_path}")
    except Exception as e:
        print(f"\n[Error] Could not save file: {e}")

def hyperparameter_optimization(strategy, mode, param, train_ds, test_ds, input_size, n_classes, n_trials, my_pruner_string):
    print("=== CREATING BENCHMARK ===")
    benchmark = create_benchmark(train_ds, test_ds, mode, param)

    print(f"Mode: {mode}, Param: {param}")

    if my_pruner_string == "MedianPruner":
        my_pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 5,
            n_warmup_steps = 2,
            interval_steps = 1
        )
    elif my_pruner_string == "HyperbandPruner":
        my_pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,      # The smallest amount of resource (e.g., 1 task or 1 epoch)
            max_resource=10,     # The maximum resource (e.g., total number of tasks in stream)
            reduction_factor=3   # How aggressively to reduce the population (keep 1/3rd)
        )
    else:
        my_pruner = None 
    
    print("=== OPTUNA ===")
    study = optuna.create_study(direction="maximize", pruner=my_pruner)
    study.optimize(
        partial(
            objective,
            device="cuda" if torch.cuda.is_available() else "cpu",
            strategy_type=strategy,
            input_size=input_size,
            n_classes=n_classes,
            benchmark=benchmark
        ),
        n_trials=n_trials
    )

    best_params = study.best_params
    best_value = study.best_value
    trial = study.best_trial.number

    print(f"Optuna[{mode}, {param}, {strategy}] -> trial={trial}, best_value={best_value}, best_params={best_params}")
    return trial, best_params, best_value

def optimization():
    # --- 2. RUNTIME FILTER (Force silence inside the function) ---
    # This overrides any reset that might happen during imports
    warnings.simplefilter("ignore", DeprecationWarning)
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", default=None, nargs="?", help="Path to save results")
    args = parser.parse_args()

    cfg = load_config()
    strategies, modes, params = [],[],[]
    modes_best_params_list = []

    output_path = args.output_path if args.output_path else cfg["optuna"].get("output_path", "output.json")
    n_trials = cfg["optuna"]["n_trials"]
    dataset = cfg["dataset"]["mode"]
    if dataset == "UNSW_NB15":
        train_ds, test_ds, label_encoder = prepare_UNSW_NB15(cfg)
    else:
        train_ds, test_ds, label_encoder = preprocessing_CICIDS()

    print(f"=== DATASET SELECTED: {dataset} ===")
    input_size = train_ds["X"].shape[1]
    n_classes = len(label_encoder.classes_)

    if cfg["optuna"]["all"] == False:
        strategy = cfg["benchmark"]["strategy"]
        mode = cfg["benchmark"].get("mode", "single")
        param = cfg["benchmark"].get("param", None)
        strategies.append(strategy)
        modes.append(mode)
        params.append(param)
    else:
        strategies = cfg["optuna"]["strategies"]
        modes = cfg["optuna"]["modes"]
        params = cfg["optuna"]["params"]

    my_pruner_string = cfg["optuna"].get("pruner", None)
    
    #
    # -- LOOP FOR EACH STRATEGIES, MODES AND PARAMS
    #
    try:
        print("Starting optimization loop... Press CTRL+C to stop and save current progress.")
        for mode, param, strategy in itertools.product(modes, params, strategies): 
                    # Skip invalid combinations if necessary (e.g. single mode with param)
                    # if mode == "single" and param is not None: continue 

                    trial, best_value, best_params = hyperparameter_optimization(
                        strategy,
                        mode,
                        param,
                        train_ds,
                        test_ds,
                        input_size,
                        n_classes,
                        n_trials,
                        my_pruner_string
                    )
                    row = {
                        'mode': mode,
                        'param': param,
                        'strategy': strategy,
                        "trial": trial,
                        "best_value": best_value,
                        "best_params": best_params
                    }
                    modes_best_params_list.append(row)
        #end loop
    except KeyboardInterrupt:
        print("\n\n!!! EXECUTION INTERRUPTED BY USER (CTRL+C) !!!")
        print("Saving current progress before exiting...")
        save_results_to_file(modes_best_params_list, output_path)
        sys.exit(0) # Exit cleanly

    #Write on file
    save_results_to_file(modes_best_params_list, output_path)
    print(f"Optimization tuna end, result saved on: {output_path}")

if __name__ == "__main__":
    optimization()