from functools import partial
from preprocessing.UNSW_NB15.preprocessing_UNSW_NB15 import prepare_UNSW_NB15
from preprocessing.CICIDS_2017.preprocessing_CICIDS_2017 import preprocessing_CICIDS
from utils.benchmark import create_benchmark
from utils.optuna_custom import objective
from utils.training import train
from utils.plotting import plot_metrics
from utils.config_loader import load_config
import logging
import time
import torch
import optuna


# def setup_logging():
#     logging.basicConfig(
#         level=print,
#         format="%(asctime)s [%(levelname)s] %(message)s"
#     )

def hyperparameter_optimization():
    # setup_logging()
    cfg = load_config()
    strategy = cfg["benchmark"]["strategy"]

    dataset = cfg["dataset"]["mode"]
    
    print(f"=== DATASET SELECTED: {dataset} ===")


    if dataset == "UNSW_NB15":
        train_ds, test_ds, label_encoder = prepare_UNSW_NB15(cfg)
    else:
        train_ds, test_ds, label_encoder = preprocessing_CICIDS()

    input_size = train_ds["X"].shape[1]
    n_classes = len(label_encoder.classes_)

    print("=== CREATING BENCHMARK ===")
    mode = cfg["benchmark"].get("mode", "single")
    param = cfg["benchmark"].get("param", None)
    benchmark = create_benchmark(train_ds, test_ds, mode, param)

    print(f"Mode: {mode}, Param: {param}")
    

    print("=== OPTUNA ===")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        partial(
            objective,
            device="cuda" if torch.cuda.is_available() else "cpu",
            strategy_type=strategy,
            input_size=input_size,
            n_classes=n_classes,
            use_sigmoid_activation=True if strategy == "ICaRL" else False,
            benchmark=benchmark
        )
    )

    print("BEST PARAMS:", study.best_params)
    print("BEST VALUE:", study.best_value)



if __name__ == "__main__":
    hyperparameter_optimization()
