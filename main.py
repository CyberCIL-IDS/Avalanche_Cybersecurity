from preprocessing.UNSW_NB15.preprocessing_UNSW_NB15 import prepare_UNSW_NB15
from utils import optuna_custom
from utils.benchmark import create_benchmark
from utils.optuna_custom import objective
from utils.training import train
from utils.plotting import plot_metrics
from utils.config_loader import load_config
from utils.parsing_best_params import parse_json
from preprocessing.CICIDS_2017.preprocessing_CICIDS_2017 import preprocessing_CICIDS

def main():
    cfg = load_config()
    dataset = cfg["dataset"]["mode"]
    print(f"=== DATASET SELECTED: {dataset} ===")
    if dataset == "UNSW_NB15":
        train_ds, test_ds, label_encoder, preprocessor = prepare_UNSW_NB15(cfg)
    else:
        train_ds, test_ds, label_encoder, preprocessor = preprocessing_CICIDS()

    input_size = train_ds["X"].shape[1]
    n_classes = len(label_encoder.classes_)
    best_params_path = cfg["benchmark"]["best_params_path"]

    all_results = parse_json(best_params_path) 


    if not cfg["all_mode_training"].get("enable", False):

        strategy = cfg["benchmark"]["strategy"]
        mode = cfg["benchmark"].get("mode", "single")
        param = cfg["benchmark"].get("param", None)
        
        print("=== CREATING BENCHMARK ===")
        benchmark = create_benchmark(train_ds, test_ds, mode, param)

        print(f"Mode: {mode}, Param: {param}")

        best_params = None
        for entry in all_results:
            meta = entry["meta"]

            if (meta["strategy"] == strategy and 
                meta["mode"] == mode and 
                meta["param"] == param):
                
                best_params = entry["best_hyperparameters"]
                print(f">>> Parametri ottimali trovati: {best_params}")
                break

        if best_params is None:
            raise ValueError(
                f"ERRORE CRITICO: Non sono stati trovati parametri nel JSON per la configurazione:\n"
                f"Strategy: {strategy}, Mode: {mode}, Param: {param}.\n"
                "Verifica che 'output_all_p1.json' contenga questa combinazione."
            )

        print("=== TRAINING ===")
        experiences, metrics = train(
            test_ds=test_ds,
            benchmark=benchmark,
            input_size=input_size,
            n_classes=n_classes,
            dataset=dataset,
            strategy_type=strategy,
            mode=mode,
            param=param,
            model_params=best_params,
            cfg=cfg,
            label_encoder=label_encoder,
            preprocessor=preprocessor
            
        )

        print("=== PLOTTING RESULTS ===")
        plot_metrics(experiences, metrics, dataset, strategy, mode, param)
    else:
        best_params = None
        for entry in all_results:
            meta = entry["meta"]

            strategy = meta["strategy"]
            mode = meta["mode"]
            param = meta["param"]
            best_params = entry["best_hyperparameters"]

            print("=== CREATING BENCHMARK ===")
            benchmark = create_benchmark(train_ds, test_ds, mode, param)

            print("=== TRAINING ===")
            experiences, metrics = train(
                test_ds=test_ds,
                benchmark=benchmark,
                input_size=input_size,
                n_classes=n_classes,
                dataset=dataset,
                strategy_type=strategy,
                mode=mode,
                param=param,
                model_params=best_params,
                cfg=cfg,
                label_encoder=label_encoder,
                preprocessor=preprocessor
            )

            print("=== PLOTTING RESULTS ===")
            plot_metrics(experiences, metrics, dataset, strategy, mode, param)

if __name__ == "__main__":
    main()
