import logging
from preprocessing.pipeline import prepare_dataset
from utils.benchmark import create_benchmark
from utils.training import train
from utils.plotting import plot_metrics
from utils.config_loader import load_config
import time


# def setup_logging():
#     logging.basicConfig(
#         level=print,
#         format="%(asctime)s [%(levelname)s] %(message)s"
#     )

def main():
    # setup_logging()
    cfg = load_config()
    strategy = cfg["strategy"]

    train_ds, test_ds, label_encoder = prepare_dataset(cfg) #preprocessor
    
    input_size = train_ds["X"].shape[1]
    n_classes = len(label_encoder.classes_)

    print("=== CREATING BENCHMARK ===")
    mode = cfg["benchmark"].get("mode", "single")
    param = cfg["benchmark"].get("param", None)
    benchmark = create_benchmark(train_ds, test_ds, mode, param)

    print(f"Mode: {mode}, Param: {param}")
    #print(f"Train shape: {train_ds['X'].shape}, Test shape: {test_ds['X'].shape}")
    #print("Dataset ready for training")

    print("=== TRAINING ===")
    experiences, metrics = train(
        benchmark=benchmark,
        input_size=input_size,
        n_classes=n_classes,
        strategy_type=strategy,
        mode=mode,
        param=param
    )

    print("=== PLOTTING RESULTS ===")
    plot_metrics(experiences, metrics, strategy, mode, param)


if __name__ == "__main__":
    main()
