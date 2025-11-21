import logging
import yaml
from preprocessing.pipeline import prepare_dataset
from utils.benchmark import create_benchmark
from utils.training import run_training
from utils.plotting import plot_metrics


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    setup_logging()
    cfg = load_config()

    train, test, label_encoder = prepare_dataset(cfg) #preprocessor
    
    input_size = train["X"].shape[1]
    n_classes = len(label_encoder.classes_)

    logging.info("=== CREATING BENCHMARK ===")
    benchmark = create_benchmark(train, test)

    logging.info(f"Train shape: {train['X'].shape}, Test shape: {test['X'].shape}")
    logging.info("Dataset ready for training")

    logging.info("=== TRAINING WITH __ ===")
    experiences, accuracy, forgetting = run_training(
        benchmark=benchmark,
        input_size=input_size,
        n_classes=n_classes
    )

    #TODO: catastrophic forgetting
    

    logging.info("=== PLOTTING RESULTS ===")
    plot_metrics(experiences, accuracy, forgetting)



if __name__ == "__main__":
    main()
