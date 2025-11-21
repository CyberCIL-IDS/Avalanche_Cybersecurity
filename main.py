import logging
import yaml
from preprocessing.pipeline import prepare_dataset
from utils.benchmark import create_benchmark
from utils.training import train
from utils.plotting import plot_metrics


# def setup_logging():
#     logging.basicConfig(
#         level=print,
#         format="%(asctime)s [%(levelname)s] %(message)s"
#     )

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # setup_logging()
    cfg = load_config()

    train_ds, test_ds, label_encoder = prepare_dataset(cfg) #preprocessor
    
    input_size = train_ds["X"].shape[1]
    n_classes = len(label_encoder.classes_)

    print("=== CREATING BENCHMARK ===")
    benchmark = create_benchmark(train_ds, test_ds)

    print(f"Train shape: {train_ds['X'].shape}, Test shape: {test_ds['X'].shape}")
    print("Dataset ready for training")

    print("=== TRAINING WITH __ ===")
    experiences, metrics = train(
        benchmark=benchmark,
        input_size=input_size,
        n_classes=n_classes
    )


    #TODO: catastrophic forgetting


    acc_exp = []
    for i in range(experiences):
        key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{i:03d}"
        if key in metrics:
            acc_exp.append(metrics[key])

    # Forgetting per experience
    forget_exp = [0.0]
    for i in range(experiences):
        key = f"ExperienceForgetting/eval_phase/test_stream/Task000/Exp{i:03d}"
        if key in metrics:
            forget_exp.append(metrics[key])


    exp_ids = list(range(1, experiences + 1))

    print("=== PLOTTING RESULTS ===")
    plot_metrics(exp_ids, acc_exp, forget_exp)


if __name__ == "__main__":
    main()
