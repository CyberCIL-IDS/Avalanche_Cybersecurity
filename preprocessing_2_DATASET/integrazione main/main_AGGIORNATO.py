import logging
import os
import time
from utils.config_loader import load_config

# --- NUOVO IMPORT: script di preprocessing ---
from preprocess_new_csvs import preprocess_and_save

from utils.benchmark import create_benchmark
from utils.training import train
from utils.plotting import plot_metrics

import pandas as pd
import joblib


def load_preprocessed_datasets(cfg):
    """
    Carica i file preprocessati generati da preprocess_new_csvs.py
    e restituisce i dizionari train_ds e test_ds compatibili col training.
    """

    output_dir = cfg["preprocessing"]["output_dir"]

    train_path = os.path.join(output_dir, "train_preprocessed.csv")
    test_path = os.path.join(output_dir, "test_preprocessed.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "I file preprocessati non esistono. "
            "Esegui prima 'preprocess_new_csvs.py'."
        )

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=["label"]).values
    y_train = df_train["label"].values

    X_test = df_test.drop(columns=["label"]).values
    y_test = df_test["label"].values

    # Carica encoder salvato
    label_encoder_path = cfg["output"]["save_label_encoder"]
    label_encoder = joblib.load(label_encoder_path)

    return (
        {"X": X_train, "y": y_train},
        {"X": X_test, "y": y_test},
        label_encoder
    )



def main():

    cfg = load_config()

    # ===========================================================================================
    # 1. SE PREPROCESSING È ABILITATO → ESEGUE preprocess_new_csvs.py
    # ===========================================================================================
    if cfg.get("preprocessing", {}).get("enabled", False):
        print("=== ESEGUO NUOVO PREPROCESSING ===")
        preprocess_and_save(cfg)
    else:
        print("=== PREPROCESSING DISABILITATO NEL CONFIG ===")

    # ===========================================================================================
    # 2. CARICAMENTO CSV GIÀ PREPROCESSATI
    # ===========================================================================================
    print("=== CARICO DATASET PREPROCESSATI ===")
    train_ds, test_ds, label_encoder = load_preprocessed_datasets(cfg)

    input_size = train_ds["X"].shape[1]
    n_classes = len(label_encoder.classes_)

    # ===========================================================================================
    # 3. CREAZIONE BENCHMARK AVALANCHE
    # ===========================================================================================
    print("=== CREATING BENCHMARK ===")
    strategy = cfg["benchmark"]["strategy"]
    mode = cfg["benchmark"].get("mode", "single")
    param = cfg["benchmark"].get("param", None)

    benchmark = create_benchmark(train_ds, test_ds, mode, param)
    print(f"Mode: {mode}, Param: {param}")

    # ===========================================================================================
    # 4. TRAINING
    # ===========================================================================================
    print("=== TRAINING ===")
    experiences, metrics = train(
        benchmark=benchmark,
        input_size=input_size,
        n_classes=n_classes,
        strategy_type=strategy,
        mode=mode,
        param=param,
    )

    # ===========================================================================================
    # 5. PLOT RISULTATI
    # ===========================================================================================
    print("=== PLOTTING RESULTS ===")
    plot_metrics(experiences, metrics, strategy, mode, param)


if __name__ == "__main__":
    main()
