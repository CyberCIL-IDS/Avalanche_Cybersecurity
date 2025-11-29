import pandas as pd
import torch
from preprocessing.feature_engineering import add_unsw_features
from preprocessing.encoder import build_preprocessor, save_object
import numpy as np
import random

def clip_outliers(df, percentile):
    for col in df.select_dtypes(include=[np.number]).columns:
        upper = df[col].quantile(percentile)
        df[col] = df[col].clip(upper=upper)
    return df

def preprocess_predict_sample(cfg, data_path="datasets/UNSW_NB15_training-set.csv", n_samples=10):
    """
    Estrae n_samples righe random dal dataset, le preprocessa e ritorna il tensore pronto per il modello.
    """
    df = pd.read_csv(data_path)

    # Estrai n righe random
    df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    # Drop target column se presente
    if "attack_cat" in df.columns:
        df = df.drop(columns=["attack_cat"])

    # Drop altre colonne non feature
    for col in cfg["preprocessing"]["drop_columns"]:
        if col in df:
            df = df.drop(columns=[col])

    # Replace unknown services
    if "service" in df.columns:
        df["service"] = df["service"].replace("-", "Unknown")

    # Feature engineering
    if cfg["preprocessing"]["add_features"]:
        df = add_unsw_features(df)

    # Clip outliers
    df = clip_outliers(df, cfg["preprocessing"]["clip_percentile"])

    # Column grouping
    categorical = cfg["preprocessing"]["categorical"]
    numerical = [c for c in df.columns if c not in categorical]

    # Build preprocessor
    preprocessor = build_preprocessor(categorical, numerical)
    X = preprocessor.fit_transform(df)

    # salva il preprocessor
    save_object(preprocessor, "preprocessor_sample.pkl")

    # Ritorna tensore PyTorch
    return torch.tensor(X, dtype=torch.float32), df, preprocessor

