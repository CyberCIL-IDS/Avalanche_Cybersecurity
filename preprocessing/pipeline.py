import logging
import numpy as np
import torch
from preprocessing.loader import load_dataset
from preprocessing.feature_engineering import add_unsw_features
from preprocessing.encoder import build_preprocessor, save_object
from sklearn.preprocessing import LabelEncoder

def clip_outliers(df, percentile):
    for col in df.select_dtypes(include=[np.number]).columns:
        upper = df[col].quantile(percentile)
        df[col] = df[col].clip(upper=upper)
    return df

def balance_samples(X, y):
    # semplice balance con undersampling
    classes, counts = np.unique(y, return_counts=True)
    min_count = min(counts)

    idx = np.hstack([np.random.choice(np.where(y == c)[0], min_count, replace=False)
                     for c in classes])

    return X[idx], y[idx]

def prepare_dataset(cfg):
    # ---------- LOAD TRAIN ----------
    df_train = load_dataset(cfg["dataset"]["train_csv"])
    df_test  = load_dataset(cfg["dataset"]["test_csv"])

    # ---------- DROP COLUMNS ----------
    for col in cfg["preprocessing"]["drop_columns"]:
        if col in df_train:
            df_train = df_train.drop(columns=[col])
        if col in df_test:
            df_test = df_test.drop(columns=[col])

    # ---------- REPLACE UNKNOWN ----------
    df_train["service"] = df_train["service"].replace("-", "Unknown")
    df_test["service"]  = df_test["service"].replace("-", "Unknown")

    # ---------- FEATURE ENGINEERING ----------
    if cfg["preprocessing"]["add_features"]:
        df_train = add_unsw_features(df_train)
        df_test  = add_unsw_features(df_test)

    # ---------- CLIP OUTLIERS ----------
    p = cfg["preprocessing"]["clip_percentile"]
    df_train = clip_outliers(df_train, p)
    df_test  = clip_outliers(df_test, p)

    # ---------- ENCODE LABELS ----------
    label_col = cfg["preprocessing"]["label_column"]
    df_train[label_col] = df_train[label_col].fillna("Unknown")
    df_test[label_col]  = df_test[label_col].fillna("Unknown")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train[label_col])
    y_test  = label_encoder.transform(df_test[label_col])

    # ---------- COLUMN GROUPS ----------
    categorical = cfg["preprocessing"]["categorical"]
    numerical = [c for c in df_train.columns if c not in categorical + [label_col]]

    # ---------- PREPROCESSOR ----------
    preprocessor = build_preprocessor(canonical_cols := categorical, numerical)

    # FIT sul train → TRANSFORM su test
    X_train = preprocessor.fit_transform(df_train.drop(columns=[label_col]))
    X_test  = preprocessor.transform(df_test.drop(columns=[label_col]))

    # OPTIONAL: BALANCING
    if cfg["preprocessing"]["balance_classes"]:
        X_train, y_train = balance_samples(X_train, y_train)

    # ---------- CONVERSIONE TENSORI ----------
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # ---------- SALVATAGGIO ----------
    # save_object(preprocessor, cfg["output"]["save_preprocessor"])
    # save_object(label_encoder, cfg["output"]["save_label_encoder"])

    return (
        {"X": X_train_tensor, "y": y_train_tensor},
        {"X": X_test_tensor,  "y": y_test_tensor},
        #preprocessor,
        label_encoder
    )
