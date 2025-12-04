import pandas as pd
import logging
import numpy as np
import torch
from scipy import sparse

from preprocessing.feature_engineering import clip_outliers
from preprocessing.encoder import build_preprocessor, save_object
from sklearn.preprocessing import LabelEncoder


def load_multiple_csv(file_list):
    """
    Carica e unisce più file CSV in un unico DataFrame.
    """
    logging.info(f"Caricamento di {len(file_list)} CSV...")
    df_list = [pd.read_csv(f) for f in file_list]
    df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Dataset combinato: {df.shape[0]} righe, {df.shape[1]} colonne")
    return df


def prepare_dataset_multi_csv(
    train_files,
    test_files,
    categorical_cols,
    label_column,
    clip_percentile,
    output_preprocessor_path,
    output_label_encoder_path,
    balance_classes=False,
    target_count=1500
):
    """
    Preprocessing completo su dataset proveniente da più file CSV.
    """

    # -----------------------------
    # 1. CARICAMENTO MULTI-CSV
    # -----------------------------
    df_train = load_multiple_csv(train_files)
    df_test = load_multiple_csv(test_files)

    # -----------------------------
    # 2. CLIP OUTLIERS
    # -----------------------------
    df_train = clip_outliers(df_train, clip_percentile)
    df_test = clip_outliers(df_test, clip_percentile)

    # -----------------------------
    # 3. ENCODE LABELS
    # -----------------------------
    df_train[label_column] = df_train[label_column].fillna("Unknown")
    df_test[label_column] = df_test[label_column].fillna("Unknown")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train[label_column])
    y_test = label_encoder.transform(df_test[label_column])

    # -----------------------------
    # 4. PREPARE COLUMN GROUPS
    # -----------------------------
    numerical_cols = [
        c for c in df_train.columns 
        if c not in categorical_cols + [label_column]
    ]

    # -----------------------------
    # 5. BUILD PREPROCESSOR
    # -----------------------------
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)

    X_train = preprocessor.fit_transform(df_train.drop(columns=[label_column]))
    X_test  = preprocessor.transform(df_test.drop(columns=[label_column]))

    # -----------------------------
    # 6. SALVATAGGIO PIPELINE
    # -----------------------------
    save_object(preprocessor, output_preprocessor_path)
    save_object(label_encoder, output_label_encoder_path)

    # -----------------------------
    # 7. OPTIONAL: BALANCE CLASSES
    # -----------------------------
    if balance_classes:
        X_train, y_train = balance_samples(X_train, y_train, target_count)

    # -----------------------------
    # 8. CONVERSIONE IN TENSORI
    # -----------------------------
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return (
        {"X": X_train_tensor, "y": y_train_tensor},
        {"X": X_test_tensor, "y": y_test_tensor},
        label_encoder
    )


# -----------------------------
# 9. BILANCIAMENTO CLASSI (stessa funzione che hai già)
# -----------------------------
def balance_samples(X, y, target_count=1500):
    classes, counts = np.unique(y, return_counts=True)
    print(f"DEBUG: Distribuzione originale: {dict(zip(classes, counts))}")

    X_balanced = []
    y_balanced = []

    for c in classes:
        idx = np.where(y == c)[0]
        cur = len(idx)

        if cur > target_count:
            new_idx = np.random.choice(idx, target_count, replace=False)
        else:
            new_idx = np.random.choice(idx, target_count, replace=True)

        X_balanced.append(X[new_idx])
        y_balanced.append(y[new_idx])

    if isinstance(X, (np.ndarray, torch.Tensor)):
        X_final = np.vstack(X_balanced)
        y_final = np.hstack(y_balanced)
    else:
        X_final = sparse.vstack(X_balanced)
        y_final = np.hstack(y_balanced)

    print(f"DEBUG: Dataset bilanciato a {target_count} campioni/classe.")

    return X_final, y_final
