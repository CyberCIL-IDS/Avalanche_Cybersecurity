import pandas as pd
import logging
import numpy as np
import torch
from scipy import sparse

from preprocessing.feature_engineering import clip_outliers
from preprocessing.encoder import build_preprocessor, save_object
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_multiple_csv(file_list):
    """
    Carica e unisce più file CSV in un unico DataFrame.
    """
    logging.info(f"Caricamento di {len(file_list)} CSV...")
    df_list = [pd.read_csv(f, sep = ',') for f in file_list]
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
    Carica TUTTI i file, li unisce e poi esegue uno split stratificato 
    per garantire che ogni classe sia presente sia in Train che in Test.
    """

    # 1. UNISCI TUTTE LE LISTE DI FILE
    all_files = train_files + test_files
    
    # 2. CARICA UN UNICO DATAFRAME GIGANTE
    # (Usa la tua funzione load_multiple_csv corretta con lo strip delle colonne)
    df_total = load_multiple_csv(all_files)

    # 3. CLIP OUTLIERS
    df_total = clip_outliers(df_total, clip_percentile)

    # 4. ENCODE LABELS (Globale)
    df_total[label_column] = df_total[label_column].fillna("Unknown")
    
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df_total[label_column])
    
    print(f"DEBUG: Classi totali trovate: {label_encoder.classes_}")

    # 5. PREPARE COLUMN GROUPS
    numerical_cols = [
        c for c in df_total.columns 
        if c not in categorical_cols + [label_column]
    ]

    # 6. BUILD PREPROCESSOR & TRANSFORM X
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    # Fit e Transform su tutto il dataset
    X_all = preprocessor.fit_transform(df_total.drop(columns=[label_column]))

    # 7. SPLIT TRAIN / TEST STRATIFICATO
    # Questo è il passaggio chiave: garantisce che 'Bot' finisca anche nel test set
    print("Eseguendo split stratificato 70/30...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, 
        test_size=0.3, 
        random_state=42, 
        stratify=y_all  # Fondamentale: mantiene le proporzioni delle classi
    )

    # 8. SALVATAGGIO PIPELINE
    save_object(preprocessor, output_preprocessor_path)
    save_object(label_encoder, output_label_encoder_path)

    # 9. OPTIONAL: BALANCE CLASSES (Solo sul Train!)
    if balance_classes:
        print("Bilanciamento classi nel Training Set...")
        X_train, y_train = balance_samples(X_train, y_train, target_count)

    # 10. CONVERSIONE IN TENSORI (Gestione sparsi/densi)
    if hasattr(X_train, "toarray"):
        X_train_np = X_train.toarray()
        X_test_np = X_test.toarray()
    else:
        X_train_np = X_train
        X_test_np = X_test

    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
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
