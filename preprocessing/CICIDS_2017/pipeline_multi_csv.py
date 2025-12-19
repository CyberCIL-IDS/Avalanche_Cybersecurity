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
    Carica più CSV, pulisce i nomi delle colonne e rimuove valori Infinito/NaN.
    """
    logging.info(f"Caricamento di {len(file_list)} CSV...")
    df_list = []
    
    for f in file_list:
        try:
            temp_df = pd.read_csv(f, sep=',')
            # 1. FIX: Rimuovi spazi dai nomi colonne (es. " Label" -> "Label")
            temp_df.columns = temp_df.columns.str.strip()
            df_list.append(temp_df)
        except Exception as e:
            logging.error(f"Errore caricamento file {f}: {e}")

    df = pd.concat(df_list, ignore_index=True)
    
    # 2. FIX CRITICO PER CICIDS2017: Gestione Infinity e NaN
    # Molte colonne (es. Flow Packets/s) contengono "Infinity" che fa esplodere PyTorch.
    initial_shape = df.shape
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    logging.info(f"Dataset pulito: {initial_shape[0] - df.shape[0]} righe con errori rimosse.")
    logging.info(f"Dataset finale: {df.shape[0]} righe, {df.shape[1]} colonne")
    
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
    target_count=1500,
    sample_fraction=0.15  # usa il 20% dei dati 
):
    """
    Preprocessing completo con Split Stratificato e Sottocampionamento opzionale.
    """

    # 1. UNISCI E PULISCI I DATI
    all_files = train_files + test_files
    df_total = load_multiple_csv(all_files)

    # 2. CLIP OUTLIERS
    df_total = clip_outliers(df_total, clip_percentile)

    # 3. ENCODE LABELS (Globale)
    df_total = df_total[df_total[label_column].notna()]
    
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df_total[label_column].astype(str))
    
    print(f"DEBUG: Classi totali trovate ({len(label_encoder.classes_)}): {label_encoder.classes_}")

    # 4. PREPARE COLUMN GROUPS
    numerical_cols = [
        c for c in df_total.columns 
        if c not in categorical_cols + [label_column]
    ]

    # 5. BUILD PREPROCESSOR & TRANSFORM X
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    X_all = preprocessor.fit_transform(df_total.drop(columns=[label_column]))

    if sample_fraction < 1.0:
        print(f"--- SOTTOCAMPIONAMENTO ATTIVO: Riduzione dataset al {sample_fraction*100}% ---")
        print(f"Dimensione originale: {X_all.shape[0]}")
        
        X_all, _, y_all, _ = train_test_split(
            X_all, y_all, 
            train_size=sample_fraction,  
            stratify=y_all,              
            random_state=42
        )
        print(f"Nuova dimensione ridotta: {X_all.shape[0]}")

    # 6. SPLIT TRAIN / TEST STRATIFICATO
    print("Eseguendo split stratificato 70/30...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, 
        test_size=0.3, 
        random_state=42, 
        stratify=y_all 
    )

    # 7. BILANCIAMENTO (Solo Train)
    if balance_classes:
        print(f"Bilanciamento classi nel Training Set (target={target_count})...")
        X_train, y_train = balance_samples(X_train, y_train, target_count)

    # 8. CONVERSIONE IN TENSORI (Gestione sparsa/densa)
    if hasattr(X_train, "toarray"):
        X_train_np = X_train.toarray()
        X_test_np = X_test.toarray()
    else:
        X_train_np = X_train
        X_test_np = X_test

    # Convertiamo in float32 e long
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 9. SALVATAGGIO
    save_object(preprocessor, output_preprocessor_path)
    save_object(label_encoder, output_label_encoder_path)

    return (
        {"X": X_train_tensor, "y": y_train_tensor},
        {"X": X_test_tensor, "y": y_test_tensor},
        label_encoder,
        preprocessor
    )


def balance_samples(X, y, target_count=1500):
    classes, counts = np.unique(y, return_counts=True)
    # print(f"DEBUG: Distribuzione pre-balance: {dict(zip(classes, counts))}")

    X_balanced = []
    y_balanced = []

    for c in classes:
        idx = np.where(y == c)[0]
        cur = len(idx)
        
        if cur == 0: continue

        if cur > target_count:
            # Undersampling: prendine solo target_count
            new_idx = np.random.choice(idx, target_count, replace=False)
        else:
            # Oversampling: duplica finché non arrivi a target_count
            new_idx = np.random.choice(idx, target_count, replace=True)

        X_balanced.append(X[new_idx])
        y_balanced.append(y[new_idx])

    if isinstance(X, (np.ndarray, torch.Tensor)):
        X_final = np.vstack(X_balanced)
        y_final = np.hstack(y_balanced)
    else:
        # Gestione matrici sparse
        X_final = sparse.vstack(X_balanced)
        y_final = np.hstack(y_balanced)

    return X_final, y_final