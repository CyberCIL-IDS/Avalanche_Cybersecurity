import logging
import numpy as np
import torch
from scipy import sparse
from preprocessing.loader import load_dataset
from preprocessing.encoder import build_preprocessor, save_object
from sklearn.preprocessing import LabelEncoder
from preprocessing.feature_engineering import clip_outliers, add_unsw_features

import numpy as np
from scipy import sparse
import torch

def balance_samples(X, y, target_count=5000, majority_multiplier=4):
    """
    Bilanciamento Ibrido Intelligente:
    - Identifica la classe maggioritaria (es. Normal/Benign).
    - Per la classe maggioritaria: Usa un target più alto (target_count * majority_multiplier).
    - Per le altre classi: Usa target_count standard.
    
    Questo preserva la varianza della classe "Normal" evitando che collassi.
    """
    classes, counts = np.unique(y, return_counts=True)
    
    # 1. Identifica automaticamente la classe più numerosa (es. Normal)
    major_class_idx = np.argmax(counts)
    major_class = classes[major_class_idx]
    
    print(f"DEBUG: Distribuzione originale: {dict(zip(classes, counts))}")
    print(f"DEBUG: Classe maggioritaria rilevata: {major_class} (moltiplicatore x{majority_multiplier})")
    
    X_balanced = []
    y_balanced = []

    for c in classes:
        idx = np.where(y == c)[0]
        current_count = len(idx)
        
        # 2. Logica differenziata per il limite di campioni
        if c == major_class:
            # Per la classe "Normal", teniamo molti più campioni (es. 20.000 invece di 5.000)
            current_target = target_count * majority_multiplier
        else:
            # Per gli attacchi, usiamo il target standard (es. 5.000)
            current_target = target_count
        
        # 3. Campionamento (Undersampling o Oversampling)
        if current_count > current_target:
            # Se ne abbiamo troppi, ne prendiamo un sottoinsieme (Undersampling)
            random_idx = np.random.choice(idx, current_target, replace=False)
        else:
            # Se ne abbiamo pochi, li duplichiamo (Oversampling)
            random_idx = np.random.choice(idx, current_target, replace=True)
            
        X_balanced.append(X[random_idx])
        y_balanced.append(y[random_idx])

    # Ricostruzione dataset
    if isinstance(X, np.ndarray) or isinstance(X, torch.Tensor):
        # Se è tensore, vstack di numpy potrebbe richiedere conversione, 
        # ma spesso funziona se il tensore è su CPU. Per sicurezza:
        if isinstance(X_balanced[0], torch.Tensor):
            X_final = torch.vstack(X_balanced) # Usa torch.vstack per i tensori
            y_final = torch.hstack(y_balanced)
        else:
            X_final = np.vstack(X_balanced)
            y_final = np.hstack(y_balanced)
    else:
        # Gestione matrici sparse
        X_final = sparse.vstack(X_balanced)
        y_final = np.hstack(y_balanced)

    print(f"DEBUG: Dataset bilanciato. Normal: ~{target_count * majority_multiplier}, Attacchi: ~{target_count}")
    return X_final, y_final

def prepare_UNSW_NB15(cfg):
    df_train = load_dataset(cfg["dataset"]["train_csv"])
    df_test  = load_dataset(cfg["dataset"]["test_csv"])

    for col in cfg["preprocessing"]["drop_columns"]:
        if col in df_train:
            df_train = df_train.drop(columns=[col])
        if col in df_test:
            df_test = df_test.drop(columns=[col])

    df_train["service"] = df_train["service"].replace("-", "Unknown")
    df_test["service"]  = df_test["service"].replace("-", "Unknown")

    if cfg["preprocessing"]["add_features"]:
        df_train = add_unsw_features(df_train)
        df_test  = add_unsw_features(df_test)

    p = cfg["preprocessing"]["clip_percentile"]
    df_train = clip_outliers(df_train, p)
    df_test  = clip_outliers(df_test, p)

    label_col = cfg["preprocessing"]["label_column"]
    df_train[label_col] = df_train[label_col].fillna("Unknown")
    df_test[label_col]  = df_test[label_col].fillna("Unknown")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train[label_col])
    y_test  = label_encoder.transform(df_test[label_col])

    categorical = cfg["preprocessing"]["categorical"]
    numerical = [c for c in df_train.columns if c not in categorical + [label_col]]

    preprocessor = build_preprocessor(canonical_cols := categorical, numerical)

    X_train = preprocessor.fit_transform(df_train.drop(columns=[label_col]))
    X_test  = preprocessor.transform(df_test.drop(columns=[label_col]))

    if cfg["preprocessing"]["balance_classes"]:
        X_train, y_train = balance_samples(X_train, y_train)

    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    save_object(preprocessor, cfg["output"]["save_preprocessor"])
    save_object(label_encoder, cfg["output"]["save_label_encoder"])

    return (
        {"X": X_train_tensor, "y": y_train_tensor},
        {"X": X_test_tensor,  "y": y_test_tensor},
        label_encoder,
        preprocessor
    )
