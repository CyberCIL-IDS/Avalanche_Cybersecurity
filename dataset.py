import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

def get_dataloaders(train_csv, test_csv, batch_size=64):
    # Caricamento CSV
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # Encoding colonne categoriali
    cat_cols = ['proto', 'service', 'state', 'attack_cat']
    for col in cat_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col].astype(str))
        df_test[col] = df_test[col].map(lambda x: x if x in le.classes_ else 'unknown')
        le.classes_ = np.append(le.classes_, 'unknown')
        df_test[col] = le.transform(df_test[col])

    # Scaling numerico
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train.drop(columns=['label', 'id']).values)
    X_test = scaler.transform(df_test.drop(columns=['label', 'id']).values)

    # Labels
    y_train = df_train['label'].astype(int).values
    y_test = df_test['label'].astype(int).values

    # Conversione in Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Dataset
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, X_train.shape[1], len(torch.unique(y_train))
