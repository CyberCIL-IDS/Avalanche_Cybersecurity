import os
import pandas as pd
import yaml
import logging
from sklearn.preprocessing import LabelEncoder
from preprocessing.feature_engineering import clip_outliers
from preprocessing.encoder import build_preprocessor, save_object

logging.basicConfig(level=logging.INFO)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def list_csv_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]

def preprocess_and_save(config):

    input_dir = config["preprocessing"]["input_dir"]
    output_dir = config["preprocessing"]["output_dir"]
    num_train = config["preprocessing"]["num_train"]
    num_test = config["preprocessing"]["num_test"]

    cat_cols = config["preprocessing"]["categorical"]
    label_col = config["preprocessing"]["label_column"]
    clip_p = config["preprocessing"]["clip_percentile"]
    balance_classes = config["preprocessing"]["balance_classes"]

    # Output paths
    preproc_path = config["output"]["save_preprocessor"]
    label_enc_path = config["output"]["save_label_encoder"]

    # ---------------------------------------------------------------------------------------
    # 1. LISTA CSV
    # ---------------------------------------------------------------------------------------
    all_csv = list_csv_files(input_dir)

    if len(all_csv) < num_train + num_test:
        raise ValueError("Non ci sono abbastanza CSV nella cartella input_dir!")

    train_files = all_csv[:num_train]
    test_files = all_csv[num_train:num_train + num_test]

    logging.info(f"Train files: {train_files}")
    logging.info(f"Test files:  {test_files}")

    # ---------------------------------------------------------------------------------------
    # 2. Caricamento
    # ---------------------------------------------------------------------------------------
    df_train = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
    df_test = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)

    # ---------------------------------------------------------------------------------------
    # 3. Clip outliers
    # ---------------------------------------------------------------------------------------
    df_train = clip_outliers(df_train, clip_p)
    df_test = clip_outliers(df_test, clip_p)

    # ---------------------------------------------------------------------------------------
    # 4. Label encoding della colonna target
    # ---------------------------------------------------------------------------------------
    df_train[label_col] = df_train[label_col].fillna("Unknown")
    df_test[label_col] = df_test[label_col].fillna("Unknown")

    label_encoder = LabelEncoder()
    df_train["encoded_label"] = label_encoder.fit_transform(df_train[label_col])
    df_test["encoded_label"] = label_encoder.transform(df_test[label_col])

    # ---------------------------------------------------------------------------------------
    # 5. Individua colonnne numeriche
    # ---------------------------------------------------------------------------------------
    num_cols = [c for c in df_train.columns if c not in cat_cols + [label_col, "encoded_label"]]

    # ---------------------------------------------------------------------------------------
    # 6. Preprocessor sklearn (OneHot + scaling)
    # ---------------------------------------------------------------------------------------
    preprocessor = build_preprocessor(cat_cols, num_cols)

    X_train = preprocessor.fit_transform(df_train[num_cols + cat_cols])
    X_test = preprocessor.transform(df_test[num_cols + cat_cols])

    # ---------------------------------------------------------------------------------------
    # 7. Salva pipeline
    # ---------------------------------------------------------------------------------------
    save_object(preprocessor, preproc_path)
    save_object(label_encoder, label_enc_path)

    # ---------------------------------------------------------------------------------------
    # 8. Salvataggio CSV preprocessati
    # ---------------------------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # Convert sparse → array
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    # Convert arrays → DataFrame
    df_train_out = pd.DataFrame(X_train_dense)
    df_train_out["label"] = df_train["encoded_label"]

    df_test_out = pd.DataFrame(X_test_dense)
    df_test_out["label"] = df_test["encoded_label"]

    train_out_path = os.path.join(output_dir, "train_preprocessed.csv")
    test_out_path = os.path.join(output_dir, "test_preprocessed.csv")

    df_train_out.to_csv(train_out_path, index=False)
    df_test_out.to_csv(test_out_path, index=False)

    logging.info(f"Salvato: {train_out_path}")
    logging.info(f"Salvato: {test_out_path}")

    logging.info("Preprocessing completato con successo!")


if __name__ == "__main__":
    config = load_config("config.yaml")
    preprocess_and_save(config)
