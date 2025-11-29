import torch
import yaml
import pandas as pd
from preprocessing.loader import load_object
from preprocessing.feature_engineering import clip_outliers
from preprocessing.feature_engineering import add_unsw_features
from preprocessing.loader import load_model

def predict(model, X_tensor):
    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
    return preds.numpy()


if __name__ == "__main__":
    
    cfg_path = "config.yaml"
    with open(cfg_path) as f:               # Carcica configurazione
        cfg = yaml.safe_load(f)

    model_path = "checkpoints/model_checkpoint_Replay_incremental_2.pth" # Verifica il nome del file
    preprocessor_path = cfg["output"]["save_preprocessor"]
    label_encoder_path = cfg["output"]["save_label_encoder"]
    data_path = cfg["dataset"]["train_csv"]  # Puoi usare anche test_csv
    output_path = "datasets/predictions.csv"
    
    print("Caricamento preprocessor e label encoder...")
    preprocessor = load_object(preprocessor_path)
    label_encoder = load_object(label_encoder_path)

    print("Estrazione dati casuali...")
    df_sample = pd.read_csv(data_path).sample(n=10, random_state=42).reset_index(drop=True)
    
    df_output = df_sample.copy()
    
    label_col = cfg["preprocessing"]["label_column"]
    if label_col in df_sample.columns:
        df_sample = df_sample.drop(columns=[label_col])
        
    for col in cfg["preprocessing"]["drop_columns"]:
        if col in df_sample:
            df_sample = df_sample.drop(columns=[col])
            
    if "service" in df_sample.columns:
        df_sample["service"] = df_sample["service"].replace("-", "Unknown")

    if cfg["preprocessing"]["add_features"]:
        df_sample = add_unsw_features(df_sample)

    df_sample = clip_outliers(df_sample, cfg["preprocessing"]["clip_percentile"])
    
    try:
        X_sparse = preprocessor.transform(df_sample)
        X_tensor = torch.tensor(X_sparse.toarray(), dtype=torch.float32)
    except ValueError as e:
        print(f"ERRORE DI TRASFORMAZIONE: {e}")
        exit()

    input_size = X_tensor.shape[1]
    num_classes = len(label_encoder.classes_) 
    
    model = load_model(model_path, input_size, num_classes)
    
    preds_indices = predict(model, X_tensor)
    
    preds_labels = label_encoder.inverse_transform(preds_indices)
    
    df_output['predicted_idx'] = preds_indices
    df_output['predicted_attack_cat'] = preds_labels

    cols_to_print = ['predicted_attack_cat'] 
    
    if label_col in df_output.columns:
        cols_to_print.append(label_col) 

    print("\n=== RISULTATI PREDIZIONE ===")
    print(df_output[cols_to_print])

    # Salva su file
    df_output.to_csv(output_path, index=False)
    print(f"\nPredizioni salvate in: {output_path}")