import torch
import yaml
import pandas as pd
import numpy as np
from preprocessing.loader import load_object
from preprocessing.feature_engineering import clip_outliers, add_unsw_features
from preprocessing.loader import load_model

# Importa la tua nuova funzione di plotting
from utils.plotting import plot_confusion_matrix 

def predict(model, X_tensor):
    model.eval() # Buona norma mettere eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
    return preds.numpy()

if __name__ == "__main__":
    
    cfg_path = "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Percorsi
    model_path = f"checkpoints/model_checkpoint_{cfg['benchmark']['strategy']}_{cfg['benchmark']['mode']}_{cfg['benchmark']['param']}.pth"
    preprocessor_path = cfg["output"]["save_preprocessor"]
    label_encoder_path = cfg["output"]["save_label_encoder"]
    data_path = cfg["dataset"]["test_csv"] 
    output_path = "datasets/predictions.csv"
    cm_output_path = "utils/plot/confusion_matrix.png" # Percorso salvataggio immagine
    
    print("Caricamento risorse...")
    preprocessor = load_object(preprocessor_path)
    label_encoder = load_object(label_encoder_path)

    N_SAMPLES = 2000 
    print(f"Estrazione {N_SAMPLES} campioni casuali per validazione...")
    
    df_sample = pd.read_csv(data_path).sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)
    df_output = df_sample.copy()
    
    label_col = cfg["preprocessing"]["label_column"]
    y_true = None
    
    if label_col in df_sample.columns:
        raw_labels = df_sample[label_col].fillna("Unknown").values
        
        known_mask = np.isin(raw_labels, label_encoder.classes_)
        
        if not known_mask.all():
            print(f"⚠️ Warning: { (~known_mask).sum() } campioni hanno label sconosciute e saranno ignorati.")
            df_sample = df_sample[known_mask].reset_index(drop=True)
            df_output = df_output[known_mask].reset_index(drop=True)
            raw_labels = raw_labels[known_mask]
            
        y_true = label_encoder.transform(raw_labels)
        
        df_sample = df_sample.drop(columns=[label_col])
    else:
        print("Warning: Colonna label non trovata. Impossibile generare Matrice di Confusione.")

    for col in cfg["preprocessing"]["drop_columns"]:
        if col in df_sample:
            df_sample = df_sample.drop(columns=[col])
            
    if "service" in df_sample.columns:
        df_sample["service"] = df_sample["service"].replace("-", "Unknown")

    if cfg["preprocessing"]["add_features"]:
        df_sample = add_unsw_features(df_sample)

    df_sample = clip_outliers(df_sample, cfg["preprocessing"]["clip_percentile"])
    
    # Transform
    try:
        X_sparse = preprocessor.transform(df_sample)
        X_tensor = torch.tensor(X_sparse.toarray(), dtype=torch.float32)
    except ValueError as e:
        print(f"Errore di trasformazione: {e}")
        exit()

    input_size = X_tensor.shape[1]
    num_classes = len(label_encoder.classes_) 
    
    model = load_model(model_path, input_size, num_classes)
    
    preds_indices = predict(model, X_tensor)
    preds_labels = label_encoder.inverse_transform(preds_indices)
    
    # Salva risultati nel CSV
    df_output['predicted_idx'] = preds_indices
    df_output['predicted_attack_cat'] = preds_labels
    df_output.to_csv(output_path, index=False)
    print(f"Predizioni salvate in: {output_path}")

    if y_true is not None:
        print("Generazione Matrice di Confusione...")
        plot_confusion_matrix(
            y_true=y_true, 
            y_pred=preds_indices, 
            classes=label_encoder.classes_, 
            filename=cm_output_path
        )
        
        cols_to_print = ['predicted_attack_cat']
        if label_col in df_output.columns: cols_to_print.append(label_col)
        print("\n=== ANTEPRIMA (Primi 10) ===")
        print(df_output[cols_to_print].head(10))