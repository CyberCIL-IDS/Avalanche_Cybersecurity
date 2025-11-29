import torch
import yaml
import pandas as pd
import yaml
from preprocessing.preprocessing_predict import preprocess_predict_sample


def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def predict(model, X_tensor):
    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
    return preds.numpy()

if __name__ == "__main__":
    # Config
    cfg_path = "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_path = "checkpoints/model_checkpoint_Replay_incremental_2.pth"
    data_path = "datasets/UNSW_NB15_training-set.csv"
    output_path = "predictions.csv"

    # Preprocessing sample
    X_tensor, df_sample, _ = preprocess_predict_sample(cfg, data_path, n_samples=10)

    # Load modello
    model = torch.load(model_path)
    model.eval()

    # Predizione
    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)

    # Salva predizioni
    df_sample['predicted_attack_cat'] = preds.numpy()
    df_sample.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
