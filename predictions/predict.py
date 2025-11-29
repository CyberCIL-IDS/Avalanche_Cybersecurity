import torch
import pandas as pd
from preprocessing_predict import preprocess_predict
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def predict(model, X):
    """
    Returns the predicted labels.
    """
    with torch.no_grad():
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
    return preds.numpy()

if __name__ == "__main__":
    # Percorsi
    model_path = "saved_model.pth"   # modello salvato
    data_path = "data_to_predict.csv"  # dati senza label           
    output_path = "predictions.csv"
    
    #TODO: selezionare random i sample di dati da predire


    # Carica dati
    df = pd.read_csv(data_path)
    df_processed = preprocess_predict(df)
    
    # Optional: standardizzazione (da usare se il modello è stato addestrato con StandardScaler)
    scaler = StandardScaler()
    df_processed = pd.DataFrame(scaler.fit_transform(df_processed), columns=df_processed.columns)
    
    # Carica modello
    model = load_model(model_path)
    
    # Predizione
    preds = predict(model, df_processed)
    
    # Salva risultati
    df['predicted_attack_cat'] = preds
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
