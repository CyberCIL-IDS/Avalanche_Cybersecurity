import pandas as pd
import logging
import pickle
import torch
from models.neural_network import NeuralNetwork

def load_dataset(csv_path):
    logging.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def load_object(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def load_model(model_path, input_size, num_classes, device="cpu"):
    logging.info(f"Loading model from {model_path}")
    
    model = NeuralNetwork(input_size, num_classes)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"]
    
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        if "train_classifier" in new_key:
            new_key = new_key.replace("train_classifier", "classifier")
            new_key = new_key.replace(".0.", ".")
            
        if "eval_classifier" in new_key:
            continue
            
        new_state_dict[new_key] = value

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Modello caricato correttamente (Strict Mode).")
    except RuntimeError as e:
        print(f"Warning caricamento strict: {e}")
        print("Riprovo con strict=False per caricare i pesi parziali...")
        model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.eval()
    return model