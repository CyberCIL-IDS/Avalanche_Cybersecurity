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
    model = NeuralNetwork(input_size, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

