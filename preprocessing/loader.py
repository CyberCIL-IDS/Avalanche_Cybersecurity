import pandas as pd
import logging

def load_dataset(csv_path):
    logging.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df
