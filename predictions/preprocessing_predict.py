import pandas as pd

def preprocess_predict(df):
    """
    Preprocess the input dataframe for prediction.
    Drops unnecessary columns and the target column 'attack_cat' if present.
    """
    # Drop target column if present
    if 'attack_cat' in df.columns:
        df = df.drop(columns=['attack_cat'])
    
    # Drop other non-feature columns (esempio: timestamp, flow_id, ecc.)
    cols_to_drop = ['id', 'label']  # modifica in base alle tue colonne
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Fill missing values if any
    df = df.fillna(0)
    
    return df
