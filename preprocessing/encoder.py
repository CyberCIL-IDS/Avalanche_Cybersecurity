import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def build_preprocessor(categorical_cols, numerical_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numerical", StandardScaler(), numerical_cols),
        ]
    )
    return preprocessor

def save_object(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
