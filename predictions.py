import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from model import NeuralNetwork

# --- Parametri ---
test_csv = "datasets/UNSW_NB15_testing-set(in).csv"
checkpoint_path = "checkpoint.pth"
num_samples = 10  # numero di campioni da predire
random_state = np.random.randint(0, 100)

# --- Leggi CSV e prendi campioni ---
df_test = pd.read_csv(test_csv)
sample_df = df_test.sample(n=num_samples, random_state=random_state)

# --- Preprocessing: encode e scale ---
cat_cols = ['proto', 'service', 'state', 'attack_cat']

for col in cat_cols:
    le = LabelEncoder()
    le.fit(df_test[col].astype(str))  # fit su tutto il test dataset
    sample_df[col] = le.transform(sample_df[col].astype(str))

scaler = StandardScaler()
X_sample = scaler.fit_transform(sample_df.drop(columns=['label', 'id']).values)

# --- Converti in tensore ---
X_sample = torch.tensor(X_sample, dtype=torch.float32)

# --- Carica modello ---
device = "cuda" if torch.cuda.is_available() else "cpu"
input_size = X_sample.shape[1]
num_classes = 2  # oppure quante classi hai
model = NeuralNetwork(input_size, num_classes).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- Predizioni ---
with torch.no_grad():
    X_sample = X_sample.to(device)
    preds = model(X_sample)
    predicted_classes = torch.argmax(preds, dim=1).cpu().numpy()

# --- Stampa predizioni e label ---
true_labels = sample_df['label'].values
print("Predicted classes:", predicted_classes)
print("True labels:", true_labels)

# --- Calcolo accuracy ---
accuracy = (predicted_classes == true_labels).sum() / len(true_labels)
print(f"\nAccuracy sui {len(true_labels)} campioni: {accuracy*100:.2f}%")
