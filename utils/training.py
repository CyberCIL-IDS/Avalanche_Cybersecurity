import os
import torch
import pandas as pd
import numpy as np
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from torch.optim.lr_scheduler import MultiStepLR
from utils.checkpoint_custom_plugin import CheckpointPlugin
from models.neural_network import NeuralNetwork
from utils.strategy import getStrategy

# Import per la predizione interna
from preprocessing.feature_engineering import clip_outliers, add_unsw_features
from utils.plotting import plot_confusion_matrix

def run_internal_prediction(model, df_test, preprocessor, label_encoder, cfg, device):
    """Funzione helper per eseguire predizioni sul modello in memoria"""
    print("\n[PREDICT] Avvio validazione su campione di test set grezzo...")
    
    # 1. Preprocessing "al volo" (Simile a predict.py)
    df_sample = df_test.copy()
    label_col = cfg["preprocessing"]["label_column"]
    y_true = None

    # Gestione Label
    if label_col in df_sample.columns:
        raw_labels = df_sample[label_col].fillna("Unknown").values
        known_mask = np.isin(raw_labels, label_encoder.classes_)
        if not known_mask.all():
            df_sample = df_sample[known_mask].reset_index(drop=True)
            raw_labels = raw_labels[known_mask]
        y_true = label_encoder.transform(raw_labels)
        df_sample = df_sample.drop(columns=[label_col])

    # Drop & Features
    for col in cfg["preprocessing"]["drop_columns"]:
        if col in df_sample: df_sample = df_sample.drop(columns=[col])
            
    if "service" in df_sample.columns:
        df_sample["service"] = df_sample["service"].replace("-", "Unknown")

    if cfg["preprocessing"]["add_features"]:
        df_sample = add_unsw_features(df_sample)

    df_sample = clip_outliers(df_sample, cfg["preprocessing"]["clip_percentile"])

    # Transform
    try:
        X_sparse = preprocessor.transform(df_sample)
        X_tensor = torch.tensor(X_sparse.toarray(), dtype=torch.float32).to(device)
    except ValueError as e:
        print(f"[PREDICT ERROR] Trasformazione fallita: {e}")
        return

    # Inferenza
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()

    # Metriche & Plot
    if y_true is not None:
        acc = (preds == y_true).mean()
        print(f"--> VALIDATION ACCURACY (Raw Data Sample): {acc:.4f}")
        
        cm_filename = f"utils/plot/confusion_matrix_{cfg['benchmark']['strategy']}_final.png"
        plot_confusion_matrix(
            y_true=y_true, 
            y_pred=preds, 
            classes=label_encoder.classes_, 
            filename=cm_filename
        )
        print(f"[PREDICT] Matrice di confusione salvata in: {cm_filename}")


def train(benchmark, input_size, n_classes, mode, param, model_params, 
          # NUOVI ARGOMENTI OPZIONALI PER PREDIZIONE
          cfg=None, preprocessor=None, label_encoder=None,
          strategy_type="Replay", train_epochs=15, momentum=0.9, weight_decay=1e-4): 
    
    # --- 1. PULIZIA PARAMETRI (Codice esistente) ---
    params_copy = model_params.copy()
    if "lr" not in params_copy: raise ValueError("Manca 'lr'")
    lr = params_copy.pop("lr")
    batch_size = params_copy.pop("batch_size", 32)
    if "weight_decay" in params_copy: weight_decay = params_copy.pop("weight_decay")
    
    use_sigmoid_activation = (strategy_type == "ICaRL")
    
    # --- 2. ISTANZIAZIONE MODELLO ---
    model = NeuralNetwork(
        input_size=input_size, 
        num_classes=n_classes, 
        use_sigmoid=use_sigmoid_activation,
        **params_copy 
    )

    # --- CONFIGURAZIONE STRATEGIA (Codice esistente) ---
    if strategy_type == "ICaRL":
        current_epochs = 100 if train_epochs < 50 else train_epochs
        milestones = [int(current_epochs * 0.6), int(current_epochs * 0.8)]
    elif strategy_type == "MER":
        current_epochs = 5 if train_epochs == 15 else train_epochs 
        milestones = [2, 4] 
    else:
        current_epochs = 5
        milestones = [10, 13]
        momentum = 0.0

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scheduler_plugin = LRSchedulerPlugin(scheduler)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()]
    )
    
    checkpoint_plugin = CheckpointPlugin(strategy_type, mode, param)
    plugins_list = [checkpoint_plugin, scheduler_plugin]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Config: {strategy_type} | LR: {lr} | WD: {weight_decay} | Batch: {batch_size} | Epochs: {current_epochs}")

    strategy = getStrategy(
        strategy_type=strategy_type,
        model=model,
        optimizer=optimizer,
        current_epochs=current_epochs,
        eval_plugin=eval_plugin,
        device=device,
        plugins_list=plugins_list,
        batch_size=batch_size
    )

    # --- TRAINING LOOP ---
    for i, experience in enumerate(benchmark.train_stream):
        print(f"Training Exp {i}: {experience.classes_in_this_experience}")
        strategy.train(experience)
        print("Evaluation...")
        strategy.eval(benchmark.test_stream)

    # ========================================================
    # === NUOVO BLOCCO: PREDIZIONE FINALE INTEGRATA ===
    # ========================================================
        
    if cfg and preprocessor and label_encoder:
        try:
            # Carichiamo un campione del test set originale
            test_csv = cfg["dataset"]["test_csv"]
            if os.path.exists(test_csv):
                df_test = pd.read_csv(test_csv)
                # Campioniamo 2000 righe per fare veloce
                if len(df_test) > 2000:
                    df_test = df_test.sample(n=10000, random_state=42).reset_index(drop=True)
                
                print(f"[PREDICT] Eseguo predizione interna sul test set campionato ({len(df_test)} righe)...")
                run_internal_prediction(model, df_test, preprocessor, label_encoder, cfg, device)
            else:
                print(f"[PREDICT WARNING] File test set non trovato: {test_csv}")
        except Exception as e:
            print(f"[PREDICT ERROR] Errore durante la validazione finale: {e}")

    return len(benchmark.train_stream), eval_plugin.get_last_metrics()