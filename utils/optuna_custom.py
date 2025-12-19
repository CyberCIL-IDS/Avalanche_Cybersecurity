import optuna
from avalanche.training import Replay, ICaRL, DER
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import TextLogger 
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from torch.optim.lr_scheduler import MultiStepLR
from models.neural_network import NeuralNetwork
from utils.strategy import getStrategy
import torch
import os
import sys

# Rimuoviamo 'use_sigmoid_activation' dagli argomenti e lo calcoliamo dentro
def objective(trial, device, strategy_type, input_size, n_classes, benchmark, current_epochs=None, use_sigmoid_activation=None):

    # --- 1. CONFIGURAZIONE DINAMICA IPERPARAMETRI ---
    if strategy_type == "ICaRL":
        opt_lr = trial.suggest_float("lr", 0.005, 0.1, log=True)
    elif strategy_type == "MER":
        opt_lr = trial.suggest_float("lr", 0.001, 0.05, log=True)
    else:
        opt_lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)

    # if strategy_type == "MER":
    #     batch_size = trial.suggest_categorical("batch_size", [32, 64])
    # else:
    #     batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    if strategy_type == "MER":
        # MER è pesante in memoria, stiamo un po' più bassi ma comunque alziamo
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    else:
        # ICaRL, DER e Replay volano con batch alti
        batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096])

    h1 = trial.suggest_int("h1", 128, 512) 
    h2 = trial.suggest_int("h2", 128, 256)
    h3 = trial.suggest_int("h3", 0, 128) 
    h4 = trial.suggest_int("h4", 0, 128)
    
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # --- 2. CONFIGURAZIONE EPOCHE ---
    if current_epochs is None:
        if strategy_type == "ICaRL":
            current_epochs = 60 
        elif strategy_type == "MER":
            current_epochs = 35 
        else:
            current_epochs = 30 

    milestones = [int(current_epochs * 0.7)]

    # --- 3. FIX CRITICO: DETERMINAZIONE SIGMOIDE ---
    # Calcoliamo qui se serve la sigmoide, ignorando argomenti esterni potenzialmente errati.
    # ICaRL richiede TASSATIVAMENTE output [0,1] per la BCELoss.
    force_sigmoid = True if strategy_type == "ICaRL" else False

    # Gestione layer opzionali (h3, h4 possono essere 0)
    # Assumiamo che la tua classe NeuralNetwork accetti h3, h4. 
    # Se supporti layer variabili, gestiscili qui, altrimenti forziamo un minimo.
    if h3 == 0: h3 = 64
    if h4 == 0: h4 = 64

    model = NeuralNetwork(
        input_size=input_size, 
        num_classes=n_classes, 
        use_sigmoid=force_sigmoid,  
        h1=h1, h2=h2, h3=h3, h4=h4, 
        dropout=dropout
    ).to(device)

    # --- 4. OPTIMIZER ---
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=opt_lr, 
        momentum=0.9, 
        weight_decay=weight_decay
    )
    
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scheduler_plugin = LRSchedulerPlugin(scheduler)

    logger = TextLogger(open(os.devnull, 'w'))
    
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True), 
        loggers=[logger] 
    )

    # --- 5. STRATEGY ---
    strategy = getStrategy(
        strategy_type=strategy_type,
        model=model,
        optimizer=optimizer,
        current_epochs=current_epochs,
        eval_plugin=eval_plugin,
        device=device,
        plugins_list=[scheduler_plugin],
        batch_size=batch_size
    )

    # --- 6. TRAINING LOOP & PRUNING ---
    mean_acc_final = 0.0

    try:
        for i, exp in enumerate(benchmark.train_stream):
            strategy.train(exp)
            results = strategy.eval(benchmark.test_stream)

            stream_acc = results.get("Top1_Acc_Stream/eval_phase/test_stream/Task000")
            
            if stream_acc is None:
                keys = [k for k in results.keys() if "Top1_Acc_Stream" in k]
                stream_acc = results[keys[0]] if keys else 0.0

            mean_acc_final = stream_acc 

            trial.report(stream_acc, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
    except RuntimeError as e:
        if "all elements of input should be between 0 and 1" in str(e):
            print("\n[CRITICAL ERROR] ICaRL ha ricevuto Logits invece di Probabilità.")
            print("Verifica che NeuralNetwork stia usando nn.Sigmoid() alla fine.")
            # Ritorna 0 per dire a Optuna che questo trial è fallito male
            return 0.0
        raise e

    return mean_acc_final