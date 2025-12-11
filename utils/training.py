import os
import torch
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from torch.optim.lr_scheduler import MultiStepLR
from utils.checkpoint_custom_plugin import CheckpointPlugin
from models.neural_network import NeuralNetwork
from utils.strategy import getStrategy

def train(benchmark, input_size, n_classes, mode, param, model_params, strategy_type="Replay", train_epochs=15, momentum=0.9, weight_decay=1e-4): 
    params_copy = model_params.copy()
    
    if "lr" not in params_copy:
         raise ValueError("ERRORE: Il parametro 'lr' deve essere presente in model_params!")
    
    lr = params_copy.pop("lr") # Rimuove 'lr' da params_copy e lo salva nella variabile
    batch_size = params_copy.pop("batch_size", 32)
    
    required_keys = ["h1", "h2", "h3", "h4", "dropout"]
    for k in required_keys:
        if k not in params_copy:
            raise ValueError(f"ERRORE: Manca il parametro '{k}' in model_params!")

    use_sigmoid_activation = (strategy_type == "ICaRL")
    
    model = NeuralNetwork(
        input_size=input_size, 
        num_classes=n_classes, 
        use_sigmoid=use_sigmoid_activation,
        **params_copy 
    )

    # --- CONFIGURAZIONE STRATEGIA ---
    if strategy_type == "ICaRL":
        current_epochs = 100 if train_epochs < 50 else train_epochs
        # LR è gestito parametricamente ora
        milestones = [int(current_epochs * 0.6), int(current_epochs * 0.8)]
    elif strategy_type == "MER":
        current_epochs = 5 if train_epochs == 15 else train_epochs 
        milestones = [2, 4] 
    else:
        current_epochs = 5
        milestones = [10, 13]
        momentum = 0.0
        weight_decay = 1e-3

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr,
        momentum=momentum, 
        weight_decay=weight_decay      
    )

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
    print(f"Config: {strategy_type} | LR: {lr} | Epochs: {current_epochs} | Device: {device}")
    print(f"Hyperparameters: {model_params}")

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

    return len(benchmark.train_stream), eval_plugin.get_last_metrics()