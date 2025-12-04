import os
import torch
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from torch.optim.lr_scheduler import MultiStepLR
from utils.checkpoint_custom_plugin import CheckpointPlugin
from models.neural_network import NeuralNetwork
from utils.strategy import getStrategy

def train(benchmark, input_size, n_classes, mode, param, strategy_type="Replay", use_checkpoint=False, train_epochs=15): 
    
    use_sigmoid_activation = (strategy_type == "ICaRL")
    model = NeuralNetwork(input_size, n_classes, use_sigmoid=use_sigmoid_activation)

    # --- CONFIGURAZIONE ---
    if strategy_type == "ICaRL":
        # ICaRL richiede molte epoche e LR alto gestito da scheduler
        current_epochs = 100 if train_epochs < 50 else train_epochs
        lr = 2.0
        milestones = [int(current_epochs * 0.6), int(current_epochs * 0.8)]
    elif strategy_type == "MER":
        # NOTA: MER è computazionalmente molto più pesante di Replay.
        # Spesso bastano meno epoche perché fa "n_inner_steps" per ogni batch.
        current_epochs = 5 if train_epochs == 15 else train_epochs 
        lr = 0.01 # MER lavora bene con LR standard o leggermente aggressivi (0.05 - 0.1)
        milestones = [2, 4] # Scheduler accorciato
    else:
        # Configurazione standard per DER o altri
        current_epochs = 15
        lr = 0.01
        milestones = [10, 13]

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr,            
        momentum=0.9, 
        weight_decay=1e-4      
    )

    # Scheduler allineato alle epoche correnti
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

    strategy = getStrategy(
        strategy_type=strategy_type,
        model=model,
        optimizer=optimizer,
        current_epochs=current_epochs,
        eval_plugin=eval_plugin,
        device=device,
        plugins_list=plugins_list
    )

    # --- RESUME LOGIC (INVARIATA) ---
    chk_path = f"checkpoints/model_checkpoint_{strategy_type}_{mode}_{param}.pth"
    start_exp = 0
    if use_checkpoint and os.path.exists(chk_path):
        print(f"Checkpoint trovato: {chk_path}")
        checkpoint = torch.load(chk_path, map_location=device, weights_only=False) 
        
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        if "storage_policy" in checkpoint and checkpoint["storage_policy"] is not None:
            strategy.storage_policy = checkpoint["storage_policy"]
            print(f"Buffer ripristinato: {len(strategy.storage_policy.buffer)} esempi.")

        if "exp_counter" in checkpoint:
            start_exp = checkpoint["exp_counter"]
            print(f"Ripresa dalla Exp: {start_exp}")

    # --- TRAINING LOOP ---
    for i, experience in enumerate(benchmark.train_stream):
        if i < start_exp:
            continue
            
        print(f"Training Exp {i}: {experience.classes_in_this_experience}")
        strategy.train(experience)
        
        print("Evaluation...")
        strategy.eval(benchmark.test_stream)

    return len(benchmark.train_stream), eval_plugin.get_last_metrics()