import optuna
from avalanche.training import Replay, ICaRL, DER
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import TextLogger 
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from torch.optim.lr_scheduler import MultiStepLR
from models.neural_network_optuna import NeuralNetworkOptuna
from utils.strategy import getStrategy
import torch
import os

def objective(trial, device, strategy_type, input_size, n_classes, use_sigmoid_activation, benchmark, train_epochs=15):

    # ---- 1. OPTUNA HYPERPARAMETERS (Define these FIRST) ----
    # Suggest these first so they are tracked correctly
    opt_lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    h1 = trial.suggest_int("h1", 256, 1024)
    h2 = trial.suggest_int("h2", 128, 512)
    h3 = trial.suggest_int("h3", 64, 256)
    h4 = trial.suggest_int("h4", 64, 256)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # ---- 2. STRATEGY CONFIGURATION ----
    if strategy_type == "ICaRL":
        # ICaRL is heavy. For HPO, limit epochs to something reasonable (e.g., 20) 
        # unless you are doing the final run.
        current_epochs = 20  
        
        # FIX: Don't overwrite opt_lr if you want to search for it. 
        # If ICaRL *requires* lr=2.0, do not use suggest_float above.
        # Assuming you want to SEARCH for LR, we use opt_lr.
        # If you want fixed LR for ICaRL, uncomment below:
        # opt_lr = 2.0 
        
        milestones = [int(current_epochs * 0.6), int(current_epochs * 0.8)]
    else:
        current_epochs = 15
        milestones = [10, 13]

    # ---- 3. MODEL SETUP ----
    model = NeuralNetworkOptuna(
        input_size=input_size, 
        num_classes=n_classes, 
        use_sigmoid=use_sigmoid_activation, 
        h1=h1, h2=h2, h3=h3, h4=h4, 
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt_lr)
    
    # Scheduler
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scheduler_plugin = LRSchedulerPlugin(scheduler)

    # ---- 4. LIGHTWEIGHT EVALUATION ----
    # Use TextLogger (file/stdout) instead of InteractiveLogger for speed
    # Print to stdout only occasionally to avoid clogging
    logger = TextLogger(open(os.devnull, 'w')) # Complete silence for speed
    
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=False),
        loggers=[logger] 
    )

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

    # ---- 5. TRAINING LOOP WITH PRUNING ----
    accuracies = []

    for i, exp in enumerate(benchmark.train_stream):
        strategy.train(exp)
        
        # Evaluate ONLY on the current experience (much faster) 
        # OR evaluate on full stream if metric requires it. 
        # Ideally, evaluating on full stream is safer for CL, but slow.
        results = strategy.eval(benchmark.test_stream)

        # Get accuracy for the current experience ONLY to check immediate performance
        # (This key assumes you want to check if it learned the LATEST task)
        # OR calculate average accuracy of all tasks seen so far:
        
        current_acc_list = []
        for k in range(i + 1): # Check all previous tasks
            key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{k:03d}"
            val = results.get(key)
            if val is not None:
                current_acc_list.append(val)
        
        if current_acc_list:
            mean_acc_so_far = sum(current_acc_list) / len(current_acc_list)
        else:
            mean_acc_so_far = 0

        # ---- CRITICAL: PRUNING ----
        # Report the current mean accuracy to Optuna
        trial.report(mean_acc_so_far, step=i)

        # Handle Pruning
        if trial.should_prune():
            # This stops the trial IMMEDIATELY if it's performing poorly
            raise optuna.TrialPruned()

    # ---- RETURN FINAL METRIC ----
    # Return the average accuracy across all experiences at the end
    return mean_acc_so_far