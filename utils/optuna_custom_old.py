import utils.optuna_custom as optuna_custom
from avalanche.training import Replay, ICaRL, DER
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from torch.optim.lr_scheduler import MultiStepLR
from models.neural_network import NeuralNetwork
from utils.strategy import getStrategy
import torch

def objective(trial, device, strategy_type, input_size, n_classes, use_sigmoid_activation, benchmark, train_epochs=15):

    if strategy_type == "ICaRL":
        current_epochs = 100 if train_epochs < 50 else train_epochs
        lr = 2.0
        milestones = [int(current_epochs * 0.6), int(current_epochs * 0.8)]
    else:
        current_epochs = 15
        lr = 0.01  
        milestones = [10, 13]

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    h1 = trial.suggest_int("h1", 256, 1024)
    h2 = trial.suggest_int("h2", 128, 512)
    h3 = trial.suggest_int("h3", 64, 256)
    h4 = trial.suggest_int("h4", 64, 256)

    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    model = NeuralNetwork(
        input_size=input_size, 
        num_classes=n_classes, 
        use_sigmoid=use_sigmoid_activation, 
        h1=h1, 
        h2=h2, 
        h3=h3, 
        h4=h4, 
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Scheduler allineato alle epoche correnti
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scheduler_plugin = LRSchedulerPlugin(scheduler)

    plugins_list = [scheduler_plugin]

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=False),
        loggers=[InteractiveLogger()]
    )

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

    accuracies = []

    for exp in benchmark.train_stream:
        strategy.train(exp)
        results = strategy.eval(benchmark.test_stream)

        key = (
            f"Top1_Acc_Exp/"
            f"eval_phase/test_stream/Task000/Exp{exp.current_experience}"
        )

        acc = results.get(key, None)

        if acc is not None:
            accuracies.append(acc)

    if len(accuracies) == 0:
        return 0.0

    return sum(accuracies) / len(accuracies)  # average accuracy across experiences
