import torch
from avalanche.training import Replay, ICaRL, DER
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger

from models.neural_network import NeuralNetwork         # benchmark già preprocessato e pronto


def train(benchmark, input_size, n_classes, strategy_type="Replay"):
    # Istanzia il modello
    model = NeuralNetwork(input_size=input_size, num_classes=n_classes)

    # Imposta Algoritmo di ottimizzazione
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Plugin per logging ed evaluation
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),  # <-- Add forgetting metric
        loggers=[InteractiveLogger()]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if strategy_type == "Replay":
        strategy = Replay(
            model = model,
            optimizer = optimizer,
            criterion = torch.nn.CrossEntropyLoss(),    #funzione di loss: funzione che misura quanto il modello “sbaglia” durante il training.
            train_mb_size = 64,     # Dimensione batch training - numero di esempi per ogni passo di addestramento
            train_epochs = 1,       # Epoche per esperienza - quante volte l’esperienza viene ripassata
            eval_mb_size = 64,      # Dimensione batch per valutazione - per ridurre uso memoria durante test
            evaluator = eval_plugin,
            device = device,
            mem_size = 2000   # buffer di replay
        )
    elif strategy_type == "ICaRL":
        strategy = ICaRL(
            feature_extractor=model.features_extractor,
            classifier=model.classifier,
            optimizer=optimizer,
            train_mb_size=64,
            train_epochs=1,
            eval_mb_size=64,
            evaluator=eval_plugin,
            device=device,
            memory_size=2000,
            buffer_transform=None,     # nuovo
            fixed_memory=True          # nuovo
        )
    elif strategy_type == "DER":
        strategy = DER(
            model=model,
            optimizer=optimizer,
            criterion=torch.nn.CrossEntropyLoss(),
            train_mb_size=64,
            train_epochs=1,
            eval_mb_size=64,
            evaluator=eval_plugin,
            device=device,
            mem_size=2000,   # <-- sostituisce buffer_size
            alpha=0.3 # peso della loss sui dati del buffer rispetto a quelli correnti
        )
    else:
        print(f"Strategy {strategy_type} not recognized.")
        SystemExit(1)
    

    # Training sulle esperienze incrementali - Questo ciclo simula perfettamente il continual learning.
    for experience in benchmark.train_stream:       #Per ogni esperienza experience
        print(f"\n Training su: {experience}")
        strategy.train(experience)  #addestra la rete sull'esperienza attuale - mescola i dati correnti con quelli del replay buffer - aggiorna il buffer con nuovi esempi

        print("\nEvaluation:")
        strategy.eval(benchmark.test_stream)    #valuta il modello su tutte le esperienze viste e non viste - mostra metriche grazie all’InteractiveLogger

    print("\nTraining completato con Replay Strategy + NeuralNetwork.")
    

    n_experiences = len(benchmark.train_stream)

    metrics = eval_plugin.get_last_metrics()

    return n_experiences, metrics#experiences, accuracy, forgettinreturn n_experiences, eval_plugin.accuracy_metrics.get_all(), eval_plugin.forgetting_metrics.get_all