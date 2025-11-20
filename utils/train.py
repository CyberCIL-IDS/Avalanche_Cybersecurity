import torch
from avalanche.training.strategies import Replay
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger

from model import NeuralNetwork       
from benchmark import get_benchmark   # benchmark già preprocessato e pronto


def train():

    # Carica il benchmark
    benchmark = get_benchmark()

    # Istanzia il modello
    model = NeuralNetwork()

    # Imposta Algoritmo di ottimizzazione
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Plugin per logging ed evaluation
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger()]
    )

    # Strategy Replay — continual learning
    strategy = Replay(
        model = model,
        optimizer = optimizer,
        criterion = torch.nn.CrossEntropyLoss(),    #funzione di loss: funzione che misura quanto il modello “sbaglia” durante il training.
        train_mb_size = 64,     # Dimensione batch training - numero di esempi per ogni passo di addestramento
        train_epochs = 5,       # Epoche per esperienza - quante volte l’esperienza viene ripassata
        eval_mb_size = 64,      # Dimensione batch per valutazione - per ridurre uso memoria durante test
        evaluator = eval_plugin,
        device = "cuda" if torch.cuda.is_available() else "cpu",
        mem_size = 2000   # buffer di replay
    )

    # Training sulle esperienze incrementali - Questo ciclo simula perfettamente il continual learning.
    for experience in benchmark.train_stream:       #Per ogni esperienza experience
        print(f"\n Training su: {experience}")
        strategy.train(experience)  #addestra la rete sull'esperienza attuale - mescola i dati correnti con quelli del replay buffer - aggiorna il buffer con nuovi esempi

        print("\nEvaluation:")
        strategy.eval(benchmark.test_stream)    #valuta il modello su tutte le esperienze viste e non viste - mostra metriche grazie all’InteractiveLogger

    print("\nTraining completato con Replay Strategy + NeuralNetwork.")


if __name__ == "__main__":
    train()
