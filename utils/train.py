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

    # Imposta ottimizzatore
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Plugin per logging ed evaluation
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger()]
    )

    # Strategy Replay — continual learning
    strategy = Replay(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=64,
        train_epochs=5,
        eval_mb_size=64,
        evaluator=eval_plugin,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mem_size=2000   # buffer di replay
    )

    # Training sulle esperienze incrementali
    for experience in benchmark.train_stream:
        print(f"\n Training su: {experience}")
        strategy.train(experience)

        print("\nEvaluation:")
        strategy.eval(benchmark.test_stream)

    print("\nTraining completato con Replay Strategy + NeuralNetwork.")


if __name__ == "__main__":
    train()
