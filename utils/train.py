# utils/train.py (o train.py)

import torch
from avalanche.training.strategies import Replay
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger

from model import NeuralNetwork       # <-- usa il tuo modello
from benchmark import get_benchmark   # <-- benchmark già preprocessato e pronto


def train():

    # 1️⃣ Carica il benchmark (già preprocessato dai tuoi script)
    benchmark = get_benchmark()

    # 2️⃣ Istanzia il modello
    model = NeuralNetwork()

    # 3️⃣ Imposta ottimizzatore
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4️⃣ Plugin per logging ed evaluation
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger()]
    )

    # 5️⃣ Strategy Replay — continual learning con memory buffer
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

    # 6️⃣ Training sulle esperienze incrementali
    for experience in benchmark.train_stream:
        print(f"\n▶️ Training su: {experience}")
        strategy.train(experience)

        print("\n📊 Evaluation:")
        strategy.eval(benchmark.test_stream)

    print("\n✅ Training completato con Replay Strategy + NeuralNetwork.")


if __name__ == "__main__":
    train()
