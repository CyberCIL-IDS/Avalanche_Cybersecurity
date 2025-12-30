from avalanche.benchmarks import nc_benchmark
from torch.utils.data import TensorDataset
import torch
import math


def create_benchmark(train_ds, test_ds, mode="single", param=None):

    train_dataset = TensorDataset(train_ds["X"], train_ds["y"])
    test_dataset = TensorDataset(test_ds["X"], test_ds["y"])

    unique_classes = torch.unique(train_ds["y"]).tolist()
    unique_classes.sort()
    print("Unique classess:", unique_classes)
    n_classes = len(unique_classes)
    print(f"Total number of classes: {n_classes}")

    if mode == "fixed":
        if param is None:
            raise ValueError("mode 'fixed' requires param")

        n_exp = math.ceil(n_classes / param)
        class_splits = []
        idx = 0
        for _ in range(n_exp):
            class_splits.append(unique_classes[idx:idx+param])
            idx += param
    elif mode == "incremental":
        if param is None:
            raise ValueError("mode 'incremental' requires param")

        first = param                 # e.g., 2
        second = param + 1            # e.g., 3
        third = n_classes - (second + 2)

        if third < 0:
            raise ValueError("param too large: incremental split impossible")

        if n_classes > 10:
            fourth = n_classes - (third + 3)
            split_sizes = [first, second, third, fourth]
        else:
            split_sizes = [first, second, third]

        idx = 0
        class_splits = []
        for size in split_sizes:
            class_splits.append(unique_classes[idx:idx+size])
            idx += size
    elif mode == "half":
        half = n_classes // 2 # integer division
        class_splits = [unique_classes[:half], unique_classes[half:]]

    else:
        raise ValueError("Unsupported mode")
    
    per_exp_classes = {i: len(exp) for i, exp in enumerate(class_splits)}
    n_experiences = len(class_splits)

    return nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=n_experiences,
        per_exp_classes = per_exp_classes,
        task_labels=False,
        shuffle=False
    )
