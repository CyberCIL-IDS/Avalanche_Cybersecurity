from avalanche.benchmarks import nc_benchmark
from torch.utils.data import TensorDataset
import torch
import math


def create_benchmark(train_ds, test_ds, mode="single", param=None):
    """
    Create CIL benchmark with 5 possible modes:
    
    Modes:
        - "single": all classes in one exp
        - "two": split into 2 experiences (balanced)
        - "three": split into 3 experiences (balanced-ish)
        - "fixed": fixed number of classes per exp (param required)
        - "incremental": experiences grow: param, param+1, rest

    Args:
        train_ds, test_ds : dict {"X": Tensor, "y": Tensor}
        mode : str
        param : int (required for "fixed" and "incremental")
    """
    
    train_dataset = TensorDataset(train_ds["X"], train_ds["y"])
    test_dataset = TensorDataset(test_ds["X"], test_ds["y"])

    unique_classes = torch.unique(train_ds["y"]).tolist()
    unique_classes.sort()
    print("Unique classess:", unique_classes)
    n_classes = len(unique_classes)
    print(f"Total number of classes: {n_classes}")

    # -------------------------------------------------------
    # MODE 1: all classes in 1 experience
    # -------------------------------------------------------
    if mode == "single":
        class_splits = [unique_classes]

    # -------------------------------------------------------
    # MODE 2: split into 2 experiences (balanced)
    # -------------------------------------------------------
    elif mode == "two":
        split_sizes = [n_classes // 2, n_classes - n_classes // 2]
        idx = 0
        class_splits = []
        for size in split_sizes:
            class_splits.append(unique_classes[idx:idx+size])
            idx += size

    # -------------------------------------------------------
    # MODE 3: split into 3 experiences (balanced-ish)
    # -------------------------------------------------------
    elif mode == "three":
        base = n_classes // 3
        split_sizes = [base, base, n_classes - 2 * base]
        idx = 0
        class_splits = []
        for size in split_sizes:
            class_splits.append(unique_classes[idx:idx+size])
            idx += size

    # -------------------------------------------------------
    # MODE 4: fixed-size increments (requires param)
    # Example: 10 classes, param=2 => 2+2+2+2+2
    # -------------------------------------------------------
    elif mode == "fixed":
        if param is None:
            raise ValueError("mode 'fixed' requires param")

        n_exp = math.ceil(n_classes / param)
        class_splits = []
        idx = 0
        for _ in range(n_exp):
            class_splits.append(unique_classes[idx:idx+param])
            idx += param

    # -------------------------------------------------------
    # MODE 5: incremental size increments (requires param)
    # Example: 10 classes, param=2 => 2 + 3 + 5
    # -------------------------------------------------------
    elif mode == "incremental":
        if param is None:
            raise ValueError("mode 'incremental' requires param")

        first = param                 # e.g., 2
        second = param + 1            # e.g., 3
        third = n_classes - (first + second)

        if third < 0:
            raise ValueError("param too large: incremental split impossible")

        split_sizes = [first, second, third]
        idx = 0
        class_splits = []
        for size in split_sizes:
            class_splits.append(unique_classes[idx:idx+size])
            idx += size

    else:
        raise ValueError("Unsupported mode")
    
    per_exp_classes = {i: len(exp) for i, exp in enumerate(class_splits)}
    n_experiences = len(class_splits)

    # --------- Create Avalanche benchmark ----------
    return nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=n_experiences,
        per_exp_classes = per_exp_classes,
        task_labels=False,
        shuffle=False
    )
