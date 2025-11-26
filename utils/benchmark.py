from avalanche.benchmarks import nc_benchmark
from torch.utils.data import TensorDataset
import torch

def create_benchmark(train_ds, test_ds, split_type="single"):    
    train_dataset=TensorDataset(train_ds["X"], train_ds["y"])
    test_dataset=TensorDataset(test_ds["X"], test_ds["y"])

    unique_classes = torch.unique(train_ds["y"]).tolist()
    unique_classes.sort()
    n_classes = len(unique_classes)

    if split_type == "single":
        class_splits = [unique_classes] #all classes in one experience
    elif split_type == "three":
        split_sizes = [n_classes // 3, n_classes // 3, n_classes - 2 * (n_classes // 3)]
        class_splits = []
        index = 0
        for size in split_sizes:
            class_splits.append(unique_classes[index:index + size])
            index += size
    elif split_type == "two":
        split_sizes = [n_classes // 2, n_classes // 2]
        class_splits = []
        index = 0
        for size in split_sizes:
            class_splits.append(unique_classes[index:index + size])
            index += size
    else:
        raise ValueError(f"Unsupported split_type: {split_type}")
    
    
    benchmark = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        #n_experiences=num_experiences,
        class_ids_per_experience=class_splits,
        task_labels=False,        # No task ID in CIL
        shuffle=True,
        #seed=123
    )
    
    return benchmark