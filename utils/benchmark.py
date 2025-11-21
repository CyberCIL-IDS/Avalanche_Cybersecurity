from avalanche.benchmarks import nc_benchmark
from torch.utils.data import TensorDataset

def create_benchmark(train_ds, test_ds):
    
    train_dataset=TensorDataset(train_ds["X"], train_ds["y"])
    test_dataset=TensorDataset(test_ds["X"], test_ds["y"])

    benchmark = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=5,
        task_labels=False,        # No task ID in CIL
        shuffle=True,
        seed=123
    )
    
    return benchmark