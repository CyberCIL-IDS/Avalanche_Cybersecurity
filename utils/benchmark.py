from avalanche.benchmarks import nc_benchmark

def create_benchmark(train_ds, test_ds):
    benchmark = nc_benchmark(
        train_dataset=train_ds,
        test_dataset=test_ds,
        n_experiences=5,
        task_labels=False,        # No task ID in CIL
        shuffle=True,
        seed=123
    )
    
    return benchmark