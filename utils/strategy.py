import torch
from avalanche.training import ICaRL, DER, MER
import os

def getStrategy(strategy_type, model, optimizer, current_epochs, eval_plugin, device, plugins_list=None, batch_size=-1):

    plugins_list = [] if plugins_list is None else plugins_list

    NUM_WORKERS = 12       
    PIN_MEMORY = True     
    HIGH_EVAL_BATCH_SIZE = 4096 

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile fallito (proseguo senza): {e}")        

    if strategy_type == "ICaRL":
        if batch_size < 512:
            print(f"WARN: ICaRL Batch size aumentato da {batch_size} a 512 per velocità.")
            batch_size = 512
        
        current_epochs = max(current_epochs, 60)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if strategy_type == "MER":
        if batch_size == -1: batch_size = 32
        
        strategy = MER(
            model=model,
            optimizer=optimizer,
            criterion=torch.nn.CrossEntropyLoss(),
            train_mb_size=batch_size,
            train_epochs=current_epochs,
            eval_mb_size=HIGH_EVAL_BATCH_SIZE, 
            mem_size=10000,          
            batch_size_mem=batch_size,       
            n_inner_steps=2,         
            beta=0.1, 
            evaluator=eval_plugin,
            device=device,
            plugins=plugins_list
        )

    elif strategy_type == "ICaRL":
        TARGET_BATCH = 8192 
        
        if batch_size < TARGET_BATCH:
            batch_size = TARGET_BATCH
            
        strategy = ICaRL(
            feature_extractor=model.feature_extractor, 
            classifier=model.classifier,
            optimizer=optimizer,
            train_mb_size=batch_size,   
            train_epochs=current_epochs, 
            eval_mb_size=HIGH_EVAL_BATCH_SIZE,
            evaluator=eval_plugin,
            device=device,
            memory_size=20000,    
            buffer_transform=None,
            fixed_memory=True,
            plugins=plugins_list
        )

    elif strategy_type == "DER":
        if batch_size == -1: batch_size = 32
        
        strategy = DER(
            model=model,
            optimizer=optimizer,
            criterion=torch.nn.CrossEntropyLoss(),
            train_mb_size=batch_size,
            train_epochs=current_epochs,
            eval_mb_size=HIGH_EVAL_BATCH_SIZE,
            evaluator=eval_plugin,
            device=device,
            mem_size=10000,
            alpha=0.3,
            plugins=plugins_list
        )

    else:
        print(f"Strategy {strategy_type} not recognized.")
        SystemExit(1)

    loader_args = {
        'num_workers': NUM_WORKERS, 
        'pin_memory': PIN_MEMORY,
        'persistent_workers': True,  
        'prefetch_factor': 8         
    }

    try:
        strategy.train_mb_num_workers = NUM_WORKERS
        strategy.eval_mb_num_workers = NUM_WORKERS
        strategy.train_mb_pin_memory = PIN_MEMORY
        strategy.eval_mb_pin_memory = PIN_MEMORY
        
        if hasattr(strategy, 'dataloader_kwargs'):
             strategy.dataloader_kwargs = loader_args

        if hasattr(strategy, '_dataloader_kwargs'):
             strategy._dataloader_kwargs = loader_args
             
    except AttributeError:
        pass

    return strategy