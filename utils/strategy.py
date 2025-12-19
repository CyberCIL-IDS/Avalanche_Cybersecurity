import torch
from avalanche.training import ICaRL, DER, MER
import os

def getStrategy(strategy_type, model, optimizer, current_epochs, eval_plugin, device, plugins_list=None, batch_size=-1):

    plugins_list = [] if plugins_list is None else plugins_list

    # --- CONFIGURAZIONE PERFORMANCE (CPU -> GPU) ---
    # Usa tanti worker quanti sono i core fisici (es. 4 o 8)
    NUM_WORKERS = 8  
    
    # Pin Memory velocizza il trasferimento RAM -> VRAM
    PIN_MEMORY = True 

    # Valutazione sempre gigante
    HIGH_EVAL_BATCH_SIZE = 4096 

    # --- 1. CREAZIONE STRATEGIA ---
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
            n_inner_steps=5,         
            beta=0.1, 
            evaluator=eval_plugin,
            device=device,
            plugins=plugins_list
            # Non passiamo num_workers qui per evitare crash su v0.6
        )

    elif strategy_type == "ICaRL":
        if batch_size == -1: batch_size = 128 

        strategy = ICaRL(
            feature_extractor=model.feature_extractor, 
            classifier=model.classifier,
            optimizer=optimizer,
            train_mb_size=batch_size,   
            train_epochs=current_epochs, 
            eval_mb_size=HIGH_EVAL_BATCH_SIZE,
            evaluator=eval_plugin,
            device=device,
            memory_size=10000,    
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

    # --- 2. ATTIVAZIONE PARALLELISMO (WORKAROUND v0.6) ---
    # Impostiamo i worker "da fuori" per bypassare il costruttore rigido
    try:
        strategy.train_mb_num_workers = NUM_WORKERS
        strategy.eval_mb_num_workers = NUM_WORKERS
        strategy.train_mb_pin_memory = PIN_MEMORY
        strategy.eval_mb_pin_memory = PIN_MEMORY
        
        # Opzionale: Persistent workers evita di ricreare i processi ad ogni epoca
        # Funziona se Avalanche passa questi kwargs al DataLoader
        if hasattr(strategy, 'dataloader_kwargs'):
             strategy.dataloader_kwargs = {
                 'num_workers': NUM_WORKERS, 
                 'pin_memory': PIN_MEMORY,
                 'persistent_workers': True # <--- Boost extra
             }
             
    except AttributeError:
        pass

    return strategy