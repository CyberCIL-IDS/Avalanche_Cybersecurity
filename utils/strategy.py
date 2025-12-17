import torch
from avalanche.training import ICaRL, DER, MER

def getStrategy(strategy_type, model, optimizer, current_epochs, eval_plugin, device, plugins_list=None, batch_size=-1):

    plugins_list = [] if plugins_list is None else plugins_list

    # --- STRATEGIA MER ---
    if strategy_type == "MER":
        # Default se batch_size non viene passato (es. -1)
        if batch_size == -1:
            batch_size = 32  # MER preferisce batch piccoli per fare più update
        
        strategy = MER(
            model=model,
            optimizer=optimizer,
            criterion=torch.nn.CrossEntropyLoss(),
            
            # FIX 1: Usa il batch_size parametrico, non 256 fisso!
            train_mb_size=batch_size,
            train_epochs=current_epochs,
            eval_mb_size=64,
            
            mem_size=10000,          
            
            # FIX 2: Usa lo stesso batch size per la memoria
            batch_size_mem=batch_size,       
            
            # FIX 3: Aumenta i passi interni. 
            # 1 è troppo poco (è quasi Replay). 5 è il valore standard per avere benefici.
            n_inner_steps=5,         
            
            beta=0.1,  # Aumentato leggermente per dare più peso al meta-update
            evaluator=eval_plugin,
            device=device,
            plugins=plugins_list
        )

    # --- STRATEGIA ICaRL ---
    elif strategy_type == "ICaRL":
        if batch_size == -1:
            batch_size = 128 # ICaRL regge batch più grandi

        strategy = ICaRL(
            feature_extractor=model.feature_extractor, 
            classifier=model.classifier,
            optimizer=optimizer,
            train_mb_size=batch_size,   
            train_epochs=current_epochs, 
            eval_mb_size=batch_size,
            evaluator=eval_plugin,
            device=device,
            memory_size=10000,    
            buffer_transform=None,
            fixed_memory=True,
            plugins=plugins_list 
        )

    # --- STRATEGIA DER ---
    elif strategy_type == "DER":
        if batch_size == -1:
            batch_size = 32
        
        strategy = DER(
            model=model,
            optimizer=optimizer,
            criterion=torch.nn.CrossEntropyLoss(),
            train_mb_size=batch_size,
            train_epochs=current_epochs,
            eval_mb_size=batch_size,
            evaluator=eval_plugin,
            device=device,
            mem_size=10000,
            alpha=0.3,
            plugins=plugins_list
        )

    else:
        print(f"Strategy {strategy_type} not recognized.")
        SystemExit(1)
        
    return strategy