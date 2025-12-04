import torch
from avalanche.training import ICaRL, DER, MER

def getStrategy(strategy_type, model, optimizer, current_epochs, eval_plugin, device, plugins_list=None, batch_size=-1):

    plugins_list = [] if plugins_list is None else plugins_list

    if strategy_type == "MER":
        #batch_size = 64
        if batch_size == -1:
            batch_size = 64
        
        if strategy_type == "MER":
            strategy = MER(
                model=model,
                optimizer=optimizer,
                criterion=torch.nn.CrossEntropyLoss(),
                train_mb_size=256,
                train_epochs=current_epochs,
                eval_mb_size=64,
                mem_size=10000,          # Dimensione totale del buffer
                batch_size_mem=batch_size,       # Quanti esempi di memoria usare per step (Specifico MER)
                n_inner_steps=1,         # Passi di ottimizzazione interna (Specifico MER)
                beta=0.05,                # Parametro di update meta-learning (Specifico MER)
                evaluator=eval_plugin,
                device=device,
                plugins=plugins_list
            )

    elif strategy_type == "ICaRL":
        #batch_size = 128
        if batch_size == -1:
            batch_size = 128

        strategy = ICaRL(
            feature_extractor=model.feature_extractor, 
            classifier=model.classifier,
            optimizer=optimizer,
            train_mb_size=batch_size,   
            train_epochs=current_epochs, 
            eval_mb_size=batch_size,
            evaluator=eval_plugin,
            device=device,
            memory_size=5000,    
            buffer_transform=None,
            fixed_memory=True,
            plugins=plugins_list 
        )

    elif strategy_type == "DER":
        #batch_size = 64
        if batch_size == -1:
            batch_size = 64
        
        strategy = DER(
            model=model,
            optimizer=optimizer,
            criterion=torch.nn.CrossEntropyLoss(),
            train_mb_size=batch_size,
            train_epochs=current_epochs,
            eval_mb_size=batch_size,
            evaluator=eval_plugin,
            device=device,
            mem_size=5000,
            alpha=0.3,
            plugins=plugins_list
        )

    else:
        print(f"Strategy {strategy_type} not recognized.")
        SystemExit(1)
    return strategy