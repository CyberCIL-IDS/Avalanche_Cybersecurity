import json
import logging
import os

# Configurazione base del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_json(path="output_all_p1.json"):
    """
    Carica un file JSON contenente i risultati dell'ottimizzazione e restituisce
    una struttura organizzata con i migliori parametri per ogni configurazione.
    """
    
    parsed_results = []

    # 1. Verifica esistenza file
    if not os.path.exists(path):
        logging.error(f"Il file non è stato trovato al percorso: {path}")
        return []

    try:
        with open(path, 'r') as file:
            data = json.load(file)
            logging.info(f"File '{path}' caricato con successo. Trovate {len(data)} entry.")

        # 2. Parsing delle variabili
        for entry in data:
            meta_info = {
                "mode": entry.get("mode"),
                "param": entry.get("param"),
                "strategy": entry.get("strategy"),
                "trial": entry.get("trial")
            }
            
            hyperparameters = entry.get("best_value", {})  
            score = entry.get("best_params")              

            # Creazione oggetto strutturato
            parsed_item = {
                "meta": meta_info,
                "best_hyperparameters": hyperparameters, 
                "best_score": score                      
            }
            parsed_results.append(parsed_item)

        # 3. Stampa risultati
        print("\n--- RISULTATI ESTRATTI ---")
        for v in parsed_results:
            strat = v['meta']['strategy']
            mode = v['meta']['mode']
            param = v['meta']['param']
            
            print(f"CONFIG: {strat} | Mode: {mode} | Param: {param}")
            print(f"SCORE:  {v['best_score']}")
            print(f"PARAMS: {v['best_hyperparameters']}")
            print("-" * 50)

        return parsed_results

    except json.JSONDecodeError:
        logging.error(f"Errore nella decodifica del file JSON: {path}")
        return []
    except Exception as e:
        logging.error(f"Errore inaspettato durante il parsing: {e}")
        return []

if __name__ == "__main__":
    parse_json()
