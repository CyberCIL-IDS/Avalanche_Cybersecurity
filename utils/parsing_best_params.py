import json
import logging
import os

try:
    from preprocessing.UNSW_NB15.preprocessing_UNSW_NB15 import prepare_UNSW_NB15
    from utils import optuna_custom
    from utils.benchmark import create_benchmark
    from utils.optuna_custom import objective
    from utils.training import train
    from utils.plotting import plot_metrics
    from utils.config_loader import load_config
    from preprocessing.CICIDS_2017.preprocessing_CICIDS_2017 import preprocessing_CICIDS
except ImportError as e:
    print(f"[WARNING] Some custom libraries were not found (error: {e}). "
          "The script will still proceed with parsing the JSON.")

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_json(path="output_all_p1.json"):
    """
    It uploads a JSON file containing the optimization results and returns
    an organized structure with the best parameters for each configuration. 
    """
    
    variabili = []

    # 1. Open the file from the path
    if not os.path.exists(path):
        logging.error(f"The file was not found at the path: {path}")
        return []

    try:
        with open(path, 'r') as file:
            data = json.load(file)
            logging.info(f"File '{path}' uploaded successfully. Find {len(data)} entry.")

        # 2. Parsing variables
        for entry in data:
            # Extracting the context (strategy, modality, etc.)
            meta_info = {
                "mode": entry.get("mode"),
                "param": entry.get("param"),
                "strategy": entry.get("strategy"),
                "trial": entry.get("trial")
            }
            
            # Extracting the params (lr, hidden layers, ecc.)
            b_value = entry.get("best_value", {})
            
            # Extracting the score
            b_params = entry.get("best_params")

            # Creating a structured object
            parsed_item = {
                "meta": meta_info,
                "value": b_value,
                "params": b_params
            }
            variabili.append(parsed_item)

        # 3. Print variables
        print("\n--- PARSATE VARIABLES ---")
        for v in variabili:
            strat = v['meta']['strategy']
            mode = v['meta']['mode']
            p_id = v['meta']['param']
            print(f"BEST VALUE: {v['value']}")
            print(f"BEST PARAMS: {v['params']}\n")

        return variabili

    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON file: {path}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error while parsing: {e}")
        return []

if __name__ == "__main__":
    parse_json()