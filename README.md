# 🇮🇹 Avalanche Cybersecurity - IDS Continual Learning
[🇬🇧 **Read this in English**](README_ENG.md)

Questo progetto implementa un sistema di **Intrusion Detection System (IDS)** basato su tecniche di **Continual Learning** utilizzando la libreria [Avalanche](https://avalanche.continualai.org/). Il sistema è progettato per adattarsi a nuovi tipi di attacchi nel tempo senza dimenticare le conoscenze apprese in precedenza (prevenzione del *Catastrophic Forgetting*).

## 🚀 Funzionalità Principali

* **Dataset Supportati**:
    * **UNSW-NB15**: Dataset per la rilevazione di intrusioni di rete.
    * **CICIDS-2017**: Dataset completo per attacchi moderni (DDoS, Brute Force, ecc.).
* **Strategie di Continual Learning**:
    * **DER** (Dark Experience Replay)
    * **MER** (Meta-Experience Replay)
    * **ICaRL** (Incremental Classifier and Representation Learning)
* **Modalità di Benchmark**:
    * **Fixed**: Apprendimento su task fissi.
    * **Incremental**: Apprendimento incrementale (Class-Incremental o Domain-Incremental).
    * **Half**: Apprendimento diviso (metà classi nel primo step, metà nel secondo).
* **Ottimizzazione Automatica**: Integrazione con **Optuna** per la ricerca degli iperparametri ottimali.
* **Preprocessing Avanzato**: Pipeline automatizzate per pulizia dati, feature engineering e bilanciamento delle classi.

## 📂 Struttura del Progetto

```text
.
├── checkpoints/       # Modelli addestrati (.pth)
├── datasets/          # File CSV dei dataset (UNSW_NB15, CICIDS_2017)
├── models/            # Architetture delle reti neurali
├── preprocessing/     # Script per caricamento e trasformazione dati
├── utils/             # Moduli di utilità (benchmark, plotting, training)
│   ├── optuna/        # Risultati dell'ottimizzazione
│   └── strategy.py    # Implementazione strategie (DER, MER, ICaRL)
├── config.yaml        # File di configurazione principale
├── main.py            # Script principale per il training
├── optimization.py    # Script per l'ottimizzazione degli iperparametri
├── predict.py         # Script per l'inferenza sui dati di test
└── requirements.txt   # Dipendenze Python
```

## 🛠️ Installazione

Assicurati di avere Python installato (versione consigliata >= 3.10). Installa le dipendenze necessarie:

```bash
pip install -r requirements.txt
```

*Nota: Il progetto richiede PyTorch con supporto CUDA se si desidera utilizzare l'accelerazione GPU.*

## ⚙️ Configurazione Dettagliata

Tutta la logica di esecuzione è centralizzata nel file `config.yaml`. Di seguito viene spiegato come configurare le sezioni critiche per ottenere i migliori risultati.

### 1. Sezione `benchmark`

Questa sezione definisce come verrà eseguito il training principale (`main.py`) e quale scenario di Continual Learning simulare.

* `strategy`: La strategia di apprendimento continuo da utilizzare.
    * Opzioni: `DER`, `MER`, `ICaRL`.
* `mode`: La modalità di suddivisione del dataset e dei task.
    * `fixed`: Training standard su un set fisso di dati.
    * `incremental`: I dati vengono forniti in più step sequenziali (esperienze).
    * `half`: Il dataset viene diviso in due parti uguali (metà delle classi nel primo step, l'altra metà nel secondo).
* `param`: Parametro numerico legato alla modalità scelta. 
    * In modalità `fixed`: impostare a `1`.
    * In modalità `incremental`: Rappresenta il numero di esperienze (step) in cui dividere il training.
    * In modalità `half`: Questo valore viene ignorato.
* `best_params_path`: Percorso assoluto o relativo al file JSON contenente gli iperparametri ottimizzati (generato da Optuna). Esempio: `"utils/optuna/optuna_best_params2.json"`.

### 2. Sezione `optuna`

Questa sezione controlla lo script di ottimizzazione (`optimization.py`). È fondamentale comprendere il parametro `all` per decidere su cosa iterare.

* `n_trials`: Numero di tentativi di ottimizzazione per ogni configurazione (es. 20 per test rapidi, 100+ per risultati robusti).
* `pruner`: Algoritmo per interrompere i trial non promettenti (es. `"HyperbandPruner"` o `"MedianPruner"`).
* `all` **(IMPORTANTE)**: Determina l'ambito della ricerca.
    * `true`: Optuna **ignora** le impostazioni singole della sezione `benchmark` ed esegue un ciclo nidificato su TUTTE le liste definite sotto `optuna`.
        * Esempio: Se `modes: ["fixed", "incremental"]` e `strategies: ["DER", "MER"]`, Optuna cercherà i migliori parametri per: `(fixed+DER)`, `(fixed+MER)`, `(incremental+DER)`, ecc.
    * `false`: Optuna ottimizzerà **solamente** la specifica configurazione (`mode/strategy/param`) attualmente attiva nella sezione `benchmark`.
* **Liste di iterazione (usate solo se `all: true`)**:
    * `modes`: Lista delle modalità da testare (es. `["half", "incremental"]`).
    * `strategies`: Lista delle strategie (es. `["ICaRL", "MER"]`).
    * `params`: Lista dei parametri numerici (es. `[1, 2, 5]`).

### 3. Esempio di Workflow `config.yaml`
Se vuoi ottimizzare e poi addestrare un modello **ICaRL** in modalità **Incremental** con **5 step**:

1. Imposta `optuna.all: false`.
2. Imposta `benchmark.strategy: "ICaRL"`.
3. Imposta `benchmark.mode: "incremental"`.
4. Imposta `benchmark.param: 5`.
5. Esegui `python optimization.py`.
6. Aggiorna `benchmark.best_params_path` con il file JSON appena creato.
7. Esegui `python main.py`.

## ▶️ Utilizzo

Il flusso di lavoro tipico prevede tre fasi: Ottimizzazione, Addestramento e Predizione.

### 1. Ottimizzazione degli Iperparametri (Opzionale)
Prima di addestrare il modello finale, è consigliabile cercare i migliori iperparametri (learning rate, batch size, ecc.).

```bash
python optimization.py [output_path.json]
```

### 2. Training e Benchmark
Esegui il training utilizzando la configurazione specificata in `config.yaml`. Lo script caricherà automaticamente i migliori parametri trovati nel JSON puntato da `best_params_path`.

```bash
python main.py
```

* Il modello addestrato verrà salvato nella cartella `checkpoints/`.
* I grafici delle metriche (accuracy, forgetting) verranno salvati in `utils/plot_NOMEDATASET/`.

### 3. Predizione (Inferenza)
Per effettuare predizioni su un set di dati di test e generare la matrice di confusione:

```bash
python predict.py
```

* Assicurati che in `config.yaml` siano impostati correttamente `strategy`, `mode` e `param` corrispondenti al checkpoint che vuoi caricare.
* I risultati verranno salvati in `datasets/predictions.csv`.

## 📊 Output e Risultati

* **Modelli**: Salvati come `model_checkpoint_{strategy}_{mode}_{param}.pth`.
* **Preprocessing**: Gli oggetti `preprocessor` e `label_encoder` vengono salvati come `.pkl` per garantire la coerenza tra training e inferenza.
* **Grafici**: Vengono generate matrici di confusione e grafici dell'andamento dell'accuratezza durante le fasi incrementali.

## 📝 Requisiti
Le principali librerie utilizzate sono:
* `torch` >= 2.1.0
* `avalanche-lib` >= 0.6
* `optuna` >= 3.0.0
* `pandas`, `numpy`, `scikit-learn`
