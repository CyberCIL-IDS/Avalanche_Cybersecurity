# Avalanche Cybersecurity - IDS Continual Learning

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

## ⚙️ Configurazione

Il comportamento del sistema è controllato dal file `config.yaml`. Ecco i parametri principali da modificare:

* **dataset**: Seleziona il dataset (`UNSW_NB15` o `CICIDS_2017`) e i percorsi dei file CSV.
* **benchmark**:
    * `strategy`: La strategia da usare (`DER`, `MER`, `ICaRL`).
    * `mode`: Modalità (`fixed` o `incremental`).
    * `param`: Numero di esperienze o step incrementali.
    * `best_params_path`: Percorso del file JSON contenente gli iperparametri ottimizzati (es. `optuna/optuna_best_params2.json`).
* **optuna**: Configurazione per la ricerca degli iperparametri (numero di trial, strategie da testare).

## ▶️ Utilizzo

Il flusso di lavoro tipico prevede tre fasi: Ottimizzazione, Addestramento e Predizione.

### 1. Ottimizzazione degli Iperparametri (Opzionale)
Prima di addestrare il modello finale, è consigliabile cercare i migliori iperparametri (learning rate, batch size, ecc.) usando Optuna.

```bash
python optimization.py [output_path.json]
```

* Se non specificato, i risultati verranno salvati nel percorso definito in `config.yaml`.
* Assicurati che il file JSON prodotto sia puntato correttamente in `config.yaml` sotto `best_params_path`.

### 2. Training e Benchmark
Esegui il training utilizzando la configurazione specificata in `config.yaml`. Lo script caricherà automaticamente i migliori parametri trovati nel JSON.

```bash
python main.py
```

* Il modello addestrato verrà salvato nella cartella `checkpoints/`.
* I grafici delle metriche (accuracy, forgetting) verranno salvati in `utils/plot/`.

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
