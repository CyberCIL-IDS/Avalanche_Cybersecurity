import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_metrics(experiences, metrics, dataset, strategy, mode, param):
    
    acc_exp = []
    for i in range(experiences):
        key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{i:03d}"
        if key in metrics:
            acc_exp.append(metrics[key])

    forget_exp = [0.0]
    for i in range(experiences):
        key = f"ExperienceForgetting/eval_phase/test_stream/Task000/Exp{i:03d}"
        if key in metrics:
            forget_exp.append(metrics[key])

    exp_ids = list(range(1, experiences + 1))
    
    plt.figure(figsize=(15,10))
    plt.plot(exp_ids, acc_exp, marker='o', label="Accuracy")
    plt.plot(exp_ids, forget_exp, marker='x', label="Forgetting")
    plt.xlabel("Experience")
    plt.ylabel("Metric")
    plt.title(f"Continual Learning Performance - {strategy} - {mode} - {param}")
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)

    filename = f"utils/plot_{dataset}/metrics_plot_{strategy}_{mode}_{param}.png"
    
    print(f"accuracy: {acc_exp} forgetting: {forget_exp} experiences: {exp_ids}")
    # salva immagine
    plt.savefig(filename)
    plt.close() 

def plot_confusion_matrix(y_true, y_pred, classes, filename, strategy, mode, param):
    """
    Genera, mostra e salva la matrice di confusione normalizzata.
    
    Args:
        y_true (array): Etichette vere (indici numerici).
        y_pred (array): Etichette predette (indici numerici).
        classes (list): Lista dei nomi delle classi.
        filename (str): Percorso dove salvare l'immagine.
        title (str): Titolo del grafico.
    """
    # Calcola la matrice di confusione
    cm = confusion_matrix(y_true, y_pred)
    
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.title(f"Confusion matrix - {strategy} - {mode} - {param}", fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix salvata in: {filename}")