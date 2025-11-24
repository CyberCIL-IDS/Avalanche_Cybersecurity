import matplotlib.pyplot as plt

def plot_metrics(experiences, accuracy, forgetting, strategy, filename="utils/plot/metrics_plot_Replay.png"):
    plt.figure(figsize=(8,5))
    plt.plot(experiences, accuracy, marker='o', label="Accuracy")
    plt.plot(experiences, forgetting, marker='x', label="Forgetting")
    plt.xlabel("Experience")
    plt.ylabel("Metric")
    plt.title(f"Continual Learning Performance - {strategy}")
    plt.legend()
    plt.grid(True)
    
    # salva immagine
    plt.savefig(filename)
    plt.close()  # chiudi figura per liberare memoria
