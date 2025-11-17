import matplotlib.pyplot as plt

def plot_metrics(experiences, accuracy, forgetting):
    plt.figure(figsize=(8,5))
    plt.plot(experiences, accuracy, marker='o', label="Accuracy")
    plt.plot(experiences, forgetting, marker='x', label="Forgetting")
    plt.xlabel("Experience")
    plt.ylabel("Metric")
    plt.title("Continual Learning Performance")
    plt.legend()
    plt.grid(True)
    plt.show()
