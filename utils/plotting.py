import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_metrics(experiences, metrics, strategy, mode):
    
    acc_exp = []
    for i in range(experiences):
        key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{i:03d}"
        if key in metrics:
            acc_exp.append(metrics[key])

    # Forgetting per experience
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
    plt.title(f"Continual Learning Performance - {strategy}")
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.legend()
    plt.grid(True)

    filename = f"utils/plot/metrics_plot_{strategy}_{mode}.png"
    
    print(f"accuracy: {acc_exp} forgetting: {forget_exp} experiences: {exp_ids}")
    # salva immagine
    plt.savefig(filename)
    plt.close()  # chiudi figura per liberare memoria
