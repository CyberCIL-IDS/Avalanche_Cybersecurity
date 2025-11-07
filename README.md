# 🛡️ CIL-IDS: An Adaptive Intrusion Detection System Based on Class Incremental Learning

## 📘 Overview

**CIL-IDS** (Class Incremental Learning-based Intrusion Detection System) is a cybersecurity project that aims to build an **adaptive IDS** capable of detecting **continuously evolving cyber threats**.  
Traditional Intrusion Detection Systems struggle to adapt to new types of attacks without retraining from scratch — this project applies **Class Incremental Learning (CIL)** techniques to allow the model to **learn new attack classes over time** while mitigating **catastrophic forgetting**.

---

## 🧩 Project Architecture

The project is structured around three main components:

1. **Data Management**
2. **Model Management**
3. **Evaluation & Analysis**

Each module is designed for flexibility, experimentation, and reproducibility.

---

## ⚙️ 1. Data Management

### 1.1 Dataset Selection
Select one or more benchmark IDS datasets suitable for incremental learning:
- **CICIDS2017**
- **UNSW-NB15**
- **NSL-KDD**
- **TON_IoT**

The dataset must contain labeled attack classes to simulate the arrival of new threats.

### 1.2 Data Preprocessing
Steps:
- Data cleaning (remove duplicates, handle missing values)
- Feature normalization or standardization
- Categorical encoding (e.g., one-hot, label encoding)
- Feature selection or dimensionality reduction (e.g., PCA)
- Train/test split strategy by **incremental tasks**, e.g.:
  - Task 1 → Normal + Attack A  
  - Task 2 → Add Attack B  
  - Task 3 → Add Attack C

### 1.3 Data Loaders for CIL
Implement task-based data loaders that:
- Load data corresponding to specific tasks/classes
- Support balanced or unbalanced scenarios
- Manage incremental updates efficiently

---

## 🤖 2. Model Management

### 2.1 Model Architecture
Choose a base model depending on data type:
- **Tabular data** → MLP (Multi-Layer Perceptron)
- **Flow or sequence data** → 1D CNN or LSTM

### 2.2 Class Incremental Learning Strategies
Mitigate catastrophic forgetting using one of the following approaches:

#### Regularization-Based
- **EWC (Elastic Weight Consolidation)**
- **LwF (Learning without Forgetting)**

#### Replay-Based
- Store exemplars from previous tasks
- Rehearsal or generative replay

#### Dynamic Architecture-Based
- Progressive networks or dynamically expandable models

### 2.3 Incremental Training Loop

The core of the system is the **incremental learning process**, where new attack classes are introduced progressively.  
The model learns from new data without forgetting previous knowledge.

#### Training Flow Example

```python
for task_id, task_data in enumerate(incremental_tasks):
    print(f"Training on Task {task_id + 1}")
    
    # Load data for the current task
    train_loader, test_loader = load_task_data(task_data)
    
    # Train model incrementally
    model.train_incrementally(train_loader)
    
    # Evaluate on all seen tasks so far
    evaluate_model(model, seen_tasks)
    
    # Save model checkpoint
    save_model_checkpoint(model, task_id)
```

## ⚙️ 2.4 Forgetting Management

To ensure that the IDS maintains performance on previously learned attack classes, it is essential to monitor and mitigate **catastrophic forgetting**.

### 🎯 Goals
- Track **old class performance degradation** after learning new tasks.  
- Use **rehearsal (replay)** or **regularization-based** strategies to reduce forgetting.

### 🧩 Common Approaches
- **Replay-based:** Store a small memory buffer of old samples and replay them during new training phases.
- **Regularization-based:** Penalize updates to weights that are crucial for previously learned tasks (e.g., Elastic Weight Consolidation).

---

## 📊 3. Evaluation & Analysis

### 3.1 Metrics

Key evaluation metrics for CIL-IDS include:

- **Accuracy per task**
- **Average accuracy** across all tasks
- **Forgetting measure (F)**

Block formula:
```math
F = \frac{1}{T-1} \sum_{i=1}^{T-1} \max_{t < T} (A_{i,t} - A_{i,T})
```

where:
- (A<sub>i,t</sub>) = accuracy on task *i* after learning task *t*
- (A<sub>i,T</sub>) = accuracy on task *i* after completing all *T* tasks

### Detection Metrics
For intrusion detection performance:
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**

### 3.2 Visualization

To interpret performance trends and diagnose forgetting, visualize:

- 📈 **Accuracy vs. Number of Tasks**  
  Shows how overall performance evolves across incremental steps.
  
- 📉 **Forgetting Curve**  
  Illustrates the extent of degradation in old task accuracy as new tasks are introduced.
  
- 🔍 **Confusion Matrix per Task**  
  Displays classification behavior and misclassifications per task or attack class.

### 3.3 Ablation & Comparison

Perform comparative experiments to understand model and method behavior:

- With vs. without **replay** mechanisms  
- Across different **CIL methods** (e.g., EWC vs. LwF vs. Replay)  
- Using various **model architectures** (e.g., MLP, CNN, LSTM)  

Ablation helps identify which strategies best mitigate forgetting while maintaining adaptability.

---

## 🧩 4. (Optional) System Integration

To make your project more **real-world ready** and maintainable:

### 🧱 Modular Architecture
Organize your code into independent components:
- **Data Management** → datasets, preprocessing, loaders  
- **Model Management** → base model, CIL strategies, training loop  
- **Evaluation Module** → metrics, visualization, analysis  

### 💻 Experiment Control
Add a **CLI or dashboard** to:
- Select datasets and incremental configurations  
- Switch between CIL methods (EWC, LwF, Replay)  
- Run experiments with custom parameters  

### 📋 Logging & Tracking
Integrate experiment tracking tools:
- [Weights & Biases (wandb)](https://wandb.ai/)  
- [MLflow](https://mlflow.org/)  
- or custom CSV/JSON logging

These tools allow reproducibility, visualization, and better comparison of experiments.

---

## 🚀 Example Workflow Summary

A typical workflow for the **CIL-IDS** system:

1. **Load Dataset**
2. **Split into Incremental Tasks**
3. **Initialize Base Model**
4. **Loop over Tasks:**
   - Train model incrementally  
   - Evaluate on all seen tasks  
   - Compute forgetting and accuracy metrics  
   - Save checkpoints and logs  
5. **Visualize and Analyze Results**

This workflow ensures the IDS evolves with new data while retaining its ability to detect older threats effectively.

---

## 🧰 5. Tools & Frameworks

To efficiently develop, train, and evaluate your CIL-IDS model, consider using the following frameworks and libraries:

### 🔥 Core Machine Learning Framework

- **[PyTorch](https://pytorch.org/)** – Open-source deep learning framework ideal for research and experimentation.  
  📘 *Guide:* [PyTorch Tutorials](https://pytorch.org/tutorials/)  
  - Supports dynamic computation graphs  
  - Easily integrates with CIL libraries like Avalanche  
  - Widely used for neural network models in cybersecurity and AI research

---

### 🧩 Incremental Learning Frameworks

- **[Avalanche](https://avalanche.continualai.org/)** – A powerful library for **continual learning and CIL experiments**.  
  📘 *Getting Started:* [Avalanche Tutorials](https://avalanche.continualai.org/getting-started/tutorials)  
  - Built on top of PyTorch  
  - Provides ready-to-use CIL strategies (EWC, LwF, Replay, etc.)  
  - Offers standardized benchmarking and evaluation tools  

---

### 📊 Experiment Tracking & Visualization

- **[Weights & Biases (wandb)](https://wandb.ai/)** – For tracking experiments, hyperparameters, and model performance visually.  
  📘 *Guide:* [W&B Documentation](https://docs.wandb.ai/)

- **[MLflow](https://mlflow.org/)** – Manage experiments, model versions, and results reproducibly.  
  📘 *Guide:* [MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html)

- **[TensorBoard](https://www.tensorflow.org/tensorboard)** – For plotting metrics and inspecting model performance visually.

---

### 🧮 Data Handling & Preprocessing

- **[Pandas](https://pandas.pydata.org/)** – For data manipulation, feature engineering, and exploratory analysis.  
- **[NumPy](https://numpy.org/)** – For efficient numerical computation.  
- **[Scikit-learn](https://scikit-learn.org/stable/)** – For preprocessing (scaling, encoding, feature selection) and baseline models.  
- **[Matplotlib](https://matplotlib.org/)** / **[Seaborn](https://seaborn.pydata.org/)** – For creating accuracy/forgetting plots and confusion matrices.

---

### ⚙️ Development & Environment

- **[Google Colab](https://colab.research.google.com/)** – Free GPU environment for training small to medium models.  
- **[Jupyter Notebooks](https://jupyter.org/)** – Interactive experimentation and visualization.  
- **[VS Code](https://code.visualstudio.com/)** – IDE for managing modular codebases.  
- **[Docker](https://www.docker.com/)** – For reproducible and portable environments.

---

### 🔒 Cybersecurity Datasets

- **[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)**  
- **[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)**  
- **[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)**  
- **[TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets)**  

Use these datasets to simulate incremental attack scenarios.

---

## 📚 Keywords
`Intrusion Detection` • `Class Incremental Learning` • `PyTorch` • `Avalanche` • `Catastrophic Forgetting` • `EWC` • `LwF` • `Replay` • `Experiment Tracking`

---

📧 **Authors:**

<table>
  <tr>
    <td align="center">
      <strong>Stefano Casini</strong><br>
      MSc Software Engineering – University of Bologna<br>
      <a href="mailto:stefano.casini2@studio.unibo.it">stefano.casini2@studio.unibo.it</a>
    </td>
    <td align="center">
      <strong>Luigi Lauriola</strong><br>
      MSc Software Engineering – University of Bologna<br>
      <a href="mailto:luigi.lauriola2@studio.unibo.it">luigi.lauriola2@studio.unibo.it</a>
    </td>
    <td align="center">
      <strong>Francesco Rondini</strong><br>
      MSc Software Engineering – University of Bologna<br>
      <a href="mailto:francesco.rondini@studio.unibo.it">francesco.rondini@studio.unibo.it</a>
    </td>
  </tr>
</table>

