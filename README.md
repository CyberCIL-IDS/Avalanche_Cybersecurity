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

\[
F = \frac{1}{T-1} \sum_{i=1}^{T-1} \max_{t < T} (A_{i,t} - A_{i,T})
\]

where:

- \( A_{i,t} \) = accuracy on task *i* after learning task *t*  
- \( A_{i,T} \) = accuracy on task *i* after completing all *T* tasks

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

📚 **Keywords:**  
`Intrusion Detection` • `Incremental Learning` • `Catastrophic Forgetting` • `Replay` • `EWC` • `LwF` • `Evaluation`

