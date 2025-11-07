import torch
from dataset import get_dataloaders
from model import NeuralNetwork
from train import train, test
import torch.nn as nn
import os

# Parametri
train_csv = "datasets/UNSW_NB15_training-set(in).csv"
test_csv = "datasets/UNSW_NB15_testing-set(in).csv"
checkpoint_path = "checkpoint.pth"
start_epoch = 0
batch_size = 64
epochs = 10
lr = 1e-4

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# DataLoader e info dataset
train_loader, test_loader, input_size, num_classes = get_dataloaders(train_csv, test_csv, batch_size)

# Modello
model = NeuralNetwork(input_size, num_classes).to(device)
print(model)

# Loss e optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Carica checkpoint se esiste
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Ripreso training dall'epoca {start_epoch}")

# Training loop
for t in range(start_epoch + epochs):
    print(f"Epoch {start_epoch + t + 1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer, device)
    test(test_loader, model, loss_fn, device)

    # Salva checkpoint a fine epoca
    torch.save({
        'epoch': t,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint salvato a epoca {start_epoch + t + 1}")
