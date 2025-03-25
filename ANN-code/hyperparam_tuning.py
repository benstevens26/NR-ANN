"""
hyperparam_tuning.py - Hyperparameter and Architecture Tuning for LENRI

This script handles:
- Defining the LENRI model with dynamic architecture
- Implementing hyperparameter tuning using Optuna
- Training LENRI with various hyperparameters and architectures
- Selecting the best performing configuration
- Saving the best model and hyperparameters
"""

import torch
import torch.optim as optim
import torch.nn as nn
import optuna
from feature_preprocessing import get_dataloaders_cf4, get_dataloaders_ar_cf4

# Toggle binary classification mode
BINARY = False  # Set to False for multi-class (C, F, Ar)
features_path = "ANN-code/Data/features_Ar_CF4_3_processed.csv"
num_classes = 2 if BINARY else 3  # Adjust number of output classes
input_size = 32  # Number of input features

def save_best_model(model, trial):
    """Saves the best model with its hyperparameters."""
    model_path = f"lenri_best_{trial.number}.pth"
    torch.save(
        {"model_state_dict": model.state_dict(), "hyperparameters": trial.params},
        model_path,
    )
    print(f"Best model saved as {model_path}")

class LENRI(nn.Module):
    def __init__(
        self,
        input_size=input_size,
        hidden_layers=[128, 64, 48, 32],
        dropout_rate=0.2,
        num_classes=num_classes,
    ):
        super(LENRI, self).__init__()
        layers = []
        prev_size = input_size

        for layer_size in hidden_layers:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = layer_size

        layers.append(nn.Linear(prev_size, num_classes))  # Adjust output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_model(
    learning_rate,
    batch_size,
    dropout_rate,
    hidden_layers,
    trial,
    weight_decay,
    num_epochs=50,
    patience=5,
):
    """Trains LENRI with given hyperparameters and architecture and returns validation loss."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if BINARY:
        train_loader, val_loader, _ = get_dataloaders_cf4(features_path, batch_size=batch_size)
    else:
        train_loader, val_loader, _ = get_dataloaders_ar_cf4(features_path, batch_size=batch_size)

    model = LENRI(hidden_layers=hidden_layers, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stopping_counter = 0
            save_best_model(model, trial)
        else:
            stopping_counter += 1
            if stopping_counter >= patience:
                print("Early stopping triggered. Training halted.")
                break


        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val_loss

def objective(trial):
    """Objective function for Optuna hyperparameter tuning using validation loss."""
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    num_layers = trial.suggest_int("num_layers", 2, 5)
    hidden_layers = [trial.suggest_int(f"n_units_layer_{i}", 16, 128, log=True) for i in range(num_layers)]

    val_loss = train_model(
        learning_rate, batch_size, dropout_rate,
        hidden_layers, trial, weight_decay
    )
    
    trial.report(val_loss, step=1)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return val_loss

def objective_simple(trial):
    """Objective function for Optuna hyperparameter tuning without tuning model layer parameters."""
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    hidden_layers = [128, 64, 48, 32] # choose hidden layers

    val_loss = train_model(
        learning_rate, batch_size, dropout_rate,
        hidden_layers, trial, weight_decay
    )
    
    trial.report(val_loss, step=1)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective_simple, n_trials=30)

print("Best trial:", study.best_trial.number)
print("Best hyperparameters:", study.best_params)
