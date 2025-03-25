"""
evaluate_model.py

This script loads a trained LENRI_CF4_3 model, evaluates it on a test dataset,
and saves predictions along with filenames to a CSV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from models import LENRI_CF4_3, LENRI_Ar_CF4_3
from feature_preprocessing import get_dataloaders_cf4, get_dataloaders_ar_cf4

# File paths
model_path = "LENRI_Ar_CF4_3_opt.pth"
features_path = "ANN-code/Data/features_Ar_CF4_3_processed.csv"
save_path = "ANN-code/Data/LENRI-Ar-CF4-3"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to extract labels from filenames
def extract_label_cf4(filename):
    if re.search(r"00_C_", filename):
        return 0  # Carbon
    elif re.search(r"00_F_", filename):
        return 1  # Fluorine
    elif re.search(r"_C_", filename):
        return 0
    elif re.search(r"_F_", filename):
        return 1
    else:
        raise ValueError(f"Unexpected filename format: {filename}")

# Dataset class including filenames
class NuclearRecoilDatasetCF4(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.drop(columns=["cam_path", "ito_path", "label"]).values
        self.labels = dataframe["label"].values
        self.filenames = dataframe["cam_path"].values  # Store filenames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        filename = self.filenames[idx]
        return filename, x, y  # Return filename

# Function to get dataloaders
def get_dataloaders_cf4(csv_file, batch_size=32, verbose=False):
    df = pd.read_csv(csv_file)
    df["label"] = df["cam_path"].apply(extract_label_cf4)
    
    train, test = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
    train, val = train_test_split(train, test_size=0.1765, stratify=train["label"], random_state=42)

    train_dataset = NuclearRecoilDatasetCF4(train)
    val_dataset = NuclearRecoilDatasetCF4(val)
    test_dataset = NuclearRecoilDatasetCF4(test)

    if verbose:
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
            train_dataset,
            val_dataset,
            test_dataset
        )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )

# Load test set
_, _, test_loader = get_dataloaders_cf4(features_path, batch_size=32)

# Load trained model
model = LENRI_CF4_3().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
print("Loaded Hyperparameters:", checkpoint["hyperparameters"])
model.eval()

# Define loss function
criterion = nn.CrossEntropyLoss()

def evaluate_model():
    """Evaluate model on test set and save predictions along with filenames."""
    test_loss = 0
    correct_test = 0
    total_test = 0
    all_labels = []
    all_preds = []
    all_probs = []
    all_filenames = []  # Store filenames

    with torch.no_grad():
        for batch in test_loader:
            filenames, inputs, labels = batch  # Now includes cam_path
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            correct_test += (predictions == labels).sum().item()
            total_test += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            all_filenames.extend(filenames)  # Store filenames

    test_loss /= len(test_loader)
    test_acc = correct_test / total_test

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save predictions and filenames to CSV
    results_df = pd.DataFrame({
        "Filename": all_filenames,
        "Predict_C": [p[0] for p in all_probs],  # Softmax probability for class C
        "Predict_F": [p[1] for p in all_probs],  # Softmax probability for class F
    })

    results_csv_path = f"{save_path}/evaluation_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved evaluation results to {results_csv_path}")

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# Run evaluation
labels, preds, probs = evaluate_model()

# Print key metrics
print(f"Precision: {precision_score(labels, preds, average='binary'):.4f}")
print(f"Recall: {recall_score(labels, preds, average='binary'):.4f}")
print(f"F1 Score: {f1_score(labels, preds, average='binary'):.4f}")

# Plot confusion matrix
def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["C", "F"], yticklabels=["C", "F"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.show()

plot_confusion_matrix(labels, preds)

# Plot ROC curve
def plot_roc_curve(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    roc_data = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})
    roc_data.to_csv(f"{save_path}/roc_curve_data.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve (Binary: C vs. F)")
    plt.legend()
    plt.savefig(f"{save_path}/roc_curve.png")
    plt.show()

plot_roc_curve(labels, probs)
