"""
evaluate_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from feature_preprocessing import get_dataloaders_cf4, get_dataloaders_ar_cf4, get_dataloaders_cf4_biased
from models import LENRI_CF4_1, LENRI_Ar_CF4_1, LENRI_CF4_2, LENRI_Ar_CF4_2, LENRI_CF4_3, LENRI_Ar_CF4_3

# Toggle binary classification mode
BINARY = True  # Set to True for binary classification (C vs. F), False for multi-class (C, F, Ar)
biased = False

model_path = "LENRI_CF4_3_opt.pth"
features_path = "ANN-code/Data/features_CF4_3_processed.csv"
save_path = "ANN-code/Data/LENRI-CF4-3"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load test set
if BINARY:
    if biased:
        _, _, test_loader = get_dataloaders_cf4_biased(features_path, batch_size=32)
    else:
        _, _, test_loader = get_dataloaders_cf4(features_path, batch_size=32)
else:
    _, _, test_loader = get_dataloaders_ar_cf4(features_path, batch_size=32)

# Load trained model
model = LENRI_CF4_3().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
print("Loaded Hyperparameters:", checkpoint["hyperparameters"])
model.eval()

# Define loss function
criterion = nn.CrossEntropyLoss()


def evaluate_model():
    """Evaluate model on test set and return predictions and true labels."""
    test_loss = 0
    correct_test = 0
    total_test = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
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

    test_loss /= len(test_loader)
    test_acc = correct_test / total_test

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds):
    """Plot confusion matrix based on classification mode."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    if BINARY:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["C", "F"],
            yticklabels=["C", "F"],
        )
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["C", "F", "Ar"],
            yticklabels=["C", "F", "Ar"],
        )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.show()


def plot_confidence_distribution(probs):
    """Plot distribution of model confidence based on classification mode."""
    plt.hist(probs, bins=20, alpha=0.7, label=["C", "F"] if BINARY else ["C", "F", "Ar"], stacked=True)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Distribution of Model Confidence in Predictions")
    plt.legend()
    plt.savefig(f"{save_path}/confidence_distribution.png")
    plt.show()

def plot_roc_curve(labels, probs):
    """Plot ROC curve and compute AUC.
    
    - If BINARY = True: Plots a single ROC curve for class "F".
    - If BINARY = False: Plots an ROC curve for each class (C, F, Ar) on the same figure.
    """
    plt.figure(figsize=(7, 5))

    if BINARY:
        # Binary classification: Compute ROC for class "F" (index 1)
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        print(roc_auc)
        # Save fpr and tpr to CSV
        roc_data = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})
        roc_data.to_csv(f"roc_curve_data.csv", index=False)

        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color="blue")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random classifier line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title("ROC Curve (Binary: C vs. F)")
    
    else:
        # Multi-class: Compute ROC curve for each class
        n_classes = probs.shape[1]
        colors = ["blue", "red", "green"]
        class_labels = ["C", "F", "Ar"]

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((labels == i).astype(int), probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_labels[i]} AUC = {roc_auc:.4f}", color=colors[i])

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random classifier line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title("ROC Curve (Multi-Class: C, F, Ar)")

    plt.legend()
    plt.savefig(f"{save_path}/roc_curve.png")
    plt.show()

# Run evaluation
labels, preds, probs = evaluate_model()
if BINARY:
    print(f"Precision: {precision_score(labels, preds, average='binary'):.4f}")
    print(f"Recall: {recall_score(labels, preds, average='binary'):.4f}")
    print(f"F1 Score: {f1_score(labels, preds, average='binary'):.4f}")
else:
    print(f"Precision: {precision_score(labels, preds, average='weighted'):.4f}")
    print(f"Recall: {recall_score(labels, preds, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(labels, preds, average='weighted'):.4f}")

    print("Precision per class:", precision_score(labels, preds, average=None))
    print("Recall per class:", recall_score(labels, preds, average=None))
    print("F1 Score per class:", f1_score(labels, preds, average=None))

plot_confusion_matrix(labels, preds)
plot_confidence_distribution(probs)
plot_roc_curve(labels, probs)

