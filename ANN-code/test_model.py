"""
test_model.py - Model Evaluation Script for LENRI

This script handles:
- Loading the trained LENRI model
- Evaluating the model on the test set
- Computing test loss and accuracy
- Calculating precision, recall, and F1-score
- Providing insights into model performance
"""

import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from feature_preprocessing import get_dataloaders_cf4, get_dataloaders_ar_cf4
from models import LENRI_CF4_1, LENRI_Ar_CF4_1, LENRI_CF4_2, LENRI_Ar_CF4_2, LENRI_CF4_3, LENRI_Ar_CF4_3

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "lenri_model_best.pth"
features_path = "ANN-code/Data/features_Ar_CF4_3_processed.csv"
model = LENRI_Ar_CF4_3().to(device)
binary = False

# Load test set
if binary:
    _, _, test_loader = get_dataloaders_cf4(features_path, batch_size=32)
else:
    _, _, test_loader = get_dataloaders_ar_cf4(features_path, batch_size=32)

# Load trained model
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

# Print hyperparameters
if 'hyperparameters' in checkpoint:
    hyperparameters = checkpoint['hyperparameters']
    print("Hyperparameters:", hyperparameters)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Evaluate on test set
test_loss = 0
correct_test = 0
total_test = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        correct_test += (predictions == labels).sum().item()
        total_test += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predictions.cpu().numpy())

# Compute test loss and accuracy
test_loss /= len(test_loader)
test_acc = correct_test / total_test

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Compute additional metrics
if binary:
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    # Print results
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
else:
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    print('[Carbon, Fluorine, Argon]')
    print(f"Precision per class: {precision_per_class}")
    print(f"Recall per class: {recall_per_class}")
    print(f"F1 Score per class: {f1_per_class}")




