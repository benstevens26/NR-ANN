import torch
import numpy as np
from sklearn.inspection import permutation_importance
from models import LENRI_CF4_3
from feature_preprocessing import get_dataloaders_cf4

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load trained model
model_path = "LENRI_CF4_3_opt.pth"
features_path = "ANN-code/Data/features_CF4_3_processed.csv"

model = LENRI_CF4_3().to(device)
# Load the model checkpoint correctly
checkpoint = torch.load(model_path, map_location=device)

# Extract only the model state_dict
model.load_state_dict(checkpoint["model_state_dict"])  # Only load model weights

# Put the model in evaluation mode
model.eval()

# Get test dataset
_, _, test_dataloader = get_dataloaders_cf4(features_path, batch_size=32)
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from copy import deepcopy

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from copy import deepcopy

def compute_permutation_importance(model, dataloader, criterion, device):
    """
    Computes permutation feature importance for a PyTorch model.

    Parameters:
    - model: Trained PyTorch model
    - dataloader: Dataloader for test/validation set
    - criterion: Loss function (e.g., torch.nn.CrossEntropyLoss())
    - device: "cpu" or "cuda"

    Returns:
    - feature_importances: Dictionary with feature importance scores
    """
    model.eval()
    original_preds, original_labels = [], []
    total_loss = 0

    # Get baseline performance
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            original_preds.extend(preds)
            original_labels.extend(labels.cpu().numpy())

    baseline_acc = accuracy_score(original_labels, original_preds)
    baseline_f1 = f1_score(original_labels, original_preds, average="weighted")
    baseline_loss = total_loss / len(dataloader)

    feature_importances = {}

    # Iterate over each feature index
    for feature_idx in range(inputs.shape[1]):
        permuted_preds, permuted_labels = [], []
        total_loss = 0

        # Create a copy of the dataset with shuffled feature values
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs_permuted = inputs.clone()
            inputs_permuted[:, feature_idx] = inputs_permuted[torch.randperm(inputs_permuted.size(0)), feature_idx]

            outputs = model(inputs_permuted)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            permuted_preds.extend(preds)
            permuted_labels.extend(labels.cpu().numpy())

        permuted_acc = accuracy_score(permuted_labels, permuted_preds)
        permuted_f1 = f1_score(permuted_labels, permuted_preds, average="weighted")
        permuted_loss = total_loss / len(dataloader)

        # Importance = Performance drop
        acc_drop = baseline_acc - permuted_acc
        f1_drop = baseline_f1 - permuted_f1
        loss_increase = permuted_loss - baseline_loss

        feature_importances[feature_idx] = {
            "accuracy_drop": acc_drop,
            "f1_drop": f1_drop,
            "loss_increase": loss_increase
        }

    return feature_importances





criterion = torch.nn.CrossEntropyLoss()

importance_scores = compute_permutation_importance(model, test_dataloader, criterion, device)


for feature, scores in importance_scores.items():
    print(f"Feature {feature}: {scores}")



# plot the feature importance scores
import matplotlib.pyplot as plt

# Sort features by importance
sorted_features = sorted(importance_scores.items(), key=lambda x: x[1]["accuracy_drop"], reverse=True)

# Extract feature names from pandas read in features path
import pandas as pd
features = pd.read_csv(features_path).columns[2:]

import matplotlib.pyplot as plt

# Create figure
plt.figure(figsize=(8, 6))

# Extract feature names and importance scores
feature_names = [features[feature[0]] for feature in sorted_features]
importance_scores = [feature[1]["accuracy_drop"] for feature in sorted_features]

# Plot horizontal bar chart
plt.barh(feature_names, importance_scores, color='#000080')

# Formatting
plt.xlabel("Accuracy Drop", fontsize=18)
plt.ylabel("Feature", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle="--", linewidth=0.5, axis='x')
plt.gca().invert_yaxis()  # Highest importance at the top

# Save the figure
plt.savefig("ANN-code/Data/LENRI_CF4_3/feature_importance.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()



import matplotlib.pyplot as plt

# Select top 10 features
top_features = sorted_features[:10]
feature_names = [features[feature[0]] for feature in top_features]
importance_scores = [feature[1]["accuracy_drop"] for feature in top_features]

# Create figure with larger size
plt.figure(figsize=(10, 8))

# Plot horizontal bar chart
plt.barh(feature_names, importance_scores, color='#000080')

# Formatting
plt.xlabel("Accuracy Drop", fontsize=18)
plt.ylabel("Feature", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.grid(True, linestyle="--", linewidth=0.5, axis='x')
plt.gca().invert_yaxis()  # Highest importance at the top

# Adjust layout to fit long feature names
plt.tight_layout()

# Save the figure
plt.savefig("ANN-code/Data/LENRI_CF4_3/feature_importance_top10.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()

