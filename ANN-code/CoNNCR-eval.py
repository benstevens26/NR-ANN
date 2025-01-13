import tensorflow as tf
import numpy as np
from cnn_processing import load_data
import json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Load the pre-trained model
model_path = "Data/CoNNCR.keras"
CoNNCR = tf.keras.models.load_model(model_path)


# Predictions and metrics
y_true = []  # True labels
y_pred = []  # Predicted labels
y_pred_prob = []  # Predicted probabilities

# Calculate performance metrics
cm = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print("Confusion Matrix:\n", cm)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# (Optional) Save evaluation results
evaluation_results = {
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "confusion_matrix": cm.tolist(),
}
with open("CoNNCR_evaluation.json", "w") as file:
    json.dump(evaluation_results, file)
