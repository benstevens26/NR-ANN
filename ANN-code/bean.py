"""
Module contains BEAN (Boosted Ensemble for Analysing Nuclear recoils)
"""

import pandas as pd
import re
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# load data
file_path = "ANN-code/Data/features_CF4_3_processed.csv"
df = pd.read_csv(file_path)
binary = True

# label extraction function
def extract_label_cf4(filename):
    if re.search(r"00_C_", filename):
        return 0  # Carbon
    elif re.search(r"00_F_", filename):
        return 1  # Fluorine
    elif re.search(r"_C_", filename):
        return 0
    elif re.search(r"_F_", filename):
        return 1
    return None

# Extract labels
df["label"] = df["cam_path"].apply(extract_label_cf4)

# Drop rows where label could not be determined
df = df.dropna(subset=["label"])

# Drop non-feature columns
df = df.drop(columns=["cam_path", "ito_path"])

# Split features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost classifier
xgb_clf = XGBClassifier(eval_metric="logloss")
xgb_clf.fit(X_train, y_train)

# Predictions
y_pred = xgb_clf.predict(X_test)

# Evaluate performance
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
}

# Print results
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")


