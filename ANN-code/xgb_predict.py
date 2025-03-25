import pandas as pd
import re
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "ANN-code/Data/features_CF4_3_processed.csv"
df = pd.read_csv(file_path)

# Define label extraction function
def extract_label_cf4(filename):
    if re.search(r"00_C_", filename):
        return 0  # Carbon
    elif re.search(r"00_F_", filename):
        return 1  # Fluorine
    elif re.search(r"_C_", filename):
        return 0
    elif re.search(r"_F_", filename):
        return 1
    return None  # Handle unexpected cases

# Extract labels
df["label"] = df["cam_path"].apply(extract_label_cf4)

# Drop rows where label could not be determined
df = df.dropna(subset=["label"])

# Preserve filenames before dropping non-feature columns
df["Filename"] = df["cam_path"]

# Drop non-feature columns
df = df.drop(columns=["cam_path", "ito_path"])

# Split features and labels
X = df.drop(columns=["label", "Filename"])





y = df["label"]
filenames = df["Filename"]  # Save filenames for reference

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
    X_scaled, y, filenames, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost classifier
xgb_clf = XGBClassifier(eval_metric="logloss")
xgb_clf.fit(X_train, y_train)

# Get probability predictions
y_probs = xgb_clf.predict_proba(X_test)  # Probabilities for each class

# Create DataFrame with Filename, Predict_C, Predict_F
predictions_df = pd.DataFrame({
    "Filename": filenames_test.values,
    "Predict_C": y_probs[:, 0],  # Probability of being Carbon
    "Predict_F": y_probs[:, 1],  # Probability of being Fluorine
})

# Save to CSV
output_file = "xgb_predictions_CF4.csv"
predictions_df.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")
