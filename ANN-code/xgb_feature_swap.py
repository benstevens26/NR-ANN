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

# Define feature groups
feature_groups = {
    "simulation": False,  # drift_length
    "pixels_voxels": True,  # sum_intensity, max_intensity, track_area, track_volume
    "axis_needed": True,  # recoil_angle
    "energy_deposition": False,  # dE/dx features
}

# Define feature categories
features_dict = {
    "simulation": ["drift_length"],
    "pixels_voxels": [
        "sum_intensity_cam",  "sum_intensity_ito", "sum_intensity_R",
        "max_intensity_cam", "max_intensity_ito", "max_intensity_R",

    ],
    "axis_needed": ["recoil_angle_cam", "recoil_angle_ito", "recoil_angle_3d_2"],
    "energy_deposition": [
        "track_area_cam", "track_area_ito", "track_volume_3d",
        "recoil_length_cam", "recoil_length_ito", "recoil_length_3d",
        "mean_energy_deposition_cam", "std_energy_deposition_cam",
        "skew_energy_deposition_cam", "kurt_energy_deposition_cam",
        "max_energy_deposition_cam", "mean_energy_deposition_ito",
        "std_energy_deposition_ito", "skew_energy_deposition_ito",
        "kurt_energy_deposition_ito", "max_energy_deposition_ito",
        "mean_energy_deposition_3d", "std_energy_deposition_3d",
        "skew_energy_deposition_3d", "kurt_energy_deposition_3d",
        "max_energy_deposition_3d", "bragg_peak_location_3d"
    ],
}

# Select features based on the active groups
selected_features = [
    feature for group, active in feature_groups.items() if active
    for feature in features_dict[group]
]

# Ensure selected features exist in dataframe (in case some are missing)
selected_features = [feat for feat in selected_features if feat in df.columns]

# Split features and labels
X = df[selected_features]
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
output_file = "xgb_predictions_CF4_no_axis_no_dedx.csv"
predictions_df.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")
