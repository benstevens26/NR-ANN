""" BEAN
Boosted
Ensemble
Algorithm
(for) Nuclear (recoil identification)
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# Load the data
data = pd.read_csv("../more_features_noisy.csv")


# Extract species label from 'name' column: 0 for carbon, 1 for fluorine
data["species"] = data["name"].apply(lambda x: 0 if "C" in x else 1)

# Define features and target
features = [
    "length",
    "total_intensity",
    "max_den",
    "recoil_angle",
    "int_mean",
    "int_skew",
    "int_kurt",
    "int_std",
]
X = data[features]
y = data["species"]

# Handle class imbalance by resampling the minority class (carbon)
carbon = data[data["species"] == 0]
fluorine = data[data["species"] == 1]

# Upsample the minority class to match the majority class
carbon_upsampled = resample(
    carbon, replace=True, n_samples=len(fluorine), random_state=42
)
data_balanced = pd.concat([fluorine, carbon_upsampled])

# Redefine X and y from the balanced dataset
X_balanced = data_balanced[features]
y_balanced = data_balanced["species"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)

# Initialize the Gradient Boosting Classifier
bdt = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42
)

# Train the classifier
bdt.fit(X_train, y_train)

# Make predictions
y_pred = bdt.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Prepare the features and labels
def extract_species(name):

    return 0 if "C" in name else 1


data["species"] = data["name"].apply(extract_species)

feature_columns = [
    "length",
    "total_intensity",
    "max_den",
    "recoil_angle",
    "int_mean",
    "int_skew",
    "int_kurt",
    "int_std",
]
X = data[feature_columns].values
y = data["species"].values

# Define the parameter grid
param_grid = {
    "n_estimators": [10, 20, 50],
    "learning_rate": [0.005, 0.01, 0.02, 0.05],
    "max_depth": [4],
}

# Initialize the GradientBoostingClassifier
bdt = GradientBoostingClassifier(random_state=42)

# Set up the grid search with cross-validation
grid_search = GridSearchCV(
    estimator=bdt, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2
)

# Run the grid search
grid_search.fit(X, y)

# Display the best parameters and the best score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)
