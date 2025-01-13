#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.math import confusion_matrix
import performance as pf
from sklearn.metrics import roc_curve, auc
import pickle
# import keras-tuner

# Data Preparation

# Load CSV data
data = pd.read_csv("Data/more_features_noisy.csv")  # Change to file path

# #Trying to match carbon and fluorine data amounts
# carbon_events = data[data["name"].str.contains("C")]
# fluorine_events = data[data["name"].str.contains("F")].sample(n=len(carbon_events),random_state=42)
# balanced_data = pd.concat([carbon_events,fluorine_events]).reset_index(drop=True)
# data = balanced_data.copy()


# Extract features and labels
def extract_species(name):
    """Extract species label from event name."""
    return 0 if "C" in name else 1


data["species"] = data["name"].apply(extract_species)
X = data.iloc[
    :, 2:10  # CHANGE WHEN MORE FEATURES ADDED
].values  # Select columns with feature data (assuming columns 1-4 are features)
y = data["species"].values

# Split into training and testing sets
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 - train_ratio, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test,
    y_test,
    test_size=test_ratio / (test_ratio + validation_ratio),
    random_state=42,
)


# Scale features for better convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


# Convert labels to categorical (one-hot encoding for binary classification)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)


# %%

# Define the LENRI model architecture
LENRI = Sequential(
    [
        Dense(
            32, input_shape=(8,), activation="leaky_relu"
        ),  # Input layer with 4 features. CHANGE WHEN MORE FEATURES ADDED
        Dropout(0.2),  # Dropout for regularisation
        Dense(16, activation="leaky_relu"),  # Hidden layer
        Dropout(0.2),  # Dropout for regularisation
        Dense(8, activation="leaky_relu"),  # Another hidden layer
        Dense(2, activation="softmax"),  # Output layer for binary classification
    ]
)

# # updated hyperparams: (worse)
# LENRI = Sequential(
#     [
#         Dense(
#             64, input_shape=(8,), activation="leaky_relu"
#         ),  # Input layer with 4 features. CHANGE WHEN MORE FEATURES ADDED
#         Dropout(0.3),  # Dropout for regularisation
#         Dense(48, activation="leaky_relu"),  # Hidden layer
#         Dropout(0.4),  # Dropout for regularisation
#         Dense(8, activation="leaky_relu"),  # Another hidden layer
#         Dropout(0.4), # Another dropout
#         Dense(2, activation="softmax"),  # Output layer for binary classification
#     ]
# )
# Compile LENRI
LENRI.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# K.set_value(LENRI.optimizer.learning_rate, 0.03) # grid searched


# LENRI
LENRI.summary()

# %%

# Train LENRI
history = LENRI.fit(
    X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val)
)

# %%
# Saving LENRI
model_save_path = "old_models/LENRIv1.keras"
LENRI.save(model_save_path)

# Saving LENRI's training history
history_save_path = "old_models/LENRIv1_history.pkl"
with open(history_save_path, "wb") as file:
    pickle.dump(history.history, file)


# # For loading the files:
# model_save_path = "saved_models/LENRI_model.keras"
# history_save_path = "saved_models/LENRI_history.pkl"

# # Load the saved model
# LENRI_loaded = load_model(model_save_path)

# # Load the training history
# with open(history_save_path, "rb") as file:
#     loaded_history = pickle.load(file)
# %%
# LENRI Evaluation
test_loss, test_accuracy = LENRI.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
y_pred = np.argmax(LENRI.predict(X_test), axis=1)  # For multi-class classification
y_pred_prob = LENRI.predict(X_test)[:, 1]  # Probability for class 1
y_true = np.argmax(y_test, axis=1)  # Assuming y_test is one-hot encoded
cm = confusion_matrix(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, average="weighted"
)  # Use 'macro', 'micro', or 'weighted' as needed
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")


pf.plot_model_performance(
    "LENRI",
    history.history["accuracy"],
    history.history["loss"],
    history.history["val_accuracy"],
    history.history["val_loss"],
    cm,
    precision,
    recall,
    f1,
)

first_layer_weights = LENRI.layers[0].get_weights()[0]
names = [i for i in data.columns[2:10]]

pf.weights_plotter(first_layer_weights, names)
pf.roc_plotter(y_true, y_pred_prob)
