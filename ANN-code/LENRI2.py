import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

def load_and_prepare_data(files, test_ratio=0.1, validation_ratio=0.15, train_ratio=0.75):
    """
    Load and prepare data for training and testing.
    Parameters:
        files (list of str): List of file paths to the CSV datasets.
        test_ratio (float): Proportion of data to use as the test set.
        validation_ratio (float): Proportion of data to use as the validation set.
        train_ratio (float): Proportion of data to use as the training set.
    Returns:
        tuple: Scaled training, validation, and test data and their respective labels.
    """
    # Read and concatenate CSV files
    data_frames = [pd.read_csv(file) for file in files]
    combined_df = pd.concat(data_frames, ignore_index=True)

    # shuffle dataframe
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Extract true labels from the `name` column
    def extract_species(name):
        return 0 if "C" in name else 1

    combined_df["species"] = combined_df["name"].apply(extract_species)

    # Split into carbon and fluorine data
    carbon_events = combined_df[combined_df["species"] == 0]
    fluorine_events = combined_df[combined_df["species"] == 1]

    # Balanced training set: 50% carbon, 50% fluorine
    fluorine_train = fluorine_events.sample(n=len(carbon_events), random_state=42)
    balanced_training_data = pd.concat([carbon_events, fluorine_train]).reset_index(drop=True)

    # Create unbalanced test set: 80% fluorine, 20% carbon
    fluorine_test = fluorine_events.sample(frac=0.8, random_state=42)
    carbon_test = carbon_events.sample(frac=0.2, random_state=42)
    unbalanced_test_data = pd.concat([fluorine_test, carbon_test]).reset_index(drop=True)

    # Extract features and labels for training and test sets
    def split_features_and_labels(data):
        X = data.drop(columns=["name", "species"]).values  # Exclude non-feature columns
        y = data["species"].values
        return X, y

    # Balanced training data
    X_train, y_train = split_features_and_labels(balanced_training_data)

    # Unbalanced test data
    X_test, y_test = split_features_and_labels(unbalanced_test_data)

    # Validation set from the training set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_ratio / (train_ratio + validation_ratio),
        random_state=42,
    )

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_lenri_model(input_shape):
    """
    Build the LENRI model architecture.
    Parameters:
        input_shape (tuple): Shape of the input features.
    Returns:
        keras.Sequential: Compiled LENRI model.
    """
    model = Sequential(
        [
            Dense(32, input_shape=input_shape, activation="leaky_relu"),
            Dropout(0.2),
            Dense(16, activation="leaky_relu"),
            Dropout(0.2),
            Dense(8, activation="leaky_relu"),
            Dense(2, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_lenri_model(model, X_train, y_train, X_val, y_val, save_path):
    """
    Train the LENRI model.
    Parameters:
        model (keras.Sequential): Compiled LENRI model.
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        save_path (str): Path to save the trained model.
    Returns:
        keras.callbacks.History: Training history.
    """
    # Define a callback for saving the best model
    checkpoint = ModelCheckpoint(
        save_path, monitor="val_loss", save_best_only=True, verbose=1
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint],
    )
    return history