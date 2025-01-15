import tensorflow as tf
# import numpy as np
# import os
# import random
# import scipy.ndimage as nd
# from convert_sim_ims import convert_im, get_dark_sample
# import pickle
# from cnn_processing import (
#     bin_image,
#     smooth_operator,
#     noise_adder,
#     pad_image,
#     parse_function,
#     load_data,
# )
# import json
# from tensorflow.keras.callbacks import EarlyStopping

# Check if a GPU is available and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid taking all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("No GPU found. Running on CPU.")

exit()
# Define base directories and batch size

base_dirs = [
    "/vols/lz/tmarley/GEM_ITO/run/im0",
    "/vols/lz/tmarley/GEM_ITO/run/im1/C",
    "/vols/lz/tmarley/GEM_ITO/run/im1/F",
    "/vols/lz/tmarley/GEM_ITO/run/im2",
    "/vols/lz/tmarley/GEM_ITO/run/im3",
    "/vols/lz/tmarley/GEM_ITO/run/im4",
]  # List your data directories here
# base_dirs = ['Data/im2']  # List your data directories here
batch_size = 32
dark_list_number = 0
binning = 1
# dark_dir="/vols/lz/MIGDAL/sim_ims/darks"
dark_dir = "Data/darks"
m_dark = np.load(f"{dark_dir}/master_dark_{str(binning)}x{str(binning)}.npy")
example_dark_list_unbinned = np.load(
    f"{dark_dir}/quest_std_dark_{dark_list_number}.npy"
)

# Load the dataset
full_dataset = load_data(base_dirs, batch_size, example_dark_list_unbinned, m_dark)

dataset_size = len(full_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size  # Ensure all data is used

train_dataset = full_dataset.take(train_size)  # First 70%
remaining = full_dataset.skip(train_size)  # Remaining 30%
val_dataset = remaining.take(val_size)  # Next 15%
test_dataset = remaining.skip(val_size)  # Final 15%

CoNNCR = tf.keras.Sequential(
    [
        # Input Layer
        tf.keras.layers.Input(shape=(572, 768, 1)),
        # Convolutional Block 1
        tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),  # Downsampling by 2x
        # Convolutional Block 2
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),  # Further downsampling
        # Convolutional Block 3
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Convolutional Block 4
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Global Average Pooling for Feature Aggregation
        tf.keras.layers.GlobalAveragePooling2D(),  # Reduces spatial dimensions to one value per feature map
        # Fully Connected Layer
        tf.keras.layers.Dense(32, activation="relu"),  # Lightweight dense layer
        tf.keras.layers.Dropout(0.5),  # Regularization to prevent overfitting
        # Output Layer
        tf.keras.layers.Dense(2, activation="softmax"),  # Binary classification
    ]
)

# Compile the model
CoNNCR.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print(CoNNCR.summary())

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor="val_loss",  # Metric to monitor (e.g., 'val_loss', 'val_accuracy')
    patience=5,  # Number of epochs with no improvement before stopping
    restore_best_weights=True,  # Restore the best weights at the end of training
)

# Train the model with EarlyStopping
history = CoNNCR.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,  # Set a high maximum epoch count
    callbacks=[early_stopping],  # Add EarlyStopping callback here
)

# Save the trained model
CoNNCR.save("CoNNCR.keras")

# Save model training history

history_dict = CoNNCR.history.history
with open("CoNNCR_history.json", "w") as file:
    json.dump(history_dict, file)

exit()
# %%

# # For loading the files:
# model_save_path = "saved_models/LENRI_model.keras"
# history_save_path = "saved_models/LENRI_history.pkl"

# # Load the saved model
# LENRI_loaded = load_model(model_save_path)

# # Load the training history
# with open(history_save_path, "rb") as file:
#     loaded_history = pickle.load(file)

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
