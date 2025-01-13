from cnn_processing import load_events_bb as load_events
import tensorflow as tf
from cnn_processing import bin_image, smooth_operator, noise_adder, pad_image, load_data
import pickle
import matplotlib.pyplot as plt
import numpy as np


# Define base directories and batch size
# base_dirs = ['/vols/lz/MIGDAL/sim_ims/C', '/vols/lz/MIGDAL/sim_ims/F']  # List your data directories here
base_dirs = ["Data/C", "Data/F"]
batch_size = 32
dark_list_number = 0
binning = 1
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

# Define the model
miniCoNNCR = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=((572, 768), 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),  # Reduce filters
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),  # Reduce filters
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.GlobalAveragePooling2D(),  # Replaces Flatten
        tf.keras.layers.Dense(32, activation="relu"),  # Reduce neurons
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)

# Compile the model
miniCoNNCR.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

print(miniCoNNCR.summary())
exit()
# # Train the model
# miniCoNNCR.fit(train_dataset, validation_data=val_dataset, epochs=20)
#
# # Save the trained model
# miniCoNNCR.save('miniCoNNCR.keras')
#
# # Save model training history
# history_save_path = "miniCoNNCR_history.pkl"
# with open(history_save_path, "wb") as file:
#     pickle.dump(miniCoNNCR.history, file)


from tensorflow.keras.models import load_model
import pickle
from tensorflow.math import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
import performance as pf

model_save_path = "miniCoNNCR.keras"
history_save_path = "miniCoNNCR_history.pkl"

LENRI = load_model(model_save_path)

with open(history_save_path, "rb") as file:
    loaded_history = pickle.load(file)

X_test, y_test = train_dataset


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
    LENRI.history["accuracy"],
    LENRI.history["loss"],
    LENRI.history["val_accuracy"],
    LENRI.history["val_loss"],
    cm,
    precision,
    recall,
    f1,
)

first_layer_weights = LENRI.layers[0].get_weights()[0]
# names = [i for i in data.columns[2:10]]

# pf.weights_plotter(first_layer_weights, names)
pf.roc_plotter(y_true, y_pred_prob)
