import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cnn_processing import load_data_yield
import performance as pf
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score





model = tf.keras.models.load_model("/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/test/test_CoNNCR-R.keras", custom_objects={"softmax_v2": tf.keras.activations.softmax})

if True: # load model and dataset
    base_dirs = [
        "/vols/lz/tmarley/GEM_ITO/run/im0/C",
        "/vols/lz/tmarley/GEM_ITO/run/im0/F",
        "/vols/lz/tmarley/GEM_ITO/run/im1/C",
        "/vols/lz/tmarley/GEM_ITO/run/im1/F",
        "/vols/lz/tmarley/GEM_ITO/run/im2/C",
        "/vols/lz/tmarley/GEM_ITO/run/im2/F",
        "/vols/lz/tmarley/GEM_ITO/run/im3/C",
        "/vols/lz/tmarley/GEM_ITO/run/im3/F",
        "/vols/lz/tmarley/GEM_ITO/run/im4/C",
        "/vols/lz/tmarley/GEM_ITO/run/im4/F",
    ]
    batch_size = 16
    dark_list_number = 0
    binning = 1
    dark_dir = "/vols/lz/MIGDAL/sim_ims/darks"
    # dark_dir="Data/darks"
    m_dark = np.load(f"{dark_dir}/master_dark_{str(binning)}x{str(binning)}.npy")
    example_dark_list_unbinned = np.load(
        f"{dark_dir}/quest_std_dark_{dark_list_number}.npy"
    )
    m_dark_tensor = tf.convert_to_tensor(m_dark, dtype=tf.float32)
    example_dark_tensor = tf.convert_to_tensor(example_dark_list_unbinned[:1], dtype=tf.float32)
    full_dataset = tf.data.Dataset.from_generator(
        lambda: load_data_yield(base_dirs, example_dark_tensor, m_dark_tensor, 3),
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),  # MAY NEED TO CHANGE
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    dataset_size = 99366 # 99989 without the  # CHANGE DEPENDING ON DATA USED
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size  # Ensure all data is used

    # train_dataset = full_dataset.take(train_size).cache("/vols/lz/twatson/CNN_cache").repeat().batch(batch_size, drop_remainder=True) # First 70%
    # remaining = full_dataset.skip(train_size)  # Remaining 30%
    # val_dataset = full_dataset.skip(train_size).take(val_size).cache("/vols/lz/twatson/CNN_cache").batch(batch_size, drop_remainder=False) # Next 15%
    test_dataset = full_dataset.skip(train_size + val_size + test_size - 1).batch(batch_size, drop_remainder=False) # Final 15%
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

##############################################################
# need to get a filename list and shuffle it in the same way #
##############################################################



# test_loss, test_accuracy = model.evaluate(test_dataset)
# print(f"Test Loss: {test_loss:.4f}")
# print(f"Test Accuracy: {test_accuracy:.4f}")
# y_pred = np.argmax(model.predict(X_test), axis=1)  # For multi-class classification
# y_pred_prob = model.predict(X_test)[:, 1]  # Probability for class 1
# y_true = np.argmax(y_test, axis=1)  # Assuming y_test is one-hot encoded
# cm = confusion_matrix(y_true, y_pred)
# precision = precision_score(
#     y_true, y_pred, average="weighted"
# )  # Use 'macro', 'micro', or 'weighted' as needed
# recall = recall_score(y_true, y_pred, average="weighted")
# f1 = f1_score(y_true, y_pred, average="weighted")



# pf.roc_plotter(y_true, y_pred_prob)

print("starting evaluation")
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Initialize lists to store predictions and true labels
y_true_list = []
y_pred_list = []
y_pred_prob_list = []

# Process test dataset batch by batch
for batch_images, batch_labels in test_dataset:
    # Get model predictions
    batch_pred_prob = model.predict(batch_images)  # Probability outputs
    batch_pred = np.argmax(batch_pred_prob, axis=1)  # Class predictions

    # Store batch results
    y_true_list.extend(batch_labels.numpy())  # Convert labels to numpy and store
    y_pred_list.extend(batch_pred)
    y_pred_prob_list.extend(batch_pred_prob[:, 1] if batch_pred_prob.shape[1] > 1 else batch_pred_prob.flatten())

# Convert lists to NumPy arrays
y_true = np.array(y_true_list)
y_pred = np.array(y_pred_list)
y_pred_prob = np.array(y_pred_prob_list)

# Compute confusion matrix and metrics
cm = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot ROC curve
pf.roc_plotter(y_true, y_pred_prob)