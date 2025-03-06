import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cnn_processing import load_data_yield
import performance as pf
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from cnn_processing import noise_adder, smooth_operator, bin_image
import os
import csv
import pandas as pd


model = tf.keras.models.load_model(
    "/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/test/test_CoNNCR-R.keras",
    custom_objects={"softmax_v2": tf.keras.activations.softmax},
)
small = False
if True:  # load model and dataset
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
    example_dark_tensor = tf.convert_to_tensor(
        example_dark_list_unbinned[:1], dtype=tf.float32
    )
    full_dataset = tf.data.Dataset.from_generator(
        lambda: load_data_yield(base_dirs, example_dark_tensor, m_dark_tensor, 3),
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),  # MAY NEED TO CHANGE
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    dataset_size = 99366  # 99989 without the  # CHANGE DEPENDING ON DATA USED
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size  # Ensure all data is used

    # train_dataset = full_dataset.take(train_size).cache("/vols/lz/twatson/CNN_cache").repeat().batch(batch_size, drop_remainder=True) # First 70%
    # remaining = full_dataset.skip(train_size)  # Remaining 30%
    # val_dataset = full_dataset.skip(train_size).take(val_size).cache("/vols/lz/twatson/CNN_cache").batch(batch_size, drop_remainder=False) # Next 15%
    test_dataset = full_dataset.skip(train_size + val_size).batch(
        batch_size, drop_remainder=True
    )  # Final 15%
    if small:
        test_dataset = test_dataset.take(3)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

##############################################################
# need to get a filename list and shuffle it in the same way #
##############################################################


def preprocess_file_path(
    image, m_dark=m_dark_tensor, example_dark_list=example_dark_tensor
):
    image1 = noise_adder(image, m_dark=m_dark, example_dark_list=example_dark_list)
    image2 = smooth_operator(image1)
    image3 = image2.astype(np.float32)
    max_val = np.max(image3)
    if max_val > 0:
        image3 = 255 * image3 / max_val
    image3 = np.repeat(image3[:, :, np.newaxis], 3, axis=-1)
    image4 = tf.image.resize_with_pad(image3, 224, 224)
    image5 = tf.keras.applications.vgg16.preprocess_input(image4)
    image5 /= np.max(abs(image5))
    image5 = tf.expand_dims(image5, axis=0)
    return image5


def get_file_list(seed=77):
    uncropped_error = np.loadtxt(
        "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/uncropped_error.csv",
        delimiter=",",
        dtype=str,
    )
    min_dim_error = np.loadtxt(
        "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/min_dim_error.csv",
        delimiter=",",
        dtype=str,
    )
    # Get all the .npy files from base_dirs
    errors = np.concatenate((uncropped_error, min_dim_error))

    # Get all the .npy files from base_dirs
    file_list = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            files = [
                f
                for f in files
                if (f.endswith(".npy") and os.path.join(root, f) not in errors)
            ]
            file_list.extend([os.path.join(root, file) for file in files])

    file_list.sort()
    np.random.seed(seed)
    np.random.shuffle(file_list)
    return file_list


file_list = get_file_list()
test_file_list = file_list[-test_size:]
other_file_list = file_list[:-test_size]
# for file_path in test_file_list:
#     image = np.load(file_path)
#     image = preprocess_file_path(image)
#     prediction = model.predict(image)
#     print(f"file path: {file_path}")
#     print(f"prediction: {prediction[0]}")


with open("CoNNCR-R_predictions_1.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write header row
    writer.writerow(["file_path", "prediction"])

    for file_path in test_file_list:
        image = np.load(file_path)
        image = preprocess_file_path(image)
        prediction = model.predict(image)

        # Write data to CSV file
        writer.writerow([file_path, prediction[0]])


df = pd.read_csv("/vols/lz/twatson/ANN/NR-ANN/ANN-code/CoNNCR-R_predictions_1.csv")

true_labels = []
predicted_probs = []
for index, row in df.iterrows():
    file_path = row["file_path"]
    prediction = np.array(
        row["prediction"].strip("[]").split()
    )  # Convert the string to an array
    prediction = prediction.astype(float)

    true_label = 0 if "C" in os.path.basename(file_path) else 1

    prob_class_1 = prediction[1]  # Probabilities for class 1 (b)

    true_labels.append(true_label)
    predicted_probs.append(prob_class_1)

true_labels = np.array(true_labels)
predicted_probs = np.array(predicted_probs)

fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig("ROC", dpi=300)
plt.show()

roc_data = pd.DataFrame({"fpr": fpr, "tpr": tpr})

# Save the DataFrame to a CSV file
roc_data.to_csv("roc_curve_data.csv", index=False)


with open("CoNNCR-R_train_val_predictions_1.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write header row
    writer.writerow(["file_path", "prediction"])

    for file_path in other_file_list:
        image = np.load(file_path)
        image = preprocess_file_path(image)
        prediction = model.predict(image)

        # Write data to CSV file
        writer.writerow([file_path, prediction[0]])


# print("starting evaluation")
# # test_loss, test_accuracy = model.evaluate(test_dataset)
# print(f"Test Loss: {test_loss:.4f}")
# print(f"Test Accuracy: {test_accuracy:.4f}")

# # Initialize lists to store predictions and true labels
# y_true_list = []
# y_pred_list = []
# y_pred_prob_list = []

# # Process test dataset batch by batch
# for batch_images, batch_labels in test_dataset:
#     # Get model predictions
#     batch_pred_prob = model.predict(batch_images)  # Probability outputs
#     batch_pred = np.argmax(batch_pred_prob, axis=1)  # Class predictions

#     # Store batch results
#     y_true_list.extend(batch_labels.numpy())  # Convert labels to numpy and store
#     y_pred_list.extend(batch_pred)
#     y_pred_prob_list.extend(batch_pred_prob[:, 1] if batch_pred_prob.shape[1] > 1 else batch_pred_prob.flatten())

# # Convert lists to NumPy arrays
# y_true = np.array(y_true_list)
# y_pred = np.array(y_pred_list)
# y_pred_prob = np.array(y_pred_prob_list)

# # Compute confusion matrix and metrics
# cm = confusion_matrix(y_true, y_pred)
# precision = precision_score(y_true, y_pred, average="weighted")
# recall = recall_score(y_true, y_pred, average="weighted")
# f1 = f1_score(y_true, y_pred, average="weighted")

# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")

# # Plot ROC curve
# pf.roc_plotter(y_true, y_pred_prob)
