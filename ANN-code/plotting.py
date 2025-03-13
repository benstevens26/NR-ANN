import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import tensorflow as tf
import os
import csv
from tqdm import tqdm
import re


# -=+ model +=-
# which = "L"
which = "C"
# which = "both"


# -=+ plotting +=-
save = False

ROC_curve = True
confusion_matrix = True
prediction_with_energy = True

# -=+ dataset +=-
biased = False
exclude_low_energies = False

# -=+ details +=-
make_predictions = False
CoNNCR_version = 4
predictions_file_path = f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/CoNNCR-Rv{CoNNCR_version}_predictions.csv"
if CoNNCR_version <= 3:
    use_unscaled = False
else:
    use_unscaled = True



# Load model(s) of choice
if which == "L":
    models = tf.keras.models.load_model(
            "/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/LENRIv1.keras",
        )
      # load LENRI

    base_dir_list = [
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
elif which == "C":
    model = tf.keras.models.load_model(
            f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v{CoNNCR_version}/CoNNCR-R.keras",
            custom_objects={"softmax_v2": tf.keras.activations.softmax},
        )
      # load CoNNCR

    base_dir_list = [
        (
            "/vols/lz/twatson/ANN/preprocessed_images_unscaled"
            if use_unscaled
            else "/vols/lz/twatson/ANN/preprocessed_images"
        )
    ]
elif which == "both":
    models = [
        tf.keras.models.load_model(
            "/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/LENRIv1.keras"
        ),
        tf.keras.models.load_model(
            f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v{CoNNCR_version}/CoNNCR-R.keras",
            custom_objects={"softmax_v2": tf.keras.activations.softmax},
        ),
    ]
    base_dir_list = [
        [
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
        ],
        [
            (
                "/vols/lz/twatson/ANN/preprocessed_images_unscaled"
                if use_unscaled
                else "/vols/lz/twatson/ANN/preprocessed_images"
            )
        ],
    ]


# get test dataset
def get_file_list(seed=77, test_only=False, which=which, use_unscaled=use_unscaled):
    # use CoNNCR dataset and match LENRI filepaths as required
    base_dirs = [
        (
            "/vols/lz/twatson/ANN/preprocessed_images_unscaled"
            if use_unscaled
            else "/vols/lz/twatson/ANN/preprocessed_images"
        )
    ]

    # Get all the .npy files from base_dirs
    file_list = []
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            files = [f for f in files if (f.endswith(".npy"))]
            file_list.extend([os.path.join(root, file) for file in files])

    file_list.sort()
    np.random.seed(seed)
    np.random.shuffle(file_list)
    if which == "C":
        return file_list[-14912:] if test_only else file_list
    elif which == "L":
        raise NotImplementedError
    elif which == "both":
        raise NotImplementedError


file_list = get_file_list(test_only=True)


save_test_paths = False
if save_test_paths:
    with open("test_names.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        for file_path in file_list:
            # Get batch file paths
            file_name = os.path.basename(file_path)
            # Write each file's path with its respective prediction
            writer.writerow([file_name]) 



if make_predictions:
    batch_size = 32  # Adjust as needed based on available memory
    batched_paths = []
    with open(predictions_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["file_path", "prediction"])  # Write header

        for file_index in range(0, len(file_list), batch_size):
            # Get batch file paths
            batched_paths = file_list[file_index:file_index + batch_size]

            # Load images for the current batch
            batched_images = [np.load(path) for path in batched_paths]  # Shape: (batch_size, 224, 224, 3)
            batched_images = np.stack(batched_images, axis=0)  # Convert to numpy array

            # Run batch prediction
            predictions = model.predict(batched_images, verbose=0)  # Shape: (batch_size, 2)

            # Write each file's path with its respective prediction
            for path, pred in zip(batched_paths, predictions):
                writer.writerow([path, list(pred)]) 

data = []  # List to store extracted data
with open(predictions_file_path, mode="r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header

    for row in reader:
        file_path, prediction_str = row  # Extract columns
        
        # Convert prediction string "[0.3, 0.7]" into a list [0.3, 0.7]
        prediction = eval(prediction_str)  # Safely convert string to list

        # Determine ground truth class from filename
        true_class = 1 if "F" in os.path.basename(file_path) else 0  # Assign class based on "F" or "C"
        energy = float(re.search(r'unscaled/([\d.]+)keV', file_path).group(1))
        # Store information in a structured format
        data.append([file_path, true_class, energy, prediction[1]])

# Exclude data as desired
if exclude_low_energies:
    data = [event for event in data if not ((event[1] == 1 and event[2] < 170) or (event[1] == 0 and event[2] < 130))]

if biased:
    NotImplementedError





if prediction_with_energy:
    labels = np.array([i[1] for i in data])
    energies = np.array([i[2] for i in data])
    predictions = np.array([i[3] for i in data])
    
    colours = np.array(["green" if labels[i] == round(predictions[i]) else "brown" for i in range(len(labels))])

    mask_o = labels == 0  # Array of True/False values
    mask_x = labels == 1 

    # Plot all "o" markers in one go
    plt.scatter(energies[mask_o], predictions[mask_o], c=colours[mask_o], marker=".",label="C")

    # Plot all "x" markers in one go
    plt.scatter(energies[mask_x], predictions[mask_x], c=colours[mask_x], marker="x",label="F")

    plt.grid()
    plt.legend()
    
    
    plt.show()


    # plt.scatter(energies,predictions,marker=markers,c=colours)
    # plt.show()



accuracy = sum(row[1] == round(row[3]) for row in data) / len(data)
print(f"Accuracy: {accuracy:.2%}")