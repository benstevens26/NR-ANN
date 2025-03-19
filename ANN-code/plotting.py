import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import tensorflow as tf
import os
import csv
from tqdm import tqdm
import re
from preprocess import preprocess_file_path_unscaled



# -=+ model +=-
# which = "L"
which = "C"
# which = "both"


# -=+ plotting +=-
save = False

ROC_curve = False
confusion_matrix = False
prediction_with_energy = False
gradcam = False
blank_analyis = False
noise_analysis = False
example_recoils = True

# -=+ dataset +=-
biased = False
exclude_low_energies = False
save_sets = [False, False, False] # train, val, test


# -=+ details +=-
make_predictions = False
CoNNCR_version = 5
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
    model = tf.keras.models.load_model(
            f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v5/initial_training/CoNNCR-R_partially_tuned.keras",
            custom_objects={"softmax_v2": tf.keras.activations.softmax},
        )

    base_dir_list = [
        (
            "/vols/lz/twatson/ANN/final_ims"
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
                "/vols/lz/twatson/ANN/final_ims"
                if use_unscaled
                else "/vols/lz/twatson/ANN/preprocessed_images"
            )
        ],
    ]


# get test dataset
def get_file_list(seed=77, test_only=False, which=which, use_unscaled=use_unscaled,base_dirs = base_dir_list):
    # use CoNNCR dataset and match LENRI filepaths as required
    # base_dirs = [
    #     (
    #         "/vols/lz/twatson/ANN/final_ims"
    #         if use_unscaled
    #         else "/vols/lz/twatson/ANN/preprocessed_images"
    #     )
    # ]

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


file_list = get_file_list(test_only=False)

batch_size = 16
dataset_size = 99366 # 99989 without the  # CHANGE DEPENDING ON DATA USED
train_size = (int(0.7 * dataset_size)//batch_size)*batch_size
val_size = (int(0.15 * dataset_size)//batch_size)*batch_size
test_size = ((dataset_size - train_size - val_size)//batch_size)*batch_size  # Ensure all data is used

train_list = file_list[:train_size]
val_list = file_list[train_size:train_size + val_size]
test_list = file_list[-test_size:]

if any(save_sets):
    if save_sets[0]: # train
        with open("/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/train_names.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            for file_path in train_list:
                # Get batch file paths
                file_name = os.path.basename(file_path)
                # Write each file's path with its respective prediction
                writer.writerow([file_name]) 
    if save_sets[1]: # val
        with open("/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/val_names.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            for file_path in val_list:
                # Get batch file paths
                file_name = os.path.basename(file_path)
                # Write each file's path with its respective prediction
                writer.writerow([file_name]) 
    if save_sets[2]: # test
        with open("/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/test_names.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            for file_path in test_list:
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
        energy = float(re.search(r'_ims/([\d.]+)keV', file_path).group(1))
        # Store information in a structured format
        data.append([file_path, true_class, energy, prediction[1]])




# Exclude data as desired
if exclude_low_energies:
    data = [event for event in data if not ((event[1] == 1 and event[2] < 170) or (event[1] == 0 and event[2] < 130))]

if biased:
    CF_ratio = 7.33
    num_F = int(sum(1 for row in data if row[1] == 1))
    num_C = int(num_F//CF_ratio)
    
    C_data = [row for row in data if row[1] == 0]
    F_data = [row for row in data if row[1] == 1]
    
    if len(C_data) > num_C:
        # Randomly shuffle and select only num_C elements
        np.random.seed(77)
        np.random.shuffle(C_data)
        C_data = C_data[:num_C]
    
    data = C_data + F_data
    np.random.shuffle(data)

def undo_preprocess(img):
    # Add back the mean pixel values
    img[:, :, 0] += 103.939  # Blue
    img[:, :, 1] += 116.779  # Green
    img[:, :, 2] += 123.68   # Red

    # Convert from BGR back to RGB
    img = img[:, :, ::-1]

    # Rescale to 0-255
    min_val = np.min(img)
    max_val = np.max(img)
    img = (img - min_val) / (max_val - min_val) * 255

    return img.astype(np.uint8)


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
print(f"Biased? {biased}")
print(f"Exclude low energy? {exclude_low_energies}")
print(f"Accuracy: {accuracy:.2%}")



    

if blank_analyis:
    
    blanks = [np.zeros((np.random.randint(40,60), np.random.randint(40,60))) for i in range(10)]
    preprocessed = [preprocess_file_path_unscaled(blank) for blank in blanks]
    predictions = []
    for file in preprocessed:
        plt.imshow(file.numpy().astype(np.uint8))
        plt.show()
        preds = model.predict(np.expand_dims(file,axis=0))
        print("Predicted:", preds[0])
        predictions.append(preds[0])


if gradcam:
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import keras
    from gradcam import get_img_array, make_gradcam_heatmap, save_and_display_gradcam
    
    img_size=(224, 224)
    last_conv_layer_name = "block5_conv3"
    
    img_path = "/vols/lz/twatson/ANN/final_ims/1.005keV_0.000_0.000_F_1.799cm_2152_im.npy"
    # needs to be "batched"
    img_array = np.load(img_path)
    img_array = np.expand_dims(img_array, axis=0)

    # Remove last layer's softmax
    model.layers[-1].activation = None
    preds = model.predict(img_array)
    print("Predicted:", preds[0])

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    plt.imshow(img_array[0, :, :, 0].astype(np.uint8), cmap="gray")
    plt.show()
    plt.matshow(heatmap)
    plt.show()
    
    save_and_display_gradcam(img_array[0], heatmap)

if noise_analysis:
    event = np.load('/vols/lz/tmarley/GEM_ITO/run/im2/F/197.711keV_0.000_0.000_F_2.207cm_5465_im.npy')
    copies = [preprocess_file_path_unscaled(event) for i in range(10)]
    fig, axs = plt.subplots(10,2, figsize=(3, 15))
    for i in range(10):
        img_array = np.expand_dims(copies[i],axis=0)
        pred = model.predict(img_array)[0][1]
        model.layers[-1].activation = None
        
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        axs[i][0].imshow(img_array[0, :, :, 0].astype(np.uint8), cmap="gray")
        axs[i][0].set_ylabel(f"{pred:.2f}")
        axs[i][1].matshow(heatmap)
    for axis in axs:
        for axis2 in axis:
            # axis2.set_axis_off()
            axis2.set_frame_on(True)
            axis2.get_xaxis().set_visible(False)
            # axis2.get_yaxis().set_visible(False)
    fig.show()

        # save_and_display_gradcam(img_array[0], heatmap)



if example_recoils:
    event_paths = ["/vols/lz/tmarley/GEM_ITO/run/im2/C/61.373keV_0.000_0.000_C_0.768cm_3062_im.npy",
                   "/vols/lz/tmarley/GEM_ITO/run/im2/C/226.142keV_0.000_0.000_C_2.085cm_1510_im.npy",
                   "/vols/lz/tmarley/GEM_ITO/run/im2/C/428.461keV_0.000_0.000_C_2.179cm_8041_im.npy",
                   "/vols/lz/tmarley/GEM_ITO/run/im2/F/60.844keV_0.000_0.000_F_1.251cm_4732_im.npy",
                   "/vols/lz/tmarley/GEM_ITO/run/im2/F/226.624keV_0.000_0.000_F_1.965cm_5710_im.npy",
                   "/vols/lz/tmarley/GEM_ITO/run/im2/F/428.668keV_0.000_0.000_F_0.557cm_8712_im.npy"]
    fig, axs = plt.subplots(3,2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1, 1]})
    for e, i in enumerate(axs):
        i[0].imshow(np.load(event_paths[e]))
        i[1].imshow(np.load(event_paths[e + 3]))
        
        i[0].set_frame_on(True)
        i[0].get_xaxis().set_visible(False)
        i[0].get_yaxis().set_visible(False)
        i[1].set_frame_on(True)
        i[1].get_xaxis().set_visible(False)
        i[1].get_yaxis().set_visible(False)

    axs[0][0].set_title("Carbon",fontsize = 15)    
    axs[0][1].set_title("Fluorine",fontsize = 15)    

    # fig.text(0.04, 0.75, r"$\mathrm{E} \sim 50keV$", va='center', ha='center', fontsize=10)
    # fig.text(0.04, 0.5, r"$\mathrm{E} \sim 200keV$", va='center', ha='center', fontsize=10)
    # fig.text(0.04, 0.25, r"$\mathrm{E} \sim 450keV$", va='center', ha='center', fontsize=10)

    fig.tight_layout()
    if save:
        fig.savefig("example_recoils",dpi=300)
    fig.show()
