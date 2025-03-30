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
from sklearn.metrics import roc_curve, auc, confusion_matrix

backdoor = False

# -=+ model +=-
# which = "L"
which = "C"
# which = "both"


# -=+ plotting +=-
save = False

roc = False
conf_mat = False
prediction_with_energy = False
gradcam = False
blank_analyis = False
noise_analysis = False
example_recoils = False
preprocess_figure = False
acc_loss_epochs = False
occlusion_analysis = False
accuracy_with_energy = False

# -=+ dataset +=-
biased = False
exclude_low_energies = False
save_sets = [False, False, False] # train, val, test


# -=+ details +=-
make_predictions = False
CoNNCR_version = 12
if which=="C":
    predictions_file_path = f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v{CoNNCR_version}/CoNNCR-Rv{CoNNCR_version}_predictions.csv"
    # predictions_file_path = f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v{5}/CoNNCR-Rv{5}_predictions.csv"

elif which == "L":
    predictions_file_path = "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/LENRI-CF4-3_predictions.csv"
if CoNNCR_version <= 3:
    use_unscaled = False
else:
    use_unscaled = True

if backdoor:
    predictions_file_path = "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/LENRI-CF4-3_predictions.csv"


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
    # model = tf.keras.models.load_model(
    #         f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/CoNNCR-R.keras",
    #         custom_objects={"softmax_v2": tf.keras.activations.softmax},
    #     )

    base_dir_list = [
        (
            "/vols/lz/twatson/ANN/old_final_ims"
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
from cnn_processing import get_file_list

base_dirs = ["/vols/lz/twatson/ANN/final_ims"]

file_list = get_file_list(base_dirs)
file_list = sorted(file_list)
np.random.seed(77) 
np.random.shuffle(file_list)


batch_size = 16
dataset_size = len(file_list) # i might be stupid lmao
train_size = (int(0.7 * dataset_size)//batch_size)*batch_size
val_size = (int(0.15 * dataset_size)//batch_size)*batch_size
test_size = ((dataset_size - train_size - val_size)//batch_size)*batch_size  # Ensure all data is used

train_list = file_list[:train_size]
val_list = file_list[train_size:train_size + val_size]
test_list = file_list[train_size + val_size : train_size + val_size + test_size]
print(f"FIRST AND LAST ELEMENTS OF TEST SET: {test_list[0], test_list[-1]}")

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

        for file_index in range(0, len(test_list), batch_size):
            # Get batch file paths
            batched_paths = test_list[file_index:file_index + batch_size]

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
    if backdoor:
        reader = csv.reader(file, delimiter="\t")
    else:
        reader = csv.reader(file)
    next(reader)  # Skip header

    if backdoor:
        for row in reader:
            file_path, prediction_C, prediction_F = row  # Extract columns
            
            # Convert prediction string "[0.3, 0.7]" into a list [0.3, 0.7]
              # Safely convert string to list

            # Determine ground truth class from filename
            true_class = 1 if "F" in os.path.basename(file_path) else 0  # Assign class based on "F" or "C"
            energy = float(re.search(r'/([\d.]+)keV', file_path).group(1))
            # Store information in a structured format
            data.append([file_path, true_class, energy, float(prediction_F)])
    else:
        for row in reader:
            file_path, prediction_str = row  # Extract columns
            
            # Convert prediction string "[0.3, 0.7]" into a list [0.3, 0.7]
            prediction = eval(prediction_str)  # Safely convert string to list

            # Determine ground truth class from filename
            true_class = 1 if "F" in os.path.basename(file_path) else 0  # Assign class based on "F" or "C"
            energy = float(re.search(r'/([\d.]+)keV', file_path).group(1))
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



if roc:
    y_true = [i[1] for i in data]
    y_scores = [i[3] for i in data]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color="blue", lw=2, label = f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0,1],[0,1],color="gray",linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    if save:
        plt.savefig("ROC.png",dpi=300)
    plt.show()
    

if conf_mat:
    import seaborn as sns
    y_true = [i[1] for i in data]
    y_pred = [round(i[3]) for i in data]
    cm = confusion_matrix(y_true,y_pred)
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=["C", "F"],yticklabels=["C", "F"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    if save:
        plt.savefig("confusion_matrix.png",dpi=300)
    plt.show()

if preprocess_figure:
    raw = np.load("/vols/lz/tmarley/GEM_ITO/run/im2/F/300.867keV_0.000_0.000_F_1.446cm_3539_im.npy")
    noisy = preprocess_file_path_unscaled(raw,steps=[1])
    smooth = preprocess_file_path_unscaled(raw,steps=[1,2])
    thresholded = preprocess_file_path_unscaled(raw,steps=[1,2,3])
    stacked = preprocess_file_path_unscaled(raw,steps=[1,2,4]).astype(np.uint8)
    resized = preprocess_file_path_unscaled(raw,steps=[1,2,4,5]).numpy().astype(np.uint8)
    preprocessed = preprocess_file_path_unscaled(raw,steps=[1,2,4,5,6]).numpy().astype(np.uint8)
    
    fig = plt.matshow(raw)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if save:
        plt.savefig("raw.png",dpi=300)
    fig = plt.matshow(noisy)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if save:
        plt.savefig("noisy.png",dpi=300)
    fig = plt.matshow(smooth)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if save:
        plt.savefig("smooth.png",dpi=300)
    fig = plt.matshow(thresholded)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if save:
        plt.savefig("thresholded.png",dpi=300)
    fig = plt.matshow(stacked)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if save:
        plt.savefig("stacked.png",dpi=300)
    fig = plt.matshow(resized)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if save:
        plt.savefig("resized.png",dpi=300)
    fig = plt.matshow(preprocessed)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if save:
        plt.savefig("preprocessed.png",dpi=300)
    
    

if prediction_with_energy:
    labels = np.array([i[1] for i in data])
    energies = np.array([i[2] for i in data])
    predictions = np.array([i[3] for i in data])
    colours = np.array(["green" if labels[i] == round(predictions[i]) else "brown" for i in range(len(labels))])

    mask_C = labels == 0  # Array of True/False values
    mask_F = labels == 1 

    # Plot all "o" markers in one go
    plt.scatter(energies[mask_C], predictions[mask_C], c=colours[mask_C], marker=".",label="C")

    # Plot all "x" markers in one go
    plt.scatter(energies[mask_F], predictions[mask_F], c=colours[mask_F], marker="+",label="F")

    plt.grid()
    plt.legend()
    
    
    plt.show()


    # plt.scatter(energies,predictions,marker=markers,c=colours)
    # plt.show()



accuracy = sum(row[1] == round(row[3]) for row in data) / len(data)
print(f"Biased? {biased}")
print(f"Exclude low energy? {exclude_low_energies}")
print(f"Accuracy: {accuracy:.2%}")
from sklearn.metrics import precision_recall_fscore_support
labels, preds = zip(*[(d[1], round(d[3])) for d in data])
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
print(f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1 Score: {f1:.3f}')




    

if blank_analyis:
    
    blanks = [np.zeros((np.random.randint(100,130), np.random.randint(100,130))) for i in range(25)]
    preprocessed = [preprocess_file_path_unscaled(blank,steps=[1,2,4,5,6]) for blank in blanks]
    predictions = []
    for file in preprocessed:
        # plt.imshow(file.numpy().astype(np.uint8))
        # plt.show()
        preds = model.predict(np.expand_dims(file,axis=0))
        # print("Predicted:", preds[0])
        predictions.append(preds[0][1] if CoNNCR_version < 7 else preds[0][0])
    fig, axs = plt.subplots(1,2,figsize=(10,6))
    no_C = True
    no_F = True
    i=0
    while no_C or no_F:
        if round(predictions[i]) == 0:
            axs[0].matshow(undo_preprocess(preprocessed[i].numpy()).astype(np.uint8))
            axs[0].set_title(f"Prediction: {predictions[i]}")
            axs[0].get_xaxis().set_visible(False)
            axs[0].get_yaxis().set_visible(False)
            no_C = False
        elif round(predictions[i]) == 1:
            axs[1].matshow(undo_preprocess(preprocessed[i].numpy()).astype(np.uint8))
            axs[1].set_title(f"Prediction: {predictions[i]}")
            axs[1].get_xaxis().set_visible(False)
            axs[1].get_yaxis().set_visible(False)
            no_F = False
        i+=1
        
    if save:
        fig.savefig("blank_predictions",dpi=300)
    fig.show()            


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
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import keras
    from gradcam import get_img_array, make_gradcam_heatmap, save_and_display_gradcam
    
    img_size=(224, 224)
    last_conv_layer_name = "block5_conv3"
    
    event = np.load('/vols/lz/tmarley/GEM_ITO/run/im2/F/197.711keV_0.000_0.000_F_2.207cm_5465_im.npy')
    event = np.load("/vols/lz/tmarley/GEM_ITO/run/im1/C/164.808keV_0.000_0.000_C_0.741cm_7190_im.npy")    
    copies = [preprocess_file_path_unscaled(event,steps=[1,2,4,5,6]) for i in range(10)]
    fig, axs = plt.subplots(10,2, figsize=(3, 15))
    for i in range(10):
        img_array = np.expand_dims(copies[i],axis=0)
        pred = model.predict(img_array)[0][1 if CoNNCR_version < 7 else 0] 
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
if False: # version for 2x2 plot
    #     event = np.load('/vols/lz/tmarley/GEM_ITO/run/im2/F/197.711keV_0.000_0.000_F_2.207cm_5465_im.npy')
    # copies = [preprocess_file_path_unscaled(event,steps=[1,2,4,5,6]) for i in range(2)]
    # fig, axs = plt.subplots(2,2, figsize=(6, 6))
    # for i in range(2):
    #     img_array = np.expand_dims(copies[i],axis=0)
    #     pred = model.predict(img_array)[0][1]
    #     model.layers[-1].activation = None
        
    #     heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    #     axs[i][0].imshow(img_array[0, :, :, 0].astype(np.uint8), cmap="gray")
    #     axs[i][0].set_ylabel(f"{pred:.2f}")
    #     axs[i][1].matshow(heatmap)
    # for axis in axs:
    #     for axis2 in axis:
    #         # axis2.set_axis_off()
    #         axis2.set_frame_on(True)
    #         axis2.get_xaxis().set_visible(False)
    #         # axis2.get_yaxis().set_visible(False)
    # fig.savefig("noise_analysis.png",dpi=300)
    # fig.show()
    pass


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


if acc_loss_epochs:
    import json
    import glob

    file_paths = [f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v{CoNNCR_version}/history.json",
                f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v{CoNNCR_version}/finetuned_history.json",
                f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v{CoNNCR_version}/finetuned_history_2.json",

                ]

    # Initialize combined lists
    combined_data = {
        "accuracy": [],
        "loss": [],
        "val_accuracy": [],
        "val_loss": []
    }

    # Loop through each file and append the data
    for file_path in file_paths:
        with open(file_path, "r") as f:
            history_data = json.load(f)
            for key in combined_data.keys():
                combined_data[key].extend(history_data.get(key, []))
            if file_path == file_paths[0]:
                epoch_marker = len(history_data["accuracy"])

    # Debugging: Print lengths to ensure correctness
    # for key, values in combined_data.items():
    #     print(f"{key}: {len(values)} entries")

    # Extract values from dictionary
    accuracy = combined_data.get("accuracy", [])
    loss = combined_data.get("loss", [])
    val_accuracy = combined_data.get("val_accuracy", [])
    val_loss = combined_data.get("val_loss", [])

    # Create a single figure
    fig, ax1 = plt.subplots()
    lines = []
    # Plot accuracy and validation accuracy
    if accuracy:
        lines += ax1.plot(accuracy, label="Accuracy", color="blue")
    if val_accuracy:
        lines += ax1.plot(val_accuracy, label="Validation Accuracy", color="blue", linestyle="--")

    # Create secondary y-axis for loss
    axloss = ax1.twinx()
    if loss:
        # axloss.grid()
        axloss.yaxis.label.set_color("red")
        lines += axloss.plot(loss, label="Loss", color="red")
        axloss.tick_params(axis="y", colors="red")
    if val_loss:
        lines += axloss.plot(val_loss, label="Validation Loss", color="red", linestyle="--")

    ymin_acc, ymax_acc = ax1.get_ylim()
    # ymin_loss, ymax_loss = axloss.get_ylim()

    ax1.vlines(epoch_marker, 0, 1, color="black", linestyle=":")
    ax1.set_ylim(ymin_acc, ymax_acc)
    

    xticks = list(ax1.get_xticks())  # Get existing x-ticks
    xticks.append(epoch_marker)  # Add marker
    ax1.set_xticks(sorted(xticks))  # Set new x-ticks

    # Convert x-ticks to labels, replacing epoch_marker with a custom label
    xtick_labels = [str(int(tick)) if tick != epoch_marker else f"{epoch_marker}" for tick in sorted(xticks)]
    ax1.set_xticklabels(xtick_labels)

    # Restore x-limits to Matplotlib’s auto-determined values
    ax1.set_xlim(0, len(combined_data["accuracy"]))
    axloss.set_xlim(0, len(combined_data["accuracy"]))


    ax1.vlines(25, 0, 1, color="black", linestyle=":")

    # Configure labels and title
    
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left")
    ax1.set_title("Accuracy and Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy", color="blue")
    ax1.tick_params(axis="y", colors="blue")
    ax1.grid(True, linestyle="dotted")
    axloss.grid(False)  # Remove grid from secondary axis

    axloss.set_ylabel("Loss", color="red")


    plt.show()


if occlusion_analysis:
    size = 56
    
    num = 224//size
    image = np.load("/vols/lz/twatson/ANN/final_ims/199.882keV_0.000_0.000_C_2.452cm_2699_im.npy")
    image = np.load("/vols/lz/twatson/ANN/final_ims/56.825keV_0.000_0.000_F_0.604cm_5803_im.npy")
    image = np.load("/vols/lz/twatson/ANN/final_ims/351.803keV_0.000_0.000_F_1.522cm_5677_im.npy")
    image = np.load('/vols/lz/twatson/ANN/old_final_ims/164.808keV_0.000_0.000_C_0.741cm_7190_im.npy')
    
    fig, axs = plt.subplots(num, 2*num, figsize = (10,10))
    vgg_mean = np.array([-103.939,  -116.779, -123.68])

    for i in range(num):
        for j in range(num):
            occluded = image.copy()
            x_start, y_start = i * size, j * size
            x_end, y_end = x_start + size, y_start + size
            
            
            
            occluded[y_start:y_end, x_start:x_end, :] = vgg_mean
            preds = model.predict(np.expand_dims(occluded,axis=0))
            predicted_class = np.argmax(preds)  # Get the class with highest probability
            confidence = np.max(preds)  # Get confidence score

            # Plot occluded image with prediction
            ax = axs[j, 2*i]
            ax.imshow(undo_preprocess(occluded))
            ax.set_title(f"Class: {predicted_class}, Conf: {confidence:.2f}")
            ax.axis("off")
            
            ax = axs[j, 2*i+1]
            os.environ["KERAS_BACKEND"] = "tensorflow"
            import keras
            from gradcam import get_img_array, make_gradcam_heatmap, save_and_display_gradcam
            
            img_size=(224, 224)
            last_conv_layer_name = "block5_conv3"
            
            pred = preds[0][1 if CoNNCR_version < 7 else 0]
            model.layers[-1].activation = None
            
            heatmap = make_gradcam_heatmap(np.expand_dims(occluded,axis=0), model, last_conv_layer_name)
            ax.matshow(heatmap)
            ax.axis("off")


    plt.tight_layout()
    plt.show()


if accuracy_with_energy:
    # histogram approach
    data_arr = np.array(data, dtype=object)

    # Extract energies, true labels, and predictions.
    energies = data_arr[:, 2].astype(float)
    true_labels = data_arr[:, 1].astype(int)
    predictions = data_arr[:, 3].astype(float)

    # Determine the predicted class (using 0.5 as the decision threshold).
    threshold = 0.5
    predicted_labels = (predictions >= threshold).astype(int)

    # Calculate a boolean array for whether each prediction is correct.
    correct = (predicted_labels == true_labels)

    # Choose the number of bins for energy. Here we use 10 bins.
    num_bins = 10
    bins = np.linspace(energies.min(), energies.max(), num_bins + 1)

    # Digitize the energies into bins.
    bin_indices = np.digitize(energies, bins)

    # Calculate the bin centers (for plotting on the x-axis).
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Prepare lists for binned accuracy, uncertainties, and counts.
    acc_list = []
    err_list = []
    counts = []

    # Loop over each bin to calculate the accuracy and the uncertainty.
    for i in range(1, len(bins)):
        idx = np.where(bin_indices == i)[0]  # indices for events in the current bin
        n = len(idx)
        if n == 0:
            # If there are no events in this bin, record NaN values.
            acc_list.append(np.nan)
            err_list.append(np.nan)
            counts.append(0)
        else:
            n_correct = np.sum(correct[idx])
            accuracy = n_correct / n
            acc_list.append(accuracy)
            counts.append(n)
            # Compute the binomial uncertainty.
            err = np.sqrt(accuracy * (1 - accuracy) / n)
            err_list.append(err)

    # Plot the binned accuracy versus energy with error bars.
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, acc_list, width=(bins[1]-bins[0])*0.9, align='center', label='Binned Accuracy')
    plt.errorbar(bin_centers, acc_list, yerr=err_list, fmt='none', ecolor='black', capsize=5, label='Uncertainty')

    plt.xlabel('Energy (keV)')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs. Energy (Histogram)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xlim(0,710)
    plt.ylim(0.5,1.05)
    if save:
        plt.savefig("energy_accuracy_hist.png",dpi=300)
    plt.show()
    