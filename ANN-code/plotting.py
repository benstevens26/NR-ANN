import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import tensorflow as tf
import os
import csv
from tqdm import tqdm
import re
from preprocess import preprocess_file_path_unscaled, get_file_list
from sklearn.metrics import roc_curve, auc, confusion_matrix





backdoor = True

# -=+ model +=-
# which = "L"
which = "C"
# which = "both"


# -=+ plotting +=-
save = False
rc_params = True

roc = False
conf_mat = False
prediction_with_energy = False
gradcam = True
blank_analyis = False
noise_analysis = False
example_recoils = False
initial_preprocess_figure = False
save_all_preprocess = False
cnn_preprocess_figure = False
learning_curve = False
learning_curve_2 = False
occlusion_analysis = False
accuracy_with_energy = False
energy_v_angle_for_species_unlogged = False
energy_v_angle_for_species_logged = False
double_roc = False
intensity_v_energy = False
lay_summary_example_recoils = False
empty_frames = False
feature_maps = False


if rc_params:
    # view the list with plt.rcParams.keys()
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.labelsize'] = 'large'
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['legend.fontsize'] = 'large'
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.directory'] = '/vols/lz/twatson/ANN/NR-ANN/ANN-code/images'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.labelsize'] = 'large'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.labelsize'] = 'large'
    plt.rcParams['ytick.right'] = True

# -=+ dataset +=-
biased = True
exclude_low_energies = True
threshold_with_intensity = True
save_sets = [False, False, False] # train, val, test
min_acc_set = False # Also produces the accuracy vs energy plot
use_biased_LENRI = False
use_bayesian = True

# -=+ details +=-
C_threshold = 130
F_threshold = 170
F_low_threshold_int = 485599.1264847403
F_high_threshold_int = 1564652.6704808741


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

# if backdoor:
#     predictions_file_path = "/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/LENRI-CF4-3_predictions.csv"


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


def bayesian_adjustment(predict_F, test_ratio=4.04):
    # True priors in the test region
    P_test_C = 1 / (1 + test_ratio)
    P_test_F = test_ratio / (1 + test_ratio)

    # Apply Bayes' rule for adjustment
    numerator_C = (1-predict_F) * P_test_C
    numerator_F = predict_F * P_test_F
    denominator = numerator_C + numerator_F

    adjusted_F = numerator_F / denominator

    return adjusted_F


if backdoor:
    import pandas as pd
    df = pd.read_csv("/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/features_CF4_3_raw.csv", delimiter="\t")
    # Extract energy from cam_path
    df["energy"] = df["cam_path"].apply(
        lambda x: float(re.search(r'/([\d.]+)keV', x).group(1)) if re.search(r'/([\d.]+)keV', x) else None
    )

    # Read predictions into a new DataFrame
    predictions = []
    if use_biased_LENRI:
        with open("/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/lenri_cf4_weighted_tuned_predictions.csv", mode="r") as file:
            reader = csv.reader(file, delimiter=",")
            next(reader)  # Skip header
            for row in reader:
                # print(row)
                file_path, _, _, prediction_F = row
                true_class = 1 if "F" in os.path.basename(file_path) else 0  # Assign class based on "F" or "C"
                predictions.append({"cam_path": file_path, "species": true_class, "prediction": prediction_F})
        if use_bayesian:
            print("WARNING: using both bias and bayesian")
    else:
        with open("/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/LENRI-CF4-3_predictions.csv", mode="r") as file:
            reader = csv.reader(file, delimiter="\t")
            next(reader)  # Skip header
            for row in reader:
                # print(row)
                file_path, prediction_F = row
                true_class = 1 if "F" in os.path.basename(file_path) else 0  # Assign class based on "F" or "C"
                predictions.append({"cam_path": file_path, "species": true_class, "prediction": prediction_F})
    
    # Convert predictions list to DataFrame
    df_predictions = pd.DataFrame(predictions)

    # Merge the DataFrames based on cam_path
    df = df.merge(df_predictions, on="cam_path", how="inner")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["correctness"] = 1 - np.abs(df["species"] - df["prediction"])
    if use_bayesian:
            df["adjusted_preds"] = np.where(
                ((df["sum_intensity_cam"] >= 485599.1264847403) & (df["sum_intensity_cam"] <= 1564652.6704808741)),
                bayesian_adjustment(df["prediction"]),
                df["prediction"]
            )

data = []  # List to store extracted data
with open(predictions_file_path, mode="r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        file_path, prediction_str = row  # Extract columns
        
        # Convert prediction string "[0.3, 0.7]" into a list [0.3, 0.7]
        prediction = eval(prediction_str)[1]  # Safely convert string to list

        # Determine ground truth class from filename
        true_class = 1 if "F" in os.path.basename(file_path) else 0  # Assign class based on "F" or "C"
        energy = float(re.search(r'/([\d.]+)keV', file_path).group(1))
        if use_bayesian and (energy < 473 and energy > 170): # should use intensity but fuck it
            prediction = bayesian_adjustment(prediction)

        # Store information in a structured format
        data.append([file_path, true_class, energy, prediction])
        





# Exclude data as desired

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
    

if exclude_low_energies and not threshold_with_intensity:
    data = [event for event in data if not ((event[1] == 1 and event[2] < F_threshold) or (event[1] == 0 and event[2] < C_threshold))]
elif exclude_low_energies and threshold_with_intensity:
    data = [event for event in data if not ((event[1] == 1 and event[2] < F_threshold) or (event[1] == 0 and event[2] < C_threshold))]


# Create minimum accuracy set:
if min_acc_set:
    if backdoor:
        full_df = df.copy()
        if biased:
            CF_ratio = 7.33
            num_F = (df["species"] == 1).sum()  # Count F cases
            num_C = int(num_F // CF_ratio)  # Compute number of C cases

            C_data = df[df["species"] == 0]  # Select C cases
            F_data = df[df["species"] == 1]  # Select F cases

            if len(C_data) > num_C:
                # Randomly shuffle and select only num_C elements
                C_data = C_data.sample(n=num_C, random_state=77)

            # Combine F_data and filtered C_data
            df = pd.concat([C_data, F_data]).sample(frac=1, random_state=77)  # Shuffle dataset

        # Apply energy thresholds
        if exclude_low_energies and not threshold_with_intensity:
            df = df[~((df["species"] == 1) & (df["energy"] < F_threshold)) & 
                    ~((df["species"] == 0) & (df["energy"] < C_threshold))]
        elif exclude_low_energies and threshold_with_intensity: # This doesn't actually threshold with intesnity at the moment - might want to come back here
            df = df[~((df["species"] == 1) & (df["energy"] < F_threshold)) & 
                    ~((df["species"] == 0) & (df["energy"] < C_threshold))]
    else:
        raise Exception("Need to know sum_intensity for the min_acc_set, so please enable backdoor and use LENRI's results!")

    df["min_acc_pred"] = ((df["sum_intensity_cam"] >= 485599.1264847403) & 
                          (df["sum_intensity_cam"] <= 1564652.6704808741)).astype(int)
    num_bins = 30 # if using biased + bayesian, 43 is best to minimise min_acc winning and 26 is best for maximising lenri winning 
                  # if using neither, 32 is best to minimise min_acc and 30 is best to maximise LENRI
                  # if just bayesian, 39 for min min_acc 25 for max LENRI
    bin_edges = np.linspace(df["energy"].min(), df["energy"].max(), num_bins + 1)
    # Assign energy values to bins
    df["energy_bin"] = pd.cut(df["energy"], bins=bin_edges, labels=False)



    # Calculate accuracy per bin for each classifier:
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins
    
    
    accuracy_per_bin = (
        df.groupby("energy_bin")
          .apply(lambda x: (x["species"] == x["min_acc_pred"]).mean())
          .sort_index()
    )
    LENRI_accuracy_per_bin = (
        df.groupby("energy_bin")
          .apply(lambda x: (x["species"] == (x["adjusted_preds" if use_bayesian else "prediction"].round())).mean())
          .sort_index()
    )
    F_ratio_by_bin = (
        df.groupby("energy_bin")["species"].mean()
          .sort_index()
    )

    df_old = pd.DataFrame(data, columns=["file_path_cam", "species", "energy", "prediction"])
    df_old["energy_bin"] = pd.cut(df_old["energy"], bins=bin_edges, labels=False)
    CoNNCR_accuracy_per_bin = (
        df_old.groupby("energy_bin")
              .apply(lambda x: (x["species"] == (x["prediction"].round())).mean())
              .sort_index()
    )
    
    
    n_bootstrap = 1000  # Number of bootstrap iterations

    # For min_acc_pred (Intensity Cuts)
    bootstrap_errors_min_acc = []
    for i in range(num_bins):
        bin_df = df[df["energy_bin"] == i]
        n = len(bin_df)
        if n > 1:
            boot_accs = []
            for _ in range(n_bootstrap):
                sample = bin_df.sample(n=n, replace=True)
                boot_acc = (sample["species"] == sample["min_acc_pred"]).mean()
                boot_accs.append(boot_acc)
            bootstrap_errors_min_acc.append(np.std(boot_accs))
        else:
            bootstrap_errors_min_acc.append(0)
    bootstrap_errors_min_acc = np.array(bootstrap_errors_min_acc)

    # For LENRI (using 'prediction' from df, rounded)
    bootstrap_errors_LENRI = []
    for i in range(num_bins):
        bin_df = df[df["energy_bin"] == i]
        n = len(bin_df)
        if n > 1:
            boot_accs = []
            for _ in range(n_bootstrap):
                sample = bin_df.sample(n=n, replace=True)
                boot_acc = (sample["species"] == sample["adjusted_preds" if use_bayesian else "prediction"].round()).mean()
                boot_accs.append(boot_acc)
            bootstrap_errors_LENRI.append(np.std(boot_accs))
        else:
            bootstrap_errors_LENRI.append(0)
    bootstrap_errors_LENRI = np.array(bootstrap_errors_LENRI)

    # For CoNNCR (using df_old)
    bootstrap_errors_CoNNCR = []
    for i in range(num_bins):
        bin_df = df_old[df_old["energy_bin"] == i]
        n = len(bin_df)
        if n > 1:
            boot_accs = []
            for _ in range(n_bootstrap):
                sample = bin_df.sample(n=n, replace=True)
                boot_acc = (sample["species"] == sample["prediction"].round()).mean()
                boot_accs.append(boot_acc)
            bootstrap_errors_CoNNCR.append(np.std(boot_accs))
        else:
            bootstrap_errors_CoNNCR.append(0)
    bootstrap_errors_CoNNCR = np.array(bootstrap_errors_CoNNCR)






    # Compute best classifier per bin (for the discrete colour bar)
    best_colors = []
    # Define classifier colors corresponding to each curve
    colors = {"acc": "white", "LENRI": "#FF4500", "CoNNCR": "#3383FF"}
    # Loop over each bin index:
    for bin_idx in range(num_bins):
        # Extract accuracies. In case some bins are missing, use 0 as default.
        acc1 = accuracy_per_bin.get(bin_idx, 0)
        acc2 = LENRI_accuracy_per_bin.get(bin_idx, 0)
        acc3 = CoNNCR_accuracy_per_bin.get(bin_idx, 0)
        # Find which classifier has the highest accuracy
        max_acc = max(acc1, acc2, acc3)
        if max_acc == acc1:
            best_colors.append(colors["acc"])
        elif max_acc == acc2:
            best_colors.append(colors["LENRI"])
        else:
            best_colors.append(colors["CoNNCR"])

    # Create subplots: one for the main plot and a smaller one for the colour bar.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                   gridspec_kw={'height_ratios': [4, 0.25]}, sharex=True)

    # Plot the accuracy curves on the first axis:
    # ax1.plot(bin_centers, accuracy_per_bin, '-', color='black', zorder=1); ax1.errorbar(bin_centers, accuracy_per_bin, yerr=bootstrap_errors_min_acc, fmt='o', markerfacecolor=colors["acc"], markeredgecolor='black', ecolor='black', ms=6, label="Intensity Cuts", zorder=2)
    # ax1.errorbar(bin_centers, LENRI_accuracy_per_bin, yerr=bootstrap_errors_LENRI, fmt='.', color=colors["LENRI"], linestyle='-', label="LENRI", mec='black', ms=12)
    # ax1.errorbar(np.array(bin_centers), CoNNCR_accuracy_per_bin, yerr=bootstrap_errors_CoNNCR, fmt='.', color=colors["CoNNCR"], linestyle='-', label="CoNNCR", mec='black', ms=12)
    
    # Intensity Cuts (black line + colored markers)
    ax1.plot(np.array(bin_centers) - 1.5, accuracy_per_bin, '-', color='black', zorder=1)
    ax1.errorbar(np.array(bin_centers) - 1.5, accuracy_per_bin, yerr=bootstrap_errors_min_acc, fmt='o', 
                markerfacecolor=colors["acc"], markeredgecolor='black', ecolor='black', 
                ms=6, label="Intensity Cuts", zorder=2)

    # LENRI
    ax1.plot(bin_centers, LENRI_accuracy_per_bin, '-', color=colors["LENRI"], zorder=1)
    ax1.errorbar(bin_centers, LENRI_accuracy_per_bin, yerr=bootstrap_errors_LENRI, fmt='o', 
                markerfacecolor=colors["LENRI"], markeredgecolor='black', ecolor=colors["LENRI"], 
                ms=6, label="LENRI", zorder=2)

    # CoNNCR
    ax1.plot(np.array(bin_centers) + 1.5, CoNNCR_accuracy_per_bin, '-', color=colors["CoNNCR"], zorder=1)
    ax1.errorbar(np.array(bin_centers) + 1.5, CoNNCR_accuracy_per_bin, yerr=bootstrap_errors_CoNNCR, fmt='o', 
                markerfacecolor=colors["CoNNCR"], markeredgecolor='black', ecolor=colors["CoNNCR"], 
                ms=6, label="CoNNCR", zorder=2)
    
    
    # ax1.plot(bin_centers, F_ratio_by_bin.values, linestyle='-.', color='black', alpha=0.4, label="F ratio",lw=2)
    ax1.fill_between(bin_centers, F_ratio_by_bin.values, color='black', alpha=0.2, label="F ratio")
    ax1.set_ylabel("Accuracy",fontsize=18)
    ax1.set_title("Accuracy vs. Energy (Bayesian Adjusted)" if use_bayesian else "Accuracy vs. Energy",fontsize=22)
    ax1.axvline(x=170, color='black', linestyle='--', linewidth=1.5)
    ax1.axvline(x=473, color='black', linestyle='--', linewidth=1.5)
    ax1.text(451.5, 0.35, "E = 473 keV", rotation=90, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.1'))
    ax1.text(151.5, 0.78, "E = 170 keV", rotation=90, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.1'))

    
    
    ax1.grid(True)
    ax1.set_xlim(130, 707)
    ax1.set_ylim(0.22, 1.05)
    ax1.legend(loc="best", prop={'size': 18})

    # Plot the discrete colour bar on the second axis:
    # We will plot a bar for each energy bin.
    bin_widths = np.diff(bin_edges)
    for i in range(num_bins):
        ax2.bar(bin_centers[i], 1, width=bin_widths[i],
                color=best_colors[i], align='center', edgecolor='none')
    ax2.set_yticks([])
    ax2.set_ylabel("Best", rotation=0, labelpad=30, va='center')
    ax2.set_xlabel("Energy (keV)",fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    # ax2.xaxis.set_label_position('top')
    # ax2.xaxis.tick_top()
    # ax1.xaxis.set_visible(False)    
    ax2.set_xlim(130, 707)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', which='both', length=0)


    plt.tight_layout()
    if save:
        plt.savefig("accuracy_v_energy_bayesian.png" if use_bayesian else "accuracy_v_energy.png",dpi=300)
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    


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


################################==FIGURES==################################
# for i in data:
#     energy = i[2]
#     if energy < 170:
#         pred = 0
#     elif energy > 170 and energy < 473:
#         pred = 1
#     elif energy > 473:
#         pred = 0
#     else:
#         print("something went wrong")
#     i.append(pred)
# min_acc = sum(row[1] == row[4] for row in data) / len(data)
# print("MINIMUM ACCURACY: " + str(min_acc))
print(f"Length of test set = {len(data)}")
accuracy = sum(row[1] == round(row[3]) for row in data) / len(data)
print(f"Biased? {biased}")
print(f"Exclude low energy? {exclude_low_energies}")
print(f"Accuracy: {accuracy:.2%}")
from sklearn.metrics import precision_recall_fscore_support
labels, preds = zip(*[(d[1], round(d[3])) for d in data])
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
print(f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1 Score: {f1:.3f}')



if roc:
    y_true = [i[1] for i in data]
    y_scores = [i[3] for i in data]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color="blue", lw=2, label = f"ROC curve (AUC = {roc_auc:.3f})")
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

if initial_preprocess_figure:
    raw = np.load("/vols/lz/tmarley/GEM_ITO/run/im2/F/182.736keV_0.000_0.000_F_2.478cm_7164_im.npy")
    noisy = preprocess_file_path_unscaled(raw,steps=[1])
    smooth = preprocess_file_path_unscaled(raw,steps=[1,2])
    thresholded = preprocess_file_path_unscaled(raw,steps=[1,2,3])
    stacked = preprocess_file_path_unscaled(raw,steps=[1,2,4]).astype(np.uint8)
    resized = preprocess_file_path_unscaled(raw,steps=[1,2,4,5]).numpy().astype(np.uint8)
    preprocessed = preprocess_file_path_unscaled(raw,steps=[1,2,4,5,6]).numpy().astype(np.uint8)
    
    if save_all_preprocess:
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
    
    fig, axs = plt.subplots(1, 4, figsize=(12, 6))
    axs[0].imshow(raw)
    axs[0].set_title("Raw")
    axs[0].axis("off")
    axs[1].imshow(noisy)
    axs[1].set_title("Noise")
    axs[1].axis("off")
    axs[2].imshow(smooth)
    axs[2].set_title("Smooth")
    axs[2].axis("off")
    axs[3].imshow(thresholded)
    axs[3].set_title("Threshold")
    axs[3].axis("off")
    if save:
        plt.savefig("initial_preprocess_figure.png",dpi=300)
    

if cnn_preprocess_figure:
    raw = np.load("/vols/lz/tmarley/GEM_ITO/run/im2/F/182.736keV_0.000_0.000_F_2.478cm_7164_im.npy")
    noisy = preprocess_file_path_unscaled(raw,steps=[1])
    smooth = preprocess_file_path_unscaled(raw,steps=[1,2])
    thresholded = preprocess_file_path_unscaled(raw,steps=[1,2,3])
    stacked = preprocess_file_path_unscaled(raw,steps=[1,2,4]).astype(np.uint8)
    resized = preprocess_file_path_unscaled(raw,steps=[1,2,4,5]).numpy().astype(np.uint8)
    preprocessed = preprocess_file_path_unscaled(raw,steps=[1,2,4,5,6]).numpy().astype(np.uint8)

    fig, axs = plt.subplots(1, 4, figsize=(12, 6))
    axs[0].imshow(raw)
    axs[0].set_title("Initial")
    axs[0].axis("off")
    axs[1].imshow(stacked)
    axs[1].set_title("Stacked")
    axs[1].axis("off")
    axs[2].imshow(resized)
    axs[2].set_title("Resized")
    axs[2].axis("off")
    axs[3].imshow(preprocessed)
    axs[3].set_title("Preprocessed")
    axs[3].axis("off")
    if save:
        plt.savefig("cnn_preprocess_figure.png",dpi=300)


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
    from gradcam import make_gradcam_heatmap, superimpose_array
    
    last_conv_layer_name  = 'block5_conv3'  
    img_size = (224, 224)
    event = np.load('/vols/lz/twatson/ANN/old_final_ims/197.711keV_0.000_0.000_F_2.207cm_5465_im.npy')
    event = np.expand_dims(event, axis=0)
    event = event/np.max(event)*255

    img_array = event
    # model.layers[-1].activation = None
    preds = model.predict(img_array)
    print("Predicted:", preds[0])
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    event_sup = superimpose_array(img_array[0], heatmap,alpha=0.6)
    # plt.matshow(event_sup)
    
    

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


if learning_curve:
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
    fig, ax1 = plt.subplots(figsize=(10, 8))
    lines = []
    # Plot accuracy and validation accuracy
    if accuracy:
        lines += ax1.plot(accuracy, label="Accuracy", color="blue",lw=3)
    if val_accuracy:
        lines += ax1.plot(val_accuracy, label="Validation Accuracy", color="blue", linestyle="--",lw=3)

    # Create secondary y-axis for loss
    axloss = ax1.twinx()
    if loss:
        # axloss.grid()
        axloss.yaxis.label.set_color("red")
        lines += axloss.plot(loss, label="Loss", color="red",lw=3)
        axloss.tick_params(axis="y", colors="red")
    if val_loss:
        lines += axloss.plot(val_loss, label="Validation Loss", color="red", linestyle="--",lw=3)

    ymin_acc, ymax_acc = ax1.get_ylim()
    ymin_loss, ymax_loss = axloss.get_ylim()

    axloss.vlines(epoch_marker, 0, 1, color="black", linestyle="--",lw=2,zorder=2)
    axloss.vlines(25, 0, 1, color="black", linestyle="--",lw=2,zorder=2)
    ax1.set_ylim(ymin_acc, ymax_acc)
    

    xticks = list(ax1.get_xticks())  # Get existing x-ticks
    xticks.append(epoch_marker)  # Add marker
    ax1.set_xticks(sorted(xticks))  # Set new x-ticks

    # Convert x-ticks to labels, replacing epoch_marker with a custom label
    xtick_labels = [str(int(tick)) if tick != epoch_marker else f"{epoch_marker}" for tick in sorted(xticks)]
    ax1.set_xticklabels(xtick_labels)

    # Restore x-limits to Matplotlib’s auto-determined values
    ax1.set_xlim(0, len(combined_data["accuracy"]))
    axloss.set_xlim(0, len(combined_data["accuracy"])-1)


    axloss.text(7.5,0.625,"Phase 1",fontsize=20)
    axloss.text(19,0.625,"Phase 2",fontsize=20)
    axloss.text(26.5,0.625,"Phase 3",fontsize=20)
    axloss.set_ylim(ymin_loss, ymax_loss)
    # Configure labels and title
    
    labels = [l.get_label() for l in lines]
    axloss.legend(lines, labels, loc="center left")
    ax1.set_title("CoNNCR Learning Curve", fontsize=25)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy", color="blue")
    ax1.tick_params(axis="y", colors="blue")
    ax1.grid(True, linestyle="dotted")
    axloss.grid(False)  # Remove grid from secondary axis

    axloss.set_ylabel("Loss", color="red")


    plt.show()


if learning_curve_2:
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    # Define patience and version
    patience = 5  # EarlyStopping patience monitoring validation loss
    CoNNCR_version = CoNNCR_version  # define this in your environment

    # File paths for the three phases
    file_paths = [
        f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v{CoNNCR_version}/history.json",
        f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v{CoNNCR_version}/finetuned_history.json",
        f"/vols/lz/twatson/ANN/NR-ANN/ANN-code/old_models/CoNNCR-R/v{CoNNCR_version}/finetuned_history_2.json",
    ]

    # Load raw histories
    histories = [json.load(open(fp, 'r')) for fp in file_paths]

    # Insert starting checkpoint values for phases 2 and 3,
    # using metrics from `patience` epochs before the end of the prior phase
    for i in range(1, len(histories)):
        prev = histories[i - 1]
        # index of loaded checkpoint = last_epoch_index - patience
        chk_idx = len(prev['val_loss']) - 1 - patience
        for key in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
            start_val = prev[key][chk_idx]
            histories[i][key] = [start_val] + histories[i][key]

    # Compute lengths and x-offsets so each phase begins at the prior phase's last epoch (vertical jump only)
    lengths = [len(h['accuracy']) for h in histories]
    offsets = [0] + [sum(lengths[:i]) - i for i in range(1, len(lengths))]

    # Initialize figure and twin axes
    fig, ax_acc = plt.subplots(figsize=(12, 8), dpi=300)
    ax_loss = ax_acc.twinx()
    # Disable grid on loss axis
    ax_loss.grid(False)

    # Plot style settings
    colors = {'acc': 'blue', 'val_acc': 'blue', 'loss': 'red', 'val_loss': 'red'}
    styles = {'acc': '-', 'val_acc': '--', 'loss': '-', 'val_loss': '--'}
    labels = {'acc': 'Train Accuracy', 'val_acc': 'Validation Accuracy',
            'loss': 'Train Loss', 'val_loss': 'Validation Loss'}

    # Plot each phase
    for i, hist in enumerate(histories):
        x = offsets[i] + np.arange(len(hist['accuracy']))
        # Plot accuracy curves
        ax_acc.plot(x, hist['accuracy'], color=colors['acc'], linestyle=styles['acc'],
                    label=labels['acc'] if i == 0 else None)
        ax_acc.plot(x, hist['val_accuracy'], color=colors['val_acc'], linestyle=styles['val_acc'],
                    label=labels['val_acc'] if i == 0 else None)
        # Plot loss curves
        ax_loss.plot(x, hist['loss'], color=colors['loss'], linestyle=styles['loss'],
                    label=labels['loss'] if i == 0 else None)
        ax_loss.plot(x, hist['val_loss'], color=colors['val_loss'], linestyle=styles['val_loss'],
                    label=labels['val_loss'] if i == 0 else None)
        # Mark phase transition (vertical dashed line)
        if i < len(histories) - 1:
            trans_x = offsets[i + 1]
            ax_acc.axvline(trans_x, color='black', linestyle='--')
            
    # ax_acc.plot([offsets[1],offsets[1] - 4],[histories[0]["accuracy"][-5], histories[0]["accuracy"][-5]], ":",c="grey")
    # ax_acc.plot([offsets[2],offsets[2] - 4],[histories[1]["accuracy"][-5], histories[1]["accuracy"][-5]], ":",c="grey")

    # Add minor xticks at phase transition epochs and label with the epoch number
    transition_positions = [offsets[j] for j in range(1, len(offsets))]
    ax_acc.set_xticks(transition_positions, minor=True)
    ax_acc.set_xticklabels([str(x) for x in transition_positions], minor=True)
    ax_acc.tick_params(axis='x', which='minor', length=6, pad=8)

    # Axis labels, colors, and ticks
    ax_acc.set_xlabel('Epoch', fontsize=22)
    ax_acc.set_ylabel('Accuracy', color='blue', fontsize=22)
    ax_loss.set_ylabel('Loss', color='red', fontsize=22)
    ax_acc.set_title(f"CoNNCR Learning Curve", fontsize=25)
    ax_acc.tick_params(axis='y', labelcolor='blue')
    ax_loss.tick_params(axis='y', labelcolor='red')
    ax_acc.set_xlim(0, sum(lengths) - 3)
    

    ax_acc.text(8,0.79,"Phase 1",fontsize=20,ha="center",bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.1'))
    ax_acc.text(19.5,0.79,"Phase 2",fontsize=20,ha="center",bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.1'))
    ax_acc.text(28,0.79,"Phase 3",fontsize=20,ha="center",bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.1'))
    # Combine legends from both axes
    lines_acc, labels_acc = ax_acc.get_legend_handles_labels()
    lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
    ax_loss.legend(lines_acc + lines_loss, labels_acc + labels_loss, loc='center left', fontsize='large')

    # Final layout adjustments and save
    fig.tight_layout()
    if save:
        fig.savefig(f"learning_curve_v{CoNNCR_version}.png")
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





if energy_v_angle_for_species_unlogged:
    def scattering(alpha, species, m = 1.67e-27, En = 2.47*1.6e-13):
        if species == 1 or species == "F":
            M = 18.998403 * 1.67e-27
        elif species == 0 or species == "C":
            M = 12.011 * 1.67e-27
        else:
            raise Exception("species isn't carbon or fluorine")
        return En*(4*m*M)/((m+M)**2)*((np.cos(alpha))**2)
    
    alpha = np.linspace(0, np.pi/2, 1000)
    # make list of energies between 0 and 707
    energies_C = scattering(alpha, 0)
    energies_F = scattering(alpha, 1)
    # plot alpha vs energy
    plt.figure(figsize=(8, 6))
    # scale energies to keV
    energies_C = energies_C/(1.6e-19)*1e-3
    energies_F = energies_F/(1.6e-19)*1e-3
    # convert alpha to degrees
    alpha = alpha*(180/np.pi)
    
    # plt.ylim(10,707)
    # plt.xlim(0,90)
    plt.plot(alpha, energies_C, label="Carbon", color="black",lw=3)
    plt.plot(alpha, energies_F, label="Fluorine", color="red",lw=3)

    # horizontal dashed line across the plot at E = 473kev
    plt.axhline(y=473, color='grey', linestyle='--', linewidth=1.5)
    plt.xlabel("Scattering Angle (degrees)")
    plt.ylabel("Energy (keV)")
    plt.title("Energy vs. Scattering Angle")
    plt.grid(True, alpha=0.7)
    plt.legend()
    plt.xlim(0, 90)
    plt.ylim(0, 707)
    if save:
        plt.savefig("scattering_angle_unlogged.png",dpi=300)
    plt.show()

if energy_v_angle_for_species_logged:
    def scattering(alpha, species, m = 1.67e-27, En = 2.47*1.6e-13):
        if species == 1 or species == "F":
            M = 18.998403 * 1.67e-27
        elif species == 0 or species == "C":
            M = 12.011 * 1.67e-27
        else:
            raise Exception("species isn't carbon or fluorine")
        return En*(4*m*M)/((m+M)**2)*((np.cos(alpha))**2)
    
    alpha = np.linspace(0, np.pi/2, 1000)
    # make list of energies between 0 and 707
    energies_C = scattering(alpha, 0)
    energies_F = scattering(alpha, 1)
    # plot alpha vs energy
    
    # scale energies to keV
    energies_C = energies_C/(1.6e-19)*1e-3
    energies_F = energies_F/(1.6e-19)*1e-3
    # convert alpha to degrees
    alpha = alpha*(180/np.pi)
    
    # use full set:
    full_df = pd.read_csv("/vols/lz/twatson/ANN/NR-ANN/ANN-code/logs/features_CF4_3_raw.csv", delimiter="\t")
    # Extract energy from cam_path
    full_df["energy"] = full_df["cam_path"].apply(
            lambda x: float(re.search(r'/([\d.]+)keV', x).group(1)) if re.search(r'/([\d.]+)keV', x) else None
        )
    full_df["species"] = full_df["cam_path"].apply(lambda x: 1 if "F" in os.path.basename(x) else 0)


    fig, axs = plt.subplots(1,2,figsize=(15, 6))
    from matplotlib import colors
    ytick_vals = [10, 20, 30, 40, 60, 100, 200, 300, 400, 600]
    axs[0].hist2d(
        full_df[full_df["species"] == 0]["recoil_angle_3d_2"], 
        full_df[full_df["species"] == 0]["energy"], 
        bins=100, 
        density=True, 
        cmap='viridis', 
        norm=colors.LogNorm()
    )
    axs[0].set_title("Carbon")
    axs[1].hist2d(
        full_df[full_df["species"] == 1]["recoil_angle_3d_2"], 
        full_df[full_df["species"] == 1]["energy"], 
        bins=100, 
        density=True, 
        cmap='viridis', 
        norm=colors.LogNorm()
    )
    axs[1].set_title("Fluorine")
    axs[0].set_xlabel("Scattering Angle (degrees)")
    axs[0].set_ylabel("Energy (keV)")
    axs[1].set_xlabel("Scattering Angle (degrees)")
    axs[1].set_ylabel("Energy (keV)")
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    axs[0].set_yticks(ytick_vals, [str(y) for y in ytick_vals])  # Set ticks and labels
    axs[1].set_yticks(ytick_vals, [str(y) for y in ytick_vals])  # Set ticks and labels
    axs[0].set_xlim(0, 90)
    axs[1].set_xlim(0, 90)
    axs[0].set_ylim(10, 707)
    axs[1].set_ylim(10, 473)
    
    axs[0].plot(alpha, energies_C, label="Carbon", color="black",lw=3)
    axs[0].plot(alpha, energies_F, label="Fluorine", color="red",lw=3)
    axs[1].plot(alpha, energies_C, label="Carbon", color="black",lw=3)
    axs[1].plot(alpha, energies_F, label="Fluorine", color="red",lw=3)
   
    # axs[0].legend(loc="center left")
    # axs[1].legend(loc="center left")

    # fig.suptitle("Energy vs. Scattering Angle",fontsize=22)
    axs[0].grid(False)
    axs[1].grid(False)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=True, fontsize=16, bbox_to_anchor=(0.5, 0.95)).get_frame().update(dict(facecolor='white', edgecolor='grey', alpha=1))
    # plt.xlim(0, 90)
    # plt.ylim(0, 707)
    if save:
        plt.savefig("scattering_angle_logged.png",dpi=300)
    plt.show()




if double_roc:
    if not backdoor:
        print("WARNING: need to enable backdoor for LENRI results to plot double roc")
    # Extract true labels and predicted probabilities
    y_true = np.array([i[1] for i in data])
    y_scores = np.array([i[3] for i in data])

    # Compute ROC curve and ROC area for the model
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    # Get point corresponding to threshold ~ 0.5
    idx_thr_05 = np.argmin(np.abs(thresholds - 0.5))
    fpr_05 = fpr[idx_thr_05]
    tpr_05 = tpr[idx_thr_05]

    
    # compute corresponding ROC curve for info from df
    # Extract true labels and predicted probabilities from df
    y_true_lenri = df["species"].values
    y_scores_lenri = df["adjusted_preds" if use_bayesian else "prediction"].values
    # Compute ROC curve and ROC area for the model
    fpr_df, tpr_df, thresholds_df = roc_curve(y_true_lenri, y_scores_lenri)
    roc_auc_df = auc(fpr_df, tpr_df)
    colors = {"acc": "white", "LENRI": "#FF4500", "CoNNCR": "#3383FF"}
    idx_thr_05_df = np.argmin(np.abs(thresholds_df - 0.5))
    fpr_05_df = fpr_df[idx_thr_05_df]
    tpr_05_df = tpr_df[idx_thr_05_df]


    # Plot ROC curve
    plt.figure(figsize=(8, 6))
        # Add red markers at threshold=0.5
    plt.plot(fpr_05, tpr_05, 'o', label='Threshold = 0.5',color="black",zorder=3)
    plt.plot(fpr_05_df, tpr_05_df, 'o',color="black",zorder=3)



    plt.plot(fpr, tpr, color=colors["CoNNCR"], lw=2, label=f"CoNNCR (AUC = {roc_auc:.2f})")
    plt.plot(fpr_df, tpr_df, color=colors["LENRI"], lw=2, label=f"LENRI (AUC = {roc_auc_df:.2f})")

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Bayesian Adjusted)" if use_bayesian else "ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save:
        plt.savefig("double_ROC_bayes.png" if use_bayesian else "double_ROC.png", dpi=300)
    
    plt.show()







if intensity_v_energy:
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # plot full_df["sum_intensity_cam"] against full_df["energy"] on the left with a line of best fit
    axs[0].scatter(full_df["energy"], full_df["sum_intensity_cam"]/(0.5e6*5.5), s=1, alpha=0.5, color="tab:blue",label="Events")
    axs[0].set_ylabel("Intensity (a.u.)")
    fig.suptitle("Intensity vs. Energy",fontsize=22,bbox=dict(facecolor='white', alpha=0, edgecolor='grey', boxstyle='round,pad=0.2'),y=1.02)
    axs[0].set_title("Full Test Set")
    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(0, 707)
    axs[0].grid(True)
    axs[0].set_xlabel("Energy (keV)")
    z = np.polyfit(full_df["energy"], full_df["sum_intensity_cam"], 1)
    p = np.poly1d(z)
    axs[0].plot(full_df["energy"], p(full_df["energy"])/(0.5e6*5.5), color="red", lw=2,label="Linear Best Fit")
    axs[0].legend(markerscale = 5,fontsize="medium")




    # plot full_df["sum_intensity_cam"] against full_df["energy"] on the right with horizontal lines at F_low_threshold_int and F_high_threshold_int and vertical lines at E=170 and E=473
    axs[1].scatter(df["energy"], df["sum_intensity_cam"]/(0.5e6*5.5), s=1, alpha=0.5, color="tab:blue",label="Events")
    axs[1].set_ylabel("Intensity (a.u.)")
    axs[1].set_xlabel("Energy (keV)")
    axs[1].set_title("Pruned Test Set")






    # axs[1].set_ylim(0, 1.2)
    axs[1].set_xlim(100, 707)
    axs[1].grid(True)
    # plot horizontal lines at F_low_threshold_int and F_high_threshold_int
    axs[1].axhline(y=F_low_threshold_int/(0.5e6*5.5), color='black', linestyle=':', lw=2,label="Intensity Cuts")
    axs[1].axhline(y=F_high_threshold_int/(0.5e6*5.5), color='black', linestyle=':', lw=2)
    # plot vertical lines at E=170 and E=473
    axs[1].axvline(x=170, color='black', linestyle='-.', lw=2,label="Energy Cuts")
    axs[1].axvline(x=473, color='black', linestyle='-.', lw=2)
    # axs[1].text(451.5, 0.35, "E = 473 keV", rotation=90, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.1'))
    # axs[1].text(151.5, 0.78, "E = 170 keV", rotation=90, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.1'))
    # plot vertical lines at F_low_threshold_int and F_high_threshold_int
    # axs[1].text(0.5, F_low_threshold_int, "F_low_threshold_int", rotation=0, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.1'))
    # axs[1].text(0.5, F_high_threshold_int, "F_high_threshold_int", rotation=0, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.1'))

    axs[1].legend(markerscale = 5,fontsize="medium")
    if save:
        plt.savefig("intensity_v_energy.png",dpi=300)




if lay_summary_example_recoils:
    frame_size = 600

    C = np.load("/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF40/C/227.362keV_0.000_0.000_C_1.843cm_8876_im.npy")
    F = np.load("/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF40/F/246.796keV_0.000_0.000_F_2.093cm_9835_im.npy")
    Ar = np.load("/vols/lz/tmarley/GEM_ITO/run/im_Ar_CF40/Ar/231.888keV_0.000_0.000_Ar_2.299cm_341_im.npy")
    events = [C, F, Ar]
    preprocessed_events = [preprocess_file_path_unscaled(event,steps = [1,2]) for event in events]
    frame = preprocess_file_path_unscaled(np.zeros((frame_size, frame_size)),steps = [1])
    # add each event to a random place in the frame
    np.random.seed(i)
    for event in preprocessed_events:
        x = np.random.randint(0, frame_size-event.shape[0])
        y = np.random.randint(0, frame_size-event.shape[1])
        frame[x:x+event.shape[0], y:y+event.shape[1]] += event
    plt.matshow(frame)
    plt.grid(False)
    plt.axis("off")
    plt.title("Example Recoils")
    # enable frame around the image
    plt.gca().set_frame_on(True)
    if save:
        plt.savefig("example_recoils_lay_summary.png", dpi=300)
    plt.show()



if empty_frames:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax in axs:
        ax.grid(False)
        ax.set_ylim(-4.5, 1)
        ax.set_xlim(-1.5, 4)
        ax.set_ylabel("y (mm)")
        ax.set_xlabel("x (mm)")
        ax.set_aspect('equal')
    if save:
        fig.savefig("empty_frames.png", dpi=300)




if feature_maps:
    
    x = np.load("/vols/lz/twatson/ANN/final_ims/351.821keV_0.000_0.000_C_2.224cm_6515_im.npy")

    x = np.expand_dims(x,axis=0)




    preds = model.predict(x)
    top_preds = [(None, "carbon", preds[0][0]),(None, "flourine", preds[0][1])]
    print("\nTop 2 predictions:")
    for i, (imagenet_id, label, prob) in enumerate(top_preds):
        print(f"{i+1}. {label}: {prob:.4f}")



    FEATURE_LAYER = 'block1_conv1'
    def show_feature_maps(layer_name=FEATURE_LAYER):
        feature_extractor = tf.keras.Model(inputs=model.input,
                                        outputs=model.get_layer(layer_name).output)
        features = feature_extractor(x)

        num_features = features.shape[-1]
        size = features.shape[1]

        cols = 8
        rows = min(num_features // cols + 1, 8)  # Show at most 64 feature maps
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

        for i in range(rows * cols):
            if i >= num_features:
                axes[i // cols, i % cols].axis('off')
                continue
            ax = axes[i // cols, i % cols]
            ax.imshow(features[0, :, :, i], cmap='viridis')
            ax.axis('off')

        plt.suptitle(f"Feature maps from layer: {layer_name}")
        plt.tight_layout()
        plt.show()

    show_feature_maps()



