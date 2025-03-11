#!/usr/bin/env python3
"""
SSH SCRIPT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm
import os
import sys
from image_preprocessing import noise_adder, gaussian_smoothing
from bb_event import Event3D
from feature_extraction import preprocess_3d, extract_R, extract_axis_3d, extract_recoil_angle_3d


# Get the job number from the argument (HTCondor passes $(PROCESS))
if len(sys.argv) > 1:
    job_number = int(sys.argv[1])  # Process ID from HTCondor
else:
    raise ValueError("No job number provided. Ensure the script is run via HTCondor.")

print(f"Running job {job_number}")

# feature extraction parameters CHANGE CHANGE CHANGE DONT FORGET CHANGE !!!
name = "recoil_angles_CF4_"+str(job_number)
matched_files = "/vols/lz/bstevens/NR-ANN/ANN-code/matched_file_paths_CF4.csv"
dark_dir = "/vols/lz/MIGDAL/sim_ims/darks"
num_jobs = 500

# Load matched file paths
df_matched = pd.read_csv(matched_files)

# Split files among jobs
total_files = len(df_matched)
files_per_job = total_files // num_jobs
start_idx = job_number * files_per_job
end_idx = total_files if job_number == num_jobs - 1 else (job_number + 1) * files_per_job

df_job = df_matched.iloc[start_idx:end_idx].reset_index(drop=True)

print(f"Processing {len(df_job)} files in job {job_number}")

# Make event objects with images preprocessed

file_paths = np.array(df_job)
events = []
dark_list_number = np.random.randint(0, 10)
m_dark = np.load(f"{dark_dir}/master_dark_1x1.npy")
example_dark_list = np.load(f"{dark_dir}/quest_std_dark_{dark_list_number}.npy")

print("---------------------------------")
print("Instantiating events and preprocessing images")
print("---------------------------------")


for cam_path, ito_path in file_paths:
    cam_image = noise_adder(np.load(cam_path), m_dark, example_dark_list)
    ito_image = np.load(ito_path)
    events.append(Event3D(cam_path, cam_image, ito_path, ito_image))


for event in tqdm(events):
    event.cam_image, event.ito_image = preprocess_3d(event.cam_image, event.ito_image)


print("---------------------------------")
print("Preprocessing complete")
print("---------------------------------")

# Define columns for the features dataframe
features = [
    "file_name",
    "recoil_angle_3d"
]

features_dataframe = pd.DataFrame(columns=features)

print("---------------------------------")
print("Starting feature extraction")
print("---------------------------------")

for event in tqdm(events):

    try:
        R = extract_R(cam_image, ito_image)
    except:
        continue

    if np.sum(R) == 0: # weird event has sumR = 0
        continue

    try:
        axis, centroid = extract_axis_3d(R)
    except:
        continue

    try:
        recoil_angle_3d = extract_recoil_angle_3d(axis)
    except:
        continue

    filename = event.cam_path

    # Append features to dataframe
    features_dataframe = features_dataframe._append(
        {
            "file_name": filename,
            "recoil_angle_3d": recoil_angle_3d
        },
        ignore_index=True,
    )

print("---------------------------------")
print("Features extracted, saving to csv")
print("---------------------------------")

output_dir = "job_outputs"
os.makedirs(output_dir, exist_ok=True)
features_dataframe.to_csv(f"{output_dir}/features_job_{job_number}.csv", index=False)

print("---------------------------------")
print("Features saved to csv")
print("---------------------------------")