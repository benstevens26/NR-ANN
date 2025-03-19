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
import re
from image_preprocessing import noise_adder, gaussian_smoothing
from bb_event import Event3D
from feature_extraction import preprocess_3d, extract_R, extract_axis_3d, extract_recoil_angle_3d, extract_intensity_profile_3d
from feature_extraction import extract_track_area, extract_track_volume, extract_length_simple
from feature_extraction import extract_axis, extract_intensity_profile, extract_recoil_angle, compute_alpha

# Get the job number from the argument (HTCondor passes $(PROCESS))
if len(sys.argv) > 1:
    job_number = int(sys.argv[1])  # Process ID from HTCondor
else:
    raise ValueError("No job number provided. Ensure the script is run via HTCondor.")

print(f"Running job {job_number}")

# feature extraction parameters CHANGE CHANGE CHANGE DONT FORGET CHANGE !!!
name = "features3_CF4_"+str(job_number)
matched_files = "/vols/lz/bstevens/NR-ANN/ANN-code/matched_file_paths_CF4.csv"
dark_dir = "/vols/lz/MIGDAL/sim_ims/darks"
num_jobs = 200

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
dark_list_number = np.random.randint(0, 10)
m_dark = np.load(f"{dark_dir}/master_dark_1x1.npy")
example_dark_list = np.load(f"{dark_dir}/quest_std_dark_{dark_list_number}.npy")

print("---------------------------------")
print("Instantiating events, preprocessing images, and extracting features")
print("---------------------------------")

# Define columns for the features dataframe
features = [
    # From filename
    "cam_path",
    "ito_path",
    "drift_length",

    # Just from pixels/voxels
    "sum_intensity_cam",
    "sum_intensity_ito",
    "sum_intensity_R",
    "max_intensity_cam",
    "max_intensity_ito",
    "max_intensity_R",
    "track_area_cam",
    "track_area_ito",
    "track_volume_3d",

    # Axis needed
    "recoil_angle_cam",
    "recoil_angle_ito",
    "recoil_angle_3d_2",

    # dE/dx needed
    "recoil_length_cam",
    "recoil_length_ito",
    "recoil_length_3d",
    "mean_energy_deposition_cam",
    "std_energy_deposition_cam",
    "skew_energy_deposition_cam",
    "kurt_energy_deposition_cam",
    "max_energy_deposition_cam",
    "mean_energy_deposition_ito",
    "std_energy_deposition_ito",
    "skew_energy_deposition_ito",
    "kurt_energy_deposition_ito",
    "max_energy_deposition_ito",
    "mean_energy_deposition_3d",
    "std_energy_deposition_3d",
    "skew_energy_deposition_3d",
    "kurt_energy_deposition_3d",
    "max_energy_deposition_3d",
    "bragg_peak_location_3d",
]


features_dataframe = pd.DataFrame(columns=features)

for cam_path, ito_path in tqdm(file_paths, desc="Feature Extraction"): # add noise, load images, and create event objects
    cam_raw = noise_adder(np.load(cam_path), m_dark, example_dark_list)
    ito_raw = np.load(ito_path)
    filename = cam_path

    # preprocessing and extracting necessary objects
    cam_image, ito_image = preprocess_3d(cam_raw, ito_raw, pad_style='match_x_at_zero')
    R = extract_R(cam_image, ito_image, downsample=True, downsample_factor=2).astype(np.float32)
    
    axis_3d, centroid_3d = extract_axis_3d(R)
    cam_axis, cam_centroid = extract_axis(cam_image)
    ito_axis, ito_centroid  = extract_axis(ito_image)

    try:
        distances_3d, intensities_3d = extract_intensity_profile_3d(R, principal_axis=axis_3d, centroid=centroid_3d)
        distances_cam, intensities_cam = extract_intensity_profile(cam_image, principal_axis=cam_axis, centroid=cam_centroid)
        distances_ito, intensities_ito = extract_intensity_profile(ito_image, principal_axis=ito_axis, centroid=ito_centroid)
        
        distances_3d = distances_3d[np.nonzero(intensities_3d)] # remove zeros as they affect stats
        intensities_3d = intensities_3d[np.nonzero(intensities_3d)]
        distances_cam = distances_cam[np.nonzero(intensities_cam)]
        intensities_cam = intensities_cam[np.nonzero(intensities_cam)]
        distances_ito = distances_ito[np.nonzero(intensities_ito)]
        intensities_ito = intensities_ito[np.nonzero(intensities_ito)]
    except:
        distances_3d, intensities_3d = None, None
        distances_cam, intensities_cam = None, None
        distances_ito, intensities_ito = None, None
    
    # from filename
    match = re.search(r'_(\d+\.\d+)cm_', filename)
    if match:
        drift_length = float(match.group(1))

    # from pixels/voxels    
    sum_intensity_cam = np.sum(cam_raw) 
    sum_intensity_ito = np.sum(ito_raw)
    sum_intensity_R = np.sum(R)

    max_intensity_cam = np.max(cam_image)
    max_intensity_ito = np.max(ito_image)
    max_intensity_R = np.max(R)

    try:
        track_area_cam = extract_track_area(cam_image)
        track_area_ito = extract_track_area(ito_image)
        track_volume_3d = extract_track_volume(R, downsample_factor=2)
    except:
        track_area_cam = np.nan
        track_area_ito = np.nan
        track_volume_3d = np.nan    

    # principal axis needed
    try:
        recoil_angle_cam = extract_recoil_angle(cam_axis)
        recoil_angle_ito = extract_recoil_angle(ito_axis)
        recoil_angle_3d_2 = compute_alpha(recoil_angle_cam, recoil_angle_ito)
    except:
        recoil_angle_cam = np.nan
        recoil_angle_ito = np.nan
        recoil_angle_3d_2 = np.nan

    # dE/dx needed
    try:
        recoil_length_3d = extract_length_simple(distances_3d, intensities_3d)
        recoil_length_cam = extract_length_simple(distances_cam, intensities_cam)
        recoil_length_ito = extract_length_simple(distances_ito, intensities_ito)
    except:
        recoil_length_3d = np.nan
        recoil_length_cam = np.nan
        recoil_length_ito = np.nan

    try:
        mean_energy_deposition_cam = np.mean(intensities_cam)
        std_energy_deposition_cam = np.std(intensities_cam)
        skew_energy_deposition_cam = sp.stats.skew(intensities_cam)
        kurt_energy_deposition_cam = sp.stats.kurtosis(intensities_cam)
        max_energy_deposition_cam = np.max(intensities_cam)

        mean_energy_deposition_ito = np.mean(intensities_ito)
        std_energy_deposition_ito = np.std(intensities_ito)
        skew_energy_deposition_ito = sp.stats.skew(intensities_ito)
        kurt_energy_deposition_ito = sp.stats.kurtosis(intensities_ito)
        max_energy_deposition_ito = np.max(intensities_ito)

        mean_energy_deposition_3d = np.mean(intensities_3d)
        std_energy_deposition_3d = np.std(intensities_3d)
        skew_energy_deposition_3d = sp.stats.skew(intensities_3d)
        kurt_energy_deposition_3d = sp.stats.kurtosis(intensities_3d)
        max_energy_deposition_3d = np.max(intensities_3d)

        bragg_peak_location_3d = np.argmax(intensities_3d)/len(intensities_3d)
    except:
        mean_energy_deposition_cam = np.nan
        std_energy_deposition_cam = np.nan
        skew_energy_deposition_cam = np.nan
        kurt_energy_deposition_cam = np.nan
        max_energy_deposition_cam = np.nan

        mean_energy_deposition_ito = np.nan
        std_energy_deposition_ito = np.nan
        skew_energy_deposition_ito = np.nan
        kurt_energy_deposition_ito = np.nan
        max_energy_deposition_ito = np.nan

        mean_energy_deposition_3d = np.nan
        std_energy_deposition_3d = np.nan
        skew_energy_deposition_3d = np.nan
        kurt_energy_deposition_3d = np.nan
        max_energy_deposition_3d = np.nan

        bragg_peak_location_3d = np.nan
        
    # Append features to dataframe
    features_dataframe = features_dataframe._append(
        {
            "cam_path": cam_path,
            "ito_path": ito_path,
            "drift_length": drift_length,
            "sum_intensity_cam": sum_intensity_cam,
            "sum_intensity_ito": sum_intensity_ito,
            "sum_intensity_R": sum_intensity_R,
            "max_intensity_cam": max_intensity_cam,
            "max_intensity_ito": max_intensity_ito,
            "max_intensity_R": max_intensity_R,
            "track_area_cam": track_area_cam,
            "track_area_ito": track_area_ito,
            "track_volume_3d": track_volume_3d,
            "recoil_angle_cam": recoil_angle_cam,
            "recoil_angle_ito": recoil_angle_ito,
            "recoil_angle_3d_2": recoil_angle_3d_2,
            "recoil_length_cam": recoil_length_cam,
            "recoil_length_ito": recoil_length_ito,
            "recoil_length_3d": recoil_length_3d,
            "mean_energy_deposition_cam": mean_energy_deposition_cam,
            "std_energy_deposition_cam": std_energy_deposition_cam,
            "skew_energy_deposition_cam": skew_energy_deposition_cam,
            "kurt_energy_deposition_cam": kurt_energy_deposition_cam,
            "max_energy_deposition_cam": max_energy_deposition_cam,
            "mean_energy_deposition_ito": mean_energy_deposition_ito,
            "std_energy_deposition_ito": std_energy_deposition_ito,
            "skew_energy_deposition_ito": skew_energy_deposition_ito,
            "kurt_energy_deposition_ito": kurt_energy_deposition_ito,
            "max_energy_deposition_ito": max_energy_deposition_ito,
            "mean_energy_deposition_3d": mean_energy_deposition_3d,
            "std_energy_deposition_3d": std_energy_deposition_3d,
            "skew_energy_deposition_3d": skew_energy_deposition_3d,
            "kurt_energy_deposition_3d": kurt_energy_deposition_3d,
            "max_energy_deposition_3d": max_energy_deposition_3d,
            "bragg_peak_location_3d": bragg_peak_location_3d,
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