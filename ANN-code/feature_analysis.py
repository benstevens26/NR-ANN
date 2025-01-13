import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

with open("matplotlibrc.json", "r") as file:
    custom_params = json.load(file)

plt.rcParams.update(custom_params)
model_name = "LENRI"


def feature_analysis(path):
    df = pd.read_csv(path)

    # choose actual features (so ignore name and dark frame columns)
    features = df.drop(df.columns[:2], axis=1)

    carbon = df[df["name"].str.contains("C")]
    fluorine = df[df["name"].str.contains("F")]

    for feature in features:
        plt.figure(figsize=(10, 6))

        sns.histplot(
            carbon[feature],
            label="Carbon",
            color="blue",
            alpha=0.5,
            bins=50,
            stat="density",
        )
        sns.histplot(
            fluorine[feature],
            label="Fluorine",
            color="orange",
            alpha=0.5,
            bins=50,
            stat="density",
        )

        plt.title("Comparison of " + feature + ": Carbon vs Fluorine")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend()
        # plt.xlim(0, 2)
        # plt.savefig("figures/feature-analysis/" + model_name + "/" + feature + ".png")
        plt.show()


# Load the datasets
# original = pd.read_csv('more_features_noisy.csv')  # Original dataset
# binned = pd.read_csv('all_2x2_binned_features.csv')  # Second dataset with missing chunk

# # Set 'name' column as the index for alignment
# original.set_index('name', inplace=True)
# binned.set_index('name', inplace=True)

# # Ensure columns are aligned and perform division
# result = original.divide(binned)

# # Optionally drop rows with missing values resulting from unmatched names
# result.dropna(inplace=True)

# # Reset the index if you want 'name' back as a regular column
# result.reset_index(inplace=True)

# # Save or view the result
# result.to_csv('2x2_binning_comparison.csv', index=False)
# print(result.head())


# ---------------------------


# # Merge the datasets on the 'name' column to align matching rows
# merged = original.merge(binned, on='name', suffixes=('_og', '_bin'))

# # Divide corresponding columns from df1 and df2
# # Skip 'name' column and divide only relevant numerical columns
# result = merged[['name']].copy()  # Start with 'name' column

# # List of columns to divide, without 'name' column
# columns_to_divide = ['noise_index', 'length', 'total_intensity', 'max_den', 'recoil_angle',
#                      'int_mean', 'int_skew', 'int_kurt', 'int_std']

# for col in columns_to_divide:
#     result[col] = merged[f"{col}_bin"] / merged[f"{col}_og"]

# # Save or inspect the result
# result.to_csv('2x2_binning_comparison.csv', index=False)
# print(result.head())


feature_analysis(
    "/vols/lz/twatson/ANN/NR-ANN/ANN-code/Data/30_segs_2x2_binned_features.csv"
)
