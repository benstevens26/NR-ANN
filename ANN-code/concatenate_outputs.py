import pandas as pd
import glob
import os

# Define input and output directories
input_dir = 'job_outputs'
output_file = '3d_angles.csv'

# Get all CSV files in the directory
csv_files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))

# Concatenate all CSV files into one DataFrame
df_list = [pd.read_csv(file) for file in csv_files]
df_aggregated = pd.concat(df_list, ignore_index=True)

# Save to CSV
df_aggregated.to_csv(output_file, index=False)

print(f"Aggregated {len(csv_files)} files into {output_file} containing {len(df_aggregated)} entries.")

