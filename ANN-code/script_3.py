import pandas as pd
import re

# File path
file_path = "ANN-code/Data/LENRI_CF4_3/evaluation_results.csv"

# Load CSV
df = pd.read_csv(file_path)

# Extract True_Label
df['True_Label'] = df['Filename'].apply(lambda x: 0 if '_C_' in x else 1)

# Extract Pred_Label
df['Pred_Label'] = df[['Predict_C', 'Predict_F']].idxmax(axis=1).apply(lambda x: 0 if x == 'Predict_C' else 1)

# Function to extract energy from filename
def extract_energy(filename):
    match = re.search(r'/(\d+\.\d+)keV_', filename)
    if match:
        return float(match.group(1))
    return None

# Apply energy extraction
df['Energy'] = df['Filename'].apply(extract_energy)

# Separate carbon and fluorine events
df_c = df[df['True_Label'] == 0]  # All Carbon events
df_f = df[df['True_Label'] == 1]  # All Fluorine events

# Define the desired ratio (1:7.33)
max_f_count = len(df_f)
desired_c_count = int(max_f_count / 7.33)  # Max Carbon count possible for the ratio

# Ensure we don’t sample more than available
c_count = min(len(df_c), desired_c_count)  
f_count = min(len(df_f), int(c_count * 7.33))  

# Sample the required number of events
df_c_sampled = df_c.sample(n=c_count, random_state=42)
df_f_sampled = df_f.sample(n=f_count, random_state=42)

# Create the ratio-corrected DataFrame
df_ratio = pd.concat([df_c_sampled, df_f_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# Energy filtering: C > 130 keV, F > 170 keV
df_energy = df[((df['True_Label'] == 0) & (df['Energy'] > 130)) |
               ((df['True_Label'] == 1) & (df['Energy'] > 170))]

# Now apply ratio filtering AFTER energy filtering
df_c_filtered = df_energy[df_energy['True_Label'] == 0]
df_f_filtered = df_energy[df_energy['True_Label'] == 1]

max_f_count_filtered = len(df_f_filtered)
desired_c_count_filtered = int(max_f_count_filtered / 7.33)

# Ensure we don’t exceed available data
c_count_filtered = min(len(df_c_filtered), desired_c_count_filtered)
f_count_filtered = min(len(df_f_filtered), int(c_count_filtered * 7.33))

# Sample events for final dataset
df_c_final = df_c_filtered.sample(n=c_count_filtered, random_state=42)
df_f_final = df_f_filtered.sample(n=f_count_filtered, random_state=42)

# Create the final DataFrame (Energy + Ratio filter)
df_combined = pd.concat([df_c_final, df_f_final]).sample(frac=1, random_state=42).reset_index(drop=True)

print(df_ratio.shape)
print(df_energy.shape)
print(df_combined.shape)


from sklearn.metrics import accuracy_score, precision_score, recall_score

# Function to compute metrics
def evaluate_metrics(df, name):
    accuracy = accuracy_score(df['True_Label'], df['Pred_Label'])
    precision = precision_score(df['True_Label'], df['Pred_Label'], pos_label=0)  # Precision for Carbon
    recall = recall_score(df['True_Label'], df['Pred_Label'], pos_label=0)  # Recall for Carbon
    return {"Dataset": name, "Accuracy": accuracy, "Precision (C)": precision, "Recall (C)": recall}

# Compute metrics for each dataset
results = [
    evaluate_metrics(df, "All Data"),
    evaluate_metrics(df_ratio, "C:F Ratio (1:7.33)"),
    evaluate_metrics(df_energy, "Energy Filtered (C >130 keV, F >170 keV)"),
    evaluate_metrics(df_combined, "Energy + Ratio Filtered")
]

# Convert results to DataFrame for display
import pandas as pd
results_df = pd.DataFrame(results)


print(results_df)


