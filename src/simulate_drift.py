import pandas as pd
import numpy as np
import os

INPUT_CSV = r"D:\manual_mlops_project\data\raw\v1_raw.csv"
OUTPUT_DIR = r"D:\manual_mlops_project\data\production"
OUTPUT_CSV = r"D:\manual_mlops_project\data\production\day2_data.csv"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# Simulate drift (example: modify numeric features)
df_day2 = df.sample(frac=0.3, random_state=42).copy()

numeric_cols = df_day2.select_dtypes(include=np.number).columns
df_day2[numeric_cols] *= 1.05  # small drift

df_day2.to_csv(OUTPUT_CSV, index=False)

print(" Day-2 drifted dataset saved at:", OUTPUT_CSV)
