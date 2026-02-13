import pandas as pd
import os


INPUT_CSV = r"D:\manual_mlops_project\data\raw\v1_raw.csv"
OUTPUT_DIR = r"D:\manual_mlops_project\data\production"
OUTPUT_CSV = r"D:\manual_mlops_project\data\production\day2_data.csv"

TRAIN_SIZE = 7000  # First 7000 for training
TOTAL_EXPECTED = 10000


os.makedirs(OUTPUT_DIR, exist_ok=True)


df = pd.read_csv(INPUT_CSV)

if len(df) != TOTAL_EXPECTED:
    print(f"Warning: Expected {TOTAL_EXPECTED} rows, found {len(df)}")

df_day2 = df.iloc[TRAIN_SIZE:].copy()

df_day2.to_csv(OUTPUT_CSV, index=False)

print("Chronological drift simulation completed.")
print(f"Training data: first {TRAIN_SIZE} rows")
print(f"Production data: last {len(df_day2)} rows")
print("Day-2 dataset saved at:", OUTPUT_CSV)
