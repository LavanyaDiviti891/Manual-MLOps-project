import pandas as pd
import yaml
from datetime import datetime
import os
import re


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_PATH = config["data"]["raw_path"]
PROCESSED_DIR = config["data"]["processed_dir"]

os.makedirs(PROCESSED_DIR, exist_ok=True)


existing_files = os.listdir(PROCESSED_DIR)

version_numbers = []

for file in existing_files:
    match = re.match(r"v(\d+)_cleaned\.csv", file)
    if match:
        version_numbers.append(int(match.group(1)))

if version_numbers:
    next_version = max(version_numbers) + 1
else:
    next_version = 2   # v1 assumed raw

cleaned_file = f"v{next_version}_cleaned.csv"
cleaned_path = os.path.join(PROCESSED_DIR, cleaned_file)


df = pd.read_csv(RAW_PATH)


df_cleaned = df.dropna()


df_cleaned.to_csv(cleaned_path, index=False)


manifest_path = "data/manifest.txt"

with open(manifest_path, "a") as f:
    f.write(
        f"\nVersion: {cleaned_file}\n"
        f"Created_on: {datetime.now()}\n"
        f"Script: src/data_prep.py\n"
        f"Input: {RAW_PATH}\n"
        f"Description: Dropped missing values\n"
        f"{'-'*60}\n"
    )

print(f"Phase A completed. Created: {cleaned_file}")
