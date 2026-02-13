import pandas as pd
import joblib
import json
import requests
import yaml
from pathlib import Path

# -----------------------------
# Load Config
# -----------------------------
CONFIG_PATH = "config.yaml"

if not Path(CONFIG_PATH).exists():
    raise FileNotFoundError("config.yaml not found.")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["data"]["raw_path"]

MODEL_DIR = Path("models")
FEATURES_PATH = MODEL_DIR / "feature_names.pkl"

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
OUTPUT_LOG = LOGS_DIR / "production_predictions.jsonl"

API_URL = "http://127.0.0.1:8000/predict"
TARGET_COL = "Machine failure"


df = pd.read_csv(DATA_PATH)
df = df.iloc[7000:]


if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

feature_names = joblib.load(FEATURES_PATH)


results = []
success_count = 0
error_count = 0

for _, row in df.iterrows():
    actual_label = row[TARGET_COL]

    # Prepare input data
    input_data = {}

    for col in feature_names:
        if col in row:
            input_data[col] = row[col]

   
    if "Type" in df.columns:
        input_data["Type"] = row["Type"]

    try:
        response = requests.post(API_URL, json={"data": input_data})
        response_json = response.json()

        prediction = response_json.get("prediction")

        if prediction is not None:
            results.append({
                "prediction": int(prediction),
                "actual": int(actual_label)
            })
            success_count += 1
        else:
            error_count += 1

    except Exception:
        error_count += 1


with open(OUTPUT_LOG, "w") as f:
    for entry in results:
        f.write(json.dumps(entry) + "\n")

print(f"Day-2 inference complete.")
print(f"Valid predictions logged: {success_count}")
print(f"Errors encountered: {error_count}")
print(f"Logs written to: {OUTPUT_LOG}")
