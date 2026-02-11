import pandas as pd
import joblib
import json
from pathlib import Path
import requests

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = r"D:\manual_mlops_project\data\production\day2_data.csv"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
FEATURES_PATH = MODEL_DIR / "feature_names.pkl"

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
OUTPUT_LOG = LOGS_DIR / "production_predictions.jsonl"

API_URL = "http://127.0.0.1:8000/predict"  # Make sure your API is running

# -----------------------------
# Load Day-2 data
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# Load model artifacts
# -----------------------------
feature_names = joblib.load(FEATURES_PATH)
encoder = joblib.load(ENCODER_PATH)

# -----------------------------
# Clean column names
# -----------------------------
df.columns = [c.strip() for c in df.columns]

# -----------------------------
# Prepare input
# -----------------------------
results = []

for idx, row in df.iterrows():
    payload = row.to_dict()

    # Remove unexpected columns
    input_data = {k: v for k, v in payload.items() if k in feature_names or k == "Type"}

    # Make API request
    try:
        response = requests.post(API_URL, json={"data": input_data}).json()
        prediction = response.get("prediction", None)
        if prediction is None:
            results.append({"payload": payload, "error": response})
        else:
            results.append({"payload": payload, "prediction": prediction, "actual": payload.get("Machine failure")})
    except Exception as e:
        results.append({"payload": payload, "error": str(e)})

# -----------------------------
# Save logs
# -----------------------------
with open(OUTPUT_LOG, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"Day-2 inference complete. Logs written to: {OUTPUT_LOG}")
