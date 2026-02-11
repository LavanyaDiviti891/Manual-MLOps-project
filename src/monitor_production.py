import pandas as pd
import json
import joblib
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
LOGS_DIR = Path("logs")
PROD_LOG = LOGS_DIR / "production_predictions.jsonl"

FEATURES_PATH = Path("models/feature_names.pkl")

# -----------------------------
# Load feature names
# -----------------------------
feature_names = joblib.load(FEATURES_PATH)

# -----------------------------
# Read predictions log
# -----------------------------
valid_preds = []
skipped = 0

with open(PROD_LOG, "r") as f:
    for line in f:
        entry = json.loads(line)
        # Check for valid prediction
        if "prediction" in entry:
            valid_preds.append(entry)
        else:
            skipped += 1
            # Optional: print skipped errors
            print(f"Skipping log entry due to error: {entry.get('error', {})}")

# -----------------------------
# Compute production accuracy
# -----------------------------
if len(valid_preds) == 0:
    print("No valid predictions found. Check API and Day-2 data!")
else:
    y_true = [r.get("actual") for r in valid_preds]
    y_pred = [r.get("prediction") for r in valid_preds]
    accuracy = sum([t == p for t, p in zip(y_true, y_pred)]) / len(y_true)
    print(f"Production accuracy: {accuracy:.4f}")
    print(f"Skipped entries due to errors: {skipped}")
