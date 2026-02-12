import json
import joblib
import yaml
from pathlib import Path

# --------------------------------------------------
# Load config
# --------------------------------------------------
CONFIG_PATH = "config.yaml"

if not Path(CONFIG_PATH).exists():
    raise FileNotFoundError("config.yaml not found.")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# --------------------------------------------------
# Paths from config
# --------------------------------------------------
MODEL_PATH = Path(config["deployment"]["model_path"])
MODEL_DIR = MODEL_PATH.parent
LOG_PATH = Path("logs/production_predictions.jsonl")

FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"

# --------------------------------------------------
# Load feature names
# --------------------------------------------------
if not FEATURE_NAMES_PATH.exists():
    raise FileNotFoundError("feature_names.pkl not found.")

feature_names = joblib.load(FEATURE_NAMES_PATH)

# --------------------------------------------------
# Read production log
# --------------------------------------------------
valid_preds = []
skipped = 0

if not LOG_PATH.exists():
    raise FileNotFoundError("Production log not found.")

with open(LOG_PATH, "r") as f:
    for line in f:
        entry = json.loads(line)
        if "prediction" in entry:
            valid_preds.append(entry)
        else:
            skipped += 1

# --------------------------------------------------
# Calculate accuracy
# --------------------------------------------------
if len(valid_preds) == 0:
    print("No valid predictions found.")
else:
    y_true = [r["actual"] for r in valid_preds]
    y_pred = [r["prediction"] for r in valid_preds]

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    print(f"Production accuracy: {accuracy:.4f}")
    print(f"Skipped entries due to errors: {skipped}")
