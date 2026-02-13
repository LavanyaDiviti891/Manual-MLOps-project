import json
import joblib
import yaml
import os
from pathlib import Path
from datetime import datetime
import csv
CONFIG_PATH = "config.yaml"

if not Path(CONFIG_PATH).exists():
    raise FileNotFoundError("config.yaml not found.")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)



MODEL_PATH = Path(config["deployment"]["model_path"])
THRESHOLD = config["deployment"]["threshold"]

LOG_PATH = Path("logs/production_predictions.jsonl")
MONITOR_LOG = Path("logs/monitoring_log.csv")

valid_preds = []
skipped = 0

if not LOG_PATH.exists():
    raise FileNotFoundError("Production log not found.")

with open(LOG_PATH, "r") as f:
    for line in f:
        entry = json.loads(line)
        if "prediction" in entry and "actual" in entry:
            valid_preds.append(entry)
        else:
            skipped += 1


if len(valid_preds) == 0:
    print("No valid predictions found.")
else:
    y_true = [r["actual"] for r in valid_preds]
    y_pred = [r["prediction"] for r in valid_preds]

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    print(f"Production accuracy: {accuracy:.4f}")
    print(f"Threshold from config: {THRESHOLD}")
    print(f"Skipped entries: {skipped}")

    MONITOR_LOG.parent.mkdir(exist_ok=True)

    file_exists = MONITOR_LOG.exists()

    with open(MONITOR_LOG, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(["timestamp", "accuracy", "threshold", "retrain_triggered"])

        retrain_flag = accuracy < THRESHOLD
        writer.writerow([
            datetime.now(),
            round(accuracy, 4),
            THRESHOLD,
            retrain_flag
        ])


    
    if accuracy < THRESHOLD:
        print("Performance below threshold. Retraining triggered.")
        os.system("python src/train.py")
    else:
        print("Model performance is acceptable. No retraining needed.")
