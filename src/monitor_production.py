import json
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
import csv

# -----------------------------
# Load Config
# -----------------------------
CONFIG_PATH = "config.yaml"

if not Path(CONFIG_PATH).exists():
    raise FileNotFoundError("config.yaml not found.")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

THRESHOLD = config["deployment"]["threshold"]

LOG_PATH = Path("logs/production_predictions.jsonl")
MONITOR_LOG = Path("logs/monitoring_log.csv")

# -----------------------------
# Validate Log File
# -----------------------------
if not LOG_PATH.exists():
    raise FileNotFoundError("Production log not found.")

valid_preds = []
skipped = 0

with open(LOG_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        entry = json.loads(line)

        if "prediction" in entry and "actual" in entry:
            try:
                pred = int(entry["prediction"])
                actual = int(entry["actual"])
                valid_preds.append((pred, actual))
            except Exception:
                skipped += 1
        else:
            skipped += 1


# -----------------------------
# Monitoring Logic
# -----------------------------
if len(valid_preds) == 0:
    print("No valid predictions found.")
    print(f"Skipped entries: {skipped}")
else:
    y_pred = [p for p, _ in valid_preds]
    y_true = [a for _, a in valid_preds]

    accuracy = sum(p == a for p, a in zip(y_pred, y_true)) / len(y_true)

    print(f"\nProduction accuracy: {accuracy:.4f}")
    print(f"Threshold from config: {THRESHOLD}")
    print(f"Total evaluated samples: {len(valid_preds)}")
    print(f"Skipped entries: {skipped}")

    # -----------------------------
    # Log Monitoring Results
    # -----------------------------
    MONITOR_LOG.parent.mkdir(exist_ok=True)
    file_exists = MONITOR_LOG.exists()

    retrain_flag = accuracy < THRESHOLD

    with open(MONITOR_LOG, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(["timestamp", "accuracy", "threshold", "retrain_triggered"])

        writer.writerow([
            datetime.now().isoformat(),
            round(accuracy, 4),
            THRESHOLD,
            retrain_flag
        ])

    # -----------------------------
    # Retrain Trigger
    # -----------------------------
    if retrain_flag:
        print("\nPerformance below threshold. Retraining triggered...")
        subprocess.run(["python", "src/train.py"])
    else:
        print("\nModel performance is acceptable. No retraining needed.")
