import pandas as pd
import json
import joblib
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


# -----------------------------
# Load Config
# -----------------------------
CONFIG_PATH = "config.yaml"

if not Path(CONFIG_PATH).exists():
    raise FileNotFoundError("config.yaml not found.")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["data"]["raw_path"]
CURRENT_VERSION = config["data"]["current_version"]

TRAIN_SIZE = 7000  # Requirement: first 7000 for training

MODEL_PATH = Path(config["deployment"]["model_path"])
MODEL_DIR = MODEL_PATH.parent
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_ESTIMATORS = config["model_params"]["n_estimators"]
MAX_DEPTH = config["model_params"]["max_depth"]
RANDOM_STATE = config["model_params"]["random_state"]

TARGET_COL = "Machine failure"
ID_COLS = ["UDI", "Product ID"]
CATEGORICAL_COLS = ["Type"]


# -----------------------------
# Load Dataset
# -----------------------------
if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")

print("Dataset loaded.")
print(f"Total samples: {len(df)}")


# -----------------------------
# Chronological Split
# -----------------------------
train_df = df.iloc[:TRAIN_SIZE]
production_df = df.iloc[TRAIN_SIZE:]

print(f"Training samples: {len(train_df)}")
print(f"Production samples (simulated): {len(production_df)}")


# -----------------------------
# Feature Engineering
# -----------------------------
y_train = train_df[TARGET_COL]
X_raw_train = train_df.drop(columns=[TARGET_COL] + ID_COLS)

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_cat = encoder.fit_transform(X_raw_train[CATEGORICAL_COLS])

encoded_cat_df = pd.DataFrame(
    encoded_cat,
    columns=encoder.get_feature_names_out(CATEGORICAL_COLS)
)

X_numeric = X_raw_train.drop(columns=CATEGORICAL_COLS).reset_index(drop=True)
X_train = pd.concat([X_numeric, encoded_cat_df], axis=1)

feature_names = list(X_train.columns)

print("Training features prepared.")



model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    random_state=RANDOM_STATE,
    n_jobs=1
)

model.fit(X_train, y_train)

print("Model training completed.")


joblib.dump(model, MODEL_PATH)
joblib.dump(encoder, MODEL_DIR / "encoder.pkl")
joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")



try:
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL
    ).decode("utf-8").strip()
except Exception:
    git_hash = "git_commit_not_available"

metadata = {
    "project_name": config["project_name"],
    "training_date": datetime.now().isoformat(),
    "dataset_version": CURRENT_VERSION,
    "model_type": config["model_params"]["algorithm"],
    "train_samples": len(train_df),
    "production_samples": len(production_df),
    "n_estimators": N_ESTIMATORS,
    "max_depth": MAX_DEPTH,
    "random_state": RANDOM_STATE,
    "n_features": len(feature_names),
    "git_commit": git_hash
}

with open(MODEL_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\nTraining completed successfully.")
print("Model saved at:", MODEL_PATH)
