import pandas as pd
import json
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import subprocess

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = r"D:\manual_mlops_project\data\raw\v1_raw.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"  # NEW
METADATA_PATH = MODEL_DIR / "metadata.json"

# -----------------------------
# Columns configuration
# -----------------------------
TARGET_COL = "Machine failure"
ID_COLS = ["UDI", "Product ID"]
CATEGORICAL_COLS = ["Type"]

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("Columns in dataset:")
print(df.columns.tolist())

# -----------------------------
# Validate target column
# -----------------------------
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

# -----------------------------
# Split features / target
# -----------------------------
y = df[TARGET_COL]
X_raw = df.drop(columns=[TARGET_COL] + ID_COLS)

# -----------------------------
# Encode categorical features
# -----------------------------
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_cat = encoder.fit_transform(X_raw[CATEGORICAL_COLS])

encoded_cat_df = pd.DataFrame(
    encoded_cat,
    columns=encoder.get_feature_names_out(CATEGORICAL_COLS)
)

# Drop original categorical columns
X_numeric = X_raw.drop(columns=CATEGORICAL_COLS).reset_index(drop=True)

# Combine numeric + encoded categorical
X = pd.concat([X_numeric, encoded_cat_df], axis=1)

feature_names = list(X.columns)

print("Final feature names:")
print(feature_names)
print(f"Number of training features: {len(feature_names)}")

# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.4f}")

# -----------------------------
# Save artifacts
# -----------------------------
joblib.dump(model, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)
joblib.dump(feature_names, FEATURE_NAMES_PATH)  # NEW

# -----------------------------
# Git commit hash (if repo exists)
# -----------------------------
try:
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
    ).decode("utf-8").strip()
except Exception:
    git_hash = "not_a_git_repo"

# -----------------------------
# Save metadata
# -----------------------------
metadata = {
    "training_date": datetime.now().isoformat(),
    "dataset_version": "v1_raw.csv",
    "model_type": "RandomForestClassifier",
    "accuracy": accuracy,
    "n_features": len(feature_names),
    "feature_names": feature_names,
    "git_commit": git_hash
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print("Phase B completed successfully. Model and features saved.")
