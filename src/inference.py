# src/inference.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import yaml
from pathlib import Path


CONFIG_PATH = "config.yaml"

if not Path(CONFIG_PATH).exists():
    raise FileNotFoundError("config.yaml not found. Cannot start API.")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = Path(config["deployment"]["model_path"])
MODEL_DIR = MODEL_PATH.parent

ENCODER_PATH = MODEL_DIR / "encoder.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"

required_files = [MODEL_PATH, ENCODER_PATH, FEATURE_NAMES_PATH]

for file in required_files:
    if not file.exists():
        raise FileNotFoundError(f"Required artifact missing: {file}")

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

app = FastAPI(title="Predictive Maintenance API")

class Payload(BaseModel):
    data: dict

@app.post("/predict")
def predict(payload: Payload):
    try:
        data = pd.DataFrame([payload.data])

        
        if "Type" not in data.columns:
            return {"error": "Missing 'Type' column in input"}

        
        cat_encoded = encoder.transform(data[["Type"]])
        cat_encoded_df = pd.DataFrame(
            cat_encoded,
            columns=encoder.get_feature_names_out(["Type"])
        )

        
        X = pd.concat(
            [data.drop(columns=["Type"], errors="ignore").reset_index(drop=True),
             cat_encoded_df],
            axis=1
        )

    
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0

        X = X[feature_names]

        prediction = model.predict(X)[0]

        return {"prediction": int(prediction)}

    except Exception as e:
        return {"error": str(e)}
