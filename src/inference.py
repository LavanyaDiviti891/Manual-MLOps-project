# src/inference.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# -----------------------------#
# Load model artifacts
# -----------------------------#
MODEL_PATH = "models/model.pkl"
ENCODER_PATH = "models/encoder.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"

if not all(os.path.exists(p) for p in [MODEL_PATH, ENCODER_PATH, FEATURE_NAMES_PATH]):
    raise FileNotFoundError("Model artifacts not found in 'models/'!")

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

# -----------------------------#
# FastAPI app
# -----------------------------#
app = FastAPI(title="Day-2 Inference API")

class Payload(BaseModel):
    data: dict

@app.post("/predict")
def predict(payload: Payload):
    try:
        # Convert to DataFrame
        data = pd.DataFrame([payload.data])

        # Encode categorical feature 'Type'
        if "Type" not in data.columns:
            return {"error": "Missing 'Type' column in input data"}

        cat_encoded = encoder.transform(data[["Type"]])
        cat_encoded_df = pd.DataFrame(
            cat_encoded,
            columns=encoder.get_feature_names_out(["Type"])
        )

        # Combine numeric + encoded
        X = pd.concat([data.drop(columns=["Type"], errors="ignore").reset_index(drop=True), cat_encoded_df], axis=1)

        # Ensure all training features are present
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0  # fill missing with 0
        X = X[feature_names]  # reorder columns

        # Predict
        prediction = model.predict(X)[0]

        return {"prediction": int(prediction)}

    except Exception as e:
        return {"error": str(e)}
