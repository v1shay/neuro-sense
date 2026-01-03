import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# Load model
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.joblib")

model = joblib.load(MODEL_PATH)

app = FastAPI(title="NeuroSense API")


# -----------------------------
# Input schema
# -----------------------------
class VoiceFeatures(BaseModel):
    features: list[float]


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health():
    return {"status": "NeuroSense running"}


@app.post("/predict")
def predict(data: VoiceFeatures):
    X = np.array(data.features).reshape(1, -1)
    prediction = int(model.predict(X)[0])
    confidence = float(model.predict_proba(X)[0].max())

    return {
        "prediction": prediction,
        "confidence": confidence
    }
