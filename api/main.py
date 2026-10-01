from fastapi import FastAPI
import joblib
import pandas as pd

from api.schemas import PredictionInput

app = FastAPI(
    title="Bank Marketing Prediction API",
    version="1.0.0"
)

model = joblib.load("models/pipeline.joblib")

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/predict")
def predict(data: PredictionInput):

    input_dict = data.model_dump(
        by_alias=True
    )

    df = pd.DataFrame(
        [input_dict]
    )

    probability = model.predict_proba(df)[0, 1]

    if 0.40 <= probability <= 0.60:
        return {
            "prediction": None,
            "probability": float(probability),
            "status": "abstention",
            "reason": "confidence_insufficient"
        }

    prediction = int(
        probability >= 0.5
    )

    return {
        "prediction": prediction,
        "probability": float(probability),
        "status": "ok"
    }

@app.post("/train")
def train():
    return {
        "status": "not_implemented"
    }