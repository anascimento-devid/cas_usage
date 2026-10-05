from fastapi import FastAPI
import joblib
import pandas as pd
import logging
import time

from api.schemas import PredictionInput


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("bank_marketing_api")


app = FastAPI(
    title="Bank Marketing Prediction API",
    version="1.0.0"
)

MODEL_PATH = "models/pipeline.joblib"
MODEL_VERSION = "gb_s3_v1"

model = joblib.load(MODEL_PATH)

logger.info(
    "model_loaded | version=%s | path=%s",
    MODEL_VERSION,
    MODEL_PATH
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION
    }

@app.post("/predict")
def predict(data: PredictionInput):
    start = time.perf_counter()

    input_dict = data.model_dump(by_alias=True)
    df = pd.DataFrame([input_dict])

    probability = model.predict_proba(df)[0, 1]

    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "predict | status=ok | probability=%.4f | latency_ms=%.2f | model=%s",
        probability,
        latency_ms,
        MODEL_VERSION
    )

    return {
        "probability": float(probability),
        "status": "ok",
        "model_version": MODEL_VERSION
    }

@app.post("/train")
def train():
    logger.warning(
        "train_called | status=not_implemented"
    )

    return {
        "status": "not_implemented"
    }