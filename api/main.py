from fastapi import FastAPI, HTTPException, status
import joblib
import pandas as pd
import logging
import time

from api.schemas import PredictionInput
from pathlib import Path
import csv
from datetime import datetime, timezone
import os

MONITORING_DIR = Path(os.environ.get("MONITORING_DIR", "monitoring"))

METRICS_FILE = MONITORING_DIR / "api_metrics.csv"


def write_api_metric(
    endpoint: str,
    status_code: int,
    latency_ms: float,
    probability: float | None = None,
):
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    write_header = not METRICS_FILE.exists()

    with METRICS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "endpoint",
                "status_code",
                "latency_ms",
                "probability",
                "model_version",
            ],
        )

        if write_header:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "probability": probability,
            "model_version": MODEL_VERSION,
        })

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

    try:
        df = pd.DataFrame([data.model_dump(by_alias=True)])

        probability = float(
            model.predict_proba(df)[0, 1]
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        write_api_metric(
            endpoint="/predict",
            status_code=200,
            latency_ms=latency_ms,
            probability=probability,
        )

        return {
            "probability": probability,
            "status": "ok",
            "model_version": MODEL_VERSION,
        }

    except Exception:
        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        write_api_metric(
            endpoint="/predict",
            status_code=500,
            latency_ms=latency_ms,
        )

        raise

@app.post("/train")
def train():
    logger.warning(
        "train_called | status=not_implemented"
    )

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Not implemented"
    )
