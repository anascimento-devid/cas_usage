import tempfile

from fastapi.testclient import TestClient
import numpy as np
import tempfile
import pytest
from pathlib import Path
import json
import joblib

from api.main import app


client = TestClient(app)

@pytest.fixture(autouse=True)
def isolated_monitoring(monkeypatch):
    tmp = Path(tempfile.mkdtemp())

    from api import main

    monkeypatch.setattr(main, "METRICS_FILE", tmp / "api_metrics")


VALID_PAYLOAD = {
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191.0,
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "ok"
    assert "model_version" in result


def test_predict_returns_score(monkeypatch):
    from api import main

    monkeypatch.setattr(
        main.model,
        "predict_proba",
        lambda x: np.array([[0.20, 0.80]])
    )

    response = client.post(
        "/predict",
        json=VALID_PAYLOAD
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "ok"
    assert result["probability"] == 0.8
    assert result["model_version"] == main.MODEL_VERSION


def test_predict_low_score(monkeypatch):
    from api import main

    monkeypatch.setattr(
        main.model,
        "predict_proba",
        lambda x: np.array([[0.80, 0.20]])
    )

    response = client.post(
        "/predict",
        json=VALID_PAYLOAD
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "ok"
    assert result["probability"] == 0.2


def test_predict_mid_score_no_abstention(monkeypatch):
    from api import main

    monkeypatch.setattr(
        main.model,
        "predict_proba",
        lambda x: np.array([[0.50, 0.50]])
    )

    response = client.post(
        "/predict",
        json=VALID_PAYLOAD
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "ok"
    assert result["probability"] == 0.5


def test_predict_invalid_payload():
    invalid_payload = VALID_PAYLOAD.copy()

    del invalid_payload["euribor3m"]

    response = client.post(
        "/predict",
        json=invalid_payload
    )

    assert response.status_code == 422


def test_train_not_implemented():
    response = client.post("/train")
    assert response.status_code == 501

def test_predict_real_model_end_to_end():
    response = client.post(
        "/predict",
        json=VALID_PAYLOAD
    )

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "ok"
    assert 0.0 <= result["probability"] <= 1.0

def test_predict_rejects_invalid_category():
    invalid = {**VALID_PAYLOAD, "month": "foo"}
    response = client.post(
        "/predict",
        json=invalid
    )
    assert response.status_code == 422

def test_predict_rejects_out_of_bounds():
    invalid = {**VALID_PAYLOAD, "campaign": 0}
    response = client.post(
        "/predict",
        json=invalid
    )
    assert response.status_code == 422

def test_model_matches_metadata():
    from training.train import FEATURES
    with open("models/model_metadata.json") as f:
        meta = json.load(f)
    model = joblib.load("models/pipeline.joblib")

    assert meta["scenario"] == "S3"
    assert meta["features"] == list(model.feature_names_in_)
    assert list(model.feature_names_in_) == FEATURES

