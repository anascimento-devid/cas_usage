from fastapi.testclient import TestClient
import numpy as np
from api.main import app

client = TestClient(app)


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
    assert response.json()["status"] == "ok"


def test_predict_positive(monkeypatch):
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
    assert result["prediction"] == 1
    assert result["probability"] == 0.8


def test_predict_negative(monkeypatch):
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
    assert result["prediction"] == 0
    assert result["probability"] == 0.2


def test_predict_abstention(monkeypatch):
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

    assert result["status"] == "abstention"
    assert result["prediction"] is None
    assert result["reason"] == "confidence_insufficient"


def test_predict_invalid_payload():
    invalid_payload = VALID_PAYLOAD.copy()

    del invalid_payload["euribor3m"]

    response = client.post(
        "/predict",
        json=invalid_payload
    )

    assert response.status_code == 422