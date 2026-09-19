import pytest
from fastapi.testclient import TestClient

from app.main import app


SAMPLE_TRANSACTION = {
    "TransactionID": 2987000,
    "TransactionAmt": 50.0,
    "TransactionDT": 86400,
    "card1": 13926,
    "card4": "visa",
    "card6": "credit",
    "ProductCD": "W",
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com"
}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):

    r = client.get("/health")

    assert r.status_code == 200


def test_predict_returns_200(client):

    r = client.post(
        "/predict",
        json=SAMPLE_TRANSACTION
    )

    assert r.status_code == 200


def test_predict_schema(client):

    r = client.post(
        "/predict",
        json=SAMPLE_TRANSACTION
    )

    assert r.status_code == 200

    body = r.json()

    assert "is_fraud" in body
    assert "fraud_score" in body
    assert "threshold_used" in body
    assert "risk_level" in body
    assert "latency_ms" in body


def test_predict_invalid_amount(client):

    invalid_transaction = SAMPLE_TRANSACTION.copy()

    invalid_transaction["TransactionAmt"] = -10

    r = client.post(
        "/predict",
        json=invalid_transaction
    )

    assert r.status_code == 422


def test_latency_under_200ms(client):

    r = client.post(
        "/predict",
        json=SAMPLE_TRANSACTION
    )

    assert r.status_code == 200

    body = r.json()

    assert body["latency_ms"] < 200