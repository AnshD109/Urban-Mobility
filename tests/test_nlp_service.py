"""
tests/test_nlp_service.py
Run: pytest tests/test_nlp_service.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "nlp_service"))

import pytest
from fastapi.testclient import TestClient

DELAY_TEXT       = {"text": "My delivery was 45 minutes late due to heavy traffic."}
DAMAGE_TEXT      = {"text": "Package arrived completely crushed and broken."}
WRONG_ORDER_TEXT = {"text": "Received someone else's package entirely."}
CONDUCT_TEXT     = {"text": "Driver was extremely rude and argumentative at the door."}
POSITIVE_TEXT    = {"text": "Excellent service! Delivery arrived 10 minutes early."}
SHORT_TEXT       = {"text": "ok"}


@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)


class TestHealth:
    def test_health_ok(self, client):
        assert client.get("/health").status_code == 200


class TestNLPPredict:
    def test_valid_request(self, client):
        resp = client.post("/nlp/predict", json=DELAY_TEXT)
        assert resp.status_code in (200, 503)

    def test_response_schema(self, client):
        resp = client.post("/nlp/predict", json=DELAY_TEXT)
        if resp.status_code == 200:
            data = resp.json()
            assert "predicted_category" in data
            assert "confidence" in data
            assert "all_scores" in data
            assert 0 < data["confidence"] <= 1

    def test_all_scores_sum_to_one(self, client):
        resp = client.post("/nlp/predict", json=DELAY_TEXT)
        if resp.status_code == 200:
            total = sum(s["probability"] for s in resp.json()["all_scores"])
            assert abs(total - 1.0) < 0.01

    @pytest.mark.parametrize("payload,expected", [
        (DELAY_TEXT,       "delay"),
        (DAMAGE_TEXT,      "damage"),
        (WRONG_ORDER_TEXT, "wrong_order"),
        (CONDUCT_TEXT,     "driver_conduct"),
        (POSITIVE_TEXT,    "positive"),
    ])
    def test_category_predictions(self, client, payload, expected):
        resp = client.post("/nlp/predict", json=payload)
        if resp.status_code == 200:
            assert resp.json()["predicted_category"] == expected

    def test_too_short_text_returns_422(self, client):
        assert client.post("/nlp/predict", json={"text": "x"}).status_code == 422

    def test_empty_text_returns_422(self, client):
        assert client.post("/nlp/predict", json={"text": ""}).status_code == 422


class TestBatch:
    def test_batch_returns_list(self, client):
        payload = [DELAY_TEXT, POSITIVE_TEXT]
        resp = client.post("/nlp/predict/batch", json=payload)
        if resp.status_code == 200:
            assert isinstance(resp.json(), list)
            assert len(resp.json()) == 2

    def test_batch_over_200_returns_400(self, client):
        payload = [DELAY_TEXT] * 201
        assert client.post("/nlp/predict/batch", json=payload).status_code == 400
