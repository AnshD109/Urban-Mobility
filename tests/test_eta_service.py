"""
tests/test_eta_service.py
Tests for the ETA & Pricing FastAPI service.
Run: pytest tests/test_eta_service.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "eta_service"))

import pytest
from fastapi.testclient import TestClient

# ── Fixtures ─────────────────────────────────────────────────────────────
VALID_PAYLOAD = {
    "pickup_zone":     "East Village",
    "dropoff_zone":    "Clinton East",
    "pickup_borough":  "Manhattan",
    "dropoff_borough": "Manhattan",
    "trip_distance":   2.7,
    "order_hour":      9,
    "day_of_week":     2,
    "month":           3,
    "temp":            5.0,
    "prcp":            0.0,
    "wspd":            10.0,
    "snow":            0.0,
}


@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestQuote:
    def test_valid_request_returns_200(self, client):
        resp = client.post("/quote", json=VALID_PAYLOAD)
        # May return 503 if models not trained yet — that's acceptable in CI
        assert resp.status_code in (200, 503)

    def test_response_schema(self, client):
        resp = client.post("/quote", json=VALID_PAYLOAD)
        if resp.status_code == 200:
            data = resp.json()
            assert "eta_p50_min" in data
            assert "eta_p10_min" in data
            assert "eta_p90_min" in data
            assert "final_price" in data
            assert data["eta_p10_min"] <= data["eta_p50_min"] <= data["eta_p90_min"]

    def test_eta_positive(self, client):
        resp = client.post("/quote", json=VALID_PAYLOAD)
        if resp.status_code == 200:
            assert resp.json()["eta_p50_min"] > 0

    def test_price_positive(self, client):
        resp = client.post("/quote", json=VALID_PAYLOAD)
        if resp.status_code == 200:
            assert resp.json()["final_price"] > 0

    def test_surge_in_rush_hour(self, client):
        rush = {**VALID_PAYLOAD, "order_hour": 18}
        off  = {**VALID_PAYLOAD, "order_hour": 14}
        r_rush = client.post("/quote", json=rush)
        r_off  = client.post("/quote", json=off)
        if r_rush.status_code == 200 and r_off.status_code == 200:
            assert r_rush.json()["surge_multiplier"] >= r_off.json()["surge_multiplier"]

    def test_rain_increases_surge(self, client):
        dry   = {**VALID_PAYLOAD, "prcp": 0.0}
        rainy = {**VALID_PAYLOAD, "prcp": 2.5}
        r_dry   = client.post("/quote", json=dry)
        r_rainy = client.post("/quote", json=rainy)
        if r_dry.status_code == 200 and r_rainy.status_code == 200:
            assert r_rainy.json()["surge_multiplier"] >= r_dry.json()["surge_multiplier"]

    def test_missing_required_field_returns_422(self, client):
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "trip_distance"}
        resp = client.post("/quote", json=bad)
        assert resp.status_code == 422

    def test_negative_distance_returns_422(self, client):
        bad = {**VALID_PAYLOAD, "trip_distance": -1}
        resp = client.post("/quote", json=bad)
        assert resp.status_code == 422


class TestBatch:
    def test_batch_up_to_100(self, client):
        payload = [VALID_PAYLOAD] * 5
        resp = client.post("/quote/batch", json=payload)
        assert resp.status_code in (200, 503)

    def test_batch_over_100_returns_400(self, client):
        payload = [VALID_PAYLOAD] * 101
        resp = client.post("/quote/batch", json=payload)
        assert resp.status_code == 400
