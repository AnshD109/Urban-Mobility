"""
tests/test_maintenance_service.py
Run: pytest tests/test_maintenance_service.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "maintenance_service"))

import pytest
from fastapi.testclient import TestClient

VALID_PAYLOAD = {
    "vehicle_id":           "V0042",
    "odometer_km":          125000,
    "avg_speed_kmh":        52.0,
    "max_speed_kmh":        112.0,
    "engine_temp_mean_c":   94.0,
    "engine_temp_max_c":    108.0,
    "battery_voltage_mean": 13.1,
    "battery_voltage_min":  11.8,
    "vibration_mean_ms2":   0.9,
    "vibration_max_ms2":    2.1,
    "brake_pressure_mean":  7.8,
    "hard_brakes_count":    8,
    "idle_hours":           120.0,
    "trips_count":          310,
}

FAULTY_PAYLOAD = {
    **VALID_PAYLOAD,
    "engine_temp_mean_c":   115.0,
    "engine_temp_max_c":    135.0,
    "battery_voltage_mean": 11.5,
    "battery_voltage_min":  9.8,
    "vibration_mean_ms2":   1.8,
    "hard_brakes_count":    20,
}


@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200


class TestMaintenanceRisk:
    def test_valid_request(self, client):
        resp = client.post("/maintenance/risk", json=VALID_PAYLOAD)
        assert resp.status_code in (200, 503)

    def test_response_schema(self, client):
        resp = client.post("/maintenance/risk", json=VALID_PAYLOAD)
        if resp.status_code == 200:
            data = resp.json()
            for key in ("risk_score", "risk_level", "failure_probability", "recommendation"):
                assert key in data

    def test_risk_score_bounds(self, client):
        resp = client.post("/maintenance/risk", json=VALID_PAYLOAD)
        if resp.status_code == 200:
            score = resp.json()["risk_score"]
            assert 0 <= score <= 100

    def test_risk_level_values(self, client):
        resp = client.post("/maintenance/risk", json=VALID_PAYLOAD)
        if resp.status_code == 200:
            assert resp.json()["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_faulty_higher_risk_than_healthy(self, client):
        r_h = client.post("/maintenance/risk", json=VALID_PAYLOAD)
        r_f = client.post("/maintenance/risk", json=FAULTY_PAYLOAD)
        if r_h.status_code == 200 and r_f.status_code == 200:
            assert r_f.json()["risk_score"] >= r_h.json()["risk_score"]

    def test_missing_field_returns_422(self, client):
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "engine_temp_mean_c"}
        assert client.post("/maintenance/risk", json=bad).status_code == 422
