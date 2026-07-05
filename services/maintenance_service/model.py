import numpy as np
import joblib
from pathlib import Path
import os

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

_iso_bundle = None
_rf_bundle  = None


def _load():
    global _iso_bundle, _rf_bundle
    if _iso_bundle is None:
        _iso_bundle = joblib.load(MODEL_DIR / "maintenance_iso.joblib")
        _rf_bundle  = joblib.load(MODEL_DIR / "maintenance_rf.joblib")


FEATURES = [
    "odometer_km", "avg_speed_kmh", "max_speed_kmh",
    "engine_temp_mean_c", "engine_temp_max_c",
    "battery_voltage_mean", "battery_voltage_min",
    "vibration_mean_ms2", "vibration_max_ms2",
    "brake_pressure_mean", "hard_brakes_count",
    "idle_hours", "trips_count",
]

_RISK_THRESHOLDS = {"LOW": 30, "MEDIUM": 60, "HIGH": 80}
_RECOMMENDATIONS = {
    "LOW":      "No immediate action required. Schedule routine check in 30 days.",
    "MEDIUM":   "Monitor closely. Schedule inspection within 14 days.",
    "HIGH":     "Book workshop visit within 7 days. Component stress detected.",
    "CRITICAL": "Immediate inspection required. High failure probability — do not delay.",
}


def _risk_level(score: float) -> str:
    if score < 30:   return "LOW"
    if score < 60:   return "MEDIUM"
    if score < 80:   return "HIGH"
    return "CRITICAL"


def predict(req) -> dict:
    _load()
    x = np.array([[getattr(req, f) for f in FEATURES]])

    scaler = _iso_bundle["scaler"]
    x_s    = scaler.transform(x)

    # Anomaly score → risk (0–100)
    raw    = _iso_bundle["model"].decision_function(x_s)[0]
    # We use a fixed reference range derived from training
    risk   = float(np.clip(100 * (1 - (raw + 0.5)), 0, 100))

    # RF failure probability
    prob   = float(_rf_bundle["model"].predict_proba(x_s)[0][1])

    level  = _risk_level(risk)
    return {
        "vehicle_id":          req.vehicle_id,
        "risk_score":          round(risk, 1),
        "risk_level":          level,
        "failure_probability": round(prob, 4),
        "recommendation":      _RECOMMENDATIONS[level],
    }
