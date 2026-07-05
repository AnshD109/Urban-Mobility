"""
Model loading and inference for the ETA + pricing service.
"""
import numpy as np
import joblib
from pathlib import Path
import os

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

_rf_bundle       = None
_quantile_bundle = None
_meta            = None
_pricing_cfg     = None


def _load():
    global _rf_bundle, _quantile_bundle, _meta, _pricing_cfg
    if _rf_bundle is None:
        _meta            = joblib.load(MODEL_DIR / "feature_pipeline.joblib")
        _rf_bundle       = joblib.load(MODEL_DIR / "eta_rf.joblib")
        _quantile_bundle = joblib.load(MODEL_DIR / "eta_quantile.joblib")
        _pricing_cfg     = joblib.load(MODEL_DIR / "pricing_config.joblib")


def _build_feature_vector(req) -> np.ndarray:
    _load()
    meta = _meta

    global_mean = meta["global_mean"]
    pu_enc_val  = meta["pu_enc"].get(req.pickup_zone,  global_mean)
    do_enc_val  = meta["do_enc"].get(req.dropoff_zone, global_mean)
    bor_enc_val = meta["borough_enc"].get(req.pickup_borough, global_mean)
    route_key   = f"{req.pickup_zone}|{req.dropoff_zone}"
    route_enc   = meta["route_enc"].get(route_key, global_mean)

    is_rainy = int(req.prcp > 0)
    is_windy = int(req.wspd > 20)

    x = np.array([[
        req.trip_distance,
        np.sin(2 * np.pi * req.order_hour / 24),
        np.cos(2 * np.pi * req.order_hour / 24),
        np.sin(2 * np.pi * req.day_of_week / 7),
        np.cos(2 * np.pi * req.day_of_week / 7),
        req.month,
        int(req.day_of_week >= 5),
        int(req.order_hour in {8, 9, 17, 18, 19}),
        req.temp,
        req.prcp,
        req.wspd,
        is_rainy,
        is_windy,
        req.prcp * req.trip_distance,
        req.wspd * req.trip_distance,
        pu_enc_val,
        do_enc_val,
        bor_enc_val,
        route_enc,
    ]])
    return x


def _surge_multiplier(req) -> float:
    cfg = _pricing_cfg["surge_config"]
    mult = 1.0
    if req.order_hour in {8, 9, 17, 18, 19}:
        mult *= cfg["rush_hour"]
    if req.prcp > 0:
        mult *= cfg["rain"]
    if req.snow > 0:
        mult *= cfg["snow"]
    if req.day_of_week >= 5 and req.order_hour >= 20:
        mult *= cfg["weekend_evening"]
    return float(np.clip(mult, cfg["surge_min"], cfg["surge_max"]))


def predict(req) -> dict:
    _load()
    x = _build_feature_vector(req)

    # Median ETA from Random Forest
    eta_p50 = float(_rf_bundle["model"].predict(x)[0])

    # Quantile bounds
    scaler  = _quantile_bundle["scaler"]
    x_s     = scaler.transform(x)
    eta_p10 = float(_quantile_bundle["models"][0.1].predict(x_s)[0])
    eta_p90 = float(_quantile_bundle["models"][0.9].predict(x_s)[0])

    # Ensure ordering
    eta_p10 = min(eta_p10, eta_p50)
    eta_p90 = max(eta_p90, eta_p50)

    # Pricing
    surge      = _surge_multiplier(req)
    base_rate  = _pricing_cfg["base_rate"]
    base_min   = _pricing_cfg["base_minimum"]
    base_price = max(base_min, req.trip_distance * base_rate)
    final      = round(base_price * surge, 2)

    return {
        "eta_p50_min":      round(eta_p50, 2),
        "eta_p10_min":      round(eta_p10, 2),
        "eta_p90_min":      round(eta_p90, 2),
        "base_price":       round(base_price, 2),
        "surge_multiplier": round(surge, 3),
        "final_price":      final,
    }
