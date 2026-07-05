"""
Maintenance Risk Service — FastAPI
Port: 8001
Endpoint: POST /maintenance/risk
"""
from fastapi import FastAPI, HTTPException
import logging, time

from schemas import MaintenanceRequest, MaintenanceResponse
from model import predict

logger = logging.getLogger("maintenance_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Maintenance Risk Service",
    description="Predict vehicle failure risk using anomaly detection and calibrated scoring.",
    version="1.0.0",
)


@app.on_event("startup")
async def startup():
    try:
        from model import _load
        _load()
        logger.info("Maintenance models loaded.")
    except Exception as e:
        logger.warning(f"Model pre-load skipped: {e}")


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "service": "maintenance_service", "version": "1.0.0"}


@app.post("/maintenance/risk", response_model=MaintenanceResponse, tags=["prediction"])
def maintenance_risk(req: MaintenanceRequest):
    """
    Assess vehicle maintenance risk.

    Returns:
    - **risk_score**: 0–100 (higher = more at risk)
    - **risk_level**: LOW / MEDIUM / HIGH / CRITICAL
    - **failure_probability**: calibrated probability from RF classifier
    - **recommendation**: actionable maintenance advice
    """
    t0 = time.perf_counter()
    try:
        result = predict(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not found: {e}. "
                            "Run notebook 05_maintenance_model first.")
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))

    ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"risk | vehicle={req.vehicle_id} | score={result['risk_score']} "
                f"| level={result['risk_level']} | {ms}ms")
    return MaintenanceResponse(**result)
