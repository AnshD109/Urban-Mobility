"""
ETA & Pricing Service — FastAPI
Port: 8000
Endpoints: POST /quote, GET /health, GET /docs
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging
import time

from schemas import QuoteRequest, QuoteResponse
from model import predict

logger = logging.getLogger("eta_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="ETA & Pricing Service",
    description="Predict delivery ETA with confidence bounds and dynamic surge pricing.",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    logger.info("ETA service starting — loading models...")
    try:
        from model import _load
        _load()
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.warning(f"Model pre-load failed (will retry on first request): {e}")


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "service": "eta_service", "version": "1.0.0"}


@app.post("/quote", response_model=QuoteResponse, tags=["prediction"])
def quote(req: QuoteRequest):
    """
    Return predicted ETA (p10/p50/p90) and dynamic pricing for a delivery order.

    - **eta_p50_min**: median ETA estimate (minutes)
    - **eta_p10_min**: optimistic bound (10th percentile)
    - **eta_p90_min**: pessimistic bound (90th percentile)
    - **final_price**: surge-adjusted delivery quote (USD)
    """
    t0 = time.perf_counter()
    try:
        result = predict(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not found: {e}. "
                            "Run notebooks 02-04 first to generate models.")
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"quote | eta={result['eta_p50_min']}min | "
                f"price=${result['final_price']} | {elapsed_ms}ms")

    return QuoteResponse(**result)


@app.post("/quote/batch", tags=["prediction"])
def quote_batch(requests: list[QuoteRequest]):
    """Batch endpoint — accepts up to 100 requests at once."""
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Max 100 requests per batch.")
    return [predict(r) for r in requests]
