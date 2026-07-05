"""
NLP Triage Service — FastAPI
Port: 8002
Endpoint: POST /nlp/predict
"""
from fastapi import FastAPI, HTTPException
import logging, time

from schemas import NLPRequest, NLPResponse
from model import predict

logger = logging.getLogger("nlp_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="NLP Triage Service",
    description="Classify incident and customer feedback text into actionable categories.",
    version="1.0.0",
)


@app.on_event("startup")
async def startup():
    try:
        from model import _load
        _load()
        logger.info("NLP model loaded.")
    except Exception as e:
        logger.warning(f"Model pre-load skipped: {e}")


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "service": "nlp_service", "version": "1.0.0"}


@app.post("/nlp/predict", response_model=NLPResponse, tags=["prediction"])
def nlp_predict(req: NLPRequest):
    """
    Classify incident or feedback text.

    Categories:
    - **delay** (0): late delivery / traffic
    - **damage** (1): broken or damaged items
    - **wrong_order** (2): wrong or missing items
    - **driver_conduct** (3): driver behaviour issues
    - **positive** (4): compliment / praise
    """
    t0 = time.perf_counter()
    try:
        result = predict(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not found: {e}. "
                            "Run notebook 06_nlp_triage first.")
    except Exception as e:
        logger.exception("NLP error")
        raise HTTPException(status_code=500, detail=str(e))

    ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"nlp | category={result['predicted_category']} "
                f"| conf={result['confidence']} | {ms}ms")
    return NLPResponse(**result)


@app.post("/nlp/predict/batch", tags=["prediction"])
def nlp_batch(requests: list[NLPRequest]):
    """Batch predict up to 200 texts."""
    if len(requests) > 200:
        raise HTTPException(status_code=400, detail="Max 200 per batch.")
    return [predict(r) for r in requests]
