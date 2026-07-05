import re
import joblib
import numpy as np
from pathlib import Path
import os

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_DIR / "nlp_pipeline.joblib")


def _clean(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def predict(req) -> dict:
    _load()
    pipeline    = _bundle["pipeline"]
    label_names = _bundle["label_names"]

    clean = _clean(req.text)
    probs = pipeline.predict_proba([clean])[0]
    label = int(np.argmax(probs))

    return {
        "predicted_category": label_names[label],
        "predicted_label":    label,
        "confidence":         round(float(probs[label]), 4),
        "all_scores": [
            {"category": name, "probability": round(float(p), 4)}
            for name, p in zip(label_names, probs)
        ],
    }
