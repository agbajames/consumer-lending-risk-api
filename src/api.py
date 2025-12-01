import json
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException


from .config import PIPELINE_PATH, METADATA_PATH, DEFAULT_THRESHOLD
from .schemas import CreditApplication
from contextlib import asynccontextmanager


app = FastAPI(title="Consumer Lending Risk Demo")

_pipeline = None
_metadata: Dict[str, Any] = {}


def _payload_to_dict(app_data: CreditApplication) -> Dict[str, Any]:
    """Convert Pydantic model to dict (handles v1 and v2)."""
    if hasattr(app_data, "model_dump"):
        return app_data.model_dump()
    return app_data.dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    global _pipeline, _metadata
    try:
        _pipeline = joblib.load(PIPELINE_PATH)
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
    except Exception as exc:
        _pipeline = None
        _metadata = {"load_error": str(exc)}
    
    yield
    
    # Shutdown (if needed in future)
    pass

# Update app creation
app = FastAPI(title="Consumer Lending Risk Demo", lifespan=lifespan)


@app.get("/health")
def health():
    """Healthcheck endpoint."""
    if _pipeline is None:
        return {
            "status": "error",
            "detail": _metadata.get("load_error", "model not loaded"),
        }
    
    return {
        "status": "ok",
        "model_trained_at": _metadata.get("trained_at"),
        "model_type": _metadata.get("model_type"),
        "backend": _metadata.get("backend"),
        "roc_auc": _metadata.get("roc_auc"),
        "pr_auc": _metadata.get("pr_auc"),
        "threshold": _metadata.get("threshold", DEFAULT_THRESHOLD),
    }


@app.post("/score")
def score(app_data: CreditApplication):
    """Score a single credit application."""
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=_metadata.get("load_error", "model not loaded"),
        )
    
    features: List[str] = _metadata.get("features") or []
    if not features:
        raise HTTPException(
            status_code=500,
            detail="model metadata missing features",
        )
    
    payload = _payload_to_dict(app_data)
    
    # Validate contract
    missing = [c for c in features if c not in payload]
    extra = [c for c in payload if c not in features]
    
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"missing features: {missing}",
        )
    
    # Drop unexpected fields silently
    for c in extra:
        payload.pop(c, None)
    
    # Create DataFrame in correct column order
    X = pd.DataFrame([[payload[c] for c in features]], columns=features)
    
    # Predict
    p_vec = _pipeline.predict_proba(X)[0]
    p1 = float(p_vec[1])
    threshold = float(_metadata.get("threshold", DEFAULT_THRESHOLD))
    
    return {
        "default_probability": p1,
        "prediction": int(p1 >= threshold),
        "threshold": threshold,
        "model_info": {
            "model_type": _metadata.get("model_type"),
            "backend": _metadata.get("backend"),
            "trained_at": _metadata.get("trained_at"),
        },
    }