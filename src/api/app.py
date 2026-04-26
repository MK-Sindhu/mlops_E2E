"""
FastAPI Model Serving API
Endpoints: /health, /ready, /predict, /feedback, /explain, /stats
Now uses SQLite for persistent storage.
"""
import os
import json
import time
import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.responses import Response

from src.data.database import (
    init_db, save_prediction, get_prediction,
    save_feedback, get_model_accuracy, get_feedback_count,
    get_prediction_stats
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0",
    description="Real-time fraud detection with MLOps monitoring"
)

# ── Prometheus Metrics ───────────────────────────────────────────────
PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions", ["result"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency"
)
FRAUD_RATIO = Gauge(
    "fraud_ratio", "Rolling ratio of fraud predictions"
)
FEEDBACK_COUNT = Counter(
    "feedback_total", "Total feedback received", ["actual_label"]
)
MODEL_ACCURACY = Gauge(
    "model_real_accuracy", "Real-world accuracy from feedback"
)

# Instrument FastAPI with Prometheus
Instrumentator().instrument(app)

# ── Global State ─────────────────────────────────────────────────────
model = None
feature_names = None
scaler = None
prediction_counter = {"fraud": 0, "legit": 0}


# ── Request/Response Schemas ─────────────────────────────────────────
class PredictionRequest(BaseModel):
    features: List[float]
    transaction_id: Optional[str] = None

class PredictionResponse(BaseModel):
    transaction_id: str
    prediction: int
    fraud_probability: float
    latency_ms: float

class FeedbackRequest(BaseModel):
    transaction_id: str
    actual_label: int

class FeedbackResponse(BaseModel):
    message: str
    total_feedback: int
    current_accuracy: Optional[float] = None


# ── Startup ──────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    global model, feature_names, scaler

    # Initialize database
    init_db()

    model_path = os.getenv("MODEL_PATH", "models/best_model.joblib")
    features_path = os.getenv("FEATURES_PATH", "models/feature_names.json")
    scaler_path = os.getenv("SCALER_PATH", "data/processed/scaler.joblib")

    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

    try:
        with open(features_path, "r") as f:
            feature_names = json.load(f)
        logger.info(f"Feature names loaded: {len(feature_names)} features")
    except Exception as e:
        logger.warning(f"Could not load feature names: {e}")

    try:
        scaler = joblib.load(scaler_path)
        logger.info("Scaler loaded.")
    except Exception as e:
        logger.warning(f"Could not load scaler: {e}")


# ── Health & Readiness ───────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/ready")
def readiness_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_loaded": True}


# ── Prediction ───────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        features_array = np.array(request.features).reshape(1, -1)

        if feature_names and features_array.shape[1] != len(feature_names):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(feature_names)} features, got {features_array.shape[1]}"
            )

        df = pd.DataFrame(features_array, columns=feature_names) if feature_names else pd.DataFrame(features_array)
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    latency_ms = (time.time() - start_time) * 1000
    txn_id = request.transaction_id or f"txn_{int(time.time() * 1000)}"

    # Save to SQLite
    save_prediction(txn_id, prediction, probability, latency_ms,
                    json.dumps(request.features))

    # Update Prometheus
    label = "fraud" if prediction == 1 else "legit"
    PREDICTION_COUNT.labels(result=label).inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000)
    prediction_counter[label] = prediction_counter.get(label, 0) + 1
    total = sum(prediction_counter.values())
    if total > 0:
        FRAUD_RATIO.set(prediction_counter.get("fraud", 0) / total)

    return PredictionResponse(
        transaction_id=txn_id,
        prediction=prediction,
        fraud_probability=round(probability, 4),
        latency_ms=round(latency_ms, 2),
    )


# ── Feedback Loop ────────────────────────────────────────────────────
@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest):
    pred = get_prediction(request.transaction_id)
    if not pred:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {request.transaction_id} not found"
        )

    save_feedback(request.transaction_id, pred["prediction"], request.actual_label)
    FEEDBACK_COUNT.labels(actual_label=str(request.actual_label)).inc()

    total = get_feedback_count()
    accuracy = get_model_accuracy()
    if accuracy is not None:
        MODEL_ACCURACY.set(accuracy)

    return FeedbackResponse(
        message=f"Feedback recorded for {request.transaction_id}",
        total_feedback=total,
        current_accuracy=round(accuracy, 4) if accuracy else None,
    )


# ── Stats Endpoint ───────────────────────────────────────────────────
@app.get("/stats")
def get_stats():
    """Get prediction and feedback statistics from the database."""
    return get_prediction_stats()


# ── Prometheus Metrics ───────────────────────────────────────────────
@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)