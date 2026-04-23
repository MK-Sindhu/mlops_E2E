"""
FastAPI Model Serving API
Endpoints: /health, /ready, /predict, /feedback, /explain
Guideline: Implement /health and /ready endpoints for automated orchestration.
Guideline: Implement feedback loop to log ground truth labels.
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
predictions_log = {}  # {transaction_id: {"prediction": int, "timestamp": str}}
feedback_log = []     # [{"transaction_id": str, "predicted": int, "actual": int}]
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
    actual_label: int  # 0 = legit, 1 = fraud

class FeedbackResponse(BaseModel):
    message: str
    total_feedback: int
    current_accuracy: Optional[float] = None


# ── Startup: Load Model ─────────────────────────────────────────────
@app.on_event("startup")
def load_model():
    global model, feature_names, scaler

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
    """Basic health check — is the service alive?"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/ready")
def readiness_check():
    """Readiness check — is the model loaded and ready to serve?"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_loaded": True}


# ── Prediction ───────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Classify a transaction as fraud (1) or legit (0)."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Convert to DataFrame with feature names
        features_array = np.array(request.features).reshape(1, -1)

        if feature_names and features_array.shape[1] != len(feature_names):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(feature_names)} features, got {features_array.shape[1]}"
            )

        df = pd.DataFrame(features_array, columns=feature_names) if feature_names else pd.DataFrame(features_array)

        # Predict
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    latency_ms = (time.time() - start_time) * 1000

    # Generate transaction ID if not provided
    txn_id = request.transaction_id or f"txn_{int(time.time() * 1000)}"

    # Log prediction for feedback loop
    predictions_log[txn_id] = {
        "prediction": prediction,
        "probability": probability,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Update Prometheus metrics
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
    """
    Submit ground truth label for a past prediction.
    Used to track real-world model performance decay.
    """
    if request.transaction_id not in predictions_log:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {request.transaction_id} not found in predictions log"
        )

    predicted = predictions_log[request.transaction_id]["prediction"]
    feedback_log.append({
        "transaction_id": request.transaction_id,
        "predicted": predicted,
        "actual": request.actual_label,
        "correct": predicted == request.actual_label,
        "timestamp": datetime.utcnow().isoformat(),
    })

    FEEDBACK_COUNT.labels(actual_label=str(request.actual_label)).inc()

    # Calculate rolling accuracy
    accuracy = None
    if len(feedback_log) >= 10:
        recent = feedback_log[-100:]  # Last 100 feedback entries
        accuracy = sum(1 for f in recent if f["correct"]) / len(recent)
        MODEL_ACCURACY.set(accuracy)

    return FeedbackResponse(
        message=f"Feedback recorded for {request.transaction_id}",
        total_feedback=len(feedback_log),
        current_accuracy=round(accuracy, 4) if accuracy else None,
    )


# ── Prometheus Metrics Endpoint ──────────────────────────────────────
@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type="text/plain")


# ── Run Server ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
