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
model_source = "unloaded"   # human-readable: "registry: ..." or "file: ..."
model_version = None        # MLflow registry version, when loaded from registry
feature_names = None
scaler = None
prediction_counter = {"fraud": 0, "legit": 0}


def _load_model_from_registry(name: str, stage_or_version: str):
    """Load XGBClassifier from the MLflow Registry by downloading the joblib
    artifact for the relevant version's run.

    We deliberately avoid ``mlflow.xgboost.load_model()`` here — it can return
    a raw ``Booster`` which lacks ``predict_proba``, and the API needs both
    ``predict`` and ``predict_proba``. Loading the joblib artifact preserves
    the full sklearn-style ``XGBClassifier``.

    Returns:
        (model, version_number, run_id)
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    if stage_or_version.isdigit():
        v = client.get_model_version(name, stage_or_version)
    else:
        versions = client.get_latest_versions(name, stages=[stage_or_version])
        if not versions:
            raise ValueError(f"No version of '{name}' is in stage '{stage_or_version}'")
        v = versions[0]

    local_path = client.download_artifacts(v.run_id, "model_files/best_model.joblib")
    return joblib.load(local_path), v.version, v.run_id


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
    global model, model_source, model_version, feature_names, scaler

    # Initialize database
    init_db()

    # Resolve model source. Precedence:
    #   1. MLflow Registry (default) — by stage (e.g. "Production") or version
    #   2. Local joblib fallback — for offline / no-MLflow environments
    registered_name = os.getenv("MLFLOW_REGISTERED_NAME", "fraud-detection-xgboost")
    model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
    fallback_path = os.getenv("MODEL_PATH", "models/best_model.joblib")

    try:
        model, model_version, run_id = _load_model_from_registry(
            registered_name, model_stage,
        )
        model_source = f"registry: {registered_name}/{model_stage} (v{model_version}, run {run_id[:16]})"
        logger.info(f"Model loaded — {model_source}")
    except Exception as registry_err:
        logger.warning(
            f"Could not load model from MLflow Registry "
            f"({registered_name}/{model_stage}): {registry_err}. "
            f"Falling back to local file."
        )
        try:
            model = joblib.load(fallback_path)
            model_source = f"file: {fallback_path}"
            logger.info(f"Model loaded — {model_source}")
        except Exception as file_err:
            logger.error(f"Failed to load model from {fallback_path}: {file_err}")
            model_source = f"FAILED ({file_err})"

    # Feature names + scaler still live on disk (DVC-tracked).
    features_path = os.getenv("FEATURES_PATH", "models/feature_names.json")
    scaler_path = os.getenv("SCALER_PATH", "data/processed/scaler.joblib")

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
    return {
        "status": "ready",
        "model_loaded": True,
        "model_source": model_source,
        "model_version": model_version,
    }


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