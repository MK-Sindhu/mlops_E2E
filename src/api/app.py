"""
FastAPI Model Serving API.

Endpoints:
    GET  /health     - Liveness probe.
    GET  /ready      - Readiness probe; reports model_source + version.
    POST /predict    - Real-time fraud prediction.
    POST /feedback   - Submit ground-truth label (feedback loop).
    GET  /explain    - SHAP-based feature contributions for a past prediction.
    GET  /stats      - Aggregate prediction + feedback stats.
    GET  /metrics    - Prometheus exposition.

Model is loaded from the MLflow Registry at startup, with a local-file
fallback. Feedback + predictions persist in SQLite for the feedback loop.
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from starlette.responses import Response

from src.data.database import (
    get_feedback_count,
    get_model_accuracy,
    get_prediction,
    get_prediction_stats,
    init_db,
    save_feedback,
    save_prediction,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Global State ─────────────────────────────────────────────────────
model = None
model_source = "unloaded"   # "registry: ..." or "file: ..."
model_version = None        # MLflow registry version, when loaded from registry
feature_names = None
scaler = None
explainer = None            # SHAP TreeExplainer, lazy-init on first /explain
prediction_counter = {"fraud": 0, "legit": 0}


def _get_explainer():
    """Lazy-init SHAP TreeExplainer. Cached for the process lifetime."""
    global explainer
    if explainer is None and model is not None:
        import shap
        explainer = shap.TreeExplainer(model)
    return explainer


def _load_model_from_registry(name: str, stage_or_version: str):
    """Load XGBClassifier from the MLflow Registry by downloading the joblib
    artifact for the relevant version's run.

    Resolution order:
        1. Explicit version string ("7" → version 7).
        2. Alias ("Production" → @production).
        3. Stage (legacy "Production" stage) — fallback for older registries
           that didn't get an alias set.

    We deliberately avoid ``mlflow.xgboost.load_model()`` here — it can return
    a raw ``Booster`` which lacks ``predict_proba``, and the API needs both
    ``predict`` and ``predict_proba``. The joblib artifact preserves the full
    sklearn-style ``XGBClassifier``.

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
        # Try alias first (modern), fall back to legacy stage lookup
        try:
            v = client.get_model_version_by_alias(name, alias=stage_or_version.lower())
        except Exception:
            versions = client.get_latest_versions(name, stages=[stage_or_version])
            if not versions:
                raise ValueError(
                    f"No version of '{name}' has alias '{stage_or_version.lower()}' "
                    f"or stage '{stage_or_version}'"
                )
            v = versions[0]

    local_path = client.download_artifacts(v.run_id, "model_files/best_model.joblib")
    return joblib.load(local_path), v.version, v.run_id


# ── App lifespan (startup + shutdown hooks) ──────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Replaces the deprecated ``@app.on_event('startup')`` handler.

    Startup:
        - init SQLite for the feedback loop
        - load model from MLflow Registry, with local-file fallback
        - load feature names + scaler from disk
    Shutdown:
        - no-op (DB connections close per request)
    """
    global model, model_source, model_version, feature_names, scaler

    init_db()

    registered_name = os.getenv("MLFLOW_REGISTERED_NAME", "fraud-detection-xgboost")
    model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
    fallback_path = os.getenv("MODEL_PATH", "models/best_model.joblib")

    try:
        model, model_version, run_id = _load_model_from_registry(
            registered_name, model_stage,
        )
        model_source = (
            f"registry: {registered_name}/{model_stage} "
            f"(v{model_version}, run {run_id[:16]})"
        )
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

    yield   # ← app is running

    # Shutdown — currently nothing to clean up


# ── App Setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0",
    description="Real-time fraud detection with MLOps monitoring",
    lifespan=lifespan,
)


# ── Prometheus Metrics ───────────────────────────────────────────────
PREDICTION_COUNT = Counter("predictions_total", "Total predictions", ["result"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
FRAUD_RATIO = Gauge("fraud_ratio", "Rolling ratio of fraud predictions")
FEEDBACK_COUNT = Counter("feedback_total", "Total feedback received", ["actual_label"])
MODEL_ACCURACY = Gauge("model_real_accuracy", "Real-world accuracy from feedback")

Instrumentator().instrument(app)


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


# ── Health & Readiness ───────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


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
                detail=f"Expected {len(feature_names)} features, got {features_array.shape[1]}",
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

    save_prediction(txn_id, prediction, probability, latency_ms,
                    json.dumps(request.features))

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
            detail=f"Transaction {request.transaction_id} not found",
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


# ── Explainability ───────────────────────────────────────────────────
@app.get("/explain")
def explain_prediction(transaction_id: str, top_k: int = 10):
    """Return SHAP-based feature contributions for a previously-made prediction.

    Args:
        transaction_id: ID returned by an earlier /predict call.
        top_k: How many top contributions to return (sorted by |shap_value|).

    Response:
        - transaction_id, prediction, fraud_probability (echoed from DB)
        - base_value: the model's expected output (SHAP background)
        - top_contributions: [{feature, shap_value}, ...] sorted by impact
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pred = get_prediction(transaction_id)
    if not pred:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {transaction_id} not found",
        )

    try:
        features = json.loads(pred["features"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(
            status_code=500,
            detail="Stored features for this transaction are corrupt.",
        )

    cols = feature_names if feature_names else [f"f{i}" for i in range(len(features))]
    df = pd.DataFrame([features], columns=cols)

    try:
        ex = _get_explainer()
        shap_values = ex.shap_values(df)
        base_value = ex.expected_value
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP error: {e}")

    # Newer XGBoost binary classifiers return a 2-D array (n, n_features);
    # older versions returned a list per class. Normalise to a 1-D row.
    if isinstance(shap_values, list):
        shap_row = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])[0]
    else:
        arr = np.asarray(shap_values)
        shap_row = arr[0] if arr.ndim == 2 else arr

    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.asarray(base_value).flatten()[0])
    else:
        base_value = float(base_value)

    contributions = sorted(
        zip(cols, [float(v) for v in shap_row]),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )

    return {
        "transaction_id": transaction_id,
        "prediction": pred["prediction"],
        "fraud_probability": round(float(pred["fraud_probability"]), 4),
        "base_value": round(base_value, 4),
        "top_contributions": [
            {"feature": name, "shap_value": round(value, 6)}
            for name, value in contributions[: max(1, top_k)]
        ],
    }


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
