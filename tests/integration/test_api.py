"""
Integration tests for the FastAPI surface.

The TestClient is used as a context manager so FastAPI's startup event
fires — that's what loads the model from the MLflow Registry. Tests that
need real predictions will skip gracefully if no model is available
(e.g. CI without registry access).

Guideline: Implement unit, integration, and end-to-end tests.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient — runs the lifespan once for the whole file."""
    with TestClient(app) as c:
        yield c


def _real_features() -> list:
    """One real engineered feature row from X_test, or zeros as a fallback."""
    try:
        from src.features.feature_engineering import engineer_features

        df = pd.read_csv("data/processed/X_test.csv").head(1)
        return engineer_features(df).iloc[0].tolist()
    except Exception:
        return [0.0] * 34


# --- Health & Readiness -----------------------------------------------


class TestHealthEndpoints:

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_ready_with_model_loaded(self, client):
        """With model loaded via registry, /ready reports source + version."""
        response = client.get("/ready")
        if response.status_code == 503:
            pytest.skip("No model in registry (CI without MLflow access)")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True
        assert "model_source" in data

    def test_ready_without_model(self, client):
        """Manually setting model to None should make /ready return 503."""
        from src.api import app as app_module

        original_model = app_module.model
        app_module.model = None
        try:
            response = client.get("/ready")
            assert response.status_code == 503
        finally:
            app_module.model = original_model


# --- Predict ----------------------------------------------------------


class TestPredictEndpoint:

    def test_predict_happy_path(self, client):
        """Real input → 200 with a valid prediction payload."""
        response = client.post(
            "/predict",
            json={
                "features": _real_features(),
                "transaction_id": "test_predict_happy",
            },
        )
        if response.status_code != 200:
            pytest.skip(f"Model not loaded (status {response.status_code})")
        data = response.json()
        assert data["transaction_id"] == "test_predict_happy"
        assert data["prediction"] in (0, 1)
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert data["latency_ms"] >= 0

    def test_predict_latency_under_business_budget(self, client):
        """Single-row inference must stay under the 200 ms business SLA."""
        response = client.post(
            "/predict",
            json={
                "features": _real_features(),
                "transaction_id": "test_predict_latency",
            },
        )
        if response.status_code != 200:
            pytest.skip(f"Model not loaded (status {response.status_code})")
        assert response.json()["latency_ms"] < 200

    def test_predict_invalid_features(self, client):
        """Wrong feature count → 400/500/503 depending on model state."""
        response = client.post(
            "/predict",
            json={
                "features": [1.0, 2.0],
                "transaction_id": "test_predict_invalid",
            },
        )
        assert response.status_code in (400, 500, 503)


# --- Feedback ---------------------------------------------------------


class TestFeedbackEndpoint:

    def test_feedback_unknown_transaction(self, client):
        response = client.post(
            "/feedback",
            json={
                "transaction_id": "definitely_does_not_exist_xyz",
                "actual_label": 0,
            },
        )
        assert response.status_code == 404

    def test_predict_then_feedback(self, client):
        """End-to-end: predict, then post feedback referencing that prediction."""
        pred = client.post(
            "/predict",
            json={
                "features": _real_features(),
                "transaction_id": "test_feedback_flow",
            },
        )
        if pred.status_code != 200:
            pytest.skip(f"Model not loaded (status {pred.status_code})")

        fb = client.post(
            "/feedback",
            json={
                "transaction_id": "test_feedback_flow",
                "actual_label": pred.json()["prediction"],
            },
        )
        assert fb.status_code == 200
        data = fb.json()
        assert data["total_feedback"] >= 1


# --- Explain ----------------------------------------------------------


class TestExplainEndpoint:

    def test_explain_unknown_transaction(self, client):
        response = client.get(
            "/explain", params={"transaction_id": "nonexistent_explain"}
        )
        assert response.status_code in (404, 503)

    def test_explain_happy_path(self, client):
        """Predict first, then ask for the SHAP explanation."""
        pred = client.post(
            "/predict",
            json={
                "features": _real_features(),
                "transaction_id": "test_explain_happy",
            },
        )
        if pred.status_code != 200:
            pytest.skip(f"Model not loaded (status {pred.status_code})")

        exp = client.get(
            "/explain",
            params={
                "transaction_id": "test_explain_happy",
                "top_k": 5,
            },
        )
        assert exp.status_code == 200
        data = exp.json()
        assert data["transaction_id"] == "test_explain_happy"
        assert isinstance(data["base_value"], float)
        contribs = data["top_contributions"]
        assert len(contribs) == 5
        for c in contribs:
            assert "feature" in c and "shap_value" in c
        # Sorted by absolute SHAP value descending
        assert abs(contribs[0]["shap_value"]) >= abs(contribs[-1]["shap_value"])


# --- Stats ------------------------------------------------------------


class TestStatsEndpoint:

    def test_stats_returns_aggregates(self, client):
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "fraud_count" in data
        assert data["total_predictions"] >= 0


# --- Prometheus metrics ----------------------------------------------


class TestMetricsEndpoint:

    def test_metrics_in_prometheus_format(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        body = response.text
        assert "# HELP" in body or "predictions_total" in body
