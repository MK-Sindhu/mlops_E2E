"""
Integration Tests - Test API endpoints.
Guideline: Implement unit, integration, and end-to-end tests.
"""
import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoints:

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_ready_without_model(self, client):
        """Without a model loaded, /ready should return 503."""
        from src.api import app as app_module
        original_model = app_module.model
        app_module.model = None

        response = client.get("/ready")
        assert response.status_code == 503

        app_module.model = original_model


class TestPredictEndpoint:

    def test_predict_invalid_features(self, client):
        """Predict with wrong number of features should fail."""
        response = client.post("/predict", json={
            "features": [1.0, 2.0],  # Too few features
            "transaction_id": "test_001"
        })
        # Should either fail with 400/500 or succeed depending on model state
        assert response.status_code in [400, 500, 503]


class TestFeedbackEndpoint:

    def test_feedback_unknown_transaction(self, client):
        """Feedback for unknown transaction should return 404."""
        response = client.post("/feedback", json={
            "transaction_id": "nonexistent_txn",
            "actual_label": 0
        })
        assert response.status_code == 404


class TestMetricsEndpoint:

    def test_metrics(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "predictions_total" in response.text or "HELP" in response.text
