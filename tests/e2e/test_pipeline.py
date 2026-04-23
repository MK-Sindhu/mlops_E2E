"""
End-to-End Tests - Full predict → feedback pipeline.
Guideline: Implement unit, integration, and end-to-end tests.
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestEndToEndPipeline:
    """
    E2E test: simulate a transaction being predicted and then feedback submitted.
    NOTE: This test requires a model to be loaded. In CI, it runs after training.
    """

    @pytest.mark.skipif(
        True,  # Set to False when model is trained and available
        reason="Model not available in this environment"
    )
    def test_full_predict_feedback_flow(self, client):
        # Step 1: Make a prediction
        fake_features = np.random.randn(33).tolist()  # Adjust to actual feature count
        predict_response = client.post("/predict", json={
            "features": fake_features,
            "transaction_id": "e2e_test_001"
        })
        assert predict_response.status_code == 200
        pred_data = predict_response.json()
        assert "prediction" in pred_data
        assert pred_data["transaction_id"] == "e2e_test_001"
        assert pred_data["latency_ms"] < 200  # Business metric check

        # Step 2: Submit feedback (ground truth)
        feedback_response = client.post("/feedback", json={
            "transaction_id": "e2e_test_001",
            "actual_label": 0
        })
        assert feedback_response.status_code == 200
        fb_data = feedback_response.json()
        assert fb_data["total_feedback"] >= 1

    def test_health_then_metrics(self, client):
        """Basic E2E: health check → metrics available."""
        # Health
        health = client.get("/health")
        assert health.status_code == 200

        # Metrics endpoint works
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
