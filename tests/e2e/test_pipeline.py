"""
End-to-End Tests — full predict → explain → feedback → stats pipeline.

Skips gracefully if no model is loaded (e.g. CI without registry access).
Guideline: Implement unit, integration, and end-to-end tests.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _real_features() -> list:
    """Engineered feature row from X_test for use as request input."""
    from src.features.feature_engineering import engineer_features
    df = pd.read_csv("data/processed/X_test.csv").head(1)
    return engineer_features(df).iloc[0].tolist()


class TestEndToEndPipeline:

    def test_full_predict_explain_feedback_stats_flow(self, client):
        ready = client.get("/ready").json()
        if not ready.get("model_loaded"):
            pytest.skip("Model not loaded — skipping E2E")

        features = _real_features()

        # 1. Predict
        pred = client.post("/predict", json={
            "features": features,
            "transaction_id": "e2e_full_001",
        })
        assert pred.status_code == 200
        pred_data = pred.json()
        assert pred_data["prediction"] in (0, 1)
        assert pred_data["latency_ms"] < 200, "Inference latency exceeded business SLA"

        # 2. Explain
        exp = client.get("/explain", params={
            "transaction_id": "e2e_full_001",
            "top_k": 5,
        })
        assert exp.status_code == 200
        assert len(exp.json()["top_contributions"]) == 5

        # 3. Feedback
        fb = client.post("/feedback", json={
            "transaction_id": "e2e_full_001",
            "actual_label": pred_data["prediction"],
        })
        assert fb.status_code == 200
        assert fb.json()["total_feedback"] >= 1

        # 4. Stats reflect the new prediction
        stats = client.get("/stats")
        assert stats.status_code == 200
        assert stats.json()["total_predictions"] >= 1

    def test_health_then_metrics(self, client):
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200
