# Phase 10 — FastAPI surface + Streamlit dashboard verification

## Goal

Verify the entire HTTP API works end-to-end against the registry-loaded model from Phase 9, fill in the missing `/explain` endpoint, and confirm the Streamlit dashboard talks to the API correctly.

> Guidelines: *"Implement integration tests. Consider model explainability techniques to understand model decisions."*

## What changed

### 1. `/explain` endpoint — was a lie, now real

The API docstring and README listed `/explain` as an endpoint, but **no implementation existed** before this phase. Implemented now in [src/api/app.py](../src/api/app.py) using SHAP `TreeExplainer`:

```text
GET /explain?transaction_id=<id>&top_k=10

Response:
{
  "transaction_id":     "<id>",
  "prediction":         0 | 1,
  "fraud_probability":  0.0–1.0,
  "base_value":         <model's expected output>,
  "top_contributions": [{"feature": "V14", "shap_value": -1.42}, ...]
}
```

Implementation notes:
- **Lazy `TreeExplainer` init** — first `/explain` call constructs it (~100 ms for our model); subsequent calls reuse it. Module-level cache via `_get_explainer()`.
- **Defensive shape handling** — newer XGBoost binary classifiers return a 2-D `shap_values` array; older versions return a list-per-class. The endpoint normalises both to a 1-D row.
- **Lookup by `transaction_id`** — the SQLite store from Phase 9's `/feedback` work already persists each prediction's input features as JSON. `/explain` reads them back, so explanations are available for any past prediction without re-sending features.

### 2. Integration tests beefed up

[tests/integration/test_api.py](../tests/integration/test_api.py) went from 5 thin tests (mostly negative paths) to **12 tests** with happy-path coverage for every endpoint:

| Endpoint | Tests |
|---|---|
| `/health` | 1 (status + timestamp) |
| `/ready` | 2 (loaded + manually-unset) |
| `/predict` | 3 (happy path, latency budget, invalid input) |
| `/feedback` | 2 (404 unknown txn, predict→feedback flow) |
| `/explain` | 2 (404 unknown txn, predict→explain happy path) |
| `/stats` | 1 (aggregate fields present) |
| `/metrics` | 1 (Prometheus format) |

Two key changes that made happy-path tests possible:
- **`TestClient` used as context manager** — `with TestClient(app) as c:` triggers FastAPI's lifespan event, which loads the model from the registry. Without this, `model` stays `None` for the entire test run.
- **Real input fixture** — `_real_features()` reads one row from `data/processed/X_test.csv` and runs it through `engineer_features()`. Tests skip gracefully (with a clear reason) if the file isn't present, so CI without DVC data doesn't fail.

### 3. E2E test unblocked

[tests/e2e/test_pipeline.py](../tests/e2e/test_pipeline.py) had `@pytest.mark.skipif(True, ...)` on the meaningful E2E test — it was disabled because no model existed at the time. Now:

- Skip flag removed (replaced with a runtime check via `/ready`)
- Feature count corrected: 33 → 34 (we now have 5 engineered features)
- Test extended to cover the **full chain**: predict → explain → feedback → stats

### 4. Streamlit dashboard surfaces explanations

[src/app/streamlit_app.py](../src/app/streamlit_app.py) now calls `/explain` and renders a SHAP bar chart immediately after each successful prediction. New helper `show_explanation(txn_id)` is best-effort — silently skips if the API is unreachable.

This means the dashboard now shows **why** the model made a decision, not just **what** it decided. The bar chart of top SHAP values per prediction is a strong demo artifact for screenshots.

## Verification

```bash
# 1. Unit suite (must still pass — refactors shouldn't break anything)
pytest tests/unit/ -v
# Expect: 46 passed (no change from Phase 9)

# 2. Integration tests against the registry-loaded model
pytest tests/integration/ -v
# Expect: 12 passed when model is available; happy-path tests will skip
# if the registry has no Production version

# 3. E2E test
pytest tests/e2e/ -v
# Expect: 2 passed (full flow + health-then-metrics)

# 4. Manual: run API + Streamlit, exercise the Predict page
uvicorn src.api.app:app --port 8000 &
sleep 3
./scripts/run_streamlit.sh
# Open http://localhost:8501 → Predict → "Generate & Predict Random Sample"
# - Should see prediction + probability + latency
# - Below it: "Why this prediction (SHAP)" bar chart of top 8 contributions
```

## Outputs of this phase

- [src/api/app.py](../src/api/app.py) — `/explain` endpoint, lazy SHAP explainer, updated module docstring listing actual endpoints
- [tests/integration/test_api.py](../tests/integration/test_api.py) — rewritten, 12 tests covering all endpoints
- [tests/e2e/test_pipeline.py](../tests/e2e/test_pipeline.py) — unblocked, full predict → explain → feedback → stats flow
- [src/app/streamlit_app.py](../src/app/streamlit_app.py) — `show_explanation()` helper called after each prediction
- This document
- Tag `v0.10.0-phase10` on `main`

## What's next

Phase 11 — split [docker-compose.yml](../docker-compose.yml) so the deployment runs the API, an MLflow tracking server (with its own UI), and the monitoring stack as separate services. Sets up the topology Phase 12–14 will hang node_exporter, Grafana dashboards, and AlertManager off.
