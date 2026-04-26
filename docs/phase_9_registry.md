# Phase 9 — MLflow Model Registry: Staging → Production lifecycle

## Goal

Take the 7 versions sitting in the registry from Phase 8 and put them through a real promotion lifecycle:

1. **Inspect** all versions and their metrics.
2. **Promote** the best version to **Staging** (selected automatically by `pr_auc`).
3. **Promote** that version to **Production** after validation.
4. **API loads from the registry** so swapping models is a metadata operation — no code, config, or container restart logic depending on file paths.
5. **Archive** older versions to keep the active set tidy.
6. **Rollback** is one CLI call away.

> Guidelines: *"Implement model versioning and management. Implement rollback mechanisms for failed deployments."*

## What changed

### 1. New CLI: [scripts/promote_model.py](../scripts/promote_model.py)

```text
python scripts/promote_model.py list                                 # all versions + metrics + stage
python scripts/promote_model.py current                              # who's in Staging / Production
python scripts/promote_model.py promote --to Staging --best-by pr_auc
python scripts/promote_model.py promote --to Production --version 7
python scripts/promote_model.py promote --to Production --version 7 --archive-existing
python scripts/promote_model.py archive --version 4
```

Notable design choices:
- **Selects best version automatically** when `--version` is omitted. Default metric is `pr_auc` (the chosen headline metric for this imbalanced problem).
- **Sets BOTH a stage transition AND an alias** (`@staging`, `@production`). MLflow stages are technically deprecated since 2.9 in favour of aliases — setting both keeps us compatible with rubric expectations *and* future MLflow versions.
- **`--archive-existing`** mirrors what production CD pipelines actually do: when a new model is promoted, the old one is archived in the same atomic operation.

### 2. API loads from the registry

[src/api/app.py](../src/api/app.py) startup now resolves the model in this order:

1. **MLflow Registry** — `models:/fraud-detection-xgboost/Production` (env-overridable via `MLFLOW_MODEL_STAGE`)
2. **Local joblib fallback** — `models/best_model.joblib` (env-overridable via `MODEL_PATH`)

Loading from the registry uses a **deliberate workaround**: instead of `mlflow.xgboost.load_model()` (which can return a raw `Booster` lacking `predict_proba`), the API downloads the `model_files/best_model.joblib` artifact for the resolved version and `joblib.load`s it. This preserves the full sklearn-style `XGBClassifier`.

### 3. `/ready` exposes the model source

```bash
$ curl -s http://localhost:8000/ready | jq
{
  "status": "ready",
  "model_loaded": true,
  "model_source": "registry: fraud-detection-xgboost/Production (v7, run b2316e8039174528)",
  "model_version": "7"
}
```

This is invaluable for ops — the deployed model's identity is observable without inspecting MLflow directly.

### 4. 8 unit tests in [tests/unit/test_promote_model.py](../tests/unit/test_promote_model.py)

Cover `list_versions`, `get_metric`, `find_best_version` with `MlflowClient` fully mocked. No registry interaction during the test run.

## Promotion lifecycle (cheat sheet)

```
┌──────────┐      ┌─────────┐      ┌────────────┐      ┌──────────┐
│   None   ├─────▶│ Staging ├─────▶│ Production ├─────▶│ Archived │
└──────────┘      └─────────┘      └────────────┘      └──────────┘
   (newly                                ▲
   trained)                              │
                                         │  (rollback = re-promote
                                         │   an Archived version
                                         │   back to Production)
```

### When to promote to Staging
- A run beats the current Staging version on `pr_auc` AND
- All unit + integration tests pass on the run's commit AND
- Inference latency is within the 200 ms budget (logged as `avg_inference_latency_ms`)

### When to promote to Production
- The Staging version has been validated against held-out data AND
- Drift detection (Phase 15) shows it doesn't regress on recent feedback AND
- Manual sign-off

### How to rollback
```bash
# Find the previous Production version
python scripts/promote_model.py list

# Re-promote it (and archive the failing one)
python scripts/promote_model.py promote --to Production --version <previous> --archive-existing
```
The API will pick up the new Production version on next startup. (For zero-downtime, the docker-compose stack in Phase 11 will rolling-restart the API container.)

## Verification (run on this branch)

```bash
# 1. Unit tests
pytest tests/unit/test_promote_model.py -v

# 2. Full suite (sanity)
pytest tests/unit/ -v

# 3. Inspect what's in the registry today
python scripts/promote_model.py list

# 4. Promote the best version to Staging
python scripts/promote_model.py promote --to Staging --best-by pr_auc

# 5. (Validation here in real life — for now: promote it to Production)
python scripts/promote_model.py promote --to Production --version <staging_version> --archive-existing

# 6. Confirm what's live
python scripts/promote_model.py current

# 7. Restart the API and verify it loads from the registry
./scripts/run_streamlit.sh   # only if you want the dashboard
# in another terminal:
uvicorn src.api.app:app --port 8000
# then:
curl -s http://localhost:8000/ready | python -m json.tool
# /ready should report: "model_source": "registry: ..."
```

## Outputs of this phase

- [scripts/promote_model.py](../scripts/promote_model.py) — promotion CLI (4 subcommands)
- [src/api/app.py](../src/api/app.py) — startup loads from registry, `/ready` reports source + version
- [tests/unit/test_promote_model.py](../tests/unit/test_promote_model.py) — 8 unit tests with mocked MlflowClient
- This document
- Tag `v0.9.0-phase9` on `main`

## What's next

Phase 10 — verify FastAPI's full endpoint surface (`/health`, `/ready`, `/predict`, `/feedback`, `/explain`, `/stats`) end-to-end with the registry-loaded model, and confirm the Streamlit dashboard works against the running API.
