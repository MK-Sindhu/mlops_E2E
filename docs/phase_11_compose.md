# Phase 11 — Split `docker-compose.yml` into 5 services

## Goal

Per the guideline (II.D Container Strategy):
> *"Use docker-compose to manage a multi-container setup: one for the API (FastAPI/Flask), one for the model server, and one for monitoring."*

The previous compose file had only 3 services (`api`, `prometheus`, `grafana`), with the model living inside the API container. Phase 9 introduced an MLflow Registry — Phase 11 makes that registry **its own service** so the topology matches the guideline literally.

## What changed

### Topology now (5 services)

```text
┌────────────────┐   http://api:8000        ┌──────────────────┐
│   streamlit    │ ─────────────────────▶   │       api        │
│   (dashboard)  │                          │     (FastAPI)    │
└────────────────┘                          └────┬─────────────┘
                                                 │ http://mlflow:5000
                                                 ▼
                                          ┌──────────────────┐
                                          │      mlflow      │  ← model server
                                          │ (tracking + reg) │     (Phase 9 lifecycle
                                          └──────────────────┘      lives here now)
                                              ▲
   ┌────────────────┐   scrape /metrics       │
   │   prometheus   │ ◀────────────────────────┘
   └─────┬──────────┘                          ┌──────────────────┐
         │ datasource                          │       api        │
         ▼                                     └──────────────────┘
   ┌────────────────┐
   │     grafana    │  http://localhost:3000  (admin/admin)
   └────────────────┘
```

| Service | Image | Port | Purpose |
|---|---|---:|---|
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.10.0` (official) | 5000 | Tracking server + Model Registry |
| `api` | built from `Dockerfile` | 8000 | FastAPI — predictions, /explain, /feedback |
| `streamlit` | built from `Dockerfile` (same image, different command) | 8501 | Dashboard |
| `prometheus` | `prom/prometheus:latest` | 9090 | Metrics scraper |
| `grafana` | `grafana/grafana:latest` | 3000 | Dashboards |

### Why one `Dockerfile` for two services
`api` and `streamlit` need **the same Python deps** (uvicorn, streamlit, requests, all the data libs). Building one image and overriding `command:` for streamlit halves build time and image storage.

### Health-check chain
Compose `depends_on` with `condition: service_healthy` means start order is enforced:
```
mlflow → api → streamlit
```
- `mlflow` ready before `api` so the registry is reachable when the API's lifespan loads the model.
- `api` ready before `streamlit` so the dashboard's first `/health` check from the sidebar succeeds.

### Volume strategy
- **Source code + configs**: baked into the image (immutable per image build).
- **Data + models + mlruns**: bind-mounted from the host.
  - `./mlruns:/mlruns` — shared between mlflow + api so existing registry versions work.
  - `./data:/app/data` — feedback DB writes back here.
  - `./models:/app/models:ro`, `./data/processed:...:ro`, `./data/baselines:...:ro` — read-only into containers.
- **Grafana state**: named volume `grafana-data` so dashboard customisation persists across `docker compose down`.

### Dockerfile slimmed
Removed `COPY data/processed/` and `COPY data/baselines/` — those are mounted as volumes now. Image is faster to build and the same image works for any data version (no rebuild when data changes).

Added `libgomp1` (system dependency for XGBoost / OpenMP).

## Environment variable interface

Both API and Streamlit are 12-factor-style configurable:

| Var | Used by | Default | Compose value |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | api | `file:./mlruns` | `http://mlflow:5000` |
| `MLFLOW_REGISTERED_NAME` | api | `fraud-detection-xgboost` | (same) |
| `MLFLOW_MODEL_STAGE` | api | `Production` | (same) |
| `MODEL_PATH` (fallback) | api | `models/best_model.joblib` | (same) |
| `API_URL` | streamlit | `http://localhost:8000` | `http://api:8000` |

This means **the same code runs locally without docker** (defaults work) **and inside docker** (env overrides work).

## Verification

```bash
# 1. Build the api/streamlit image
docker compose build

# 2. Bring the whole stack up (in detached mode)
docker compose up -d

# 3. Watch services come healthy (wait until all are "running (healthy)")
docker compose ps

# 4. Smoke-test each service's port
curl -s http://localhost:5000/health           # MLflow
curl -s http://localhost:8000/health           # API
curl -s http://localhost:8000/ready            # API model_source should report registry
curl -s -I http://localhost:8501               # Streamlit
curl -s http://localhost:9090/-/ready          # Prometheus
curl -s -I http://localhost:3000               # Grafana

# 5. Open in a browser
# - http://localhost:8501  → Streamlit dashboard
# - http://localhost:5000  → MLflow UI (Experiments + Models tabs)
# - http://localhost:9090  → Prometheus
# - http://localhost:3000  → Grafana (admin/admin)

# 6. Tear down
docker compose down            # keeps volumes
docker compose down -v         # also drops grafana-data
```

## Outputs of this phase

- [docker-compose.yml](../docker-compose.yml) — rewritten for 5 services with health-checked dependency order
- [Dockerfile](../Dockerfile) — slimmed; data/ is volume-mounted, not copied; added `libgomp1`
- [src/app/streamlit_app.py](../src/app/streamlit_app.py) — `API_URL` reads from env (`API_URL`), default unchanged
- This document
- Tag `v0.11.0-phase11` on `main`

## What's next

Phase 12 — add `node_exporter` to the compose stack so Prometheus + Grafana can show host CPU/RAM/disk metrics alongside the API's prediction-rate / drift / latency metrics.
