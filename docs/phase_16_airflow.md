# Phase 16 — Airflow DAGs

## Goal

Schedule the closed loop from Phase 15 + the scraper from Phase 6 so the system runs **without human intervention**:

| DAG | Schedule | What it does |
|---|---|---|
| `fraud_retraining_check` | `@daily` | **Multi-task DAG** — `check_drift` + `check_accuracy` (parallel) → `decide_retrain` (BranchPythonOperator) → either `retrain` (auto-promote to Staging) or `skip_retrain`. |
| `fraud_stats_scrape` | `@weekly` | Single BashOperator — runs `scripts/run_scrape.py` to refresh `data/external/fraud_stats.json`. |

> Guideline: *"Automate model retraining pipelines."*

## Architecture decision: single container, SQLite, fixed admin/admin

A single Airflow container running:
1. `airflow db migrate` (idempotent — safe to re-run on every boot)
2. `airflow users create --username admin --password admin ...` (`|| true` so re-runs don't fail)
3. `airflow scheduler &` (background)
4. `exec airflow webserver` (foreground; container lives as long as webserver does)

**Why this and not `airflow standalone`?** Standalone mode ignores `_AIRFLOW_WWW_USER_*` env vars and generates a random admin password every boot. For a learning project we want **fixed `admin / admin` credentials** so anyone can log in without first grepping logs.

**Why this and not the canonical 5-container stack** (postgres + redis + worker + webserver + scheduler)?
- Our DAGs are **pure scheduling** — no fan-out, no concurrency, no celery.
- That stack adds 4 services + ~1.5 GB of images for zero functional benefit at this scale.

If this ever needs to scale (parallel DAGs, distributed workers), swap to `apache/airflow:2.10.5` with `LocalExecutor + postgres + redis + celery_worker`. Until then, single-container is the right call.

**Why no DB volume?** Airflow's SQLite metadata DB lives at the default `/opt/airflow/airflow.db` *inside* the container's writable layer. DAG run history is therefore ephemeral on `docker compose down`. Acceptable: the DAGs themselves live as committed files in `./airflow/dags/`, so re-creating the container loses *runs* (which we don't care about) but keeps *DAGs* (which we do).

Earlier attempts mounted a named volume at `/opt/airflow/data` and pointed `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` at it — that fails because Docker creates the volume's mount point as `root:root`, but the airflow container runs as user `airflow` (UID 50000), so SQLite can't open the file (`unable to open database file`). The default location bypasses this entirely.

## Custom image

[docker/airflow.Dockerfile](../docker/airflow.Dockerfile) extends `apache/airflow:2.10.5-python3.10` with:
- `libgomp1` (needed by `xgboost`)
- All deps from [requirements.txt](../requirements.txt) (so DAGs can `from src.models.train import …` etc.)

The DAGs themselves use **`BashOperator`**, not `PythonOperator`. Why: keeps Airflow uncoupled from our code's exception model. A bash exit code `≠0` is a clean failure signal Airflow can retry. If the script throws, the bash dies, the task fails, the UI shows it.

## Volumes inside the Airflow container

| Mount | Purpose |
|---|---|
| `./airflow/dags:/opt/airflow/dags:ro` | DAG files |
| `airflow-data:/opt/airflow/data` | Named volume for the SQLite metadata DB so DAG run history survives restarts |
| `./scripts:/project/scripts:ro` | Our orchestration scripts |
| `./src:/project/src:ro` | Importable modules |
| `./configs:/project/configs:ro` | `config.yaml` |
| `./models:/project/models:ro` | feature_names.json + persisted model |
| `./data:/project/data` | SQLite DB (`fraud_detection.db`) — **read/write** so retrain orchestrator can persist drift reports |
| `./mlruns:${PWD}/mlruns` | Same path-matching trick as Phase 11 — meta.yaml's absolute artifact_uri values resolve identically inside Airflow |

`PYTHONPATH=/project` makes `from src.* import …` work; `cd /project &&` in each DAG's bash command keeps relative paths in scripts (`data/processed/X_train.csv` etc.) resolving correctly.

## Verification

### Step 1 — Build the Airflow image
First-time build downloads ~1 GB. Subsequent builds use layer cache.

```bash
docker compose build airflow
```

### Step 2 — Bring up the full stack (now 8 services)
```bash
docker compose up -d
sleep 60   # Airflow takes ~45–60s to initialise SQLite + start webserver
docker compose ps
```

Expect 8 services: `mlflow`, `api`, `streamlit`, `airflow`, `node-exporter`, `alertmanager`, `prometheus`, `grafana`.

### Step 3 — Open the Airflow UI
- http://localhost:8090
- Login: `admin / admin`
- You should see two DAGs in the list:
  - `fraud_retraining_check` (`@daily`)
  - `fraud_stats_scrape` (`@weekly`)
- Both should be **unpaused** (toggle on the left should be blue, not grey).

### Step 4 — Trigger a DAG manually for the demo
For the assignment writeup, you'll want a screenshot of a successful run.

In the UI: click the DAG name → top-right **▶ "Trigger DAG"** → confirm.
Or via CLI:
```bash
docker compose exec airflow airflow dags trigger fraud_retraining_check
docker compose exec airflow airflow dags trigger fraud_stats_scrape
```

Watch the **Graph** view — the `check_and_retrain` task should turn green (or red if it errors; click it to see logs).

### Step 5 — Inspect a run's logs
- UI: DAG → Grid view → click the colored square for a run → click the task → **Logs** tab.
- CLI:
```bash
docker compose exec airflow airflow tasks logs fraud_retraining_check check_and_retrain <run_id>
```

The logs should show the same output as `python scripts/check_and_retrain.py` running locally — drift detection, retrain decision, etc.

### Step 6 — Tear down
```bash
docker compose down
# To also wipe Airflow's metadata DB:
docker compose down -v
```

## Outputs of this phase

- [docker/airflow.Dockerfile](../docker/airflow.Dockerfile) — extends official image
- [airflow/dags/fraud_retraining_dag.py](../airflow/dags/fraud_retraining_dag.py) — daily DAG
- [airflow/dags/fraud_scraping_dag.py](../airflow/dags/fraud_scraping_dag.py) — weekly DAG
- [docker-compose.yml](../docker-compose.yml) — `airflow` service + `airflow-data` named volume
- This document
- Tag `v0.16.0-phase16` on `main`

## What's next

Phase 17 — Docker Swarm deployment so the API runs as multiple replicas behind a built-in load balancer. Demonstrates horizontal scalability, the last guideline checkpoint before final docs.
