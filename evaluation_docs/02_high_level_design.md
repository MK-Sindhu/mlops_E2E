# High-Level Design (HLD) — Credit Card Fraud Detection

> **Rubric line** — "High-level design document narrating the design choices and rationale."

| Field | Value |
|---|---|
| Project | Credit Card Fraud Detection — End-to-End MLOps |
| Authors | MK Sindhu et al. |
| Version | 1.0.0 |
| Last reviewed | 2026-04-28 |
| Status | Final for evaluation |

---

## 1. Purpose & scope

The system detects fraudulent credit-card transactions in real time and
keeps the underlying ML model fresh as data distributions evolve. The
**MLOps story is the deliverable** — a single, accurate model would be
trivial; this project demonstrates the platform around it: ingestion,
validation, training, registry-backed serving, observability, drift
detection, retraining, alerting, and reproducible deployment.

In scope:

- Online inference over a REST API.
- Data validation, preprocessing, feature engineering, training, evaluation
  as a versioned pipeline.
- Experiment tracking + model registry with stage transitions.
- Container-orchestrated monitoring + alerting.
- Scheduled drift-triggered retraining with auto-promotion.
- Compose + Swarm deployments from one source-of-truth.

Out of scope (explicitly):

- Multi-region or multi-tenant deployment.
- Public TLS termination and authentication (would be added by a reverse
  proxy / API gateway at production time — see [Security posture](#13-security-posture)).
- Streaming ingest (Kafka / Kinesis).
- Online learning / continual training.

## 2. Stakeholders

| Stakeholder | Concern | What this design gives them |
|---|---|---|
| **Fraud analyst / business user** | Catch fraud, low false-alarm noise. | UI with explanations + feedback loop. |
| **Data scientist** | Iterate on models, reproduce experiments. | DVC + MLflow tracking + Registry. |
| **MLOps / SRE** | Liveness, latency, alerting, rollback. | Prometheus + Grafana + AlertManager + Swarm rollback. |
| **Auditor / grader** | Evidence for every rubric line. | This doc + [`GUIDELINES_COMPLIANCE.md`](../../GUIDELINES_COMPLIANCE.md). |

## 3. Quality attributes & non-functional requirements

| Attribute | Target | How achieved | Evidence |
|---|---|---|---|
| **Latency (p99)** | < 200 ms | XGBoost `tree_method=hist`; pandas pipeline; SHAP lazy-loaded. | `models/metrics.json:avg_inference_latency_ms = 7.71`; latency histogram in Grafana. |
| **PR-AUC** | > 0.80 | Class-imbalance handled via `scale_pos_weight=50`; dedup + scaler-leak fix. | `metrics.json:pr_auc = 0.817`. |
| **Reproducibility** | bit-exact replay of any registered model | DVC pins data; MLflow logs git commit + env; feature baseline carries `_meta.source_data_md5`. | `dvc.lock`; per-run MLflow `git_commit` tag. |
| **Availability (single-host)** | API healthcheck-driven restart | Compose `healthcheck`; Swarm `restart_policy: any` + `update_config: rollback`. | [`docker-compose.swarm.yml`](../docker-compose.swarm.yml). |
| **Observability** | metrics + traces + logs surface failures within 1 minute | Prometheus 15s scrape; AlertManager `for: 5m` → Mailtrap. | [`docker/monitoring/prometheus/alert_rules.yml`](../docker/monitoring/prometheus/alert_rules.yml). |
| **Time-to-detect drift** | within 24 h | Daily Airflow DAG runs KS-test on engineered features. | [`airflow/dags/fraud_retraining_dag.py`](../airflow/dags/fraud_retraining_dag.py). |
| **Time-to-recover from bad model** | < 1 min | `scripts/promote_model.py` re-points `production` alias; API restart re-loads. | [`scripts/promote_model.py`](../scripts/promote_model.py). |

## 4. Data engineering

### 4.1 Sources (three of them)

| # | Source | Cadence | Owner | Purpose |
|---|---|---|---|---|
| 1 | Kaggle creditcard.csv | one-shot | Worldline / ULB | training |
| 2 | Wikipedia + Kaggle fraud-stats pages | weekly Airflow DAG | external | "Threat Landscape" UI |
| 3 | User feedback (Streamlit → /feedback) | continuous | operator | retraining decision |

### 4.2 Pipeline shape

The training pipeline is a **DVC DAG with five stages** (validate →
preprocess → feature_engineering → train → evaluate). Each stage's deps and
outs are declared in [`dvc.yaml`](../dvc.yaml); `dvc repro` recomputes only
stages whose inputs changed and updates `dvc.lock`. Throughput on the
~285 k row dataset is ≈ 90 s end-to-end on a laptop.

> **Why DVC, not "just scripts"?** Without DVC we would need bespoke
> incremental-rebuild logic. With DVC, the dependency graph itself is
> versioned and the cache deduplicates intermediate artifacts.

### 4.3 Validation

[`src/data/validate.py`](../src/data/validate.py) emits a structured
`validation_report.json` covering:

- Schema (required columns + dtypes).
- Missing-value counts per column.
- Duplicate-row count.
- Class-balance summary (pre-warns of catastrophic data shifts).

Failure of any check fails the DVC stage → the pipeline halts before bad
data reaches training.

### 4.4 Drift baselines

`feature_baselines.json` is computed once during the EDA / feature stage
and committed to Git. It contains per-feature {min, max, mean, std,
quartiles} **plus** a `_meta` block:

```json
{
  "_meta": {
    "source_data_md5": "...",
    "feature_version": "1.0.0",
    "computed_at": "2026-04-25T13:09:11Z"
  }
}
```

The drift detector refuses to compare if the runtime `FEATURE_VERSION` does
not match the baseline — preventing silent comparisons across schemas.

## 5. Pipeline orchestration & visualization

The rubric awards "ML Pipeline Visualization [4]" specifically for being
able to **see** pipeline runs, errors, and successes. We satisfy this with
a **single unified Streamlit page** plus the three native MLOps consoles
behind it:

| Layer | Tool | URL | What it shows |
|---|---|---|---|
| Unified pipeline dashboard | **Streamlit `Pipeline Status` page** | http://localhost:8501 → "Pipeline Status" | Single pane of glass — DVC stage status (from `dvc.lock`), Airflow DAG state (REST API), MLflow latest runs + registry versions (REST API), Prometheus scrape-target health (REST API). |
| Training pipeline (data → model) | DVC + `dvc dag` / `dvc plots show` | n/a (CLI + HTML) | The 5-stage DAG, last run timing, failed stage. |
| Scheduled jobs (drift, retraining, scraping) | **Airflow webserver** | http://localhost:8090 | DAG graph view, task duration Gantt, retries, logs. |
| Live-traffic + host metrics | **Grafana** | http://localhost:3000 | Predictions/sec, latency p50/p99, fraud ratio, host CPU/RAM/disk; pre-provisioned dashboards. |
| Errors / failed runs | Airflow `Browse → Task Instances`; Prometheus `up==0` panel; AlertManager UI | various | Failed task list with stack traces; alert history. |

> **The Streamlit `Pipeline Status` page** stitches the four backends into
> one screen, so a non-technical user does not need to know that Airflow
> lives at `:8090` or that DVC is a CLI tool. Each section also links out
> to the native console for deep-dive investigation, satisfying both
> "separate UI screen" and "orchestrated across tools" rubric items.

### 5.1 Pipeline speed & throughput

| Pipeline | Wall-clock (laptop, no GPU) | Throughput |
|---|---|---|
| Full DVC `repro` cold | ≈ 90 s | ≈ 3,000 rows / s |
| Single XGBoost `fit` (after preprocess) | ≈ 12 s | n/a |
| `/predict` round-trip (loopback) | ≈ 7 ms | ≈ 140 req/s single replica |
| Airflow `check_drift` task | ≈ 3 s | per 1k recent predictions |

Numbers are reproduced in `models/metrics.json` and the run logs.

## 6. Frontend design

### 6.1 Pages

Seven Streamlit pages — listed in the user manual ([05_user_manual.md](05_user_manual.md))
with screenshots. The new **Pipeline Status** page is the unified
ML-pipeline visualization screen the rubric asks for: it pulls live state
from DVC (`dvc.lock`), Airflow REST, MLflow REST, and Prometheus REST
into one screen with deep-links to each tool's native UI. Design choices:

- **Sidebar nav, not tabs.** With seven pages tabs would clip on small
  screens; sidebar is responsive.
- **Health pill in sidebar.** Always-visible green/red API-connection
  badge so the user is never confused about why a button does nothing.
- **Foolproofing.**
  - Buttons are disabled when API is unreachable.
  - File uploader rejects anything that is not `.csv`.
  - Feedback page has a strict 0/1 dropdown — typing arbitrary labels is
    impossible.
  - All numeric outputs format with explicit units (`ms`, `%`).
- **Colour grammar.** Red = fraud / error, green = legit / success, blue =
  neutral info. Same grammar across all pages.

### 6.2 Loose coupling

Streamlit talks to the API only via `requests.get/post`. There is no
shared Python module. Replacing Streamlit with a React app would be a
drop-in change. This is the rubric's loose-coupling rule. The base URL
is configurable via `API_URL` env var (`http://api:8000` in compose,
`http://localhost:8000` for local dev).

## 7. Source control, CI/CD, and versioning

| What | Tool | Where |
|---|---|---|
| Code | Git | GitHub |
| Data | DVC remote | local `.dvc/cache` (configurable to S3/Azure/GCS) |
| Model artifacts | MLflow Registry + DVC | `mlruns/`, `models/` |
| Feature engineering version | `FEATURE_VERSION` constant | [`src/features/feature_engineering.py`](../src/features/feature_engineering.py) |

**CI** is GitHub Actions, four jobs (lint → test → build → validate-compose),
on every push and PR to `main` / `develop`. Coverage report is uploaded as
an artifact. CI does *not* publish to PyPI — this is an internal pipeline,
not a library.

**DVC DAG = our CI pipeline for the ML side.** `dvc dag` shows the same
shape as Fig. 2 in the architecture doc.

**Branch model.** One feature branch per phase (`v0.<N>.0-phase<N>` tag at
merge). Conventional Commits for the messages so the changelog is
machine-readable.

## 8. Experiment tracking

Every training run (manual `dvc repro` or Airflow-triggered) logs to
MLflow:

- **Params** — `n_estimators`, `max_depth`, `learning_rate`,
  `scale_pos_weight`, `tree_method`, `max_bin`, `random_state`, `feature_version`.
- **Metrics** — `f1_score`, `precision`, `recall`, `pr_auc`, `roc_auc`,
  `avg_inference_latency_ms`, model_size_kb.
- **Tags** — `git_commit`, `mlflow.source.name`, custom
  `experiment_phase`, `quantization_enabled`.
- **Artifacts** — `best_model.joblib`, SHAP feature ranking JSON,
  `feature_names.json`, environment (`requirements.txt` snapshot).

Beyond `mlflow.autolog()` we explicitly log:

- Model size on disk (relevant for the "no-cloud / quantization" rubric).
- A SHAP-derived global feature ranking.
- The dataset hash (`source_data_md5`) so a run is bit-exactly
  reproducible.
- The current `FEATURE_VERSION`.

## 9. Instrumentation & observability

### 9.1 What we measure

| Signal | Source | Why |
|---|---|---|
| `predictions_total{result=fraud|legit}` | API | Throughput + class skew. |
| `prediction_latency_seconds` (histogram) | API | Latency SLA. |
| `prediction_errors_total` | API | Reliability. |
| `fraud_ratio` (gauge) | API | Detect attack waves. |
| `model_real_accuracy` (gauge) | API | Live correctness from feedback. |
| `feedback_total{actual_label}` | API | Label distribution sanity. |
| Host CPU/RAM/disk/network | node_exporter | Capacity. |
| `up` (per scrape target) | Prometheus | Outage detection. |

### 9.2 Visualization & alerting

- **Grafana dashboards** are *provisioned* (committed YAML in
  [`docker/monitoring/grafana/provisioning/`](../docker/monitoring/grafana/provisioning/))
  — no manual setup. **Seven** dashboards auto-load:
  *Project Overview*, *API* (legacy + Endpoint Detail), *Host* (legacy +
  System Resources Detail), *Stack Health*, *ML Ops & Feedback Loop*.
  They form an incident-investigation funnel: overview →
  endpoint → host → stack → ML.
- **Alert rules** in [`alert_rules.yml`](../docker/monitoring/prometheus/alert_rules.yml):
  - `HighErrorRate` — `> 5%` 5xx for 5 minutes.
  - `HighInferenceLatency` — p99 > 200 ms for 5 minutes.
  - `DataDriftDetected` — pushed by drift DAG when KS-test trips.
  - `APIDown` — `up{job="api"} == 0` for 1 minute.
  - `MLflowDown` / `AirflowDown` / `StreamlitDown` / `GrafanaDown` —
    blackbox HTTP probe failures (`probe_success == 0`) for 2 minutes,
    so every component in the stack — not just the API — is monitored.
- **Blackbox exporter** ([`docker/monitoring/blackbox/blackbox.yml`](../docker/monitoring/blackbox/blackbox.yml))
  HTTP-pings MLflow `/health`, Airflow `/health`, Streamlit
  `/_stcore/health`, and Grafana `/api/health` every 15 s. Prometheus
  scrapes the exporter, surfacing `probe_success{job="blackbox-<svc>"}`
  for each. Combined with the API + node-exporter scrapes, this gives
  full coverage of the rubric's "all components monitored" item.
- **AlertManager → Mailtrap**: `mailtrap_smtp_password` is a Docker secret.

## 10. Software packaging & deployment

| Requirement (rubric) | Implementation |
|---|---|
| **MLflow APIification** | API loads from MLflow Registry — Production alias chain — at startup. Same model can also be served by `mlflow models serve`, but the FastAPI surface gives us SHAP, feedback, stats endpoints in one process. |
| **MLprojects for env parity** | [`MLproject`](../MLproject) at the repo root declares 10 entry points (`validate`, `preprocess`, `feature_engineering`, `train`, `evaluate`, `scrape`, `retrain`, `promote`, `full_pipeline`, `main`). Each runs the same script DVC and Airflow invoke, so `mlflow run .` reproduces a training round in a fresh venv built from [`python_env.yaml`](../python_env.yaml) + `requirements.txt`. The same `Dockerfile` then layers that closure into the API / Streamlit / Airflow images — identical dependency graph across dev / test / Airflow / `mlflow run`. |
| **FastAPI for interactive web portal** | Yes — see [LLD §3](03_low_level_design.md#3-rest-api-specification). |
| **Dockerised backend + frontend** | Single multi-stage `Dockerfile`; same image, different `command`. |
| **Compose with two services** | We exceed the minimum: 8 services in [`docker-compose.yml`](../docker-compose.yml), still split as **frontend** (`streamlit`) and **backend** (`api`) at the architectural boundary. |

### 10.1 Compose vs Swarm overlay

| Concern | Compose | Swarm overlay |
|---|---|---|
| Replicas | 1 each | api ×3, others ×1 |
| Secrets | bind-mount | Raft-encrypted |
| Healthcheck-driven restart | `restart: on-failure` | `restart_policy: any` |
| Rolling update + rollback | n/a | `update_config: failure_action=rollback` |
| Where used | local dev | replicated demo |

Both modes use the same `docker-compose.yml` as base; Swarm-only fields
overlay via `docker-compose.swarm.yml` and `docker stack config` validates
the merged manifest in CI.

## 11. Backend / model lifecycle

States in MLflow Registry (Phase 9):

```
   [None] ──promote──▶ [Staging] ──regression-test──▶ [Production]
                                                          │
                                                          ▼
                                                     [Archived]
```

Stage transitions are scripted in [`scripts/promote_model.py`](../scripts/promote_model.py)
(no manual UI clicks). The API resolves *aliases first* and falls back to
legacy *stage* lookups (Phase 9 doc explains why both: aliases are the
modern API and stages are deprecated, but third-party tools still set
stages).

## 12. Closed-loop retraining

[`fraud_retraining_check`](../airflow/dags/fraud_retraining_dag.py) DAG, daily:

```
[check_drift, check_accuracy] >> decide_retrain >> [retrain, skip_retrain]
```

`decide_retrain` is a `BranchPythonOperator`:

```python
if drift.get("drift_detected") or accuracy.get("accuracy_decay"):
    return "retrain"
return "skip_retrain"
```

`retrain` task calls `scripts.retrain.retrain(config)` which runs the
training stage, evaluates against a regression baseline (must match or
beat current Production), and **only then** promotes to Staging. Promotion
to Production requires a deliberate human run of `promote_model.py` — the
loop never auto-promotes to Production by design.

## 13. Security posture

| Layer | Implementation | Scope justification |
|---|---|---|
| Data at rest | Fernet-encrypted raw CSV ([`src/data/security.py`](../src/data/security.py)). | Mandated by rubric II.A. |
| Credentials at rest | Docker Secrets (file-sourced, mounted at `/run/secrets/`); never in env vars or Git. | Phase 17. |
| Encryption key | Out of repo (`configs/.encryption_key` is gitignored). | Standard hygiene. |
| Auth on API | None by design. Production deployments add a reverse proxy (nginx, Traefik) or an API gateway for AuthN/Z + TLS. | Out of scope for academic project; documented in README "Security posture". |
| Encryption in transit | Inter-service over Docker overlay; no TLS at the API edge. | Same scope decision. Documented honestly. |

## 14. Failure modes & mitigations

| Failure | Detection | Mitigation |
|---|---|---|
| Model load fails at startup | `/ready` returns 503; alert fires | Local-file fallback at startup; Swarm rollback to previous image. |
| Drift on a single feature | Daily DAG, KS-test p < 0.05 | Retrain branch; Slack/email via Mailtrap. |
| `/predict` 5xx spike | Prometheus rule `HighErrorRate` for 5 min | Page on AlertManager; Swarm replica restart. |
| MLflow store corruption | Periodic backup of `mlruns/` (manual) | Restore from snapshot; `dvc.lock` lets us rebuild any model. |
| Airflow DAG bug | DAG fails → email + UI red | Code rolled back via Git; DAGs are mounted read-only so no corrupted state. |
| Bad data shape upstream | Validation stage fails immediately | Pipeline halts before training; `validation_report.json` shows what failed. |

## 15. Trade-offs & deferred work

Honest list — same one we'd open the viva with:

1. **Single-node Swarm.** True multi-node scaling needs shared storage
   (NFS / CephFS) for the bind mounts. Mechanism (routing mesh, healthchecks,
   secrets) is fully exercised on one node; adding nodes is mechanical.
2. **No public TLS.** Deferred to a reverse proxy in production deploys.
3. **SQLite for everything that's stateful.** MLflow + Airflow + the
   feedback DB all use SQLite. Upgrading to Postgres is a single env-var
   change for MLflow and Airflow; the feedback DB has its own ORM-style
   wrapper to make the swap small.
4. **No streaming ingest.** Daily DAGs are sufficient for the fraud
   problem statement; real-time scoring is on the request path, not the
   data path.
5. **Single dataset.** Generalisation to other markets needs retraining
   with locally collected data; the pipeline supports it but we have not
   demonstrated it.

## 16. Glossary

| Term | Meaning here |
|---|---|
| **DVC stage** | A node in `dvc.yaml` with `cmd`, `deps`, `outs`. |
| **Drift** | Statistical change in input distribution detected via per-feature KS-test against `feature_baselines.json`. |
| **PR-AUC** | Area under the precision-recall curve. Headline metric because of class imbalance (1 : 577). |
| **Fernet** | symmetric authenticated encryption from the `cryptography` library. |
| **Swarm routing mesh** | Built-in L4 load balancer that distributes traffic on a published port across all replicas. |
| **Production alias** | MLflow Registry alias pointing at one model version; the API loads whatever this alias resolves to. |
