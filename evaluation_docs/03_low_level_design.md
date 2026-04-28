# Low-Level Design (LLD) — API Endpoints, I/O Specifications & Module Contracts

> **Rubric line** — "Low-level design document quoting the endpoint definitions and IO specification."

| Field | Value |
|---|---|
| Project | Credit Card Fraud Detection — End-to-End MLOps |
| Component owners | API + Pipeline + Monitoring |
| Version | 1.0.0 |
| Last reviewed | 2026-04-28 |

The HLD ([02_high_level_design.md](02_high_level_design.md)) explains *why*
the system is shaped this way. This LLD specifies *exactly* the public
contracts every module must honour.

---

## 1. Module decomposition

```
src/
├── api/
│   └── app.py                  # FastAPI surface (this doc §3)
├── app/
│   └── streamlit_app.py        # Frontend; consumes API only via HTTP
├── data/
│   ├── ingest.py               # CSV -> DataFrame loader
│   ├── validate.py             # Schema/nulls/duplicates checks
│   ├── preprocess.py           # train/test split + scaler
│   ├── security.py             # Fernet encrypt/decrypt
│   ├── scrape_fraud_stats.py   # Wikipedia + Kaggle scraper
│   └── database.py             # SQLite ORM-style helpers (this doc §6)
├── features/
│   └── feature_engineering.py  # engineer_features + compute_drift_baselines
├── models/
│   └── train.py                # build_model + train_and_log
├── monitoring/
│   └── drift_detection.py      # KS-test + reports
└── __init__.py
```

Each subpackage has a single responsibility; cross-package imports are
**one-directional** (api → data, models → features → data, monitoring →
features). No cycles.

## 2. Configuration & environment

The runtime resolves configuration in this precedence order:

1. **Process env vars** (highest priority — what compose / Swarm sets).
2. **`configs/config.yaml`** (defaults committed to Git).
3. **In-code defaults** (lowest priority — kept conservative).

| Env var | Default | Used by |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | API, training |
| `MLFLOW_REGISTERED_NAME` | `fraud-detection-xgboost` | API, training |
| `MLFLOW_MODEL_STAGE` | `Production` | API |
| `MODEL_PATH` | `models/best_model.joblib` | API (fallback) |
| `FEATURES_PATH` | `models/feature_names.json` | API |
| `SCALER_PATH` | `data/processed/scaler.joblib` | API |
| `DB_PATH` | `data/fraud_detection.db` | API, retrain |
| `API_URL` | `http://localhost:8000` | Streamlit |

## 3. REST API specification

Base URL: `http://<host>:8000`
Content-Type: `application/json`
OpenAPI (auto-generated): `http://<host>:8000/openapi.json`
Interactive docs: `http://<host>:8000/docs`

### 3.1 Endpoint summary

| Verb | Path | Purpose | Auth | SLA |
|---|---|---|---|---|
| GET  | [`/health`](#31-get-health)   | Liveness probe | none | n/a |
| GET  | [`/ready`](#32-get-ready)     | Readiness probe — model loaded? | none | n/a |
| POST | [`/predict`](#33-post-predict) | Score a single transaction | none | p99 < 200 ms |
| POST | [`/feedback`](#34-post-feedback) | Submit ground-truth label | none | < 50 ms |
| GET  | [`/explain`](#35-get-explain) | SHAP top-k contributions | none | < 200 ms |
| GET  | [`/stats`](#36-get-stats)     | Aggregate prediction stats | none | < 50 ms |
| GET  | [`/metrics`](#37-get-metrics) | Prometheus exposition | none | n/a |

> **Auth = none.** Out of scope; production deployments add a reverse-proxy
> auth layer. See [HLD §13](02_high_level_design.md#13-security-posture).

### 3.1 GET /health

**Purpose** — process liveness only. Always returns 200 unless the process
is dead.

| Field | Spec |
|---|---|
| Request body | none |
| 200 response (JSON) | `{"status": "healthy", "timestamp": "<ISO-8601>"}` |
| Errors | none |

**Source** — [`src/api/app.py:221-223`](../src/api/app.py#L221-L223).

Example:

```bash
$ curl http://localhost:8000/health
{"status":"healthy","timestamp":"2026-04-28T10:13:42.117Z"}
```

### 3.2 GET /ready

**Purpose** — readiness probe; only returns 200 once the model is loaded
from MLflow Registry (or the local-file fallback).

| Field | Spec |
|---|---|
| Request body | none |
| 200 response | `{"status": "ready", "model_loaded": true, "model_source": "<str>", "model_version": <int|null>}` |
| 503 response | `{"detail": "Model not loaded"}` |

`model_source` is a human-readable string of the form
`registry: fraud-detection-xgboost/Production (v7, run f9c8a4b1...)` or
`file: models/best_model.joblib`.

**Source** — [`src/api/app.py:226-235`](../src/api/app.py#L226-L235).

### 3.3 POST /predict

**Purpose** — synchronous fraud prediction for one transaction. Persists
the request features to SQLite for later `/explain` and `/feedback`.

#### Request

```json
{
  "features": [<float>, ... 34 floats ...],
  "transaction_id": "txn_001"   // optional; auto-generated if absent
}
```

| Field | Type | Constraints |
|---|---|---|
| `features` | `list[float]` | length must equal `len(feature_names)` (= 34); else 400 |
| `transaction_id` | `str` (optional) | Free-form; if omitted, server generates `txn_<unix_ms>` |

> **Why 34 features?** 28 PCA components (V1-V28) + `Amount` + 5
> engineered (V14×V12 interaction, log_amount, amount_bin, etc.). See
> [`src/features/feature_engineering.py`](../src/features/feature_engineering.py).

#### Responses

| Status | Schema | When |
|---|---|---|
| 200 | `PredictionResponse` (below) | Happy path |
| 400 | `{"detail": "Expected 34 features, got N"}` | Length mismatch |
| 500 | `{"detail": "Prediction error: <msg>"}` | Model raised |
| 503 | `{"detail": "Model not loaded"}` | Process not ready |

```json
PredictionResponse:
{
  "transaction_id": "txn_001",
  "prediction": 0 | 1,
  "fraud_probability": 0.0123,    // 4-decimal-rounded
  "latency_ms": 7.21              // 2-decimal-rounded
}
```

#### Side effects

- Insert row into `predictions` table (`INSERT OR REPLACE`, keyed by
  `transaction_id`).
- Increment `predictions_total{result=fraud|legit}` Counter.
- Observe `prediction_latency_seconds` Histogram.
- Update rolling `fraud_ratio` Gauge.

**Source** — [`src/api/app.py:239-288`](../src/api/app.py#L239-L288).

### 3.4 POST /feedback

**Purpose** — record the ground-truth label for a previously-scored
transaction so the system can compute live accuracy and trigger drift /
retraining.

#### Request

```json
{
  "transaction_id": "txn_001",
  "actual_label": 0 | 1
}
```

#### Responses

| Status | Schema | When |
|---|---|---|
| 200 | `FeedbackResponse` (below) | Happy path |
| 404 | `{"detail": "Transaction <id> not found"}` | No prior `/predict` for that id |

```json
FeedbackResponse:
{
  "message": "Feedback recorded for txn_001",
  "total_feedback": 423,
  "current_accuracy": 0.9128       // null if window < 10 entries
}
```

#### Side effects

- Insert into `feedback` table with `is_correct = (predicted == actual)`.
- Increment `feedback_total{actual_label}` Counter.
- Set `model_real_accuracy` Gauge.

**Source** — [`src/api/app.py:292-313`](../src/api/app.py#L292-L313).

### 3.5 GET /explain

**Purpose** — SHAP-based feature attributions for a previously-scored
transaction. Useful for analysts auditing why the model raised an alarm.

#### Query string

| Param | Type | Default | Notes |
|---|---|---|---|
| `transaction_id` | `str` | required | Must exist in `predictions` table |
| `top_k` | `int` | 10 | Number of contributions to return; clamped to ≥1 |

#### Responses

| Status | Schema | When |
|---|---|---|
| 200 | `ExplainResponse` (below) | Happy path |
| 404 | `{"detail": "Transaction <id> not found"}` | No prior `/predict` |
| 500 | `{"detail": "SHAP error: <msg>"}` | Explainer init / compute failed |
| 503 | `{"detail": "Model not loaded"}` | Process not ready |

```json
ExplainResponse:
{
  "transaction_id": "txn_001",
  "prediction": 1,
  "fraud_probability": 0.9412,
  "base_value": -3.7891,                     // SHAP background expectation
  "top_contributions": [
    {"feature": "V14",      "shap_value": 1.234567},
    {"feature": "V14_x_V12","shap_value": -0.812345},
    ...
  ]                                          // sorted by |shap_value| desc
}
```

**Source** — [`src/api/app.py:317-388`](../src/api/app.py#L317-L388).

### 3.6 GET /stats

**Purpose** — aggregate counters from SQLite for the dashboard.

| Field | Spec |
|---|---|
| Request | none |
| 200 | `{"total_predictions": <int>, "fraud_count": <int>, "legit_count": <int>, "fraud_ratio": <float>, "avg_latency_ms": <float>}` |

**Source** — [`src/api/app.py:392-395`](../src/api/app.py#L392-L395).

### 3.7 GET /metrics

**Purpose** — Prometheus-format exposition (text/plain).

Sample (truncated):

```
# HELP predictions_total Total predictions
# TYPE predictions_total counter
predictions_total{result="legit"} 1842.0
predictions_total{result="fraud"} 17.0
# HELP prediction_latency_seconds Prediction latency
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{le="0.005"} 1623.0
prediction_latency_seconds_bucket{le="0.01"}  1820.0
prediction_latency_seconds_bucket{le="+Inf"}  1859.0
prediction_latency_seconds_sum 12.0931
prediction_latency_seconds_count 1859.0
# HELP fraud_ratio Rolling ratio of fraud predictions
fraud_ratio 0.00914
# HELP model_real_accuracy Real-world accuracy from feedback
model_real_accuracy 0.91280
```

Scraped by Prometheus every 15 s (configured in
[`docker/monitoring/prometheus/prometheus.yml`](../docker/monitoring/prometheus/prometheus.yml)).

**Source** — [`src/api/app.py:399-401`](../src/api/app.py#L399-L401).

## 4. Pydantic schemas (canonical types)

```python
class PredictionRequest(BaseModel):
    features: List[float]
    transaction_id: Optional[str] = None

class PredictionResponse(BaseModel):
    transaction_id: str
    prediction: int                  # 0 = legit, 1 = fraud
    fraud_probability: float         # [0.0, 1.0]
    latency_ms: float

class FeedbackRequest(BaseModel):
    transaction_id: str
    actual_label: int                # 0 or 1

class FeedbackResponse(BaseModel):
    message: str
    total_feedback: int
    current_accuracy: Optional[float] = None
```

## 5. Internal module contracts

### 5.1 `src.data.ingest`

```python
def load_csv(path: str) -> pd.DataFrame: ...
def get_data_summary(df: pd.DataFrame) -> dict: ...   # rows, cols, dtypes, missing
```

### 5.2 `src.data.validate`

```python
def validate_schema(df, required_cols=("Time","V1",...,"V28","Amount","Class")) -> dict
def validate_no_missing(df) -> dict
def validate_no_duplicates(df) -> dict
def write_validation_report(report: dict, path: str = "data/validation_report.json") -> None
```

`write_validation_report` always writes (never silently no-ops). The
report has a top-level `passed: bool`.

### 5.3 `src.data.preprocess`

```python
def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]
```

Contract:

- `Time` is dropped before splitting.
- 9,144 exact-duplicate rows are removed (Phase 7 fix).
- Scaler is fit **on training data only**, then applied to test.
- Returns `(X_train, X_test, y_train, y_test, scaler)`.
- Persists artifacts to `data/processed/` when called from
  `scripts/run_preprocessing.py`.

### 5.4 `src.features.feature_engineering`

```python
FEATURE_VERSION: str = "1.0.0"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame: ...
def compute_drift_baselines(
    X: pd.DataFrame,
    out_path: str = "data/baselines/feature_baselines.json",
) -> None
```

Contract:

- `engineer_features` adds 5 new columns; total goes from 29 → 34.
- `compute_drift_baselines` writes a JSON of the form
  `{"<feature>": {min, max, mean, std, q25, q50, q75}, "_meta": {...}}`.
- `_meta.feature_version` always equals the runtime constant.

### 5.5 `src.models.train`

```python
def load_config(path: str = "configs/config.yaml") -> dict
def build_model(config: dict) -> XGBClassifier
def train_and_log(config: dict) -> tuple[XGBClassifier, dict]
```

Contract:

- `train_and_log` is the canonical entry point used by both DVC and
  Airflow. It logs to MLflow with experiment `fraud-detection`, registers
  the model under `fraud-detection-xgboost`, and writes
  `models/best_model.joblib` + `models/metrics.json`.
- Returned `dict` mirrors `metrics.json`: `f1_score`, `precision`,
  `recall`, `roc_auc`, `pr_auc`, `avg_inference_latency_ms`.

### 5.6 `src.monitoring.drift_detection`

```python
def detect_drift(
    current_features: pd.DataFrame,
    baseline_path: str = "data/baselines/feature_baselines.json",
    p_value_threshold: float = 0.05,
) -> dict
```

Contract — returned dict keys:

- `drift_detected: bool`
- `drifted_features_count: int`
- `drifted_features: list[str]`
- `per_feature: dict[str, dict]` with `ks_stat`, `p_value`, `drifted: bool`
- Refuses to run if `_meta.feature_version != FEATURE_VERSION` (raises
  `ValueError`).

### 5.7 `src.data.security`

```python
def get_fernet(key_path: str = "configs/.encryption_key") -> Fernet
def encrypt_file(in_path, out_path, key_path=None) -> None
def decrypt_file(in_path, out_path, key_path=None) -> None
```

Contract — symmetric Fernet (AES-128-CBC + HMAC-SHA256). Key is generated
on first call if missing; permissions are set to `600`.

### 5.8 `src.data.scrape_fraud_stats`

```python
def scrape() -> dict       # {"sources": [...], "_meta": {...}}
def save(data: dict, path: str = "data/external/fraud_stats.json") -> None
```

Contract — every `source` entry has `name`, `url`, `fetched_at`,
`title`, `summary`, optional `stats` (list of dicts). Network failures
are captured as a `note` field; the function never raises.

## 6. Database schema (SQLite)

```sql
CREATE TABLE predictions (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  transaction_id     TEXT    UNIQUE NOT NULL,
  prediction         INTEGER NOT NULL,         -- 0 or 1
  fraud_probability  REAL    NOT NULL,         -- [0.0, 1.0]
  latency_ms         REAL    NOT NULL,
  features           TEXT,                     -- JSON-encoded list[float]
  created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE feedback (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  transaction_id     TEXT    NOT NULL,
  predicted_label    INTEGER NOT NULL,
  actual_label       INTEGER NOT NULL,
  is_correct         INTEGER NOT NULL,         -- 0 or 1
  created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (transaction_id) REFERENCES predictions(transaction_id)
);

CREATE TABLE drift_reports (
  id                       INTEGER PRIMARY KEY AUTOINCREMENT,
  drift_detected           INTEGER NOT NULL,
  drifted_features_count   INTEGER NOT NULL,
  drifted_features         TEXT,
  report_json              TEXT,
  created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Source: [`src/data/database.py`](../src/data/database.py).

`predictions.transaction_id` is `UNIQUE` so `INSERT OR REPLACE` is idempotent
across retries from the API layer.

## 7. DVC stages — exact deps and outs

From [`dvc.yaml`](../dvc.yaml):

```yaml
validate:           cmd: python scripts/run_validate.py
preprocess:         cmd: python scripts/run_preprocessing.py
feature_engineering: cmd: python scripts/run_feature_engineering.py
train:              cmd: python scripts/run_training.py
evaluate:           cmd: python scripts/run_evaluate.py
```

Full deps/outs already shown in [01_architecture_diagram.md §3.1](01_architecture_diagram.md#31-block-level-explanation).
Every output is content-hashed in `dvc.lock` and tracked in Git.

## 8. Airflow DAG contracts

### 8.1 `fraud_retraining_check` (daily)

| Task | Operator | Pushes to XCom | Reads from XCom |
|---|---|---|---|
| `check_drift` | PythonOperator | `{drift_detected, drifted_features_count, drifted_features, skipped}` | — |
| `check_accuracy` | PythonOperator | `{accuracy, threshold, accuracy_decay, skipped}` | — |
| `decide_retrain` | BranchPythonOperator | `task_id` to run next | `check_drift`, `check_accuracy` |
| `retrain` | PythonOperator | `{promoted, version, f1_score, pr_auc, run_id, success}` | — |
| `skip_retrain` | EmptyOperator | — | — |

Schedule: `@daily`. `max_active_runs=1`. `retries=1` with 10-minute
backoff. Source: [`airflow/dags/fraud_retraining_dag.py`](../airflow/dags/fraud_retraining_dag.py).

### 8.2 `fraud_stats_scrape` (weekly)

Single PythonOperator that calls `src.data.scrape_fraud_stats.scrape`,
persists to `data/external/fraud_stats.json` (atomic write to temp +
rename).

## 9. Prometheus alert rules

| Alert | Expression | `for` | Severity |
|---|---|---|---|
| `APIDown` | `up{job="api"} == 0` | 1m | critical |
| `HighErrorRate` | `rate(prediction_errors_total[5m]) / rate(predictions_total[5m]) > 0.05` | 5m | warning |
| `HighInferenceLatency` | `histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[5m])) > 0.2` | 5m | warning |
| `DataDriftDetected` | `data_drift_detected == 1` (pushed by DAG) | 0m | warning |
| `LowDiskSpace` | `node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"} < 0.1` | 5m | warning |

Routes via AlertManager → Mailtrap SMTP. Critical alerts get the
`@critical` receiver label.

## 10. Coding standards & implementation quality

### 10.1 Style

- **PEP 8** via `flake8` (max-line=100); `black` is the formatter of
  record. Both gate the CI `lint` job.
- All public functions carry docstrings; all module headers describe the
  module's role in one paragraph.
- Type hints everywhere except for trivially-obvious returns.

### 10.2 Logging

- Stdlib `logging`; level configurable via env (default `INFO`).
- Each module gets `logger = logging.getLogger(__name__)`.
- Log lines are single-line, `key=value` formatted where structured.

### 10.3 Exception handling

- The API converts all expected failures to typed `HTTPException`s with a
  matching status code.
- Unexpected failures bubble up to FastAPI's default handler (which
  returns 500 with a request-id in the response).
- Background scripts (Airflow callables) wrap their entry points in
  try/except so a single failure does not crash the scheduler.

### 10.4 Tests

Covered comprehensively in [04_test_plan_and_report.md](04_test_plan_and_report.md).
Headlines: **47 unit + 12 integration + 2 e2e** tests; **62% coverage** on `src/`.

### 10.5 Inline documentation

- Every non-trivial function has a docstring; every "why-it-was-done-this-way"
  comment includes the phase number it was introduced in (so a reader can
  jump straight to the rationale doc).
- Comments are *load-bearing only* — no commentary that just restates the
  code below it.

## 11. Versioning & API stability

- API uses semantic versioning at the FastAPI app level
  (`title="Credit Card Fraud Detection API", version="1.0.0"`).
- The `version` field shows up in `/openapi.json` and the Swagger UI.
- Breaking changes to request/response schemas would bump the major
  version; backward-compatible additions bump the minor.
