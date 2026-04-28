# Phase 13 — Grafana Dashboards

## Goal

**Seven** provisioned dashboards that auto-load the moment the stack starts — anyone running `docker compose up` sees real-time fraud + host metrics without clicking through Grafana's UI to import anything.

> Guideline: *"Build panels for metrics. Continuously monitor model performance in production."*

## What's provisioned

### Dashboard provider config
[docker/monitoring/grafana/provisioning/dashboards/dashboards.yml](../docker/monitoring/grafana/provisioning/dashboards/dashboards.yml)

Tells Grafana to scan `/etc/grafana/provisioning/dashboards/` on startup and load every `*.json` file as a dashboard inside a folder called **"Fraud Detection"**.

### Dashboard 1 — Fraud Detection — API
[docker/monitoring/grafana/provisioning/dashboards/fraud-detection-api.json](../docker/monitoring/grafana/provisioning/dashboards/fraud-detection-api.json)

| # | Panel | Query | Why |
|---|---|---|---|
| 1 | Prediction rate | `rate(predictions_total[1m])` per `result` | Throughput by fraud/legit |
| 2 | Latency p50 / p95 / p99 | `histogram_quantile(...)` over `prediction_latency_seconds_bucket` | 200 ms business SLA visible (red threshold at 0.2 s) |
| 3 | Fraud ratio (rolling) | `fraud_ratio` | Current production fraud share. Yellow >5%, red >20% |
| 4 | Real-world model accuracy | `model_real_accuracy` | From feedback loop. Red <70%, green ≥85% |
| 5 | Total predictions | `sum(predictions_total)` | Lifetime counter |
| 6 | Feedback received | `sum(feedback_total)` | Ground-truth labels submitted |
| 7 | HTTP rate by handler | `sum by (handler) (rate(http_requests_total[1m]))` | From `prometheus_fastapi_instrumentator` — shows `/predict`, `/feedback`, `/explain`, `/metrics` traffic separately |

### Dashboard 2 — Fraud Detection — Host
[docker/monitoring/grafana/provisioning/dashboards/fraud-detection-host.json](../docker/monitoring/grafana/provisioning/dashboards/fraud-detection-host.json)

| # | Panel | Query | Why |
|---|---|---|---|
| 1 | CPU usage % (+ iowait) | `100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)` | Host load. Yellow ≥70%, red ≥90% |
| 2 | Memory usage % | `(MemTotal - MemAvailable) / MemTotal * 100` | RAM pressure |
| 3 | Load average | `node_load1 / 5 / 15` | Trend over short / medium / long windows |
| 4 | Disk usage on `/` | `(1 - avail/size) * 100` | Stat panel, red ≥90% |
| 5 | Memory available | `node_memory_MemAvailable_bytes` | Stat panel in bytes (auto-formats to GB) |
| 6 | Network I/O | rx + tx bytes/sec, excluding `lo` | Throughput on real interfaces |

### Dashboard 3 — Fraud Detection — Project Overview

[docker/monitoring/grafana/provisioning/dashboards/fraud-detection-overview.json](../docker/monitoring/grafana/provisioning/dashboards/fraud-detection-overview.json)

The "executive screen" for the demo. One page, everything that matters:
API up/down, total predictions, fraud caught, real-world accuracy,
inference p99, predictions/sec by class, fraud-ratio trend, host CPU /
memory / disk, plus quick links to every other surface (Streamlit,
MLflow, Airflow, Prometheus, AlertManager, sibling dashboards).

### Dashboard 4 — Fraud Detection — API Endpoint Detail

[docker/monitoring/grafana/provisioning/dashboards/fraud-detection-api-detail.json](../docker/monitoring/grafana/provisioning/dashboards/fraud-detection-api-detail.json)

Per-endpoint deep dive. Request rate **by handler**, status-code
distribution as stacked bars, 5xx and 4xx rates as red-thresholded stat
tiles, average request and response sizes, p99 latency split per
handler, prediction-latency **heatmap**, and a combined "predictions vs
errors per minute" timeseries.

### Dashboard 5 — Fraud Detection — System Resources Detail

[docker/monitoring/grafana/provisioning/dashboards/fraud-detection-resources.json](../docker/monitoring/grafana/provisioning/dashboards/fraud-detection-resources.json)

Goes deeper than Dashboard 2: CPU usage **by mode** (user/system/iowait/etc.)
*and* per-CPU core utilisation; memory broken down into used / buffers /
cached / free with stacking; swap; dirty/active-anon memory pressure;
disk read/write throughput plus disk-busy %; **filesystem usage and
inode usage** as bar gauges per mountpoint; per-interface network rx/tx
plus errors/dropped; open file-descriptor saturation; context switches
per second; and host uptime.

### Dashboard 6 — Fraud Detection — Stack Health

[docker/monitoring/grafana/provisioning/dashboards/fraud-detection-stack-health.json](../docker/monitoring/grafana/provisioning/dashboards/fraud-detection-stack-health.json)

Monitors the **monitoring** itself. `up{}` per scrape target as
green/red tiles, scrape duration and samples-scraped per job, API
process resident + virtual memory, process CPU, open file descriptors
(open vs max), Python GC collections per generation, API uptime, and
API process network bytes.

### Dashboard 7 — Fraud Detection — ML Ops & Feedback Loop

[docker/monitoring/grafana/provisioning/dashboards/fraud-detection-ml-ops.json](../docker/monitoring/grafana/provisioning/dashboards/fraud-detection-ml-ops.json)

The ML-specific narrative: predictions/min stacked by class, fraud
ratio over time with thresholded bands, real-world accuracy with red
zones, feedback received per minute by **actual label**, snapshot tiles
(predictions, feedback, feedback/prediction ratio, fraud rate, accuracy,
p99), cumulative predictions vs cumulative feedback, and an
"operational links" footer with the exact CLIs to trigger retraining or
promote a model.

## Why all seven are valuable together

A latency spike on the API dashboard is ambiguous on its own — *model
issue, endpoint issue, host issue, or scrape issue?* The seven
dashboards form an **incident-investigation funnel**:

1. **Project Overview** — am I in trouble?
2. **API Endpoint Detail** — which endpoint is sad?
3. **System Resources Detail** — is the host the cause?
4. **Stack Health** — is the API process itself unhealthy (FDs, GC, scrape failing)?
5. **ML Ops & Feedback Loop** — is this a real model regression vs. infrastructure?

Phase 14 turns the most important threshold crossings into actual
AlertManager alerts.

## Volume mounts (no compose change needed)

The existing [docker-compose.yml](../docker-compose.yml) already mounts:
```yaml
- ./docker/monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
```

Both new dashboard files live under that tree, so they're picked up automatically. No restart-of-restart-required dance.

## Verification

```bash
# 1. Stack up
docker compose up -d
sleep 15

# 2. Open Grafana
open http://localhost:3000     # or just navigate in your browser
# - Login: admin / admin
# - Skip the password change prompt or set a new one
# - Browse → Fraud Detection folder → two dashboards listed

# 3. Generate some load so panels have data
for i in $(seq 1 20); do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d "{\"features\": $(python -c "
import pandas as pd
from src.features.feature_engineering import engineer_features
df = pd.read_csv('data/processed/X_test.csv').sample(1)
print(engineer_features(df).iloc[0].tolist())
"), \"transaction_id\": \"load_$i\"}" >/dev/null
done

# 4. Refresh the API dashboard — Prediction rate should show a spike
```

## Outputs of this phase

- [docker/monitoring/grafana/provisioning/dashboards/dashboards.yml](../docker/monitoring/grafana/provisioning/dashboards/dashboards.yml) — provider config
- [.../fraud-detection-api.json](../docker/monitoring/grafana/provisioning/dashboards/fraud-detection-api.json) — 7-panel API dashboard
- [.../fraud-detection-host.json](../docker/monitoring/grafana/provisioning/dashboards/fraud-detection-host.json) — 6-panel host dashboard
- [.../fraud-detection-overview.json](../docker/monitoring/grafana/provisioning/dashboards/fraud-detection-overview.json) — 11-panel executive overview
- [.../fraud-detection-api-detail.json](../docker/monitoring/grafana/provisioning/dashboards/fraud-detection-api-detail.json) — 10-panel per-endpoint deep dive
- [.../fraud-detection-resources.json](../docker/monitoring/grafana/provisioning/dashboards/fraud-detection-resources.json) — 14-panel system resources detail (per-CPU, memory breakdown, disk I/O, FS bargauges, network errors, FDs, ctx switches)
- [.../fraud-detection-stack-health.json](../docker/monitoring/grafana/provisioning/dashboards/fraud-detection-stack-health.json) — 10-panel scrape + API process monitor
- [.../fraud-detection-ml-ops.json](../docker/monitoring/grafana/provisioning/dashboards/fraud-detection-ml-ops.json) — 12-panel feedback-loop dashboard
- This document
- Tag `v0.13.0-phase13` on `main`

## What's next

Phase 14 — wire Prometheus's existing `alert_rules.yml` to **AlertManager** with **Mailtrap SMTP** so threshold crossings (high error rate, high latency, drift) actually email someone.
