# Phase 14 — AlertManager + Mailtrap SMTP

## Goal

Make the alert rules from [alert_rules.yml](../docker/monitoring/prometheus/alert_rules.yml) actually go somewhere. Until now, Prometheus evaluated them but had no notification target — fired alerts were visible at `/alerts` but no one was emailed.

> Guideline: *"Configure Prometheus/Grafana to trigger alerts if error rates exceed 5% or if data drift is detected."*

## Topology

```
Prometheus  evaluates  alert_rules.yml
    │
    │ POST  http://alertmanager:9093/api/v2/alerts  (every 15s eval)
    ▼
AlertManager  groups + routes
    │
    │ smtp 2525 (TLS)
    ▼
Mailtrap sandbox inbox  ← devs/QA review messages safely (no real delivery)
```

## What changed

### 1. New `alertmanager` service in [docker-compose.yml](../docker-compose.yml)

```yaml
alertmanager:
  image: prom/alertmanager:latest
  ports: ["9093:9093"]
  volumes:
    - ./docker/monitoring/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
    - alertmanager-data:/alertmanager
```

Plus a named volume `alertmanager-data` so silences and notification state survive restarts.

### 2. Prometheus knows about AlertManager
[prometheus.yml](../docker/monitoring/prometheus/prometheus.yml) gains:
```yaml
alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]
```

### 3. AlertManager config — [alertmanager.yml](../docker/monitoring/alertmanager/alertmanager.yml)

| Section | What it does |
|---|---|
| `global.smtp_*` | Mailtrap sandbox SMTP — captures emails for review without delivering to real inboxes |
| `inhibit_rules` | When a `critical` alert fires, silences the `warning` version of the same alertname/instance — avoids duplicate noise |
| `route` | Default → `email-default`. Sub-route: `severity=critical` → `email-critical` with shorter `group_wait` (10s) and `repeat_interval` (1h) |
| `receivers` | Two receivers, both with HTML email bodies linking back to Prometheus + Grafana |

### 4. Existing alert rules now have a destination

The 4 rules in [alert_rules.yml](../docker/monitoring/prometheus/alert_rules.yml) (unchanged from earlier) now route as follows:

| Alert | Threshold | Routes to |
|---|---|---|
| `HighErrorRate` | API 5xx > 5% for 2m | `email-critical` (severity: critical) |
| `ModelPerformanceDecay` | `model_real_accuracy < 0.80` for 5m | `email-default` (severity: warning) |
| `AbnormalFraudRatio` | `fraud_ratio > 0.10` for 5m (drift signal) | `email-default` (severity: warning) |
| `HighLatency` | p95 latency > 200 ms for 2m | `email-default` (severity: warning) |

These map directly to the guideline's required alert conditions (>5% errors, drift, latency SLA breach).

## Setting up real Mailtrap credentials

The committed config has placeholder credentials so SMTP send will fail with auth error — alerts still fire (visible in the AlertManager UI) but no email is delivered. To fix:

1. Sign up at https://mailtrap.io (free tier is fine)
2. Open your sandbox → **Integration** → **SMTP** → copy `Username` and `Password`
3. Edit [docker/monitoring/alertmanager/alertmanager.yml](../docker/monitoring/alertmanager/alertmanager.yml):
   ```yaml
   smtp_auth_username: 'YOUR_REAL_USERNAME'
   smtp_auth_password: 'YOUR_REAL_PASSWORD'
   ```
4. `docker compose restart alertmanager`

For production: don't commit credentials to git. Use a `.env` file with `docker compose --env-file`, or mount a secret into the container.

## Verification

```bash
docker compose up -d
sleep 15

# 1. AlertManager itself is reachable
curl -s http://localhost:9093/-/ready
# → "OK" if config parsed cleanly

# 2. Prometheus knows about AlertManager
curl -s http://localhost:9090/api/v1/alertmanagers | python -m json.tool
# → "alertmanager:9093" listed under "activeAlertmanagers"

# 3. See currently firing/pending alerts
curl -s http://localhost:9090/api/v1/alerts | python -m json.tool

# 4. Open the UIs
#    http://localhost:9093  → AlertManager (alerts, silences, status)
#    http://localhost:9090/alerts  → Prometheus alerts page

# 5. Manually fire a test alert to verify routing reaches AlertManager
curl -s -XPOST http://localhost:9093/api/v2/alerts -d '[
  {
    "status": "firing",
    "labels": {
      "alertname": "Phase14SmokeTest",
      "severity": "critical",
      "service": "fraud-detection"
    },
    "annotations": {
      "summary": "Phase 14 smoke test fired manually via curl",
      "description": "If you can see this alert in http://localhost:9093 and a Mailtrap inbox, routing works."
    },
    "generatorURL": "http://localhost:9090"
  }
]' -H "Content-Type: application/json"

# Then check it landed
curl -s http://localhost:9093/api/v2/alerts | python -m json.tool
```

If you have real Mailtrap credentials configured, the test alert will appear in your sandbox inbox within ~30s (after the `email-critical` group_wait).

## Outputs of this phase

- [docker/monitoring/alertmanager/alertmanager.yml](../docker/monitoring/alertmanager/alertmanager.yml) — new
- [docker-compose.yml](../docker-compose.yml) — `alertmanager` service added; named `alertmanager-data` volume; Prometheus depends on AlertManager
- [docker/monitoring/prometheus/prometheus.yml](../docker/monitoring/prometheus/prometheus.yml) — `alerting:` block points Prometheus at the new service
- This document
- Tag `v0.14.0-phase14` on `main`

## What's next

Phase 15 — verify the existing drift detection module + feedback loop + retraining script form a working closed loop end-to-end. With alerts now routable, the drift detector firing → AlertManager firing → email is the full operational story.
