# Phase 15 — Closed-loop: drift + feedback + retraining

## Goal

Three components were already in the repo individually but never wired together:

| Component | File | Status before Phase 15 |
|---|---|---|
| **Drift detection** | [src/monitoring/drift_detection.py](../src/monitoring/drift_detection.py) | Worked, but reports never persisted |
| **Feedback loop** | [src/data/database.py](../src/data/database.py) + `/feedback` endpoint | Worked, accuracy queryable |
| **Retraining** | [scripts/retrain.py](../scripts/retrain.py) | **Broken** — imported `train_model` which Phase 8 renamed to `build_model` |

Phase 15 wires them into a single orchestrator the Airflow DAG (Phase 16) can call on a schedule.

> Guidelines: *"Implement a mechanism to log ground truth labels. Monitor for changes in input data distribution. Retrain when performance degrades. Implement rollback mechanisms."*

## What changed

### 1. [scripts/retrain.py](../scripts/retrain.py) — rewritten
Now a thin layer over `train_and_log` (Phase 8):

```text
retrain(config):
    1. train_and_log(config, run_name="retrain-<UTC>")  # full MLflow + registry
    2. if metrics["f1_score"] >= performance_decay_threshold:
           transition new version → Staging  (alias @staging)
       else:
           keep version registered but unpromoted; log warning
```

**Why no auto-Production**: retrains shouldn't hit live traffic without human review. If F1 ≥ threshold, the new version becomes the Staging candidate; a human still uses [scripts/promote_model.py](../scripts/promote_model.py) to promote it to Production.

**The "rollback" semantics**: if the new version fails the threshold, the **previously-promoted Staging/Production version stays in place**. Nothing to rollback explicitly — the system never deployed the bad version.

### 2. [scripts/check_and_retrain.py](../scripts/check_and_retrain.py) — new
The orchestrator that decides *when* to retrain:

```text
check_and_retrain(force=False):
    1. accuracy = SELECT … FROM feedback ORDER BY id DESC LIMIT feedback_window
    2. current = SELECT features FROM predictions ORDER BY id DESC LIMIT 1000
       reference = sample(X_train.csv, 5000)
       drift_report = detect_drift_ks_test(reference, current)
       save_drift_report(drift_report)            ← persisted to DB
    3. trigger_reasons = []
       if accuracy is below threshold:  trigger_reasons += ["accuracy_decay"]
       if drift_detected:               trigger_reasons += ["drift"]
       if --force:                      trigger_reasons += ["forced_via_flag"]
    4. if trigger_reasons: retrain(config)
       else: no-op, return summary
```

The orchestrator **always** runs the drift check (when there's enough data) and persists the report — useful trend data for analytics later, even when no retrain is triggered.

### 3. Drift reports now hit the DB
Previously [src/data/database.py](../src/data/database.py) had a `save_drift_report()` function and a `drift_reports` table, but no caller. Now `check_and_retrain.py` calls it on every drift evaluation. Trend queries are possible:

```sql
SELECT created_at, drift_detected, drifted_features_count
FROM drift_reports
ORDER BY id DESC
LIMIT 20;
```

## Verification

### Path A — happy path: no retrain triggered

```bash
# 1. Make a few predictions to populate the DB
docker compose up -d
sleep 15

for i in $(seq 1 50); do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d "{\"features\": $(python -c "
import pandas as pd
from src.features.feature_engineering import engineer_features
df = pd.read_csv('data/processed/X_test.csv').sample(1)
print(engineer_features(df).iloc[0].tolist())
"), \"transaction_id\": \"loop_$i\"}" >/dev/null
done

# 2. Run the orchestrator
python scripts/check_and_retrain.py
# Expected: drift_detected likely false (X_test ≈ X_train distribution),
# accuracy=None (no feedback yet), retrain_triggered=false
```

### Path B — force a retrain end-to-end

This actually triggers a training run + Staging promotion, so it takes a couple of minutes.

```bash
python scripts/check_and_retrain.py --force

# In summary output, look for:
#   "retrain_triggered": true
#   "retrain_outcome": {
#     "promoted": true,
#     "version": "8" (or whatever the next number is),
#     "metrics": { ... }
#   }
```

After the retrain succeeds:

```bash
python scripts/promote_model.py current
# Staging should show the new version (e.g. v8) — Production untouched (still v7)
```

### Path C — verify drift reports persisted

After running the orchestrator at least once:

```bash
python -c "
from src.data.database import get_connection
conn = get_connection()
rows = conn.execute(
    'SELECT created_at, drift_detected, drifted_features_count, drifted_features '
    'FROM drift_reports ORDER BY id DESC LIMIT 5'
).fetchall()
for r in rows:
    print(dict(r))
conn.close()
"
```

## Outputs of this phase

- [scripts/retrain.py](../scripts/retrain.py) — rewritten, integrates with `train_and_log` + MLflow Registry
- [scripts/check_and_retrain.py](../scripts/check_and_retrain.py) — closed-loop orchestrator
- This document
- Tag `v0.15.0-phase15` on `main`

(No code changes to [src/monitoring/drift_detection.py](../src/monitoring/drift_detection.py) or [src/data/database.py](../src/data/database.py) — the orchestrator just calls existing functions in their proper order.)

## What's next

Phase 16 — Airflow DAG that runs `python scripts/check_and_retrain.py` on a daily schedule, plus a separate weekly task that runs `python scripts/run_scrape.py` for the fraud-stats scraper. That converts the manual cron-style verification path here into actual automation.
