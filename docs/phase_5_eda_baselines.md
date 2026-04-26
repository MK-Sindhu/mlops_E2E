# Phase 5 — EDA & Drift Baselines

## Goal
Verify the EDA notebook covers what the guidelines require and that drift baselines are persisted in a form the monitoring code can consume.

> Guideline (Section II.A & II.B):
> *"Perform EDA to understand data characteristics, identify patterns, and detect potential issues. During EDA, calculate the statistical baseline (mean, variance, distribution) of features to be used later for drift detection. Record baseline statistics for later comparison."*

## What's in the EDA notebook

[notebooks/eda.ipynb](../notebooks/eda.ipynb) covers:

| Section | Finding |
|---|---|
| Class distribution | 284,315 legitimate vs **492 fraud** → **0.173%** positive rate, 1:577 imbalance ratio. Confirms PR-AUC must be the headline metric (not ROC-AUC). |
| Missing values & duplicates | **0 missing**, **1,081 duplicates** flagged for the preprocessing stage to handle. |
| Schema | 30 `float64` + 1 `int64` (`Class`) — matches the schema asserted in [src/data/validate.py](../src/data/validate.py). |
| Amount distribution | Heavy right tail; log transform reveals near-normal shape — motivates the `Amount_Log` engineered feature. |
| Time distribution by class | Fraud peaks visible at distinct hours (hours 11–12 and 26–27). Signal exists but `Time` is intentionally dropped from training to avoid leaking the dataset's specific 2-day window into the model. |
| Top discriminative features | V3, V14, V17, V12, V10, V7 — chosen by class-conditional mean separation. |
| Correlation with target | Strongest **negative**: V17 (-0.33), V14 (-0.30), V12 (-0.26). Strongest **positive**: V11 (+0.15), V4 (+0.13), V2 (+0.09). |
| Outliers in `Amount` | Long tail in legitimate class up to $25,691; fraud bounded around $2,000. Documented but not winsorised — XGBoost handles outliers without scaling. |

Static plots from the notebook are committed to [docs/](.):
- [class_distribution.png](class_distribution.png)
- [amount_distribution.png](amount_distribution.png)
- [time_distribution.png](time_distribution.png)
- [top_features.png](top_features.png)
- [correlation_with_target.png](correlation_with_target.png)

## Two baseline computations — by design, not bug

There are **two** places that compute statistical baselines, with different scopes:

| Source | Output location | Feature count | Purpose |
|---|---|---|---|
| `notebooks/eda.ipynb` | (only inside the notebook output) | **30** — raw V1–V28 + Amount + Time | EDA exploration; sanity check that `compute_drift_baselines` produces sensible numbers. |
| `dvc.yaml` `feature_engineering` stage → [src/features/feature_engineering.py:59](../src/features/feature_engineering.py#L59) | [data/baselines/feature_baselines.json](../data/baselines/feature_baselines.json) | **34** — V1–V28 + Amount + 5 engineered (Amount_Log, V1_V2_interaction, V_magnitude, V_mean, V_std) | **Canonical** baseline consumed by [src/monitoring/drift_detection.py](../src/monitoring/drift_detection.py). |

**The 34-feature JSON is the one production code uses.** The notebook version is purely for human EDA.

## Baseline schema

The baseline JSON is wrapped in a metadata envelope so drift detection can verify the baseline still corresponds to the data version it expects:

```json
{
  "_meta": {
    "baseline_schema_version": "1.0.0",
    "feature_version":         "1.0.0",
    "computed_at":             "2026-04-26T...Z",
    "row_count":               227845,
    "feature_count":           34,
    "source_data_path":        "data/processed/X_train.csv",
    "source_data_md5":         "<hex>"
  },
  "features": {
    "V1":     { "mean": ..., "std": ..., "variance": ..., "min": ..., "max": ..., "median": ..., "q25": ..., "q75": ... },
    "V2":     { ... },
    ...
    "V_std":  { ... }
  }
}
```

Why each `_meta` field exists:
- **`baseline_schema_version`** — bump if the *envelope* shape ever changes (so consumers can detect a format break).
- **`feature_version`** — must match `FEATURE_VERSION` in [feature_engineering.py](../src/features/feature_engineering.py); a mismatch means the baseline was computed against a different `engineer_features()` and shouldn't be trusted.
- **`source_data_md5`** — hash of the X_train.csv used. If raw data is regenerated and the hash changes, the baseline is stale.
- **`computed_at`** + **`row_count`** — sanity check / audit trail.

The per-feature stats satisfy the guideline's *"mean, variance, distribution"* requirement — quartiles + min/max give us the empirical distribution shape KS-test drift detection needs.

### Reader API

[src/monitoring/drift_detection.py](../src/monitoring/drift_detection.py) exposes:

| Function | Returns |
|---|---|
| `load_baselines()` | The inner `features` dict — drop-in compatible with all existing call sites. |
| `load_baseline_meta()` | The `_meta` dict — used by future provenance checks (Phase 15). |

## Feature engineering versioning

Per the guideline *"version their feature engineering logic separately from model logic"*, [src/features/feature_engineering.py:18](../src/features/feature_engineering.py#L18) declares:

```python
FEATURE_VERSION = "1.0.0"
```

Bump this whenever feature logic changes. A version mismatch between training-time features and inference-time features will be caught by [src/api/app.py](../src/api/app.py) at startup (planned check — verified in Phase 10).

## What's not in this phase

- **Re-running the EDA notebook** — the static outputs and PNG snapshots are committed; re-running is a Phase 7/8 concern (after preprocessing changes).
- **Drift detection itself** — that lives in [src/monitoring/drift_detection.py](../src/monitoring/drift_detection.py) and is verified end-to-end in Phase 15.

## Outputs of this phase

- [src/features/feature_engineering.py](../src/features/feature_engineering.py) — `compute_drift_baselines` now wraps stats in `{_meta, features}` and records source data MD5, timestamp, and row count.
- [src/monitoring/drift_detection.py](../src/monitoring/drift_detection.py) — `load_baselines()` returns the inner `features` dict; new `load_baseline_meta()` exposes provenance.
- [scripts/run_feature_engineering.py](../scripts/run_feature_engineering.py) — passes `source_path=X_train.csv` so MD5 is recorded.
- [data/baselines/feature_baselines.json](../data/baselines/feature_baselines.json) — regenerated with the new schema.
- [README.md](../README.md) — added link to the EDA notebook.
- This document.
- Tag `v0.5.0-phase5` on `main`.

## What's next

Phase 6 — add a BeautifulSoup scraper that pulls public fraud statistics from the web. This adds the third data source (after Kaggle CSV + user feedback DB) to satisfy the guideline's *"Identify and collect relevant data from various sources"* requirement, and gives the Streamlit dashboard a live "current threat landscape" panel.
