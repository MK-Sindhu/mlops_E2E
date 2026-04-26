# Phase 7 — Preprocessing & Feature Engineering Pipeline

## Goal
Verify the chain `validate → preprocess → feature_engineering` runs cleanly end-to-end and the resulting splits are sane.

> Guideline (II.B): *"Clean and transform the data to prepare it for model training. Handle missing values, outliers, and inconsistencies. Automate data preprocessing pipelines."*

## What changed (and why)

### 🔴 Fixed: test-set leakage in the scaler

**Before** ([src/data/preprocess.py:113–127](../src/data/preprocess.py#L113-L127) at `v0.6.0-phase6`):
```python
df_clean = clean_data(df, config)
df_scaled, scaler = scale_features(df_clean, config, fit=True)   # ← fit on FULL data
X_train, X_test, y_train, y_test = split_data(df_scaled, config) # ← then split
```

The `StandardScaler` saw test-set rows during `fit_transform`. Its `mean_` and `scale_` parameters incorporated test-set statistics. This is textbook test-set leakage — small numerical impact on this dataset, **but a guaranteed flag in any MLOps review**.

**After**:
```python
df_clean = clean_data(df, config)
X_train, X_test, y_train, y_test = split_data(df_clean, config)            # split first
X_train_scaled, scaler = scale_features(X_train, config, fit=True)         # fit on train only
X_test_scaled, _      = scale_features(X_test,  config, fit=False, scaler=scaler)
```

The saved [scaler.joblib](../data/processed/scaler.joblib) is now fit on **220,530 training rows** only, never on the test set. The scaler logs `scaler.n_samples_seen_` so a glance at the run output confirms no leakage.

### 🟡 Added: 11 unit tests for the preprocess module

[tests/unit/test_preprocess.py](../tests/unit/test_preprocess.py) covers what was previously untested:

| Group | Test | What it proves |
|---|---|---|
| `clean_data` | `test_drops_time_column` | `Time` is removed |
|              | `test_drops_nan_rows` | NaN-containing rows are dropped |
|              | `test_removes_duplicates` | Duplicate rows collapsed |
| `scale_features` | `test_fit_uses_training_statistics` | Scaler.mean_ matches training mean |
|                  | `test_inference_reuses_fitted_scaler` | `fit=False` returns the same scaler instance |
|                  | `test_fit_false_without_scaler_raises` | Defensive error path |
| `split_data` | `test_test_size_matches_config` | Test-set size honours config |
|              | `test_stratified_preserves_class_balance` | Fraud ratio nearly identical in both splits |
|              | `test_train_test_index_disjoint` | No row in both partitions |
|              | `test_target_column_excluded_from_X` | `Class` not leaked into features |
| **Leakage guard** | `test_scaler_fit_on_training_partition_only` | `scaler.n_samples_seen_ == len(X_train)`, regression guard for the leakage fix |

The leakage guard test is the most important — it's the regression-prevention test for the fix above. If anyone in the future reverts the order of operations, this test fails.

### 🟢 De-duplicated: scripts/run_preprocessing.py

The DVC entrypoint [scripts/run_preprocessing.py](../scripts/run_preprocessing.py) used to duplicate the orchestration logic from [src/data/preprocess.py](../src/data/preprocess.py). It's now a thin wrapper around `run_preprocessing(config)`. Single source of truth — if the pipeline changes, both DVC and direct invocations stay consistent automatically.

## Duplicate-count discrepancy explained

| Source | Reported duplicates |
|---|---|
| [notebooks/eda.ipynb](../notebooks/eda.ipynb) | **1,081** (counted on raw data, with `Time` included) |
| [src/data/preprocess.py](../src/data/preprocess.py) | **9,144** (counted *after* `Time` is dropped) |

Both correct. `Time` is essentially a unique-per-row counter (seconds since first transaction), so removing it lets many otherwise-identical rows collapse into duplicates. After dedup the cleaned dataset is **275,663 rows**, which split 80/20 into 220,530 train / 55,133 test (both with fraud ratio ≈ 0.0017).

## Pipeline shape (DVC dag, stages 1–3 of 5)

```
data/raw/creditcard.csv (DVC tracked)
        │
        ▼
   ┌─────────┐
   │validate │   scripts/run_validate.py        →  data/validation_report.json
   └────┬────┘
        │
        ▼
   ┌──────────┐
   │preprocess│   scripts/run_preprocessing.py  →  X_train.csv, X_test.csv,
   └────┬─────┘                                    y_train.csv, y_test.csv,
        │                                          scaler.joblib
        ▼
┌──────────────────┐
│feature_engineering│ scripts/run_feature_engineering.py → data/baselines/feature_baselines.json
└──────────────────┘
                                                 (continues to train → evaluate)
```

## Verification (run on this branch)

After the leakage fix, **the scaler.joblib and feature_baselines.json files will rotate** — different bytes, different MD5. That's expected; the new files are the *correct* artifacts.

```bash
# Run the affected stages
dvc repro --force preprocess feature_engineering

# Confirm scaler is fit on training partition only
python -c "
import joblib
s = joblib.load('data/processed/scaler.joblib')
print(f'Scaler fit on {s.n_samples_seen_} rows (should be ~220,530)')
print(f'mean_={s.mean_}, scale_={s.scale_}')
"

# Run the new tests (and the full unit suite)
pytest tests/unit/ -v
```

## Outputs of this phase

- [src/data/preprocess.py](../src/data/preprocess.py) — `run_preprocessing` reordered to fit scaler on X_train only
- [scripts/run_preprocessing.py](../scripts/run_preprocessing.py) — thin wrapper around `run_preprocessing`
- [tests/unit/test_preprocess.py](../tests/unit/test_preprocess.py) — 11 new tests
- [data/processed/](../data/processed/) — regenerated artifacts (scaler now fit on training only)
- [data/baselines/feature_baselines.json](../data/baselines/feature_baselines.json) — regenerated; new MD5 reflects the new X_train.csv
- This document
- Tag `v0.7.0-phase7` on `main`

## What's next

Phase 8 — verify model training + MLflow tracking, and **actually implement** the `quantize: true` config flag that's been declared but not honoured.
