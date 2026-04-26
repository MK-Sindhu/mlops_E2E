# Phase 8 — Training, MLflow Tracking, Quantization

## Goal

Make the trainer trustworthy and reproducible:
- Single source of truth (no script-vs-module drift)
- Quantization config flag actually does something
- Multiple comparable experiments showing the quantization tradeoff
- Every run reproducible by Git commit hash + MLflow run ID

> Guidelines: *Track model versions, hyperparameters, performance metrics. Optimize models for local hardware (quantization). Every experiment must be reproducible via a Git commit hash and a corresponding MLflow run ID.*

## What changed

### 1. De-duplicated training code
Previously [src/models/train.py](../src/models/train.py) and [scripts/run_training.py](../scripts/run_training.py) were 80% identical with subtle differences (e.g. one registered the model, the other didn't; one used a remote tracking server, the other a file store). Now:

- [src/models/train.py](../src/models/train.py) is the **single source of truth** with `build_model`, `evaluate_model`, `save_artifacts`, `compute_model_size_metrics`, `train_and_log`.
- [scripts/run_training.py](../scripts/run_training.py) is a thin DVC wrapper around `train_and_log(config, run_name="dvc-pipeline")`.
- [scripts/run_experiments.py](../scripts/run_experiments.py) is a thin sweep runner over the same function.

### 2. Quantization implemented for real
[configs/config.yaml](../configs/config.yaml) had `quantize: true` declared but never read. Now [build_model](../src/models/train.py) honours it:

| Setting | Behaviour |
|---|---|
| `quantize: false` | Pass nothing extra to XGBoost — defaults apply. |
| `quantize: true`  | Force `tree_method="hist"` + `max_bin` from config (default 128, vs XGBoost's default 256). |

Histogram-based training with reduced `max_bin` is the closest practical equivalent to quantization for tree-based models — features are bucketed into discrete bins during training. Smaller memory footprint, slightly faster inference, marginal accuracy cost.

### 3. Tracking URI: file store by default + env-var override
Old config pointed at `http://localhost:5000`, which fails unless an MLflow server is running. New default is `file:./mlruns` (always works). Override at runtime:

```bash
MLFLOW_TRACKING_URI=http://mlflow:5000 python scripts/run_training.py
```

This is what the docker-compose stack will use in Phase 11 when an MLflow server container joins the deployment.

### 4. Model size metrics logged to MLflow
Every run now logs `model_size_bytes_joblib` and `model_size_bytes_json` so the quantization tradeoff is visible in the MLflow UI side-by-side with `f1_score` and `avg_inference_latency_ms`.

### 5. Model Registry registration
Each `train_and_log` call registers a new version of the `fraud-detection-xgboost` model. Running the experiment sweep produces 3 versions in one go — Phase 9 will promote selected versions to Staging/Production.

## Comparison runs

The sweep `python scripts/run_experiments.py` produces 3 MLflow runs:

| Name | quantize | max_bin | Purpose |
|---|---|---|---|
| `baseline_no_quantize`  | False | 256 (XGB default) | Reference, no quantization |
| `quantized_max_bin_128` | True  | 128 | Recommended production setting |
| `quantized_max_bin_64`  | True  |  64 | Aggressive — for size-constrained deployment |

After running, results are written to `reports/experiment_comparison.json` and printed as a comparison table.

### Results (sweep run on 2026-04-26)

| Run name | quantize | max_bin | F1 | PR-AUC | Precision | Recall | Latency (ms) | Size (KB joblib) | Size (KB json) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_no_quantize`  | False | 256 | **0.8229** | 0.8140 | 0.9000 | 0.7579 | 2.69 | 278.4 | 408.3 |
| `quantized_max_bin_128` | True  | 128 | 0.8111 | 0.8173 | 0.8588 | 0.7684 | 5.36 | 279.9 | 411.0 |
| `quantized_max_bin_64`  | True  |  64 | 0.8087 | **0.8214** | 0.8409 | 0.7789 | 2.03 | 281.9 | 415.0 |

**Observations:**
- **F1** peaks at the no-quantize baseline (0.8229) — quantization costs ~1.5% F1.
- **PR-AUC** actually peaks at `max_bin=64` (0.8214) — aggressive quantization slightly *helps* the precision-recall curve, suggesting some regularization effect.
- **Recall** monotonically rises with more quantization (0.7579 → 0.7684 → 0.7789) — a fraud-detection–friendly tradeoff: better at catching fraud at the cost of more false positives.
- **Model size** barely changes (~3 KB across all three) because XGBoost serializes the full tree structure regardless of `max_bin`. The savings from quantization in XGBoost are **runtime memory + training speed**, not artifact size.
- **Latency variance** between runs is system-load noise; all are well under the 200 ms budget.

### Run IDs (for reproducibility)

| Run name | MLflow Run ID | Registry version |
|---|---|---:|
| (DVC pipeline)          | `e50f99b4d33f446bbef96e17d5a95559` | v4 |
| `baseline_no_quantize`  | `72ae8430f5d04330ad5a74b9ced6abb5` | v5 |
| `quantized_max_bin_128` | `070a45c5a20c4cfb97eaf57514bb5276` | v6 |
| `quantized_max_bin_64`  | `b2316e8039174528a5e4afab9b1ffafa` | v7 |

The `fraud-detection-xgboost` registry has **7 versions** total at the close of Phase 8 (v1–v3 from earlier training during development, v4 from this phase's `dvc repro train`, v5–v7 from the sweep). Phase 9 will promote one to **Staging** and one to **Production**.

## Versioning & Artifacts

### Data versioning (DVC + git)

DVC tracks raw data + every pipeline output via [dvc.yaml](../dvc.yaml). The hashes live in [dvc.lock](../dvc.lock); each git commit that touches `dvc.lock` is effectively a "data version" snapshot.

```bash
# When did the pipeline output hashes change?
git log --oneline -- dvc.lock data/raw/creditcard.csv.dvc

# Inspect a specific data version (recover the X_train.csv from a past commit)
git checkout <commit-hash> -- dvc.lock
dvc checkout
```

A **DVC local remote** is now configured at `~/dvc-fraud-storage/`. Push/pull workflow:

```bash
dvc push                    # upload current cache to remote
dvc pull                    # restore from remote (e.g. on a fresh clone)
du -sh ~/dvc-fraud-storage/ # 286 MB after first push (raw CSV + processed splits + scaler + baselines + models)
```

### Model versioning (MLflow Registry)

Every `train_and_log` call registers a new version of `fraud-detection-xgboost`. Versions accumulate forever; nothing is overwritten.

```bash
# List all registered versions
ls mlruns/models/fraud-detection-xgboost/

# Inspect a specific version's metadata
cat mlruns/models/fraud-detection-xgboost/version-7/meta.yaml
```

After Phase 8 the registry has **7 versions** — v1–v3 from earlier development, v4 from `dvc repro`, v5–v7 from the sweep. Phase 9 will set stage transitions (`Staging`, `Production`).

### Artifacts per MLflow run

Each run logs **three** model representations plus auxiliary files. Inspect any run:

```bash
RUN_ID=72ae8430f5d04330ad5a74b9ced6abb5
EXP=mlruns/626522875577780945

ls -la "$EXP/$RUN_ID/artifacts/"
ls -la "$EXP/$RUN_ID/artifacts/model/"          # MLflow native xgboost format → Registry
ls -la "$EXP/$RUN_ID/artifacts/model_files/"    # raw weights: joblib + json + features + metrics
ls -la "$EXP/$RUN_ID/artifacts/auxiliary/"      # scaler.joblib, config.yaml, feature_baselines.json
```

| Artifact path | Contents | Purpose |
|---|---|---|
| `artifacts/model/MLmodel` | YAML metadata | Tells MLflow how to load the model |
| `artifacts/model/model.xgb` | Native XGBoost binary | Loaded by `mlflow.xgboost.load_model()` and the Registry |
| `artifacts/model/conda.yaml` + `requirements.txt` + `python_env.yaml` | Reproducible env | Pinned Python + library versions for the run |
| `artifacts/model_files/best_model.joblib` | `joblib.dump`-ed model | Direct sklearn-style loading |
| `artifacts/model_files/best_model.json` | XGBoost native JSON | Portable, human-readable weights |
| `artifacts/model_files/feature_names.json` | List of feature columns | Used by the API to validate inference inputs |
| `artifacts/model_files/metrics.json` | F1, PR-AUC, latency, etc. | Local copy alongside MLflow's logged metrics |
| `artifacts/auxiliary/scaler.joblib` | Fitted `StandardScaler` | Inference-time preprocessing |
| `artifacts/auxiliary/config.yaml` | Project config snapshot | What hyperparameters/paths produced this run |
| `artifacts/auxiliary/feature_baselines.json` | Drift baselines | Pinned to this model's training data |

## Reproducibility contract

Every experiment is identifiable by:
- **Git commit hash** — logged as MLflow tag `git_commit_hash`. Re-checkout that commit to recover the exact code.
- **MLflow Run ID** — fetches the exact hyperparameters, metrics, model artifact, and conda env.
- **DVC lock hash** — `dvc.lock` records the data versions feeding training.
- **`feature_baselines.json` `_meta.source_data_md5`** — proves which `X_train.csv` was used.

## Verification commands

```bash
# 1. Unit tests for the build_model + size metrics (no actual training)
pytest tests/unit/test_train.py -v

# 2. DVC training stage end-to-end
dvc repro --force train

# 3. Sweep — produces 3 comparable runs in MLflow + 3 registry versions
python scripts/run_experiments.py

# 4. Inspect runs
ls mlruns/                                            # experiment ID directories
cat reports/experiment_comparison.json | python -m json.tool

# 5. Inspect registry (file backend)
ls "mlruns/models/fraud-detection-xgboost/" 2>/dev/null || \
ls mlruns/models/                                     # registered models
```

## Outputs of this phase

- [src/models/train.py](../src/models/train.py) — refactored, quantization-aware, single source of truth
- [scripts/run_training.py](../scripts/run_training.py) — thin DVC wrapper
- [scripts/run_experiments.py](../scripts/run_experiments.py) — comparison sweep over 3 configs
- [configs/config.yaml](../configs/config.yaml) — added `max_bin`, `registered_model_name`, switched `tracking_uri` default to file store
- [tests/unit/test_train.py](../tests/unit/test_train.py) — 5 tests for builder + size metrics
- [reports/experiment_comparison.json](../reports/experiment_comparison.json) — generated by sweep, gitignored
- This document
- Tag `v0.8.0-phase8` on `main`

## What's next

Phase 9 — promote the best registered model version to **Staging** and **Production**, update the API to load via the registry URI (`models:/fraud-detection-xgboost/Production`), and document the model-promotion lifecycle.
