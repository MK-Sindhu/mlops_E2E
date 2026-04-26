# Credit Card Fraud Detection - MLOps Pipeline

## Problem Statement
Credit card fraud is a significant financial threat, costing billions annually. This project builds an end-to-end AI system that classifies credit card transactions as fraudulent or legitimate, with a strong focus on MLOps best practices.

## Metrics
- **ML Metrics**: F1-Score, Precision, Recall, PR-AUC (due to class imbalance)
- **Business Metrics**: Inference latency < 200ms per transaction

## Dataset
- **Source**: Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (European cardholders, Sep 2013)
- **Size**: 284,807 transactions, 31 columns
- **Schema**: `Time`, `V1`–`V28` (PCA-transformed for confidentiality), `Amount`, `Class` (0 = legit, 1 = fraud)
- **Known biases / caveats**:
  - Extreme class imbalance — only 492 frauds (~0.172%). Use PR-AUC, not ROC-AUC, as the headline metric.
  - `V1`–`V28` are anonymised PCA components — feature interpretation is limited.
  - Single-region, single-time-window — distribution may not generalise to other markets.
- **Storage**: Raw CSV is encrypted at rest with Fernet (`creditcard.csv.enc`) and tracked via DVC.

## Architecture
```
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│  FastAPI     │───▶│  Model      │───▶│  Prometheus +    │
│  (API)       │    │  Server     │    │  Grafana         │
│  /predict    │    │  (XGBoost)  │    │  (Monitoring)    │
│  /health     │    │             │    │                  │
│  /feedback   │    │             │    │                  │
└─────────────┘    └─────────────┘    └─────────────────┘
```

## Tech Stack
| Component            | Tool                     |
|----------------------|--------------------------|
| Version Control      | Git + DVC                |
| Experiment Tracking  | MLflow                   |
| Model Training       | XGBoost, Scikit-learn    |
| Model Serving        | FastAPI                  |
| Containerization     | Docker + Docker Compose  |
| CI/CD                | GitHub Actions           |
| Monitoring           | Prometheus + Grafana     |
| Security             | Fernet encryption        |
| Explainability       | SHAP                     |

## Project Structure
```
credit-card-fraud-detection/
├── .github/workflows/     # CI/CD pipelines
├── configs/               # Configuration files
├── data/
│   ├── raw/               # Original dataset (DVC tracked)
│   ├── processed/         # Cleaned & transformed data
│   └── baselines/         # Statistical baselines for drift detection
├── docker/
│   ├── api/               # API Dockerfile
│   └── monitoring/        # Prometheus & Grafana configs
├── docs/                  # Documentation
├── mlruns/                # MLflow experiment runs
├── notebooks/             # EDA & experimentation notebooks
├── scripts/               # Utility scripts (train, retrain, etc.)
├── src/
│   ├── data/              # Data ingestion & validation
│   ├── features/          # Feature engineering (versioned separately)
│   ├── models/            # Model training, evaluation, registry
│   ├── api/               # FastAPI app with /predict, /health, /feedback
│   └── monitoring/        # Drift detection & alerting
├── tests/
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the data
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/raw/creditcard.csv`, or pull the DVC-tracked version:
```bash
dvc pull
```

### 3. Run the full pipeline
DVC stages (validate → preprocess → feature_engineering → train → evaluate):
```bash
dvc repro
```
Or run stages individually via the `scripts/run_*.py` wrappers.

### 4. Run with Docker Compose
```bash
docker-compose up --build
```

### 5. Launch the Streamlit dashboard (local dev)
```bash
./scripts/run_streamlit.sh
```
Then open http://localhost:8501. The wrapper forces the project venv so the dashboard never picks up a system Python with a broken pyarrow.

### 5. Test endpoints
```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -1.2, ...]}'

# Feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "txn_123", "actual_label": 1}'
```

## Exploratory Data Analysis
See [notebooks/eda.ipynb](notebooks/eda.ipynb) for the full EDA: class imbalance (1:577), discriminative-feature ranking, time/amount distributions, and outlier analysis. Static plots are committed to [docs/](docs/) for read-only viewing without running the notebook.

## Reproducibility
Every experiment is reproducible via:
- **Git commit hash** — exact code version
- **MLflow run ID** — hyperparameters, metrics, artifacts

## No Cloud
All components run locally via Docker Compose. No cloud services used.
