# Credit Card Fraud Detection - MLOps Pipeline

## Problem Statement
Credit card fraud is a significant financial threat, costing billions annually. This project builds an end-to-end AI system that classifies credit card transactions as fraudulent or legitimate, with a strong focus on MLOps best practices.

## Metrics
- **ML Metrics**: F1-Score, Precision, Recall, PR-AUC (due to class imbalance)
- **Business Metrics**: Inference latency < 200ms per transaction

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

### 2. Download & prepare data
```bash
python scripts/download_data.py
python src/data/validate.py
python src/data/preprocess.py
```

### 3. Train model
```bash
python src/models/train.py
```

### 4. Run with Docker Compose
```bash
docker-compose up --build
```

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

## Reproducibility
Every experiment is reproducible via:
- **Git commit hash** — exact code version
- **MLflow run ID** — hyperparameters, metrics, artifacts

## No Cloud
All components run locally via Docker Compose. No cloud services used.
