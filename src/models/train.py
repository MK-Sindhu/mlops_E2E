"""
Model Training Module
Train, evaluate, and register models with MLflow tracking.
Guideline: Every experiment must be reproducible via a Git commit hash and MLflow run ID.
Guideline: Track model versions, hyperparameters, and performance metrics.
"""
import os
import subprocess
import logging
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
import joblib
import yaml
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_git_commit_hash() -> str:
    """Get current Git commit hash for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def train_model(X_train: pd.DataFrame, y_train: pd.Series, config: dict) -> XGBClassifier:
    """
    Train an XGBoost classifier.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        config: Model configuration.
    
    Returns:
        Trained XGBClassifier model.
    """
    params = config["model"]["params"]
    logger.info(f"Training XGBoost with params: {params}")

    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        scale_pos_weight=params["scale_pos_weight"],
        eval_metric=params["eval_metric"],
        random_state=params["random_state"],
        n_jobs=config["model"]["optimization"]["n_jobs"],
        use_label_encoder=False,
    )

    model.fit(X_train, y_train)
    logger.info("Model training complete.")
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate model and return metrics.
    
    Returns:
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "f1_score": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
    }

    # Measure inference latency (business metric: < 200ms)
    start = time.time()
    for _ in range(100):
        model.predict(X_test.iloc[:1])
    avg_latency_ms = (time.time() - start) / 100 * 1000
    metrics["avg_inference_latency_ms"] = round(avg_latency_ms, 2)

    logger.info(f"Evaluation metrics: {json.dumps(metrics, indent=2)}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    return metrics


def train_and_log(config: dict):
    """
    Full training pipeline with MLflow logging.
    Logs: git hash, hyperparameters, metrics, model artifact.
    """
    # Setup MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Load processed data
    proc_dir = config["data"]["processed_path"]
    X_train = pd.read_csv(os.path.join(proc_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(proc_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(proc_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(proc_dir, "y_test.csv")).squeeze()

    # Apply feature engineering
    from src.features.feature_engineering import engineer_features
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    with mlflow.start_run() as run:
        # Log Git commit hash for reproducibility
        git_hash = get_git_commit_hash()
        mlflow.set_tag("git_commit_hash", git_hash)
        mlflow.set_tag("model_type", config["model"]["algorithm"])

        # Log hyperparameters
        mlflow.log_params(config["model"]["params"])

        # Train
        model = train_model(X_train, y_train, config)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model artifact
        mlflow.xgboost.log_model(model, "model")

        # Save model locally too
        os.makedirs("models", exist_ok=True)
        model_path = "models/best_model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Save feature names for serving
        feature_names = list(X_train.columns)
        with open("models/feature_names.json", "w") as f:
            json.dump(feature_names, f)
        mlflow.log_artifact("models/feature_names.json")

        logger.info(f"MLflow Run ID: {run.info.run_id}")
        logger.info(f"Git Commit: {git_hash}")
        logger.info(f"Model saved to {model_path}")

        return run.info.run_id, metrics


if __name__ == "__main__":
    config = load_config()
    run_id, metrics = train_and_log(config)
    print(f"\nRun ID: {run_id}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Inference Latency: {metrics['avg_inference_latency_ms']:.2f}ms")
