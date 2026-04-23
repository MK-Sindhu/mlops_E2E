"""
Train model with MLflow experiment tracking.
Logs: git commit hash, hyperparameters, metrics, model artifact.
Guideline: Every experiment must be reproducible via Git commit hash + MLflow run ID.
"""
import os
import sys
import json
import time
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    classification_report
)
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
import joblib
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.feature_engineering import engineer_features


def get_git_hash():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


# Load config
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# Load data
proc = config["data"]["processed_path"]
X_train = engineer_features(pd.read_csv(f"{proc}/X_train.csv"))
X_test = engineer_features(pd.read_csv(f"{proc}/X_test.csv"))
y_train = pd.read_csv(f"{proc}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{proc}/y_test.csv").squeeze()

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# MLflow setup (local file store, no server needed)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(config["mlflow"]["experiment_name"])

params = config["model"]["params"]

with mlflow.start_run() as run:
    # Log git hash
    git_hash = get_git_hash()
    mlflow.set_tag("git_commit_hash", git_hash)
    mlflow.set_tag("model_type", "xgboost")

    # Log hyperparameters
    mlflow.log_params(params)

    # Train
    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        scale_pos_weight=params["scale_pos_weight"],
        eval_metric=params["eval_metric"],
        random_state=params["random_state"],
        n_jobs=-1,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }

    # Measure inference latency
    start = time.time()
    for _ in range(100):
        model.predict(X_test.iloc[:1])
    metrics["avg_inference_latency_ms"] = round((time.time() - start) / 100 * 1000, 2)

    # Log metrics
    mlflow.log_metrics(metrics)

    # Save locally
    os.makedirs("models", exist_ok=True)

    # 1. Save model weights (joblib)
    joblib.dump(model, "models/best_model.joblib")

    # 2. Save model in native XGBoost format
    model.save_model("models/best_model.json")

    # 3. Save feature names
    feature_names = list(X_train.columns)
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # 4. Save metrics to JSON
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Log ALL artifacts to MLflow ──────────────────
    # Model via MLflow's native XGBoost logging (includes weights + conda env)
    mlflow.xgboost.log_model(
        model, "model",
        registered_model_name="fraud-detection-xgboost"
    )

    # Additional artifacts folder
    mlflow.log_artifact("models/best_model.joblib", artifact_path="model_weights")
    mlflow.log_artifact("models/best_model.json", artifact_path="model_weights")
    mlflow.log_artifact("models/feature_names.json", artifact_path="model_weights")
    mlflow.log_artifact("models/metrics.json", artifact_path="model_weights")
    mlflow.log_artifact("data/processed/scaler.joblib", artifact_path="preprocessing")
    mlflow.log_artifact("configs/config.yaml", artifact_path="config")
    mlflow.log_artifact("data/baselines/feature_baselines.json", artifact_path="baselines")

    # Print results
    print(f"\n{'='*50}")
    print(f"MLflow Run ID:  {run.info.run_id}")
    print(f"Git Commit:     {git_hash}")
    print(f"{'='*50}")
    print(f"F1 Score:       {metrics['f1_score']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"ROC-AUC:        {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:         {metrics['pr_auc']:.4f}")
    print(f"Latency:        {metrics['avg_inference_latency_ms']}ms")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))