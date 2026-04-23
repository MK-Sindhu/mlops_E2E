"""
Model Retraining Script
Triggered when drift is detected or performance decays.
Guideline: Retrain models periodically or when performance degrades due to data drift.
Guideline: Implement rollback mechanisms for failed deployments.
"""
import os
import logging
import json
import shutil
import yaml
import pandas as pd
import joblib
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def backup_current_model(model_path: str = "models/best_model.joblib"):
    """Backup current model before retraining (for rollback)."""
    if os.path.exists(model_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"models/backup/model_{timestamp}.joblib"
        os.makedirs("models/backup", exist_ok=True)
        shutil.copy2(model_path, backup_path)
        logger.info(f"Model backed up to {backup_path}")
        return backup_path
    return None


def rollback_model(backup_path: str, model_path: str = "models/best_model.joblib"):
    """Rollback to a previous model version."""
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, model_path)
        logger.info(f"Rolled back to model: {backup_path}")
    else:
        logger.error(f"Backup not found: {backup_path}")


def retrain(config: dict):
    """
    Full retraining pipeline:
    1. Backup current model
    2. Retrain on latest data
    3. Evaluate new model
    4. If better → deploy; if worse → rollback
    """
    from src.models.train import train_model, evaluate_model
    from src.features.feature_engineering import engineer_features

    logger.info("Starting model retraining...")

    # Step 1: Backup
    backup_path = backup_current_model()

    # Step 2: Load data
    proc_dir = config["data"]["processed_path"]
    X_train = pd.read_csv(os.path.join(proc_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(proc_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(proc_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(proc_dir, "y_test.csv")).squeeze()

    # Step 3: Feature engineering
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    # Step 4: Retrain
    new_model = train_model(X_train, y_train, config)
    new_metrics = evaluate_model(new_model, X_test, y_test)

    # Step 5: Compare with threshold
    threshold = config["monitoring"]["performance_decay_threshold"]
    if new_metrics["f1_score"] >= threshold:
        # Deploy new model
        model_path = "models/best_model.joblib"
        joblib.dump(new_model, model_path)
        logger.info(f"New model deployed! F1: {new_metrics['f1_score']:.4f}")
        return True, new_metrics
    else:
        # Rollback
        if backup_path:
            rollback_model(backup_path)
            logger.warning(
                f"New model F1 ({new_metrics['f1_score']:.4f}) below threshold "
                f"({threshold}). Rolled back."
            )
        return False, new_metrics


if __name__ == "__main__":
    config = load_config()
    success, metrics = retrain(config)
    print(f"Retraining {'succeeded' if success else 'failed (rolled back)'}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
