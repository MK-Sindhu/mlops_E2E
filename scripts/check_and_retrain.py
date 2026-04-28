"""
Closed-loop orchestrator.

Decides whether to retrain by inspecting two production signals:

    1. Feedback-based accuracy (from the SQLite ``feedback`` table) —
       if it falls below ``monitoring.performance_decay_threshold``, retrain.

    2. KS-test data drift on the most-recent predictions vs a sample of
       the training data — if more than the configured fraction of
       features have shifted distributions, retrain.

Drift reports are persisted to the ``drift_reports`` table regardless of
whether retraining is triggered, so trends are queryable later.

Designed to be called by Airflow (Phase 16) on a schedule, or manually:

    python scripts/check_and_retrain.py            # decide based on signals
    python scripts/check_and_retrain.py --force    # retrain regardless
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.retrain import retrain  # noqa: E402
from src.data.database import (  # noqa: E402
    get_connection,
    get_feedback_count,
    get_model_accuracy,
    save_drift_report,
)
from src.features.feature_engineering import engineer_features  # noqa: E402
from src.models.train import load_config  # noqa: E402
from src.monitoring.drift_detection import detect_drift_ks_test  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Production signals ----------------------------------------------


def get_recent_predictions_df(window: int = 1000, features_path: str = None):
    """Pull the most-recent predictions' input features into a DataFrame.

    The /predict endpoint stores each request's feature vector as a JSON
    string in ``predictions.features``; we deserialize and label with the
    feature-name list from ``api.features_path`` in config.

    Args:
        window: Number of most-recent predictions to include.
        features_path: Where to load the feature-name list from. Defaults
            to ``api.features_path`` in config.yaml.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT features FROM predictions WHERE features IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (window,),
    ).fetchall()
    conn.close()

    if not rows:
        return None

    if features_path is None:
        features_path = (
            load_config().get("api", {}).get("features_path", "models/feature_names.json")
        )
    if not os.path.exists(features_path):
        logger.warning(f"{features_path} not found; cannot label drift df")
        return None
    with open(features_path) as f:
        feature_names = json.load(f)

    feature_arrays = []
    for r in rows:
        try:
            arr = json.loads(r["features"])
            if len(arr) == len(feature_names):
                feature_arrays.append(arr)
        except (TypeError, ValueError):
            continue

    if not feature_arrays:
        return None
    return pd.DataFrame(feature_arrays, columns=feature_names)


def check_drift(config: dict):
    """Run KS-test drift detection on recent predictions vs training data.

    Returns the drift report (dict) or None if there's not enough data.
    Persists the report to the ``drift_reports`` table either way (when run).
    All knobs (window sizes, sampling, min-samples) come from
    ``monitoring.*`` in configs/config.yaml.
    """
    monitoring = config.get("monitoring", {}) or {}
    recent_window = int(monitoring.get("recent_predictions_window", 1000))
    min_recent = int(monitoring.get("drift_min_recent_samples", 30))
    ref_sample_size = int(monitoring.get("drift_reference_sample_size", 5000))
    ref_random_state = int(monitoring.get("drift_reference_random_state", 42))

    current = get_recent_predictions_df(window=recent_window)
    if current is None or len(current) < min_recent:
        logger.info(
            f"Insufficient recent predictions for drift detection "
            f"({0 if current is None else len(current)} rows; need ≥{min_recent})"
        )
        return None

    proc = config["data"]["processed_path"]
    x_train_path = os.path.join(proc, "X_train.csv")
    if not os.path.exists(x_train_path):
        logger.warning(f"{x_train_path} missing; skipping drift detection")
        return None

    # Sample training data — KS-test doesn't need the full set
    reference = engineer_features(
        pd.read_csv(x_train_path).sample(
            n=min(ref_sample_size, sum(1 for _ in open(x_train_path)) - 1),
            random_state=ref_random_state,
        )
    )

    threshold = monitoring.get("drift_threshold", 0.05)
    report = detect_drift_ks_test(reference, current, threshold=threshold)

    save_drift_report(
        drift_detected=report["drift_detected"],
        drifted_count=report["drifted_features_count"],
        drifted_features=",".join(report["drifted_features"]),
        report_json=json.dumps(report),
    )
    logger.info(
        f"Drift check: {report['drifted_features_count']}/{report['total_features']} "
        f"features drifted (drift_detected={report['drift_detected']})"
    )
    return report


def check_accuracy(config: dict):
    """Get the current feedback-based accuracy if we have enough samples."""
    feedback_count = get_feedback_count()
    min_samples = config["monitoring"].get("feedback_window", 100)
    if feedback_count < min_samples:
        logger.info(
            f"Insufficient feedback ({feedback_count} < {min_samples}); skipping accuracy check"
        )
        return None
    return get_model_accuracy()


# --- Decision + entry point ------------------------------------------


def check_and_retrain(force: bool = False) -> dict:
    config = load_config()

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feedback_count": get_feedback_count(),
        "accuracy": None,
        "drift": None,
        "retrain_triggered": False,
        "retrain_reasons": [],
    }

    # Always evaluate signals so we have a record, even if force=True
    accuracy = check_accuracy(config)
    summary["accuracy"] = accuracy

    drift_report = check_drift(config)
    summary["drift"] = (
        {
            "drift_detected": drift_report["drift_detected"],
            "drifted_features_count": drift_report["drifted_features_count"],
            "drifted_features": drift_report["drifted_features"],
            "total_features": drift_report["total_features"],
        }
        if drift_report else None
    )

    threshold = config["monitoring"]["performance_decay_threshold"]
    if accuracy is not None and accuracy < threshold:
        summary["retrain_reasons"].append(
            f"accuracy_decay (accuracy={accuracy:.4f} < {threshold})"
        )
    if drift_report and drift_report["drift_detected"]:
        summary["retrain_reasons"].append(
            f"drift ({drift_report['drifted_features_count']}/{drift_report['total_features']} features)"
        )
    if force:
        summary["retrain_reasons"].append("forced_via_flag")

    if not summary["retrain_reasons"]:
        logger.info("No retraining required; signals nominal.")
        return summary

    logger.warning(f"Retraining triggered: {', '.join(summary['retrain_reasons'])}")
    success, retrain_info = retrain(config)
    summary["retrain_triggered"] = True
    summary["retrain_outcome"] = retrain_info
    return summary


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true",
                   help="Retrain regardless of drift/accuracy signals (dry-runs the orchestration path)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = check_and_retrain(force=args.force)
    print("\n=== check_and_retrain summary ===")
    print(json.dumps(result, indent=2, default=str))
