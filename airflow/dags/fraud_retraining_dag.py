"""
Daily retraining-decision DAG (multi-task version).

Visualises the decision pipeline as actual Airflow tasks instead of one
opaque BashOperator:

    ┌────────────────┐
    │  check_drift   │──┐
    └────────────────┘  │   ┌──────────────────┐    ┌──────────┐
                         ├──▶│ decide_retrain   │──▶ │ retrain  │
    ┌────────────────┐  │   │ (BranchOperator) │    └──────────┘
    │ check_accuracy │──┘   └──────┬───────────┘
    └────────────────┘             │                ┌──────────────┐
                                   └───────────────▶│ skip_retrain │
                                                    └──────────────┘

Each task pushes its result to XCom; the decide task pulls them and routes
to ``retrain`` or ``skip_retrain``.

Requires: PYTHONPATH=/project (set in docker-compose) so ``from src.*``
and ``from scripts.*`` resolve. The chdir guarantees relative paths in
the underlying scripts (e.g. data/processed/X_train.csv) keep working.
"""
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator


PROJECT_ROOT = "/project"
sys.path.insert(0, PROJECT_ROOT)


# --- Task callables ---------------------------------------------------


def _ensure_cwd():
    """Project scripts use relative paths (data/processed/X_train.csv etc.).
    Each task callable must run from the project root."""
    if os.getcwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)


def task_check_drift(**_context) -> dict:
    """Run KS-test drift detection on recent predictions; persist report."""
    _ensure_cwd()
    from scripts.check_and_retrain import check_drift
    from src.models.train import load_config

    config = load_config()
    report = check_drift(config)
    if report is None:
        return {"drift_detected": False, "drifted_features_count": 0,
                "drifted_features": [], "skipped": True}
    return {
        "drift_detected": bool(report["drift_detected"]),
        "drifted_features_count": int(report["drifted_features_count"]),
        "drifted_features": list(report["drifted_features"]),
        "skipped": False,
    }


def task_check_accuracy(**_context) -> dict:
    """Compute feedback-based accuracy; flag decay if below threshold."""
    _ensure_cwd()
    from scripts.check_and_retrain import check_accuracy
    from src.models.train import load_config

    config = load_config()
    accuracy = check_accuracy(config)
    threshold = config["monitoring"]["performance_decay_threshold"]
    decay = accuracy is not None and accuracy < threshold
    return {
        "accuracy": float(accuracy) if accuracy is not None else None,
        "threshold": float(threshold),
        "accuracy_decay": bool(decay),
        "skipped": accuracy is None,
    }


def task_decide_retrain(**context) -> str:
    """BranchPythonOperator callable: returns the next task_id to run."""
    drift = context["ti"].xcom_pull(task_ids="check_drift") or {}
    accuracy = context["ti"].xcom_pull(task_ids="check_accuracy") or {}

    if drift.get("drift_detected") or accuracy.get("accuracy_decay"):
        reasons = []
        if drift.get("drift_detected"):
            reasons.append(f"drift in {drift.get('drifted_features_count')} features")
        if accuracy.get("accuracy_decay"):
            reasons.append(f"accuracy {accuracy.get('accuracy')} < {accuracy.get('threshold')}")
        print(f"Decision: RETRAIN ({', '.join(reasons)})")
        return "retrain"

    print("Decision: SKIP RETRAIN — both signals nominal.")
    return "skip_retrain"


def task_retrain(**_context) -> dict:
    """Run train_and_log via scripts.retrain.retrain; auto-promote to Staging."""
    _ensure_cwd()
    from scripts.retrain import retrain
    from src.models.train import load_config

    config = load_config()
    success, info = retrain(config)
    return {
        "promoted": bool(info.get("promoted", False)),
        "version": info.get("version"),
        "f1_score": info["metrics"]["f1_score"],
        "pr_auc": info["metrics"]["pr_auc"],
        "run_id": info.get("run_id"),
        "success": bool(success),
    }


# --- DAG --------------------------------------------------------------


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


with DAG(
    dag_id="fraud_retraining_check",
    description=(
        "Daily drift + feedback-accuracy check; retrain (and auto-promote to Staging) if signals demand it."
    ),
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2026, 4, 26),
    catchup=False,
    tags=["fraud-detection", "retraining", "multi-task"],
    max_active_runs=1,
) as dag:

    check_drift = PythonOperator(
        task_id="check_drift",
        python_callable=task_check_drift,
        execution_timeout=timedelta(minutes=5),
    )

    check_accuracy = PythonOperator(
        task_id="check_accuracy",
        python_callable=task_check_accuracy,
        execution_timeout=timedelta(minutes=2),
    )

    decide_retrain = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=task_decide_retrain,
    )

    retrain = PythonOperator(
        task_id="retrain",
        python_callable=task_retrain,
        execution_timeout=timedelta(minutes=10),
    )

    skip_retrain = EmptyOperator(
        task_id="skip_retrain",
    )

    # Both signals feed the branch decision, which routes to one of two leaves.
    [check_drift, check_accuracy] >> decide_retrain >> [retrain, skip_retrain]
