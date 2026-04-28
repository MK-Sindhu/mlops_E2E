"""
Model Training Module.

Single source of truth for training the fraud-detection XGBoost model.
DVC stage and the experiment-sweep script both call ``train_and_log``.

Guideline: Every experiment must be reproducible via Git commit hash + MLflow run ID.
Guideline: Track model versions, hyperparameters, and performance metrics.
Guideline: Optimize models for local hardware (quantization).
"""

import json
import logging
import os
import subprocess
import time
from typing import Optional, Tuple

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import yaml
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.features.feature_engineering import FEATURE_VERSION, engineer_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Config helpers -----------------------------------------------------


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_git_commit_hash() -> str:
    """Current Git HEAD — used as MLflow tag for full reproducibility."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def resolve_tracking_uri(config: dict) -> str:
    """Honour MLFLOW_TRACKING_URI env override, else fall back to config."""
    return os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])


# --- Model construction (where quantization lives) ---------------------


def build_model(config: dict) -> XGBClassifier:
    """Construct an XGBClassifier with optional quantization.

    Quantization here means using XGBoost's histogram-based training with a
    reduced ``max_bin``. Continuous features are bucketed into discrete bins
    during training, which is the closest practical equivalent to quantization
    for tree-based models — smaller memory footprint, slightly faster
    inference, marginal accuracy cost.

    When ``quantize`` is False, ``tree_method`` and ``max_bin`` are NOT passed,
    so XGBoost uses its own defaults (currently ``hist`` / ``max_bin=256`` in
    XGBoost 2.x).
    """
    params = config["model"]["params"]
    opt = config["model"]["optimization"]

    kwargs = dict(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        scale_pos_weight=params["scale_pos_weight"],
        eval_metric=params["eval_metric"],
        random_state=params["random_state"],
        n_jobs=opt["n_jobs"],
        use_label_encoder=False,
    )

    if opt.get("quantize", False):
        kwargs["tree_method"] = "hist"
        kwargs["max_bin"] = opt.get("max_bin", 128)
        logger.info(f"Quantization ON: tree_method=hist, max_bin={kwargs['max_bin']}")
    else:
        logger.info("Quantization OFF: using XGBoost defaults")

    return XGBClassifier(**kwargs)


# --- Evaluation --------------------------------------------------------


def measure_inference_latency_ms(
    model, X_test: pd.DataFrame, n_iter: int = 100
) -> float:
    """Measure per-call single-row inference latency."""
    start = time.perf_counter()
    for _ in range(n_iter):
        model.predict(X_test.iloc[:1])
    return round((time.perf_counter() - start) / n_iter * 1000, 2)


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, config: Optional[dict] = None
) -> dict:
    """ML metrics + business metric (latency).

    ``config['model']['latency_iterations']`` controls how many single-row
    predict() calls are timed (default 100 if config not provided).
    """
    n_iter = 100
    if config is not None:
        n_iter = int(config.get("model", {}).get("latency_iterations", n_iter))

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "f1_score": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "avg_inference_latency_ms": measure_inference_latency_ms(
            model, X_test, n_iter=n_iter
        ),
    }
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    logger.info(
        f"\n{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}"
    )
    return metrics


# --- Persistence -------------------------------------------------------


def save_artifacts(
    model, X_train: pd.DataFrame, metrics: dict, config: Optional[dict] = None
) -> dict:
    """Save model + metadata to disk, return paths.

    The model directory is derived from ``api.model_path`` in config so that
    the path lives in exactly one place (and DVC's ``outs:`` already pin
    these filenames downstream).
    """
    api_cfg = (config or {}).get("api", {}) or {}
    model_path = api_cfg.get("model_path", "models/best_model.joblib")
    features_path = api_cfg.get("features_path", "models/feature_names.json")

    model_dir = os.path.dirname(model_path) or "models"
    os.makedirs(model_dir, exist_ok=True)

    base, _ = os.path.splitext(model_path)  # e.g. "models/best_model"
    paths = {
        "joblib": model_path,
        "json": f"{base}.json",
        "features": features_path,
        "metrics": os.path.join(model_dir, "metrics.json"),
    }
    joblib.dump(model, paths["joblib"])
    model.save_model(paths["json"])
    with open(paths["features"], "w") as f:
        json.dump(list(X_train.columns), f)
    with open(paths["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)
    return paths


def compute_model_size_metrics(paths: dict) -> dict:
    """Report saved artifact sizes — visible proof quantization affects size."""
    return {
        "model_size_bytes_joblib": os.path.getsize(paths["joblib"]),
        "model_size_bytes_json": os.path.getsize(paths["json"]),
    }


# --- Orchestrator -----------------------------------------------------


def train_and_log(
    config: dict,
    run_name: Optional[str] = None,
) -> Tuple[str, dict, dict]:
    """Run full training pipeline with MLflow logging + registry registration.

    Returns:
        (run_id, metrics, size_metrics)
    """
    mlflow.set_tracking_uri(resolve_tracking_uri(config))
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    proc = config["data"]["processed_path"]
    X_train = engineer_features(pd.read_csv(os.path.join(proc, "X_train.csv")))
    X_test = engineer_features(pd.read_csv(os.path.join(proc, "X_test.csv")))
    y_train = pd.read_csv(os.path.join(proc, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(proc, "y_test.csv")).squeeze()

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    with mlflow.start_run(run_name=run_name) as run:
        # Reproducibility tags
        mlflow.set_tag("git_commit_hash", get_git_commit_hash())
        mlflow.set_tag("model_type", config["model"]["algorithm"])
        mlflow.set_tag(
            "quantized", str(config["model"]["optimization"].get("quantize", False))
        )

        # Hyperparameters
        mlflow.log_params(config["model"]["params"])
        mlflow.log_params(
            {f"opt_{k}": v for k, v in config["model"]["optimization"].items()}
        )

        # Train
        model = build_model(config)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, config=config)
        mlflow.log_metrics(metrics)

        # Persist + size
        paths = save_artifacts(model, X_train, metrics, config=config)
        size_metrics = compute_model_size_metrics(paths)
        mlflow.log_metrics(size_metrics)

        # Build a signature + input example so the registry "Schema" panel is
        # populated and the "Make Predictions" snippets are concrete.
        signature = infer_signature(X_train, model.predict(X_train.head(5)))
        input_example = X_train.head(3)

        # Register model in MLflow Model Registry
        registered_name = config["mlflow"].get("registered_model_name")
        if registered_name:
            mlflow.xgboost.log_model(
                model,
                "model",
                registered_model_name=registered_name,
                signature=signature,
                input_example=input_example,
            )
            logger.info(f"Registered as '{registered_name}'")

            # Enrich the freshly-registered version with a description + tags
            # so the Models → Version page is no longer empty. We have to
            # search by run_id because log_model returns no version info.
            try:
                client = MlflowClient()
                versions = client.search_model_versions(
                    f"name='{registered_name}' AND run_id='{run.info.run_id}'"
                )
                if versions:
                    v = versions[0].version
                    quantized = config["model"]["optimization"].get("quantize", False)
                    description_lines = [
                        f"**Run name**: `{run_name or 'unnamed'}`",
                        f"**F1**: {metrics['f1_score']:.4f} • "
                        f"**PR-AUC**: {metrics['pr_auc']:.4f} • "
                        f"**Recall**: {metrics['recall']:.4f}",
                        f"**Latency**: {metrics['avg_inference_latency_ms']} ms • "
                        f"**Size**: {size_metrics['model_size_bytes_joblib']:,} bytes",
                        f"**Quantized**: {quantized} • "
                        f"**Git commit**: `{get_git_commit_hash()[:12]}`",
                    ]
                    client.update_model_version(
                        name=registered_name,
                        version=v,
                        description="\n\n".join(description_lines),
                    )
                    # Searchable tags — show up under "Tags" on the version page.
                    tags = {
                        "run_name": run_name or "unnamed",
                        "f1_score": f"{metrics['f1_score']:.4f}",
                        "pr_auc": f"{metrics['pr_auc']:.4f}",
                        "recall": f"{metrics['recall']:.4f}",
                        "avg_inference_latency_ms": str(
                            metrics["avg_inference_latency_ms"]
                        ),
                        "quantized": str(quantized),
                        "git_commit": get_git_commit_hash()[:12],
                        "feature_version": FEATURE_VERSION,
                    }
                    for k, val in tags.items():
                        client.set_model_version_tag(
                            name=registered_name, version=v, key=k, value=val
                        )
                    logger.info(
                        f"Enriched registry v{v} with description + {len(tags)} tags"
                    )
            except Exception as e:
                # Registry enrichment is best-effort — don't fail training over it.
                logger.warning(f"Could not enrich registry version metadata: {e}")
        else:
            mlflow.xgboost.log_model(
                model, "model", signature=signature, input_example=input_example
            )

        # Auxiliary artifacts (paths derived from config, not hardcoded)
        data_cfg = config.get("data", {}) or {}
        scaler_path = os.path.join(
            data_cfg.get("processed_path", "data/processed/"), "scaler.joblib"
        )
        baseline_path = os.path.join(
            data_cfg.get("baselines_path", "data/baselines/"),
            data_cfg.get("baselines_filename", "feature_baselines.json"),
        )
        for p in paths.values():
            mlflow.log_artifact(p, artifact_path="model_files")
        for aux in (scaler_path, "configs/config.yaml", baseline_path):
            if os.path.exists(aux):
                mlflow.log_artifact(aux, artifact_path="auxiliary")

        logger.info(f"MLflow Run ID: {run.info.run_id}")
        return run.info.run_id, metrics, size_metrics


if __name__ == "__main__":
    config = load_config()
    run_id, metrics, sizes = train_and_log(config)
    print(f"\nRun ID:   {run_id}")
    print(f"F1:       {metrics['f1_score']:.4f}")
    print(f"PR-AUC:   {metrics['pr_auc']:.4f}")
    print(f"Latency:  {metrics['avg_inference_latency_ms']}ms")
    print(f"Size:     {sizes['model_size_bytes_joblib']:,} bytes (joblib)")
