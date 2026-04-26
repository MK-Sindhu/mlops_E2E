"""
Model Retraining Script.

Delegates the actual training to ``train_and_log`` (Phase 8) so retraining
goes through the same MLflow-tracked, registry-registered pipeline as a
fresh dvc-pipeline run. Adds:
    - run_name tagging so retrains are identifiable in MLflow
    - auto-promotion to Staging if F1 >= performance_decay_threshold
    - safe fallback: when below threshold, registers but does NOT promote

Production promotion remains a manual step (scripts/promote_model.py),
since retrains shouldn't ship to prod without human review.

Guidelines:
    - Retrain models when performance degrades or drift is detected.
    - Implement rollback mechanisms — here, "rollback" is implicit: the
      previously-promoted Staging/Production version stays in place when
      the new version is below threshold.
"""
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from typing import Tuple

# scripts/ isn't a package; allow `from src.*` imports when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MLflow stage transitions print deprecation warnings — quiet for CLI
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

from mlflow.tracking import MlflowClient  # noqa: E402

from src.models.train import load_config, train_and_log  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retrain(config: dict) -> Tuple[bool, dict]:
    """Run a fresh training cycle, register in MLflow, auto-promote on success.

    Returns:
        (auto_promoted, info) where info contains run_id, run_name, metrics,
        size_metrics, promoted (bool), and either ``version`` (on success) or
        ``reason`` (on skipped promotion).
    """
    run_name = f"retrain-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    logger.info(f"Starting retraining run '{run_name}'")

    run_id, metrics, sizes = train_and_log(config, run_name=run_name)

    info: dict = {
        "run_id": run_id,
        "run_name": run_name,
        "metrics": metrics,
        "size_metrics": sizes,
        "promoted": False,
    }

    threshold = config["monitoring"]["performance_decay_threshold"]
    if metrics["f1_score"] < threshold:
        logger.warning(
            f"Retrained F1 ({metrics['f1_score']:.4f}) below threshold "
            f"({threshold}). Version registered but NOT promoted. "
            f"Existing Staging/Production untouched."
        )
        info["reason"] = f"below_threshold (f1={metrics['f1_score']:.4f} < {threshold})"
        return False, info

    # Resolve the version that was just registered for this run
    client = MlflowClient()
    name = config["mlflow"]["registered_model_name"]
    versions = client.search_model_versions(f"name='{name}' AND run_id='{run_id}'")
    if not versions:
        logger.error(f"Could not find registered version for run_id={run_id}")
        info["reason"] = "registry_lookup_failed"
        return False, info

    version = versions[0].version

    # Auto-promote to Staging (alias + legacy stage). Production is gated
    # on manual review via scripts/promote_model.py.
    client.transition_model_version_stage(name=name, version=version, stage="Staging")
    client.set_registered_model_alias(name=name, alias="staging", version=version)
    logger.info(f"Promoted v{version} to Staging (alias=@staging)")

    info["promoted"] = True
    info["version"] = version
    return True, info


if __name__ == "__main__":
    cfg = load_config()
    success, info = retrain(cfg)
    print("\n=== Retrain summary ===")
    print(json.dumps(info, indent=2, default=str))
    sys.exit(0 if success else 1)
