"""
Data Drift Detection Module
Compares incoming data distribution against training baselines using KS-test.
Guideline: Monitor for changes in input data distribution.
Guideline: Configure alerts if data drift is detected.

Defaults (baselines path, KS p-value threshold, z-score threshold) come from
``configs/config.yaml`` so callers don't have to thread config explicitly.
"""

import json
import logging
import os
from typing import Dict, Optional

import pandas as pd
import yaml
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CONFIG_PATH = "configs/config.yaml"


def _load_config() -> dict:
    """Read configs/config.yaml; returns {} if file is missing."""
    if not os.path.exists(_CONFIG_PATH):
        return {}
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


def _default_baselines_path() -> str:
    """Compose ``data.baselines_path`` + ``data.baselines_filename`` from config."""
    cfg = _load_config()
    data = cfg.get("data", {}) or {}
    base = data.get("baselines_path", "data/baselines/")
    name = data.get("baselines_filename", "feature_baselines.json")
    return os.path.join(base, name)


def _default_drift_threshold() -> float:
    return float(_load_config().get("monitoring", {}).get("drift_threshold", 0.05))


def _default_z_threshold() -> float:
    return float(_load_config().get("monitoring", {}).get("drift_z_threshold", 3.0))


def load_baselines(baselines_path: Optional[str] = None) -> Dict:
    """Load the per-feature baseline statistics dict."""
    path = baselines_path or _default_baselines_path()
    with open(path, "r") as f:
        return json.load(f)["features"]


def load_baseline_meta(baselines_path: Optional[str] = None) -> Dict:
    """Load baseline provenance metadata (feature_version, source_data_md5, etc.)."""
    path = baselines_path or _default_baselines_path()
    with open(path, "r") as f:
        return json.load(f)["_meta"]


def detect_drift_ks_test(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    threshold: Optional[float] = None,
) -> Dict:
    """
    Detect data drift using the Kolmogorov-Smirnov test.
    Compares distributions of each feature between reference and current data.

    Args:
        reference_data: Training data (or a sample of it).
        current_data: New incoming data.
        threshold: p-value threshold; below this means drift detected.

    Returns:
        Dictionary with drift results per feature.
    """
    if threshold is None:
        threshold = _default_drift_threshold()

    drift_report = {}
    drifted_features = []

    common_cols = list(set(reference_data.columns) & set(current_data.columns))

    for col in common_cols:
        ks_stat, p_value = stats.ks_2samp(
            reference_data[col].dropna(), current_data[col].dropna()
        )
        # Cast scipy / numpy scalars to plain Python types so the report
        # is JSON-serialisable end-to-end (callers persist it via json.dumps).
        is_drifted = bool(p_value < threshold)
        drift_report[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "drifted": is_drifted,
        }
        if is_drifted:
            drifted_features.append(col)

    drift_summary = {
        "total_features": len(common_cols),
        "drifted_features_count": len(drifted_features),
        "drifted_features": drifted_features,
        "drift_detected": bool(len(drifted_features) > 0),
        "feature_details": drift_report,
    }

    if drifted_features:
        logger.warning(
            f"DRIFT DETECTED in {len(drifted_features)} features: {drifted_features}"
        )
    else:
        logger.info("No drift detected.")

    return drift_summary


def detect_drift_from_baselines(
    current_data: pd.DataFrame,
    baselines: Dict,
    z_threshold: Optional[float] = None,
) -> Dict:
    """
    Quick drift check using stored baselines (mean/std).
    Flags features where current mean is more than z_threshold standard deviations
    away from the training mean.

    Args:
        current_data: New incoming data.
        baselines: Stored baselines from training.
        z_threshold: Number of std deviations to flag.

    Returns:
        Dictionary with drift results.
    """
    if z_threshold is None:
        z_threshold = _default_z_threshold()

    drift_report = {}
    drifted = []

    for col in current_data.columns:
        if col not in baselines:
            continue

        baseline = baselines[col]
        current_mean = float(current_data[col].mean())
        training_mean = baseline["mean"]
        training_std = baseline["std"]

        if training_std == 0:
            z_score = 0.0
        else:
            z_score = abs(current_mean - training_mean) / training_std

        is_drifted = bool(z_score > z_threshold)
        drift_report[col] = {
            "training_mean": round(training_mean, 4),
            "current_mean": round(current_mean, 4),
            "z_score": round(z_score, 4),
            "drifted": is_drifted,
        }
        if is_drifted:
            drifted.append(col)

    return {
        "total_features": len(drift_report),
        "drifted_features_count": len(drifted),
        "drifted_features": drifted,
        "drift_detected": bool(len(drifted) > 0),
        "feature_details": drift_report,
    }


if __name__ == "__main__":
    # Example usage
    baselines = load_baselines()
    meta = load_baseline_meta()
    print(f"Loaded baselines for {len(baselines)} features")
    print(f"Baseline metadata: {meta}")
