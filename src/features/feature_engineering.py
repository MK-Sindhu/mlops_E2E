"""
Feature Engineering Module (Versioned Separately from Model Logic)
Creates new features and computes drift baselines.
Guideline: Version feature engineering logic separately from model logic.
Guideline: Calculate statistical baselines during EDA for drift detection.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature engineering version — bump this when logic changes
FEATURE_VERSION = "1.0.0"

# Schema version for the baseline JSON file (independent of FEATURE_VERSION)
BASELINE_SCHEMA_VERSION = "1.0.0"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones.

    This module is versioned independently from the model,
    so changes here can be tracked separately.

    Args:
        df: DataFrame with preprocessed data (no target column).

    Returns:
        DataFrame with additional engineered features.
    """
    logger.info(f"Running feature engineering v{FEATURE_VERSION}")
    df_feat = df.copy()

    # Feature 1: Transaction amount buckets (binned)
    if "Amount" in df_feat.columns:
        df_feat["Amount_Log"] = np.log1p(df_feat["Amount"].clip(lower=0))

    # Feature 2: Interaction between top PCA components
    if "V1" in df_feat.columns and "V2" in df_feat.columns:
        df_feat["V1_V2_interaction"] = df_feat["V1"] * df_feat["V2"]

    # Feature 3: Magnitude of PCA vector (V1-V28)
    v_cols = [c for c in df_feat.columns if c.startswith("V") and c[1:].isdigit()]
    if v_cols:
        df_feat["V_magnitude"] = np.sqrt((df_feat[v_cols] ** 2).sum(axis=1))

    # Feature 4: Mean and std of V features
    if v_cols:
        df_feat["V_mean"] = df_feat[v_cols].mean(axis=1)
        df_feat["V_std"] = df_feat[v_cols].std(axis=1)

    logger.info(f"Engineered {len(df_feat.columns) - len(df.columns)} new features")
    return df_feat


def _file_md5(path: str) -> str:
    """MD5 hash of a file's contents (chunked for large files)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_drift_baselines(
    df: pd.DataFrame,
    output_path: str,
    source_path: Optional[str] = None,
    filename: str = "feature_baselines.json",
) -> Dict:
    """
    Compute statistical baselines for drift detection.
    Saves mean, variance, and distribution info for each feature, plus
    provenance metadata so the drift detector can verify the baseline
    is still aligned with the data version it expects.

    Guideline: Calculate the statistical baseline (mean, variance, distribution)
    of features to be used later for drift detection.

    Args:
        df: Training DataFrame (features only, no target).
        output_path: Directory to save baselines JSON in.
        source_path: Optional path to the source data file used to compute
            baselines. If provided, its MD5 is stored as provenance.
        filename: Output JSON filename (matches ``data.baselines_filename``
            in configs/config.yaml).

    Returns:
        Dictionary of baseline statistics per feature (the inner ``features``
        block of the JSON, not the metadata wrapper).
    """
    logger.info("Computing drift baselines...")
    feature_stats = {}

    for col in df.columns:
        feature_stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "variance": float(df[col].var()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "q25": float(df[col].quantile(0.25)),
            "q75": float(df[col].quantile(0.75)),
        }

    meta = {
        "baseline_schema_version": BASELINE_SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_count": int(len(df)),
        "feature_count": len(feature_stats),
    }
    if source_path and os.path.exists(source_path):
        meta["source_data_path"] = source_path
        meta["source_data_md5"] = _file_md5(source_path)

    payload = {"_meta": meta, "features": feature_stats}

    os.makedirs(output_path, exist_ok=True)
    baseline_file = os.path.join(output_path, filename)
    with open(baseline_file, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        f"Saved baselines for {len(feature_stats)} features "
        f"(rows={meta['row_count']}, feature_version={FEATURE_VERSION}) "
        f"to {baseline_file}"
    )
    return feature_stats


def get_feature_names() -> list:
    """Return the list of final feature names after engineering."""
    base_features = ["Amount"] + [f"V{i}" for i in range(1, 29)]
    engineered = ["Amount_Log", "V1_V2_interaction", "V_magnitude", "V_mean", "V_std"]
    return base_features + engineered


if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load processed training data
    X_train = pd.read_csv(os.path.join(config["data"]["processed_path"], "X_train.csv"))

    # Engineer features
    X_train_feat = engineer_features(X_train)
    print(f"Features after engineering: {X_train_feat.shape[1]}")

    # Compute baselines
    baselines = compute_drift_baselines(
        X_train_feat,
        config["data"]["baselines_path"],
        filename=config["data"].get("baselines_filename", "feature_baselines.json"),
    )
    print(f"Baselines computed for {len(baselines)} features")
