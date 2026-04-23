"""
Feature Engineering Module (Versioned Separately from Model Logic)
Creates new features and computes drift baselines.
Guideline: Version feature engineering logic separately from model logic.
Guideline: Calculate statistical baselines during EDA for drift detection.
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature engineering version — bump this when logic changes
FEATURE_VERSION = "1.0.0"


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


def compute_drift_baselines(df: pd.DataFrame, output_path: str) -> Dict:
    """
    Compute statistical baselines for drift detection.
    Saves mean, variance, and distribution info for each feature.
    
    Guideline: Calculate the statistical baseline (mean, variance, distribution)
    of features to be used later for drift detection.
    
    Args:
        df: Training DataFrame (features only, no target).
        output_path: Path to save baselines JSON.
    
    Returns:
        Dictionary of baseline statistics per feature.
    """
    logger.info("Computing drift baselines...")
    baselines = {}

    for col in df.columns:
        stats = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "variance": float(df[col].var()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "q25": float(df[col].quantile(0.25)),
            "q75": float(df[col].quantile(0.75)),
        }
        baselines[col] = stats

    # Save baselines
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    baseline_file = os.path.join(output_path, "feature_baselines.json")
    with open(baseline_file, "w") as f:
        json.dump(baselines, f, indent=2)

    logger.info(f"Saved baselines for {len(baselines)} features to {baseline_file}")
    return baselines


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
    baselines = compute_drift_baselines(X_train_feat, config["data"]["baselines_path"])
    print(f"Baselines computed for {len(baselines)} features")
