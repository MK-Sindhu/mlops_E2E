"""Run feature engineering and compute drift baselines."""
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.feature_engineering import engineer_features, compute_drift_baselines
from src.data.ingest import load_config

config = load_config()

X_train_path = os.path.join(config["data"]["processed_path"], "X_train.csv")
X_train = pd.read_csv(X_train_path)
X_train_feat = engineer_features(X_train)

baselines = compute_drift_baselines(
    X_train_feat,
    config["data"]["baselines_path"],
    source_path=X_train_path,
)

print(f"Original features: {X_train.shape[1]}")
print(f"After engineering: {X_train_feat.shape[1]}")
print(f"Baselines saved for {len(baselines)} features")
