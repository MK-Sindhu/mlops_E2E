"""DVC entrypoint: run the full preprocessing pipeline."""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ingest import load_config
from src.data.preprocess import run_preprocessing

config = load_config()
X_train, X_test, y_train, y_test, scaler = run_preprocessing(config)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train fraud ratio: {y_train.mean():.4f}")
print(f"Test fraud ratio: {y_test.mean():.4f}")
print(f"Scaler fit on {scaler.n_samples_seen_} rows (training only — no leakage)")
print("Saved to data/processed/")
