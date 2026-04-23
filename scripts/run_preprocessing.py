"""Run the full preprocessing pipeline: clean, scale, split, save."""
import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ingest import load_config, load_raw_data
from src.data.preprocess import clean_data, scale_features, split_data

config = load_config()
df = load_raw_data(config["data"]["raw_path"])

# Clean → Scale → Split
df_clean = clean_data(df, config)
df_scaled, scaler = scale_features(df_clean, config, fit=True)
X_train, X_test, y_train, y_test = split_data(df_scaled, config)

# Save
out = config["data"]["processed_path"]
os.makedirs(out, exist_ok=True)
X_train.to_csv(f"{out}/X_train.csv", index=False)
X_test.to_csv(f"{out}/X_test.csv", index=False)
y_train.to_csv(f"{out}/y_train.csv", index=False)
y_test.to_csv(f"{out}/y_test.csv", index=False)
joblib.dump(scaler, f"{out}/scaler.joblib")

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train fraud ratio: {y_train.mean():.4f}")
print(f"Test fraud ratio: {y_test.mean():.4f}")
print("Saved to data/processed/")
