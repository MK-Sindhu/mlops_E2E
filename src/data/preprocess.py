"""
Data Preprocessing Module
Handles cleaning, transformation, and train/test splitting.
Guideline: Clean and transform the data. Handle missing values, outliers, and inconsistencies.
"""

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Clean raw data: drop unnecessary columns, handle missing values.

    Args:
        df: Raw DataFrame.
        config: Project configuration.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Cleaning data...")
    df_clean = df.copy()

    # Drop specified columns
    drop_cols = config["features"]["drop_columns"]
    df_clean = df_clean.drop(columns=drop_cols, errors="ignore")
    logger.info(f"Dropped columns: {drop_cols}")

    # Handle missing values (drop rows with any NaN)
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna()
    dropped_rows = initial_rows - len(df_clean)
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows with missing values")

    # Remove duplicate transactions
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dropped_dupes = initial_rows - len(df_clean)
    if dropped_dupes > 0:
        logger.info(f"Removed {dropped_dupes} duplicate rows")

    return df_clean


def scale_features(df: pd.DataFrame, config: dict, fit: bool = True, scaler=None):
    """
    Scale specified columns using StandardScaler.

    Args:
        df: DataFrame to scale.
        config: Project configuration.
        fit: Whether to fit the scaler (True for training, False for inference).
        scaler: Pre-fitted scaler (used when fit=False).

    Returns:
        Tuple of (scaled DataFrame, fitted scaler).
    """
    scale_cols = config["features"]["scale_columns"]
    df_scaled = df.copy()

    if fit:
        scaler = StandardScaler()
        df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
        logger.info(f"Fitted and applied scaler on columns: {scale_cols}")
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False")
        df_scaled[scale_cols] = scaler.transform(df_scaled[scale_cols])
        logger.info(f"Applied pre-fitted scaler on columns: {scale_cols}")

    return df_scaled, scaler


def split_data(df: pd.DataFrame, config: dict):
    """
    Split data into train and test sets.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    target_col = config["features"]["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,  # Maintain fraud ratio in both splits
    )

    logger.info(f"Train set: {len(X_train)} samples (fraud: {y_train.mean():.4f})")
    logger.info(f"Test set: {len(X_test)} samples (fraud: {y_test.mean():.4f})")

    return X_train, X_test, y_train, y_test


def run_preprocessing(config: dict):
    """Run the full preprocessing pipeline.

    Order matters: split BEFORE scaling so the StandardScaler is fit only on
    the training partition. Fitting on the full dataset would leak test-set
    statistics into training.
    """
    from src.data.ingest import load_raw_data

    # 1. Load
    df = load_raw_data(config["data"]["raw_path"])

    # 2. Clean (drop Time, NaN rows, duplicates)
    df_clean = clean_data(df, config)

    # 3. Split BEFORE scaling — prevents test-set leakage
    X_train, X_test, y_train, y_test = split_data(df_clean, config)

    # 4. Fit scaler on X_train only, then transform both partitions
    X_train_scaled, scaler = scale_features(X_train, config, fit=True)
    X_test_scaled, _ = scale_features(X_test, config, fit=False, scaler=scaler)

    # 5. Save
    output_dir = config["data"]["processed_path"]
    os.makedirs(output_dir, exist_ok=True)

    X_train_scaled.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

    logger.info(
        f"Saved processed data to {output_dir} "
        f"(scaler fit on {scaler.n_samples_seen_} training rows; no test leakage)"
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    config = load_config()
    run_preprocessing(config)
