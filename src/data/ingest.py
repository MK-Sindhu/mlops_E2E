"""
Data Ingestion Module
Downloads and loads the credit card fraud dataset.
Guideline: Version-control data-collection scripts and configurations.
"""
import os
import logging
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load raw credit card transaction data.
    
    The dataset should be downloaded from:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    and placed in data/raw/creditcard.csv
    
    Args:
        data_path: Path to the raw CSV file.
    
    Returns:
        DataFrame with raw transaction data.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please download it from Kaggle: "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in data/raw/creditcard.csv"
        )
    
    logger.info(f"Loading raw data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


if __name__ == "__main__":
    config = load_config()
    df = load_raw_data(config["data"]["raw_path"])
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud ratio: {df['Class'].mean():.4f}")
