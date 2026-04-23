"""
Data Validation Module
Automated checks for schema consistency, missing values, and data quality.
Guideline: Implement automated checks for schema consistency and missing values during ingestion.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected schema for the credit card dataset
EXPECTED_SCHEMA = {
    "Time": "float64",
    "Amount": "float64",
    "Class": "int64",
}
# V1-V28 are all float64
for i in range(1, 29):
    EXPECTED_SCHEMA[f"V{i}"] = "float64"

EXPECTED_COLUMNS = list(EXPECTED_SCHEMA.keys())


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Check if DataFrame matches expected schema."""
    errors = []

    # Check missing columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # Check extra columns
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        errors.append(f"Unexpected columns: {extra_cols}")

    # Check data types
    for col, expected_dtype in EXPECTED_SCHEMA.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if actual_dtype != expected_dtype:
                errors.append(
                    f"Column '{col}': expected {expected_dtype}, got {actual_dtype}"
                )

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_missing_values(df: pd.DataFrame) -> Tuple[bool, Dict[str, float]]:
    """Check for missing values in each column."""
    missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
    has_missing = any(v > 0 for v in missing_pct.values())
    return not has_missing, {k: v for k, v in missing_pct.items() if v > 0}


def validate_target_column(df: pd.DataFrame, target_col: str = "Class") -> Tuple[bool, str]:
    """Validate the target column has expected values (0 and 1)."""
    unique_vals = set(df[target_col].unique())
    expected_vals = {0, 1}
    if unique_vals != expected_vals:
        return False, f"Target column has unexpected values: {unique_vals}"
    return True, "Target column is valid (binary: 0, 1)"


def validate_value_ranges(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Check for suspicious value ranges."""
    warnings = []

    if (df["Amount"] < 0).any():
        warnings.append("Negative transaction amounts found")

    if (df["Time"] < 0).any():
        warnings.append("Negative time values found")

    fraud_ratio = df["Class"].mean()
    if fraud_ratio > 0.5:
        warnings.append(f"Fraud ratio unusually high: {fraud_ratio:.4f}")

    return len(warnings) == 0, warnings


def run_all_validations(df: pd.DataFrame) -> Dict:
    """Run all validation checks and return a report."""
    logger.info("Running data validation checks...")

    report = {}

    # Schema validation
    schema_valid, schema_errors = validate_schema(df)
    report["schema"] = {"valid": schema_valid, "errors": schema_errors}
    logger.info(f"Schema validation: {'PASS' if schema_valid else 'FAIL'}")

    # Missing values
    no_missing, missing_cols = validate_missing_values(df)
    report["missing_values"] = {"valid": no_missing, "missing_columns": missing_cols}
    logger.info(f"Missing values check: {'PASS' if no_missing else 'FAIL'}")

    # Target column
    target_valid, target_msg = validate_target_column(df)
    report["target"] = {"valid": target_valid, "message": target_msg}
    logger.info(f"Target validation: {'PASS' if target_valid else 'FAIL'}")

    # Value ranges
    ranges_valid, range_warnings = validate_value_ranges(df)
    report["value_ranges"] = {"valid": ranges_valid, "warnings": range_warnings}
    logger.info(f"Value range check: {'PASS' if ranges_valid else 'WARN'}")

    # Overall
    all_valid = all([schema_valid, no_missing, target_valid])
    report["overall_valid"] = all_valid

    if not all_valid:
        logger.error("Data validation FAILED. Check report for details.")
    else:
        logger.info("All data validation checks PASSED.")

    return report


if __name__ == "__main__":
    from ingest import load_config, load_raw_data

    config = load_config()
    df = load_raw_data(config["data"]["raw_path"])
    report = run_all_validations(df)
    print(report)
