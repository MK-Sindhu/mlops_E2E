"""
Unit Tests
Guideline: Implement unit, integration, and end-to-end tests.
"""

import pandas as pd
import numpy as np

from src.data.validate import (
    validate_schema,
    validate_missing_values,
    validate_target_column,
    EXPECTED_COLUMNS,
)
from src.features.feature_engineering import engineer_features, FEATURE_VERSION


# ── Data Validation Tests ────────────────────────────────────────────


class TestDataValidation:

    def _make_valid_df(self, n=100):
        """Create a valid dummy DataFrame matching expected schema."""
        data = {col: np.random.randn(n) for col in EXPECTED_COLUMNS if col != "Class"}
        labels = [0] * (n - 2) + [0, 1]  # Guarantee both classes exist
        np.random.shuffle(labels)
        data["Class"] = labels
        data["Time"] = np.abs(data.get("Time", np.random.randn(n)))
        data["Amount"] = np.abs(np.random.randn(n) * 100)
        return pd.DataFrame(data)

    def test_valid_schema(self):
        df = self._make_valid_df()
        is_valid, errors = validate_schema(df)
        assert is_valid, f"Valid data should pass schema check: {errors}"

    def test_missing_column(self):
        df = self._make_valid_df()
        df = df.drop(columns=["V1"])
        is_valid, errors = validate_schema(df)
        assert not is_valid
        assert any("V1" in str(e) for e in errors)

    def test_no_missing_values(self):
        df = self._make_valid_df()
        no_missing, _ = validate_missing_values(df)
        assert no_missing

    def test_with_missing_values(self):
        df = self._make_valid_df()
        df.loc[0, "V1"] = np.nan
        no_missing, missing_cols = validate_missing_values(df)
        assert not no_missing
        assert "V1" in missing_cols

    def test_valid_target(self):
        df = self._make_valid_df()
        is_valid, _ = validate_target_column(df)
        assert is_valid

    def test_invalid_target(self):
        df = self._make_valid_df()
        df["Class"] = 5
        is_valid, _ = validate_target_column(df)
        assert not is_valid


# ── Feature Engineering Tests ────────────────────────────────────────


class TestFeatureEngineering:

    def _make_feature_df(self, n=50):
        data = {"Amount": np.abs(np.random.randn(n) * 100)}
        for i in range(1, 29):
            data[f"V{i}"] = np.random.randn(n)
        return pd.DataFrame(data)

    def test_feature_version_exists(self):
        assert FEATURE_VERSION is not None

    def test_new_features_created(self):
        df = self._make_feature_df()
        df_feat = engineer_features(df)
        assert "Amount_Log" in df_feat.columns
        assert "V1_V2_interaction" in df_feat.columns
        assert "V_magnitude" in df_feat.columns
        assert "V_mean" in df_feat.columns
        assert "V_std" in df_feat.columns

    def test_no_nans_after_engineering(self):
        df = self._make_feature_df()
        df_feat = engineer_features(df)
        assert df_feat.isnull().sum().sum() == 0

    def test_output_shape(self):
        df = self._make_feature_df()
        original_cols = len(df.columns)
        df_feat = engineer_features(df)
        assert len(df_feat.columns) > original_cols
