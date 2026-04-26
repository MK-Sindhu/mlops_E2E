"""Unit tests for src/data/preprocess.py.

Coverage focus: cleaning correctness, scaler-leakage prevention, and
stratified-split integrity.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    clean_data,
    run_preprocessing,
    scale_features,
    split_data,
)


# --- Fixtures -----------------------------------------------------------


@pytest.fixture
def config():
    return {
        "data": {
            "raw_path": "data/raw/creditcard.csv",
            "processed_path": "data/processed/",
            "test_size": 0.2,
            "random_state": 42,
        },
        "features": {
            "drop_columns": ["Time"],
            "target_column": "Class",
            "scale_columns": ["Amount"],
        },
    }


def _make_df(n=500, seed=0):
    """Synthetic dataset matching the real schema (Time, V1-V28, Amount, Class)."""
    rng = np.random.default_rng(seed)
    data = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    data["Time"] = rng.integers(0, 100_000, n).astype(float)
    data["Amount"] = np.abs(rng.standard_normal(n) * 100)
    # Guarantee enough fraud rows for stratified splitting to work
    n_fraud = max(20, int(n * 0.05))
    labels = [1] * n_fraud + [0] * (n - n_fraud)
    rng.shuffle(labels)
    data["Class"] = labels
    return pd.DataFrame(data)


# --- clean_data ---------------------------------------------------------


class TestCleanData:

    def test_drops_time_column(self, config):
        df = _make_df()
        cleaned = clean_data(df, config)
        assert "Time" not in cleaned.columns
        assert "Amount" in cleaned.columns
        assert "Class" in cleaned.columns

    def test_drops_nan_rows(self, config):
        df = _make_df(n=100)
        df.loc[0, "V1"] = np.nan
        df.loc[5, "Amount"] = np.nan
        cleaned = clean_data(df, config)
        assert cleaned.isnull().sum().sum() == 0
        assert len(cleaned) <= len(df) - 2

    def test_removes_duplicates(self, config):
        df = _make_df(n=100)
        # Inject explicit duplicates
        df_dup = pd.concat([df, df.iloc[[0, 1, 2]]], ignore_index=True)
        cleaned = clean_data(df_dup, config)
        # Duplicates dropped, so result is at most original length
        assert len(cleaned) <= len(df)


# --- scale_features -----------------------------------------------------


class TestScaleFeatures:

    def test_fit_uses_training_statistics(self, config):
        """Scaler.mean_ must match the training data's mean — proves no extra rows leak in."""
        rng = np.random.default_rng(0)
        train = pd.DataFrame({
            "Amount": rng.standard_normal(200) * 50 + 100,
            "V1": rng.standard_normal(200),
        })
        _, scaler = scale_features(train, config, fit=True)
        assert np.isclose(scaler.mean_[0], train["Amount"].mean())

    def test_inference_reuses_fitted_scaler(self, config):
        """When fit=False, the same scaler instance is returned."""
        rng = np.random.default_rng(0)
        train = pd.DataFrame({
            "Amount": rng.standard_normal(100) * 50 + 100,
            "V1": rng.standard_normal(100),
        })
        test = pd.DataFrame({
            "Amount": rng.standard_normal(40) * 50 + 100,
            "V1": rng.standard_normal(40),
        })
        _, scaler = scale_features(train, config, fit=True)
        _, scaler2 = scale_features(test, config, fit=False, scaler=scaler)
        assert scaler2 is scaler

    def test_fit_false_without_scaler_raises(self, config):
        df = pd.DataFrame({"Amount": [1.0, 2.0], "V1": [0.0, 0.0]})
        with pytest.raises(ValueError, match="Scaler must be provided"):
            scale_features(df, config, fit=False, scaler=None)


# --- split_data ---------------------------------------------------------


class TestSplitData:

    def test_test_size_matches_config(self, config):
        df = _make_df(n=1000)
        X_train, X_test, y_train, y_test = split_data(df, config)
        actual_ratio = len(X_test) / (len(X_train) + len(X_test))
        assert actual_ratio == pytest.approx(0.2, abs=0.01)

    def test_stratified_preserves_class_balance(self, config):
        """Fraud ratio in train and test must be near-identical (stratify=y)."""
        df = _make_df(n=2000)
        X_train, X_test, y_train, y_test = split_data(df, config)
        assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.005)

    def test_train_test_index_disjoint(self, config):
        """No row may appear in both partitions."""
        df = _make_df(n=500)
        X_train, X_test, _, _ = split_data(df.reset_index(drop=True), config)
        assert len(set(X_train.index) & set(X_test.index)) == 0

    def test_target_column_excluded_from_X(self, config):
        """X partitions must not contain the target column."""
        df = _make_df(n=200)
        X_train, X_test, _, _ = split_data(df, config)
        assert "Class" not in X_train.columns
        assert "Class" not in X_test.columns


# --- end-to-end leakage guard ------------------------------------------


class TestNoLeakage:

    def test_scaler_fit_on_training_partition_only(self, config, tmp_path):
        """The scaler returned by run_preprocessing must be fit on X_train rows only,
        never on the full cleaned dataset. This is the regression guard for the
        Phase 7 leakage fix."""
        df = _make_df(n=1000)
        raw_path = tmp_path / "raw.csv"
        df.to_csv(raw_path, index=False)

        config = {
            **config,
            "data": {
                **config["data"],
                "raw_path": str(raw_path),
                "processed_path": str(tmp_path / "out"),
            },
        }

        X_train, X_test, _, _, scaler = run_preprocessing(config)

        # Scaler should have been fit on exactly the training partition
        assert scaler.n_samples_seen_ == len(X_train)
        # And NOT on the full clean dataset
        df_clean = clean_data(df, config)
        assert scaler.n_samples_seen_ < len(df_clean)
