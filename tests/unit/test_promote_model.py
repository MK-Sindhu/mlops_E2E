"""Unit tests for the model-promotion CLI helpers.

The MlflowClient is mocked so these tests don't touch a real registry or
filesystem.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# scripts/ isn't a package; load promote_model.py via importlib
import importlib.util
import pathlib

PROMOTE_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "promote_model.py"
)
spec = importlib.util.spec_from_file_location("promote_model", PROMOTE_PATH)
promote_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promote_model)


# --- helpers ----------------------------------------------------------


def _mv(version, stage="None", run_id=None, aliases=None):
    """Build a fake ModelVersion-shaped object."""
    return SimpleNamespace(
        version=str(version),
        current_stage=stage,
        run_id=run_id or f"run_{version:032d}",
        aliases=aliases or [],
    )


# --- list_versions ----------------------------------------------------


def test_list_versions_sorted_by_version_number():
    client = MagicMock()
    # Mock returns versions out of order — should come back sorted
    client.search_model_versions.return_value = [_mv(7), _mv(3), _mv(1), _mv(5)]
    result = promote_model.list_versions(client, "fraud-detection-xgboost")
    assert [v.version for v in result] == ["1", "3", "5", "7"]


def test_list_versions_passes_correct_filter():
    client = MagicMock()
    client.search_model_versions.return_value = []
    promote_model.list_versions(client, "my-model")
    client.search_model_versions.assert_called_once_with("name='my-model'")


# --- get_metric -------------------------------------------------------


def test_get_metric_returns_value_when_present():
    client = MagicMock()
    client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(metrics={"pr_auc": 0.82})
    )
    assert promote_model.get_metric(client, "run_x", "pr_auc") == 0.82


def test_get_metric_returns_none_when_missing():
    client = MagicMock()
    client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(metrics={})
    )
    assert promote_model.get_metric(client, "run_x", "pr_auc") is None


def test_get_metric_returns_none_on_exception():
    client = MagicMock()
    client.get_run.side_effect = RuntimeError("boom")
    assert promote_model.get_metric(client, "run_x", "pr_auc") is None


# --- find_best_version -----------------------------------------------


def test_find_best_version_picks_highest_metric():
    """Among versions, the one with the highest target metric wins."""
    client = MagicMock()
    client.search_model_versions.return_value = [
        _mv(5, run_id="r5"),
        _mv(6, run_id="r6"),
        _mv(7, run_id="r7"),
    ]
    metric_table = {"r5": 0.81, "r6": 0.82, "r7": 0.79}
    client.get_run.side_effect = lambda rid: SimpleNamespace(
        data=SimpleNamespace(metrics={"pr_auc": metric_table[rid]})
    )
    best, score = promote_model.find_best_version(client, "m", "pr_auc")
    assert best.version == "6"
    assert score == 0.82


def test_find_best_version_ignores_versions_missing_the_metric():
    """A version whose run never logged the metric is skipped."""
    client = MagicMock()
    client.search_model_versions.return_value = [
        _mv(1, run_id="r1"),
        _mv(2, run_id="r2"),
    ]
    metric_table = {"r1": None, "r2": 0.5}
    client.get_run.side_effect = lambda rid: SimpleNamespace(
        data=SimpleNamespace(metrics={"pr_auc": metric_table[rid]} if metric_table[rid] is not None else {})
    )
    best, score = promote_model.find_best_version(client, "m", "pr_auc")
    assert best.version == "2"
    assert score == 0.5


def test_find_best_version_returns_none_when_no_metric_found():
    client = MagicMock()
    client.search_model_versions.return_value = [_mv(1)]
    client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(metrics={})
    )
    best, score = promote_model.find_best_version(client, "m", "pr_auc")
    assert best is None
    assert score == float("-inf")
