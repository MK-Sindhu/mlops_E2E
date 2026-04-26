"""Unit tests for the training builder + size metrics. Does NOT actually train."""
import pytest

from src.models.train import build_model, compute_model_size_metrics


def _base_config():
    return {
        "model": {
            "algorithm": "xgboost",
            "params": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "scale_pos_weight": 1,
                "eval_metric": "logloss",
                "random_state": 0,
            },
            "optimization": {
                "quantize": False,
                "max_bin": 256,
                "n_jobs": 1,
            },
        }
    }


# --- build_model: quantization toggle ----------------------------------


def test_build_model_quantize_true_sets_tree_method_hist():
    cfg = _base_config()
    cfg["model"]["optimization"] = {"quantize": True, "max_bin": 128, "n_jobs": 1}
    model = build_model(cfg)
    assert model.get_params()["tree_method"] == "hist"


def test_build_model_quantize_true_applies_max_bin_from_config():
    cfg = _base_config()
    cfg["model"]["optimization"] = {"quantize": True, "max_bin": 64, "n_jobs": 1}
    model = build_model(cfg)
    assert model.get_params()["max_bin"] == 64


def test_build_model_quantize_false_leaves_xgb_defaults_in_place():
    """When quantize=False, build_model must not pass tree_method/max_bin to XGBoost.

    XGBClassifier returns None from get_params() for any kwarg that wasn't
    explicitly set at construction time.
    """
    cfg = _base_config()
    cfg["model"]["optimization"] = {"quantize": False, "n_jobs": 1}
    model = build_model(cfg)
    assert model.get_params().get("max_bin") is None
    assert model.get_params().get("tree_method") is None


# --- build_model: hyperparameter pass-through --------------------------


def test_build_model_passes_all_hyperparameters():
    cfg = _base_config()
    cfg["model"]["params"] = {
        "n_estimators": 7,
        "max_depth": 4,
        "learning_rate": 0.5,
        "scale_pos_weight": 99,
        "eval_metric": "auc",
        "random_state": 42,
    }
    model = build_model(cfg)
    p = model.get_params()
    assert p["n_estimators"] == 7
    assert p["max_depth"] == 4
    assert p["learning_rate"] == 0.5
    assert p["scale_pos_weight"] == 99
    assert p["random_state"] == 42


# --- compute_model_size_metrics ----------------------------------------


def test_compute_model_size_metrics_reports_actual_bytes(tmp_path):
    joblib_path = tmp_path / "m.joblib"
    json_path = tmp_path / "m.json"
    joblib_path.write_bytes(b"x" * 12345)
    json_path.write_bytes(b"y" * 6789)
    metrics = compute_model_size_metrics({
        "joblib": str(joblib_path),
        "json": str(json_path),
    })
    assert metrics["model_size_bytes_joblib"] == 12345
    assert metrics["model_size_bytes_json"] == 6789
