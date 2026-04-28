"""Run a sweep of training experiments comparing model + data variants.

Each variant becomes a separate MLflow run and a new version in the
``fraud-detection-xgboost`` registered model. Results are collected into
``reports/experiment_comparison.json`` for documentation.

Variants exercised:
    * Model knobs       : quantization on/off, ``max_bin``, ``max_depth``,
                          ``n_estimators``, ``learning_rate``.
    * Data version       : ``data.test_size`` — when this changes we
                          re-run preprocessing+feature_engineering before
                          training so the resulting splits actually use
                          the new ratio. (The default sweep keeps the
                          current splits to stay fast.)

Run from the project root:

    python scripts/run_experiments.py
"""
import json
import os
import subprocess
import sys
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train import load_config, train_and_log


# Each entry is (name, overrides). Overrides is a sparse dict that gets
# deep-merged onto the base config (configs/config.yaml). When a top-level
# ``data`` key appears in overrides, preprocessing+feature_engineering are
# re-run automatically.
EXPERIMENTS = [
    # 1) Baseline — no quantization, XGBoost defaults
    ("baseline_no_quantize", {
        "model": {"optimization": {"quantize": False, "max_bin": 256, "n_jobs": -1}},
    }),
    # 2) Quantized — max_bin=128 (the default in config.yaml)
    ("quantized_max_bin_128", {
        "model": {"optimization": {"quantize": True, "max_bin": 128, "n_jobs": -1}},
    }),
    # 3) Aggressively quantized — max_bin=64
    ("quantized_max_bin_64", {
        "model": {"optimization": {"quantize": True, "max_bin": 64, "n_jobs": -1}},
    }),
    # 4) Different model: shallower trees, more of them
    ("shallow_more_trees", {
        "model": {"params": {"max_depth": 4, "n_estimators": 200}},
    }),
    # 5) Different model: deeper trees, slower learning rate
    ("deep_slow_lr", {
        "model": {"params": {"max_depth": 10, "learning_rate": 0.05}},
    }),
    # 6) Different DATA version: 30% test split (vs 20% default).
    #    This re-runs preprocessing+feature_engineering before training so
    #    the X_train / X_test files actually reflect the new ratio.
    ("data_split_30pct", {
        "data": {"test_size": 0.3},
    }),
]


def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into a copy of ``base``."""
    out = deepcopy(base)
    for key, value in overrides.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def reprocess_for_data_change(cfg: dict) -> None:
    """Run the data pipeline so X_train/X_test reflect ``cfg['data']``.

    We invoke the existing scripts as subprocesses so they pick up
    *configs/config.yaml*. Caller is responsible for restoring the original
    config afterwards.
    """
    # Persist the new config to disk so the run_*.py scripts read it
    import yaml  # local import — only needed for the data path
    with open("configs/config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("  -> re-running preprocess + feature_engineering for new data split")
    for script in ("scripts/run_preprocessing.py", "scripts/run_feature_engineering.py"):
        subprocess.check_call([sys.executable, script])


def main():
    base = load_config()
    # Snapshot the on-disk config so we can restore it at the end if any
    # variant rewrote it (the data-version path does).
    with open("configs/config.yaml", "r") as f:
        original_yaml = f.read()

    results = []
    try:
        for name, overrides in EXPERIMENTS:
            cfg = deep_merge(base, overrides)
            print(f"\n{'='*70}\nExperiment: {name}\nOverrides: {overrides}\n{'='*70}")

            if "data" in overrides:
                reprocess_for_data_change(cfg)

            run_id, metrics, sizes = train_and_log(cfg, run_name=name)

            results.append({
                "name": name,
                "run_id": run_id,
                "overrides": overrides,
                "f1_score": metrics["f1_score"],
                "pr_auc": metrics["pr_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "avg_inference_latency_ms": metrics["avg_inference_latency_ms"],
                "model_size_bytes_joblib": sizes["model_size_bytes_joblib"],
                "model_size_bytes_json":   sizes["model_size_bytes_json"],
            })
    finally:
        # Always restore the original config (the data-version path overwrites it).
        with open("configs/config.yaml", "w") as f:
            f.write(original_yaml)
        print("\nOriginal configs/config.yaml restored.")

    os.makedirs("reports", exist_ok=True)
    out_path = "reports/experiment_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 96)
    print("EXPERIMENT COMPARISON")
    print("=" * 96)
    print(f"{'name':<24} {'F1':>7} {'PR-AUC':>8} {'precision':>10} "
          f"{'recall':>7} {'lat(ms)':>9} {'size(KB)':>10}")
    print("-" * 96)
    for r in results:
        print(
            f"{r['name']:<24} "
            f"{r['f1_score']:>7.4f} {r['pr_auc']:>8.4f} "
            f"{r['precision']:>10.4f} {r['recall']:>7.4f} "
            f"{r['avg_inference_latency_ms']:>9.2f} "
            f"{r['model_size_bytes_joblib']/1024:>10.1f}"
        )
    print("=" * 96)
    print(f"\nFull results saved to: {out_path}")
    print("Each run is registered in MLflow at http://localhost:5000 "
          "as a new version of 'fraud-detection-xgboost'.")


if __name__ == "__main__":
    main()
