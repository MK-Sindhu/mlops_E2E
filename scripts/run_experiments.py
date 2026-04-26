"""Run a sweep of training experiments comparing quantization tradeoffs.

Three runs are produced — each becomes a separate MLflow run *and* a new
version in the ``fraud-detection-xgboost`` registered model. Results are
collected into ``reports/experiment_comparison.json`` for documentation.
"""
import json
import os
import sys
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train import load_config, train_and_log


EXPERIMENTS = [
    ("baseline_no_quantize",    {"quantize": False, "max_bin": 256, "n_jobs": -1}),
    ("quantized_max_bin_128",   {"quantize": True,  "max_bin": 128, "n_jobs": -1}),
    ("quantized_max_bin_64",    {"quantize": True,  "max_bin": 64,  "n_jobs": -1}),
]


def main():
    base = load_config()
    results = []

    for name, opt in EXPERIMENTS:
        cfg = deepcopy(base)
        cfg["model"]["optimization"] = opt
        print(f"\n{'='*70}\nExperiment: {name}\nOptimization: {opt}\n{'='*70}")
        run_id, metrics, sizes = train_and_log(cfg, run_name=name)
        results.append({
            "name": name,
            "run_id": run_id,
            "quantize": opt["quantize"],
            "max_bin": opt["max_bin"],
            "f1_score": metrics["f1_score"],
            "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "avg_inference_latency_ms": metrics["avg_inference_latency_ms"],
            "model_size_bytes_joblib": sizes["model_size_bytes_joblib"],
            "model_size_bytes_json":   sizes["model_size_bytes_json"],
        })

    os.makedirs("reports", exist_ok=True)
    out_path = "reports/experiment_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print("EXPERIMENT COMPARISON")
    print("=" * 78)
    print(f"{'name':<26} {'quantize':>9} {'max_bin':>8} "
          f"{'F1':>7} {'PR-AUC':>8} {'lat(ms)':>9} {'size(KB)':>10}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['name']:<26} "
            f"{str(r['quantize']):>9} {r['max_bin']:>8} "
            f"{r['f1_score']:>7.4f} {r['pr_auc']:>8.4f} "
            f"{r['avg_inference_latency_ms']:>9.2f} "
            f"{r['model_size_bytes_joblib']/1024:>10.1f}"
        )
    print("=" * 78)
    print(f"\nFull results saved to: {out_path}")
    print("Each run is also registered as a new version of "
          "'fraud-detection-xgboost' in the MLflow Model Registry.")


if __name__ == "__main__":
    main()
