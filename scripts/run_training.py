"""DVC entrypoint: run training, log to MLflow, register the model."""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train import load_config, train_and_log

config = load_config()
run_id, metrics, sizes = train_and_log(config, run_name="dvc-pipeline")

print(f"Run ID:        {run_id}")
print(f"F1:            {metrics['f1_score']:.4f}")
print(f"PR-AUC:        {metrics['pr_auc']:.4f}")
print(f"Precision:     {metrics['precision']:.4f}")
print(f"Recall:        {metrics['recall']:.4f}")
print(f"Latency:       {metrics['avg_inference_latency_ms']}ms")
print(f"Size (joblib): {sizes['model_size_bytes_joblib']:,} bytes")
print(f"Size (json):   {sizes['model_size_bytes_json']:,} bytes")
