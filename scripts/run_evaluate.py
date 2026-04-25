"""Evaluate the trained model: write metrics, confusion matrix, and ROC curve."""
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.feature_engineering import engineer_features

PROCESSED = "data/processed"
REPORTS = "reports"
os.makedirs(REPORTS, exist_ok=True)

model = joblib.load("models/best_model.joblib")
X_test = engineer_features(pd.read_csv(f"{PROCESSED}/X_test.csv"))
y_test = pd.read_csv(f"{PROCESSED}/y_test.csv").squeeze()

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "f1_score": float(f1_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "pr_auc": float(average_precision_score(y_test, y_proba)),
}
with open(f"{REPORTS}/eval_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

cm_records = [
    {"actual": int(a), "predicted": int(p)}
    for a, p in zip(y_test.to_numpy(), y_pred)
]
with open(f"{REPORTS}/confusion_matrix.json", "w") as f:
    json.dump(cm_records, f)

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_records = [{"fpr": float(x), "tpr": float(y)} for x, y in zip(fpr, tpr)]
with open(f"{REPORTS}/roc_curve.json", "w") as f:
    json.dump(roc_records, f)

print(f"F1: {metrics['f1_score']:.4f}  ROC-AUC: {metrics['roc_auc']:.4f}  PR-AUC: {metrics['pr_auc']:.4f}")
print(f"Wrote {REPORTS}/eval_metrics.json, confusion_matrix.json, roc_curve.json")
