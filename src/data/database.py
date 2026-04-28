"""
Database Module — SQLite for persistent storage.
Stores predictions and feedback for the feedback loop.
Guideline: Persist pipeline state and metadata.

Path resolution precedence:
    1. ``DB_PATH`` env var
    2. ``data.database_path`` from configs/config.yaml
"""

import os
import sqlite3
import logging
from typing import Optional, Dict

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CONFIG_PATH = "configs/config.yaml"


def _load_config_section(section: str) -> dict:
    """Read a config section from disk; returns {} if file is missing."""
    if not os.path.exists(_CONFIG_PATH):
        return {}
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f).get(section, {}) or {}


def get_db_path() -> str:
    """Resolve the SQLite path: env var → config → final fallback."""
    env = os.getenv("DB_PATH")
    if env:
        return env
    cfg_path = _load_config_section("data").get("database_path")
    return cfg_path or "data/fraud_detection.db"


def get_feedback_window() -> int:
    """Resolve the feedback window size from config."""
    return int(_load_config_section("monitoring").get("feedback_window", 100))


def get_connection():
    """Get SQLite connection."""
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT UNIQUE NOT NULL,
            prediction INTEGER NOT NULL,
            fraud_probability REAL NOT NULL,
            latency_ms REAL NOT NULL,
            features TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            predicted_label INTEGER NOT NULL,
            actual_label INTEGER NOT NULL,
            is_correct INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (transaction_id) REFERENCES predictions(transaction_id)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS drift_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drift_detected INTEGER NOT NULL,
            drifted_features_count INTEGER NOT NULL,
            drifted_features TEXT,
            report_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {get_db_path()}")


def save_prediction(
    transaction_id: str,
    prediction: int,
    fraud_probability: float,
    latency_ms: float,
    features: str = None,
):
    """Store a prediction."""
    conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO predictions
           (transaction_id, prediction, fraud_probability, latency_ms, features)
           VALUES (?, ?, ?, ?, ?)""",
        (transaction_id, prediction, fraud_probability, latency_ms, features),
    )
    conn.commit()
    conn.close()


def get_prediction(transaction_id: str) -> Optional[Dict]:
    """Retrieve a prediction by transaction ID."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM predictions WHERE transaction_id = ?", (transaction_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_feedback(transaction_id: str, predicted: int, actual: int):
    """Store feedback (ground truth)."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO feedback
           (transaction_id, predicted_label, actual_label, is_correct)
           VALUES (?, ?, ?, ?)""",
        (transaction_id, predicted, actual, int(predicted == actual)),
    )
    conn.commit()
    conn.close()


def get_model_accuracy(window: Optional[int] = None) -> Optional[float]:
    """Calculate accuracy from recent feedback.

    Args:
        window: How many recent feedback rows to evaluate. If None, falls back
            to ``monitoring.feedback_window`` from config.yaml (default 100).
    """
    if window is None:
        window = get_feedback_window()
    conn = get_connection()
    rows = conn.execute(
        "SELECT is_correct FROM feedback ORDER BY id DESC LIMIT ?", (window,)
    ).fetchall()
    conn.close()

    if len(rows) < 10:
        return None
    return sum(r["is_correct"] for r in rows) / len(rows)


def get_feedback_count() -> int:
    """Total feedback entries."""
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    conn.close()
    return count


def get_prediction_stats() -> Dict:
    """Get prediction statistics."""
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    fraud = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction = 1"
    ).fetchone()[0]
    avg_latency = conn.execute("SELECT AVG(latency_ms) FROM predictions").fetchone()[0]
    conn.close()
    return {
        "total_predictions": total,
        "fraud_count": fraud,
        "legit_count": total - fraud,
        "fraud_ratio": fraud / total if total > 0 else 0,
        "avg_latency_ms": round(avg_latency, 2) if avg_latency else 0,
    }


def save_drift_report(
    drift_detected: bool, drifted_count: int, drifted_features: str, report_json: str
):
    """Store drift detection report."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO drift_reports
           (drift_detected, drifted_features_count, drifted_features, report_json)
           VALUES (?, ?, ?, ?)""",
        (int(drift_detected), drifted_count, drifted_features, report_json),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database created at {get_db_path()}")
