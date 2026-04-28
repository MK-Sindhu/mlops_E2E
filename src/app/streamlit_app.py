"""
Streamlit Demo UI — Credit Card Fraud Detection
Interactive web UI for making predictions, viewing stats, and monitoring.
"""

import json
import os
import sys

# `streamlit run` only adds the script's own directory to sys.path,
# so `from src.*` imports below would fail. Add the project root explicitly.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402


def _load_config():
    """Read configs/config.yaml; returns {} if absent (e.g. unit-test layouts)."""
    path = os.path.join(_PROJECT_ROOT, "configs", "config.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


_CFG = _load_config()
_API_CFG = _CFG.get("api", {}) or {}
_DATA_CFG = _CFG.get("data", {}) or {}

# API base URL: env wins, then config, then localhost fallback
API_URL = os.getenv("API_URL", _API_CFG.get("url", "http://localhost:8000"))
HEALTH_TIMEOUT_S = int(_API_CFG.get("client_timeout_seconds", 3))
EXPLAIN_TIMEOUT_S = int(_API_CFG.get("explain_timeout_seconds", 15))

# Pipeline-Status page dials these tools' UIs. Defaults assume docker-compose
# service DNS; override with env vars to point at remote/Swarm endpoints.
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow:5000")
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow:8080")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")


def _airflow_password():
    """Resolve Airflow admin password.

    Order: AIRFLOW_PASSWORD env > AIRFLOW_PASSWORD_FILE (docker secret) > 'admin'.
    """
    if os.getenv("AIRFLOW_PASSWORD"):
        return os.environ["AIRFLOW_PASSWORD"]
    pwd_file = os.getenv("AIRFLOW_PASSWORD_FILE")
    if pwd_file and os.path.exists(pwd_file):
        with open(pwd_file) as f:
            return f.read().strip()
    return "admin"


AIRFLOW_PASSWORD = _airflow_password()
FRAUD_STATS_PATH = os.path.join(
    _DATA_CFG.get("external_path", "data/external/"),
    _DATA_CFG.get("fraud_stats_filename", "fraud_stats.json"),
)

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")
st.title("🔍 Credit Card Fraud Detection")
st.markdown("Real-time fraud detection powered by XGBoost + MLOps pipeline")

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Predict",
        "Batch Predict",
        "Feedback",
        "Dashboard",
        "Pipeline Status",
        "Threat Landscape",
        "About",
    ],
)


# ── Helper ───────────────────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=HEALTH_TIMEOUT_S)
        return r.status_code == 200
    except Exception:
        return False


api_healthy = check_api()
if api_healthy:
    st.sidebar.success("API: Connected")
else:
    st.sidebar.error(
        "API: Not running. Start with `uvicorn src.api.app:app --port 8000`"
    )


def show_explanation(txn_id, top_k=8):
    """Fetch and display the SHAP explanation for a prediction.

    Best-effort — silently skips if /explain isn't reachable.
    """
    try:
        resp = requests.get(
            f"{API_URL}/explain",
            params={"transaction_id": txn_id, "top_k": top_k},
            timeout=EXPLAIN_TIMEOUT_S,
        )
        if resp.status_code != 200:
            return
        data = resp.json()
        st.markdown("**Why this prediction (SHAP)**")
        st.caption(
            f"Base value: {data['base_value']:.4f} • "
            f"top {len(data['top_contributions'])} contributions by |shap_value|"
        )
        contrib_df = pd.DataFrame(data["top_contributions"]).set_index("feature")
        st.bar_chart(contrib_df["shap_value"])
    except Exception:
        pass


# ── Page: Predict ────────────────────────────────────────────────────
if page == "Predict":
    st.header("Single Transaction Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Option 1: Upload CSV Row")
        uploaded = st.file_uploader("Upload a CSV with one transaction", type=["csv"])

        if uploaded and api_healthy:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head())

            if st.button("Predict (CSV)"):
                from src.features.feature_engineering import engineer_features

                if "Class" in df.columns:
                    df = df.drop(columns=["Class"])
                if "Time" in df.columns:
                    df = df.drop(columns=["Time"])
                df_feat = engineer_features(df)
                features = df_feat.values[0].tolist()
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "features": features,
                        "transaction_id": f"ui_{int(pd.Timestamp.now().timestamp())}",
                    },
                )
                result = resp.json()
                if result["prediction"] == 1:
                    st.error(
                        f"🚨 FRAUD DETECTED (probability: {result['fraud_probability']:.4f})"
                    )
                else:
                    st.success(
                        f"✅ Legitimate (probability: {result['fraud_probability']:.4f})"
                    )
                st.json(result)
                show_explanation(result["transaction_id"])

    with col2:
        st.subheader("Option 2: Random Test Sample")
        if st.button("Generate & Predict Random Sample") and api_healthy:
            x_test_path = os.path.join(
                _DATA_CFG.get("processed_path", "data/processed/"), "X_test.csv"
            )
            X_test = pd.read_csv(x_test_path)
            from src.features.feature_engineering import engineer_features

            idx = np.random.randint(0, len(X_test))
            sample = X_test.iloc[idx]
            sample_feat = engineer_features(pd.DataFrame([sample]))
            features = sample_feat.values[0].tolist()

            resp = requests.post(
                f"{API_URL}/predict",
                json={"features": features, "transaction_id": f"ui_random_{idx}"},
            )
            result = resp.json()

            if result["prediction"] == 1:
                st.error(f"🚨 FRAUD (prob: {result['fraud_probability']:.4f})")
            else:
                st.success(f"✅ Legit (prob: {result['fraud_probability']:.4f})")

            st.metric("Latency", f"{result['latency_ms']:.2f} ms")
            st.json(result)
            show_explanation(result["transaction_id"])


# ── Page: Batch Predict ──────────────────────────────────────────────
elif page == "Batch Predict":
    st.header("Batch Prediction")

    uploaded = st.file_uploader("Upload CSV with multiple transactions", type=["csv"])

    if uploaded and api_healthy:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df)} transactions")

        if st.button("Run Batch Prediction"):
            from src.features.feature_engineering import engineer_features

            if "Class" in df.columns:
                actual_labels = df["Class"].tolist()
                df = df.drop(columns=["Class"])
            else:
                actual_labels = None
            if "Time" in df.columns:
                df = df.drop(columns=["Time"])

            df_feat = engineer_features(df)
            results = []
            progress = st.progress(0)

            for i in range(len(df_feat)):
                features = df_feat.iloc[i].tolist()
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"features": features, "transaction_id": f"batch_{i}"},
                )
                results.append(resp.json())
                progress.progress((i + 1) / len(df_feat))

            results_df = pd.DataFrame(results)
            fraud_count = (results_df["prediction"] == 1).sum()

            st.metric("Total", len(results_df))
            st.metric("Fraud Detected", fraud_count)
            st.metric("Avg Latency", f"{results_df['latency_ms'].mean():.2f} ms")

            st.dataframe(results_df)


# ── Page: Feedback ───────────────────────────────────────────────────
elif page == "Feedback":
    st.header("Submit Feedback (Ground Truth)")
    st.markdown(
        "Provide the actual label for a past prediction to track model accuracy."
    )

    txn_id = st.text_input("Transaction ID", placeholder="e.g. ui_random_42")
    actual = st.selectbox(
        "Actual Label",
        [0, 1],
        format_func=lambda x: "Legit (0)" if x == 0 else "Fraud (1)",
    )

    if st.button("Submit Feedback") and api_healthy:
        resp = requests.post(
            f"{API_URL}/feedback",
            json={"transaction_id": txn_id, "actual_label": actual},
        )
        if resp.status_code == 200:
            st.success(resp.json()["message"])
            if resp.json().get("current_accuracy"):
                st.metric("Model Accuracy", f"{resp.json()['current_accuracy']:.2%}")
        else:
            st.error(resp.json().get("detail", "Error"))


# ── Page: Dashboard ──────────────────────────────────────────────────
elif page == "Dashboard":
    st.header("Monitoring Dashboard")

    if api_healthy:
        stats = requests.get(f"{API_URL}/stats").json()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", stats["total_predictions"])
        col2.metric("Fraud Detected", stats["fraud_count"])
        col3.metric("Fraud Ratio", f"{stats['fraud_ratio']:.2%}")
        col4.metric("Avg Latency", f"{stats['avg_latency_ms']:.2f} ms")

        st.markdown("---")
        st.subheader("External Links")
        st.markdown("- [Prometheus](http://localhost:9090) — Raw metrics")
        st.markdown("- [Grafana](http://localhost:3000) — Dashboards (admin/admin)")
        st.markdown("- [API Docs](http://localhost:8000/docs) — Swagger UI")
        st.markdown("- [MLflow](http://localhost:5000) — Experiment Tracking")
    else:
        st.warning("API is not running")


# ── Page: Pipeline Status ────────────────────────────────────────────
# Single pane of glass for the ML pipeline. Pulls live state from
# Airflow REST, MLflow REST, Prometheus REST and the local DVC lock —
# so a non-technical user can answer "is the pipeline healthy?" without
# opening four separate tool UIs.
elif page == "Pipeline Status":
    st.header("ML Pipeline Status")
    st.caption(
        "Live view across DVC (training pipeline), Airflow (scheduled jobs), "
        "MLflow (experiments + registry) and Prometheus (scrape targets). "
        "Refresh the page to re-poll."
    )

    # ── DVC training pipeline ────────────────────────────────────────
    st.subheader("1. DVC Training Pipeline")
    dvc_lock_path = os.path.join(_PROJECT_ROOT, "dvc.lock")
    dvc_yaml_path = os.path.join(_PROJECT_ROOT, "dvc.yaml")
    if os.path.exists(dvc_lock_path) and os.path.exists(dvc_yaml_path):
        try:
            with open(dvc_lock_path) as f:
                lock = yaml.safe_load(f) or {}
            with open(dvc_yaml_path) as f:
                yml = yaml.safe_load(f) or {}
            stage_names = list((yml.get("stages") or {}).keys())
            locked_stages = (lock.get("stages") or {})
            rows = []
            for s in stage_names:
                locked = s in locked_stages
                rows.append(
                    {
                        "stage": s,
                        "status": "✅ locked" if locked else "⏳ not yet run",
                        "cmd": (yml["stages"][s] or {}).get("cmd", ""),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                f"`dvc.lock` last modified: "
                f"{pd.Timestamp.fromtimestamp(os.path.getmtime(dvc_lock_path))}"
            )
        except Exception as exc:
            st.error(f"Could not parse dvc.yaml / dvc.lock: {exc}")
    else:
        st.warning(
            "dvc.lock or dvc.yaml not found — run `dvc repro` from the project root."
        )

    st.markdown("---")

    # ── Airflow scheduled DAGs ───────────────────────────────────────
    st.subheader("2. Airflow Scheduled DAGs")
    try:
        r = requests.get(
            f"{AIRFLOW_URL}/api/v1/dags",
            auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
            timeout=HEALTH_TIMEOUT_S,
        )
        if r.status_code == 200:
            dags = r.json().get("dags", [])
            if not dags:
                st.info("Airflow reachable, but no DAGs registered yet.")
            else:
                rows = []
                for d in dags:
                    dag_id = d["dag_id"]
                    runs = requests.get(
                        f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns?limit=1"
                        "&order_by=-execution_date",
                        auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
                        timeout=HEALTH_TIMEOUT_S,
                    )
                    last = (runs.json().get("dag_runs") or [{}])[0] if runs.ok else {}
                    rows.append(
                        {
                            "dag_id": dag_id,
                            "paused": d.get("is_paused"),
                            "schedule": d.get("schedule_interval", {}).get("value")
                            if isinstance(d.get("schedule_interval"), dict)
                            else d.get("schedule_interval"),
                            "last_state": last.get("state", "—"),
                            "last_run": last.get("execution_date", "—"),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.markdown(f"Open the [Airflow UI]({AIRFLOW_URL}) for full graph + logs.")
        else:
            st.warning(f"Airflow returned HTTP {r.status_code} — is it running?")
    except Exception as exc:
        st.warning(
            f"Airflow not reachable at {AIRFLOW_URL} ({exc.__class__.__name__}). "
            "Start the stack with `docker compose up -d airflow`."
        )

    st.markdown("---")

    # ── MLflow experiments + registry ────────────────────────────────
    st.subheader("3. MLflow — Experiments & Registry")
    try:
        # Resolve the project's experiment name → ID. The local mlruns/
        # store typically does not have a "Default" (id=0) experiment, so
        # we look up the named experiment from configs/config.yaml first.
        mlflow_cfg = _CFG.get("mlflow", {}) or {}
        experiment_name = mlflow_cfg.get("experiment_name", "fraud-detection")
        exp_lookup = requests.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/get-by-name",
            params={"experiment_name": experiment_name},
            timeout=HEALTH_TIMEOUT_S,
        )
        if exp_lookup.status_code == 200:
            experiment_id = (
                (exp_lookup.json() or {}).get("experiment", {}).get("experiment_id", "0")
            )
        else:
            experiment_id = "0"

        # Latest 5 runs in the resolved experiment
        r = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
            json={"experiment_ids": [experiment_id], "max_results": 5,
                  "order_by": ["attributes.start_time DESC"]},
            timeout=HEALTH_TIMEOUT_S,
        )
        if r.status_code == 200:
            runs = (r.json() or {}).get("runs", [])
            if runs:
                rows = []
                for run in runs:
                    info = run.get("info", {})
                    metrics_map = {
                        m["key"]: m["value"]
                        for m in (run.get("data", {}) or {}).get("metrics", [])
                    }
                    rows.append(
                        {
                            "run_id": (info.get("run_id") or "")[:8],
                            "status": info.get("status"),
                            "started": pd.Timestamp(
                                info.get("start_time", 0), unit="ms"
                            ),
                            "f1": round(metrics_map.get("f1_score", 0.0), 4),
                            "pr_auc": round(metrics_map.get("pr_auc", 0.0), 4),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No MLflow runs found yet.")

            # Registry: latest version per stage
            reg = requests.get(
                f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/get-latest-versions",
                params={"name": _CFG.get("mlflow", {}).get(
                    "registered_model_name", "fraud-detection-xgboost")},
                timeout=HEALTH_TIMEOUT_S,
            )
            if reg.status_code == 200:
                versions = (reg.json() or {}).get("model_versions", [])
                if versions:
                    st.markdown("**Registered model — latest version per stage:**")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "name": v.get("name"),
                                    "version": v.get("version"),
                                    "stage": v.get("current_stage"),
                                    "run_id": (v.get("run_id") or "")[:8],
                                }
                                for v in versions
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            st.markdown(f"Open the [MLflow UI]({MLFLOW_URL}) for run details + artifacts.")
        else:
            st.warning(f"MLflow returned HTTP {r.status_code}.")
    except Exception as exc:
        st.warning(
            f"MLflow not reachable at {MLFLOW_URL} ({exc.__class__.__name__}). "
            "Start the stack with `docker compose up -d mlflow`."
        )

    st.markdown("---")

    # ── Prometheus scrape targets (proxy for "all components healthy") ─
    st.subheader("4. Prometheus — Scrape-Target Health")
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/targets", timeout=HEALTH_TIMEOUT_S
        )
        if r.status_code == 200:
            targets = (r.json().get("data") or {}).get("activeTargets", [])
            if targets:
                rows = []
                for t in targets:
                    labels = t.get("labels", {})
                    rows.append(
                        {
                            "job": labels.get("job"),
                            "instance": labels.get("instance"),
                            "service": labels.get("service", "—"),
                            "health": "✅ up" if t.get("health") == "up" else "❌ down",
                            "last_scrape": t.get("lastScrape"),
                        }
                    )
                df = pd.DataFrame(rows)
                up = (df["health"].str.contains("up")).sum()
                st.metric("Targets up", f"{up} / {len(df)}")
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Prometheus has no active targets yet.")
            st.markdown(
                f"- [Prometheus]({PROMETHEUS_URL}) — raw metrics & alerts  \n"
                f"- [Grafana]({GRAFANA_URL}) — dashboards (admin / configured pwd)"
            )
        else:
            st.warning(f"Prometheus returned HTTP {r.status_code}.")
    except Exception as exc:
        st.warning(
            f"Prometheus not reachable at {PROMETHEUS_URL} "
            f"({exc.__class__.__name__})."
        )


# ── Page: Threat Landscape ───────────────────────────────────────────
elif page == "Threat Landscape":
    st.header("Threat Landscape")
    st.markdown("Public fraud-related context scraped from external sources.")

    if not os.path.exists(FRAUD_STATS_PATH):
        st.warning(
            "No scraped data yet. Run `python scripts/run_scrape.py` from the "
            "project root to fetch sources, then refresh this page."
        )
    else:
        with open(FRAUD_STATS_PATH) as f:
            data = json.load(f)

        meta = data.get("_meta", {})
        st.caption(
            f"Last scraped: {meta.get('scraped_at', 'unknown')} • "
            f"{meta.get('source_count', 0)} source(s) • "
            f"scraper v{meta.get('scraper_version', '?')}"
        )

        for src in data.get("sources", []):
            label = src.get("title") or src.get("name", "source")
            with st.expander(f"📄 {label}", expanded=True):
                st.markdown(f"**Source URL**: <{src.get('url', '')}>")
                st.markdown(f"**Fetched at**: {src.get('fetched_at', '?')}")
                if src.get("summary"):
                    st.markdown("**Summary**")
                    st.write(src["summary"])
                if src.get("description"):
                    st.markdown("**Description**")
                    st.write(src["description"])
                if src.get("note"):
                    st.info(src["note"])
                if src.get("sections"):
                    st.markdown("**Sections found**")
                    st.write(", ".join(src["sections"]))
                if src.get("stats"):
                    st.markdown("**Statistics extracted**")
                    st.dataframe(pd.DataFrame(src["stats"]))
                if src.get("external_links"):
                    st.markdown("**Related links**")
                    for link in src["external_links"]:
                        st.markdown(f"- {link}")


# ── Page: About ──────────────────────────────────────────────────────
elif page == "About":
    st.header("About This Project")
    st.markdown(
        """
    ### Credit Card Fraud Detection — MLOps Pipeline

    **Model:** XGBoost classifier trained on 284,807 transactions

    **Metrics:**
    | Metric | Value |
    |--------|-------|
    | F1 Score | 0.8229 |
    | Precision | 0.9000 |
    | Recall | 0.7579 |
    | ROC-AUC | 0.9817 |
    | PR-AUC | 0.8180 |

    **MLOps Stack:**
    Git, DVC, MLflow, Docker, FastAPI, Prometheus, Grafana,
    Airflow, Streamlit, SQLite, AlertManager

    **Pipeline:** `dvc repro` runs validate → preprocess → feature_engineering → train → evaluate
    """
    )
