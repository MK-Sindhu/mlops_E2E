"""
Streamlit Demo UI — Credit Card Fraud Detection
Interactive web UI for making predictions, viewing stats, and monitoring.
"""
import streamlit as st
import requests
import json
import pandas as pd
import numpy as np

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")
st.title("🔍 Credit Card Fraud Detection")
st.markdown("Real-time fraud detection powered by XGBoost + MLOps pipeline")

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Predict", "Batch Predict", "Feedback", "Dashboard", "About"])


# ── Helper ───────────────────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


api_healthy = check_api()
if api_healthy:
    st.sidebar.success("API: Connected")
else:
    st.sidebar.error("API: Not running. Start with `uvicorn src.api.app:app --port 8000`")


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
                resp = requests.post(f"{API_URL}/predict", json={
                    "features": features,
                    "transaction_id": f"ui_{int(pd.Timestamp.now().timestamp())}"
                })
                result = resp.json()
                if result["prediction"] == 1:
                    st.error(f"🚨 FRAUD DETECTED (probability: {result['fraud_probability']:.4f})")
                else:
                    st.success(f"✅ Legitimate (probability: {result['fraud_probability']:.4f})")
                st.json(result)

    with col2:
        st.subheader("Option 2: Random Test Sample")
        if st.button("Generate & Predict Random Sample") and api_healthy:
            X_test = pd.read_csv("data/processed/X_test.csv")
            from src.features.feature_engineering import engineer_features
            idx = np.random.randint(0, len(X_test))
            sample = X_test.iloc[idx]
            sample_feat = engineer_features(pd.DataFrame([sample]))
            features = sample_feat.values[0].tolist()

            resp = requests.post(f"{API_URL}/predict", json={
                "features": features,
                "transaction_id": f"ui_random_{idx}"
            })
            result = resp.json()

            if result["prediction"] == 1:
                st.error(f"🚨 FRAUD (prob: {result['fraud_probability']:.4f})")
            else:
                st.success(f"✅ Legit (prob: {result['fraud_probability']:.4f})")

            st.metric("Latency", f"{result['latency_ms']:.2f} ms")
            st.json(result)


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
                resp = requests.post(f"{API_URL}/predict", json={
                    "features": features,
                    "transaction_id": f"batch_{i}"
                })
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
    st.markdown("Provide the actual label for a past prediction to track model accuracy.")

    txn_id = st.text_input("Transaction ID", placeholder="e.g. ui_random_42")
    actual = st.selectbox("Actual Label", [0, 1], format_func=lambda x: "Legit (0)" if x == 0 else "Fraud (1)")

    if st.button("Submit Feedback") and api_healthy:
        resp = requests.post(f"{API_URL}/feedback", json={
            "transaction_id": txn_id,
            "actual_label": actual
        })
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


# ── Page: About ──────────────────────────────────────────────────────
elif page == "About":
    st.header("About This Project")
    st.markdown("""
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
    """)