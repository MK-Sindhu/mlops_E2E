"""
Weekly fraud-stats scraping DAG.

Refreshes ``data/external/fraud_stats.json`` by re-running the BeautifulSoup
scraper from Phase 6. The Streamlit dashboard's "Threat Landscape" page
reads this file, so weekly is a reasonable cadence: Wikipedia's article
doesn't change often.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_ROOT = "/project"

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=15),
    "email_on_failure": False,
}

with DAG(
    dag_id="fraud_stats_scrape",
    description="Weekly refresh of public fraud-stats sources (Wikipedia + Kaggle).",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=datetime(2026, 4, 26),
    catchup=False,
    tags=["fraud-detection", "scraping"],
    max_active_runs=1,
) as dag:

    scrape = BashOperator(
        task_id="scrape_fraud_stats",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python scripts/run_scrape.py"
        ),
        env={"PYTHONPATH": PROJECT_ROOT},
        append_env=True,
        execution_timeout=timedelta(minutes=5),
    )
