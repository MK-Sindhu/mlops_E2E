# Custom Airflow image — official 2.10.5 + our project's runtime deps so
# DAGs that call scripts/*.py can import src.* without ModuleNotFoundError.

FROM apache/airflow:2.10.5-python3.10

USER root
# libgomp1 is needed by xgboost (OpenMP). The airflow user is also remapped
# to UID 1000 so it matches the host user owning bind-mounted directories.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && usermod -u 1000 airflow \
    && chown -R airflow /home/airflow /opt/airflow

USER airflow

# Project deps — mirrored from the main requirements.txt so the DAGs can
# import src.models.train, src.monitoring.drift_detection, etc.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
