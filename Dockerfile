# Multi-purpose Dockerfile.
#   - Built once, used by both the `api` and `streamlit` services.
#   - Data + models are mounted as volumes from the host (see docker-compose.yml),
#     so the image stays small and rebuilds are fast.

FROM python:3.11-slim

WORKDIR /app

# System deps for pyarrow / xgboost / shap (some need libgomp etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps — copy requirements first so the install layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source + configs are baked in (immutable for a given image)
COPY src/      src/
COPY configs/  configs/

EXPOSE 8000 8501

# Default command — overridden by docker-compose for the streamlit service.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
