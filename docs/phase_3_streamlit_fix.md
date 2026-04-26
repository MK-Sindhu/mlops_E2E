# Phase 3 — Streamlit / pyarrow Environment Fix

## Symptom
Launching the Streamlit dashboard crashed with:

```
ImportError: libre2.so.9: cannot open shared object file: No such file or directory
File "/home/swapnil/anaconda3/lib/python3.12/site-packages/streamlit/elements/arrow.py", line 648, in dataframe
    import pyarrow as pa
File "/home/swapnil/anaconda3/lib/python3.12/site-packages/pyarrow/__init__.py", line 65, in <module>
    import pyarrow.lib as _lib
```

## Root cause

Two compounding issues:

1. **Wrong Python interpreter.** The traceback shows `/home/swapnil/anaconda3/lib/python3.12/site-packages/...` — Streamlit was running on the **system Anaconda** Python (3.12), not the project's `.venv` (3.10). Anaconda's pyarrow was linked against `libre2.so.9`, but Anaconda has since upgraded to `libre2.so.11`, breaking the link.

2. **`requirements.txt` was incomplete.** `streamlit`, `pyarrow`, and `requests` were *not* declared in [requirements.txt](../requirements.txt) — they only existed in the venv because someone `pip install`-ed them ad-hoc. Anyone reproducing the environment from `requirements.txt` would not get a working dashboard. This is an MLOps reproducibility failure.

## Fix

### 1. Pin the missing direct deps in `requirements.txt`
```
requests==2.33.1
streamlit==1.56.0
pyarrow==15.0.2
```
Versions match what the working venv actually has, so anyone running `pip install -r requirements.txt` rebuilds the same environment.

### 2. Add a launcher script that forces the venv
[scripts/run_streamlit.sh](../scripts/run_streamlit.sh) calls `.venv/bin/streamlit` directly — never `streamlit` via PATH — so even if the user has Anaconda earlier in PATH, the dashboard runs against the right Python.

### 3. Document the run command in README
[README.md](../README.md) Quick Start now includes step 5: `./scripts/run_streamlit.sh`.

## What this does NOT yet solve

Local venv parity is fixed, but **dev/staging/prod parity is not**. A user could still skip the venv and hit other surprises. That's solved properly in **Phase 11** when we containerise Streamlit alongside the API in `docker-compose.yml` — the guideline-mandated path for environment parity.

## Verification

After installing the updated requirements:

```bash
.venv/bin/pip install -r requirements.txt
./scripts/run_streamlit.sh
# open http://localhost:8501 — dashboard loads, st.dataframe(df.head()) renders without error
```

## Outputs of this phase
- [requirements.txt](../requirements.txt) — added `streamlit`, `pyarrow`, `requests` with exact pins
- [scripts/run_streamlit.sh](../scripts/run_streamlit.sh) — venv-locked launcher
- [README.md](../README.md) — Quick Start step 5 documents the launch command
- This document
- Tag `v0.3.0-phase3` on `main`

## What's next
Phase 4 — verify data ingestion, validation, and DVC tracking actually work end-to-end (`dvc repro` of the `validate` stage cleanly).
