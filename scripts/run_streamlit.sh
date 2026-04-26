#!/usr/bin/env bash
# Launch the Streamlit dashboard from the project venv.
#
# Why this wrapper exists:
#   Running `streamlit run …` with the wrong Python (e.g. system Anaconda)
#   pulls in an unrelated pyarrow/re2 install and crashes with
#   `ImportError: libre2.so.9`. Forcing the venv's interpreter prevents that.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT_ROOT/.venv"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: venv not found at $VENV" >&2
  echo "Create it with: python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# Use the venv's streamlit explicitly — never trust PATH alone.
exec "$VENV/bin/streamlit" run \
  "$PROJECT_ROOT/src/app/streamlit_app.py" \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  "$@"
