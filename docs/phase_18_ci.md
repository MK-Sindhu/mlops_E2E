# Phase 18 — CI Workflow Verification

## Goal

Bring the GitHub Actions CI workflow back into alignment with the codebase
after 17 phases of additions. Run every CI check locally first, fix anything
broken, and improve coverage where it's a quick win.

## What was actually broken

Three real issues, plus formatting drift:

| Issue | Symptom | Cause |
|---|---|---|
| **Lint step would fail.** | 33 flake8 errors, 18 files would be reformatted by `black`. | 17 phases of edits without running the formatter; some unused imports left around as code evolved. |
| **Coverage was untracked.** | No coverage number anywhere — easy to silently lose test discipline as the project grows. | The CI ran tests with `-v` but never asked for `--cov`. |
| **Swarm overlay not validated.** | `docker-compose.swarm.yml` was added in Phase 17 but never validated in CI. A typo there would only surface during real deployment. | The `validate-compose` job checked only the base file. |

The previously-suspected issue — "secrets/ files missing makes `docker compose
config --quiet` fail" — turned out to be false. `docker compose config` does
**not** check file existence for `file:`-sourced secrets, only syntax. So
Phase 17 didn't actually break the existing validate step. The CI still
seeds `secrets/` from `secrets.example/` for parity with the dev workflow
and so any future `docker compose up` step in CI would have what it needs.

## Changes

### Lint — repo-wide cleanup

- `black src/ tests/` — reformatted **18 files**. Pure formatting, no
  behavioral change.
- Removed **13 unused imports** flagged by flake8 (`F401`):
  `numpy as np` from preprocess / validate / drift_detection,
  `datetime` and `List` from database, unused `pytest` from 3 test files,
  unused `os` / `tempfile` from test_scraper and test_api,
  unused `List` / `Tuple` from drift_detection.
- Pinned `flake8==7.1.1` and `black==25.11.0` in the CI lint job so a
  point-release of black doesn't randomly break the build with new style
  rules.

### Tests — combined + coverage

Old:
```yaml
- name: Run unit tests
  run: pytest tests/unit/ -v --tb=short
- name: Run integration tests
  run: pytest tests/integration/ -v --tb=short
```

New:
```yaml
- name: Run tests with coverage
  run: |
    pytest tests/unit/ tests/integration/ \
      --cov=src \
      --cov-report=term \
      --cov-report=xml \
      -v --tb=short
- name: Upload coverage report
  uses: actions/upload-artifact@v4
  if: always()
  with:
    name: coverage-xml
    path: coverage.xml
```

Why combined: coverage is computed across both suites in one run. Running
them separately would double-execute the slow lifespan / fixture setup and
give two partial coverage reports.

`pytest-cov==4.1.0` added to `requirements.txt`.

**Baseline coverage: 62%** on `src/` (758 statements, 290 missing). High
coverage on the API surface and database layer; gaps in
`monitoring/drift_detection.py` (0% — the script is exercised only via
the running pipeline, never unit-tested) and `data/security.py` (0% — same).
Closing these gaps is its own follow-up phase, not Phase 18 work.

### Compose validation — base + Swarm overlay

```yaml
- name: Seed placeholder secrets
  run: |
    mkdir -p secrets
    cp secrets.example/airflow_admin_password \
       secrets.example/grafana_admin_password \
       secrets.example/mailtrap_smtp_password \
       secrets/
- name: Validate base compose file
  run: docker compose config --quiet
- name: Validate Swarm overlay merges cleanly
  run: |
    docker stack config \
      --compose-file docker-compose.yml \
      --compose-file docker-compose.swarm.yml \
      > /dev/null
```

`docker stack config`, not `docker compose config`, for the Swarm
validation — compose's CLI rejects `container_name` + `replicas > 1`
even for static syntax checking, but `docker stack deploy` (what we
actually use) only emits a warning. `docker stack config` is the
Swarm-aware static validator that mirrors deploy-time leniency.

### Build job — both images

The base Dockerfile produces `fraud-api`. Phase 16 added a second image
(`fraud-airflow`) from `docker/airflow.Dockerfile`. The CI build job now
builds both:

```yaml
- run: docker build -t fraud-api:${{ github.sha }} -f Dockerfile .
- run: docker build -t fraud-airflow:${{ github.sha }} -f docker/airflow.Dockerfile .
```

## What stays out of CI (and why)

- **`tests/e2e/test_pipeline.py`** — needs `data/processed/X_test.csv`
  (DVC-tracked, not in git) and skips gracefully if no model is in the
  MLflow Registry. Running it in CI without DVC remote auth and a live
  registry would just be a skip every time. Runs locally instead.
- **Network-marked scraper test** (`@pytest.mark.network`) — already
  excluded by `pytest.ini`'s default `addopts = -m "not network"`. No
  CI-specific flag needed.
- **`docker compose up` + smoke tests against the running stack** —
  would require ~3 minutes of CI time per run for a marginal signal
  beyond what `validate-compose` already provides. Reserve for a future
  `nightly` workflow if needed.

## Local dry-run before pushing

Same checks the CI runs, in one shell snippet:

```bash
# lint
flake8 src/ tests/ --max-line-length=120
black --check src/ tests/

# test + coverage
pytest tests/unit/ tests/integration/ --cov=src --cov-report=term

# compose validation
mkdir -p secrets
cp secrets.example/airflow_admin_password secrets.example/grafana_admin_password \
   secrets.example/mailtrap_smtp_password secrets/
docker compose config --quiet
docker stack config -c docker-compose.yml -c docker-compose.swarm.yml > /dev/null
```

## Outputs of this phase

- Reformatted source: 18 files (black) + 13 unused imports removed
- `requirements.txt` — adds `pytest-cov==4.1.0`
- `.github/workflows/ci.yml` — coverage, secrets seeding, swarm overlay validation
- This document
- Tag `v0.18.0-phase18` on `main`

## What's next

Phase 19 — final documentation polish + architecture diagram. The end.
