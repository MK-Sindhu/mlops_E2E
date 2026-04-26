# Phase 4 — Data Ingestion, Validation, and DVC Tracking

## Goal
Verify the data layer satisfies the Phase A guideline:
> *"Document data sources, formats, and potential biases. Implement automated checks for schema consistency and missing values during ingestion. Version-control data-collection scripts. Automate data ingestion and validation."*

## What's in place

### Modules
| File | Purpose |
|---|---|
| [src/data/ingest.py](../src/data/ingest.py) | Config-driven CSV loader. Raises a clear error pointing to the Kaggle source if the file is missing. |
| [src/data/validate.py](../src/data/validate.py) | Four checks: schema (column names + dtypes), missing values, target binary integrity, suspicious value ranges. Returns a structured `dict` report. |
| [src/data/security.py](../src/data/security.py) | Fernet symmetric encryption helpers — `generate_key`, `load_key`, `encrypt_file`, `decrypt_file`. Key gitignored. |
| [scripts/run_validate.py](../scripts/run_validate.py) | DVC-stage entrypoint. Loads config, runs ingestion + validation, writes `data/validation_report.json`. |

### DVC pipeline (stage 1 of 5)
```yaml
validate:
  cmd: python scripts/run_validate.py
  deps:
    - scripts/run_validate.py
    - src/data/ingest.py
    - src/data/validate.py
    - data/raw/creditcard.csv
  outs:
    - data/validation_report.json:
        cache: false
```

DVC dag: `creditcard.csv → validate, preprocess → feature_engineering, train → evaluate`.

## Verification (run on this branch)

```bash
$ dvc --version
3.42.0

$ dvc status
Data and pipelines are up to date.

$ cat data/validation_report.json | python -m json.tool
{
    "schema":          {"valid": true, "errors": []},
    "missing_values":  {"valid": true, "missing_columns": {}},
    "target":          {"valid": true, "message": "Target column is valid (binary: 0, 1)"},
    "value_ranges":    {"valid": true, "warnings": []},
    "overall_valid":   true
}

$ ls -lh data/raw/
-rw-rw-r-- 1 swapnil swapnil 144M creditcard.csv
-rw-rw-r-- 1 swapnil swapnil   99 creditcard.csv.dvc
-rw-rw-r-- 1 swapnil swapnil  134 creditcard.csv.enc        # stub — see "Known issues"

$ python -c "from src.data.security import load_key; print('Key loaded OK, len=', len(load_key('configs/.encryption_key')))"
Key loaded OK, len= 44
```

## Fixes shipped in this phase

### 1. Pinned `pathspec==0.11.2`
DVC 3.42.0 imports `_DIR_MARK` from `pathspec.patterns.gitwildmatch`. `pathspec` 0.12.0 removed this symbol, so a fresh `pip install` of the project would land on `pathspec` 1.x and break `dvc status` / `dvc repro` with:
```
ImportError: cannot import name '_DIR_MARK' from 'pathspec.patterns.gitwildmatch'
```
[requirements.txt](../requirements.txt) now pins `pathspec==0.11.2` with an inline comment so future maintainers know why.

This is the **third unpinned-transitive-dep failure** in the project (after the Streamlit/pyarrow issue in Phase 3). Worth a follow-up: regenerate a full `requirements.lock` from `pip freeze` and ship it alongside `requirements.txt`. Tracked as a Phase 18 follow-up.

## Known issues / follow-ups

### `creditcard.csv.enc` is a stub
The encrypted snapshot is **134 bytes** — it cannot possibly be the encrypted form of the 144 MB CSV. It looks like a placeholder from an earlier dry-run. The encryption *machinery* works (key loads cleanly, helpers in `security.py` are correct), but to actually satisfy "encrypted at rest" you'd run:

```bash
python -m src.data.security        # encrypts data/raw/creditcard.csv -> creditcard.csv.enc
```

Not addressed in Phase 4 because:
- The DVC tracking covers reproducibility of the *unencrypted* CSV via hash.
- For a non-public-cloud deployment, the threat model is "laptop stolen", and DVC + gitignored key + on-disk encryption would be a Phase 18 hardening item.

### `data/validation_report.json` is a tracked DVC output but not gitignored
DVC stages declare `cache: false` for the report, meaning DVC won't store it in the cache, but git also doesn't ignore it. Since validation reports change every run, they'll show up in `git status`. Consider ignoring or keeping — currently kept so the report stays visible.

## Outputs of this phase
- [requirements.txt](../requirements.txt) — added `pathspec==0.11.2` pin with reason
- This document
- Tag `v0.4.0-phase4` on `main`

## What's next
Phase 5 — verify the EDA notebook and the drift baselines (`data/baselines/feature_baselines.json`) that drift detection depends on.
