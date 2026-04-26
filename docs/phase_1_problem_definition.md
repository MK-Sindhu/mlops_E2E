# Phase 1 — Problem Definition & Success Metrics

## Goal
Lock down what we are building, why, and how we will measure success — *before* writing any pipeline code.

## Business problem
Credit card fraud causes billions in annual losses. We are building a system that classifies transactions as fraudulent (1) or legitimate (0) in real time, so they can be blocked or flagged for review.

## Success metrics

### ML metrics
| Metric | Why we care |
|---|---|
| **F1-Score** | Balances precision and recall — needed because false negatives (missed fraud) and false positives (blocked legit transactions) are both costly. |
| **Precision** | High precision → low rate of legitimate transactions wrongly flagged. |
| **Recall** | High recall → most fraud is caught. |
| **PR-AUC** | The headline metric. ROC-AUC is misleading under extreme class imbalance; PR-AUC reflects performance on the minority (fraud) class. |

### Business metrics
| Metric | Target |
|---|---|
| Inference latency (per transaction) | **< 200 ms** |
| API error rate | **< 5%** |
| Model F1 in production | **> 0.80** (else trigger retraining) |

## Dataset
- **Source**: Kaggle — Credit Card Fraud Detection (European cardholders, Sep 2013).
- **Size**: 284,807 rows × 31 columns.
- **Schema**: `Time`, `V1`–`V28` (PCA-transformed), `Amount`, `Class`.
- **Known biases**:
  - 0.172% positive class — extreme imbalance.
  - PCA-transformed features → low interpretability (mitigated downstream with SHAP).
  - Single region + 2-day time window → distribution is narrow.

## Reproducibility contract
Every experiment is identifiable by:
- **Git commit hash** — exact code state.
- **MLflow run ID** — params, metrics, artifacts.
- **DVC lock hash** — exact data + pipeline state.

## Outputs of this phase
- [README.md](../README.md) — added Dataset section, fixed broken `download_data.py` reference, replaced manual run steps with `dvc repro`.
- This document.

## What's next
Phase 2 — formalise the Git workflow (branches, conventional commits, tags) so every subsequent phase ships a tagged, traceable commit.
