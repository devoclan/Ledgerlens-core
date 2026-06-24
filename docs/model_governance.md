# Model Governance: SHAP Feature Importance Stability

## Overview

LedgerLens tracks per-version SHAP feature importances to detect when model
retraining produces a significantly different feature ranking. This is a
model governance safeguard — if a new model suddenly relies on different
signals than its predecessor, it warrants human review before promotion.

## Stability Threshold

The default Spearman rank correlation threshold is **ρ = 0.70**
(`SHAP_STABILITY_THRESHOLD`). When the top-10 feature rankings between
consecutive model versions drop below this threshold for any model in
the ensemble, auto-promotion is blocked.

### What to investigate when stability fails

1. **Genuine tactic shift** — wash-trading bots may have changed behavior.
   Check whether the newly promoted features (e.g., `timing_tightness_score`
   rising from rank 8 to rank 1) align with observed changes in attack patterns.

2. **Data pipeline bug** — a feature computation change or data ingestion
   issue may introduce spurious correlation. Verify that feature distributions
   are consistent with expectations.

3. **Overfitting** — the new model may be fitting noise in the retraining set.
   Compare validation AUC-PR and check whether the new top features have
   high variance across cross-validation folds.

## The `--force-promote` flag

When the stability check fails, you can override it:

```bash
python -m cli retrain-check --force-promote
```

This is logged at WARN level with the caller identity for audit purposes.
Use this only after investigating the cause of the ranking change.

## API Endpoints

- `GET /admin/feature-importance/{version}` — returns stored SHAP importances
  for a model version. Accepts `?model_name=xgboost` filter.
- `GET /admin/feature-importance/diff?old=abc&new=def` — returns the
  `StabilityReport` comparing two versions.

Both endpoints require the admin API key.
