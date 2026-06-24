# Per-Feature PSI Drift Monitoring

## Overview

LedgerLens monitors distributional drift on a per-feature basis using the
Population Stability Index (PSI). Each `retrain-check` run computes PSI for
every feature individually, stores the results as a time series, and triggers
alerts when multiple features drift simultaneously.

## PSI Computation

For each feature, PSI is computed between the training reference distribution
(`models/training_reference.csv`) and recent production scoring data stored in
the `feature_distribution_snapshots` SQLite table:

```
PSI = sum((current_i - reference_i) * ln(current_i / reference_i))
```

Bin edges are derived from the reference distribution's percentiles to ensure
consistent bucketing. A near-constant feature (all values identical) returns
PSI = 0.0 to avoid degenerate binning.

## Alert Thresholds

| Parameter                  | Default | Description                                  |
|----------------------------|---------|----------------------------------------------|
| `PSI_THRESHOLD`            | 0.20    | PSI above which a feature is considered drifted |
| `PSI_MIN_DRIFTED_FEATURES` | 3       | Minimum drifted features to trigger an alert |
| `PSI_ALERT_COOLDOWN_HOURS` | 24      | Suppress duplicate alerts within this window |

An alert is written to `degradation_alerts` with `alert_type = 'feature_drift'`
when the count of features exceeding `PSI_THRESHOLD` reaches
`PSI_MIN_DRIFTED_FEATURES` and no alert has fired in the last
`PSI_ALERT_COOLDOWN_HOURS` hours.

## PSI Heatmap

The `export_psi_heatmap()` function generates a matplotlib heatmap with:
- Y axis: feature names
- X axis: dates
- Colour scale: white (0.0) -> yellow (0.10) -> orange (0.20) -> red (0.25+)

Available via `GET /admin/psi-heatmap` (admin API key required).

## API Endpoints

- `GET /admin/psi-history?feature=chi2_24h&days=90` — per-feature PSI time series
- `GET /admin/psi-heatmap?days=90` — heatmap PNG
- `GET /admin/drift-reports` — aggregate drift check results

## Time Series Storage

PSI values are persisted in the `feature_psi_history` SQLite table with one row
per (feature, computation timestamp). This enables trend analysis and
acceleration detection beyond simple threshold checks.
