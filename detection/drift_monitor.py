"""Monitor feature distribution drift and trigger retraining when needed.

Implements Population Stability Index (PSI) computation to detect when the
distribution of features in production scoring has shifted significantly from
the training distribution. Persists scored feature vectors to SQLite and
provides drift detection thresholds.

Per-feature PSI time series are stored in the ``feature_psi_history`` table.
When 3+ features exceed PSI > 0.20 simultaneously, an alert is written to
``degradation_alerts`` with ``alert_type = 'feature_drift'``.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("ledgerlens.drift_monitor")

PSI_ALERT_COOLDOWN_HOURS: int = 24

MAX_SNAPSHOT_ROWS = 500_000
MIN_SNAPSHOT_ROWS_AFTER_PRUNE = 450_000


def _init_db(db_path: str) -> None:
    """Initialize feature_distribution_snapshots table if it doesn't exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_distribution_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet TEXT NOT NULL,
            asset_pair TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            feature_value REAL NOT NULL,
            recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_feature_recorded_at ON feature_distribution_snapshots(feature_name, recorded_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_recorded_at ON feature_distribution_snapshots(recorded_at)"
    )
    conn.commit()
    conn.close()


def record_scored_features(
    feature_vectors: list[dict],
    wallet_ids: list[str] | None = None,
    asset_pairs: list[str] | None = None,
    db_path: str | None = None,
) -> None:
    """Persist a batch of feature vectors to the distribution snapshots table.

    Args:
        feature_vectors: List of feature dicts (keys are feature names, values are floats).
        wallet_ids: Wallet IDs corresponding to each feature vector (optional).
        asset_pairs: Asset pair strings corresponding to each feature vector (optional).
        db_path: SQLite database path. Defaults to settings.db_path.
    """
    from config.settings import settings

    db_path = db_path or settings.db_path
    _init_db(db_path)

    if not feature_vectors:
        return

    wallet_ids = wallet_ids or ["unknown"] * len(feature_vectors)
    asset_pairs = asset_pairs or ["unknown"] * len(feature_vectors)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()
    rows = []

    for fv, wallet, pair in zip(feature_vectors, wallet_ids, asset_pairs):
        for feature_name, feature_value in fv.items():
            if isinstance(feature_value, (int, float)) and not np.isnan(feature_value):
                rows.append((wallet, pair, feature_name, float(feature_value), now))

    cursor.executemany(
        """
        INSERT INTO feature_distribution_snapshots (wallet, asset_pair, feature_name, feature_value, recorded_at)
        VALUES (?, ?, ?, ?, ?)
    """,
        rows,
    )
    conn.commit()

    # Enforce hard row cap: if exceeded, prune oldest rows.
    cursor.execute("SELECT COUNT(*) FROM feature_distribution_snapshots")
    count = cursor.fetchone()[0]

    if count > MAX_SNAPSHOT_ROWS:
        logger.warning(
            "Feature distribution snapshots exceeded hard cap (%d > %d); pruning oldest rows",
            count,
            MAX_SNAPSHOT_ROWS,
        )
        cursor.execute(
            f"""
            DELETE FROM feature_distribution_snapshots
            WHERE recorded_at NOT IN (
                SELECT recorded_at FROM feature_distribution_snapshots
                ORDER BY recorded_at DESC
                LIMIT {MIN_SNAPSHOT_ROWS_AFTER_PRUNE}
            )
        """
        )
        conn.commit()

    conn.close()


def compute_psi(
    training_ref: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-10,
) -> float:
    """Compute Population Stability Index (PSI) between two distributions.

    PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))

    PSI = 0 means identical distributions.
    PSI > 0.20 conventionally signals significant drift.
    PSI > 0.25 is severe drift.

    Args:
        training_ref: Training reference distribution (1D array).
        current: Current production distribution (1D array).
        bins: Number of histogram bins.
        epsilon: Small constant to avoid log(0).

    Returns:
        PSI value (float >= 0).
    """
    # Handle empty or nan-filled arrays
    training_ref = training_ref[~np.isnan(training_ref)]
    current = current[~np.isnan(current)]

    if len(training_ref) == 0 or len(current) == 0:
        return 0.0

    # Compute histogram bins from the combined range to ensure consistent binning
    min_val = min(training_ref.min(), current.min())
    max_val = max(training_ref.max(), current.max())

    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, bins + 1)

    # Compute frequencies (avoid zero-count bins with epsilon)
    ref_counts, _ = np.histogram(training_ref, bins=bin_edges)
    curr_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * bins)
    curr_pct = (curr_counts + epsilon) / (curr_counts.sum() + epsilon * bins)

    # Compute PSI
    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))

    return float(max(0.0, psi))


def run_drift_report(
    training_dataset_path: str,
    db_path: str | None = None,
    days_back: int = 30,
) -> dict[str, float]:
    """Compare training reference distribution with recent scored features.

    Loads the training dataset, computes reference distributions for all features,
    then compares with the last N days of scored features in the database.

    Args:
        training_dataset_path: Path to training CSV with feature columns.
        db_path: SQLite database path. Defaults to settings.db_path.
        days_back: Number of days of production data to compare.

    Returns:
        Dict mapping feature names to PSI values.
    """
    from config.settings import settings
    from detection.feature_engineering import FEATURE_NAMES

    db_path = db_path or settings.db_path
    _init_db(db_path)

    # Load training reference
    try:
        training_df = pd.read_csv(training_dataset_path)
    except FileNotFoundError:
        logger.warning("Training dataset not found at %s; returning empty report", training_dataset_path)
        return {}

    # Load recent scored features from database
    cutoff_time = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
    conn = sqlite3.connect(db_path)
    scored_df = pd.read_sql_query(
        """
        SELECT feature_name, feature_value
        FROM feature_distribution_snapshots
        WHERE recorded_at >= ?
        """,
        conn,
        params=(cutoff_time,),
    )
    conn.close()

    if scored_df.empty:
        logger.info("No scored features found in the last %d days", days_back)
        return {}

    scored_names = set(scored_df["feature_name"].unique())
    feature_names = [
        f for f in FEATURE_NAMES if f in training_df.columns and f in scored_names
    ]

    # Compute PSI for each feature
    report = {}
    for feature_name in feature_names:
        training_dist = training_df[feature_name].dropna().values
        scored_dist = scored_df[scored_df["feature_name"] == feature_name]["feature_value"].values

        if len(training_dist) == 0 or len(scored_dist) == 0:
            report[feature_name] = 0.0
        else:
            psi = compute_psi(training_dist, scored_dist)
            report[feature_name] = psi

    return report


def compute_psi_for_feature(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI between reference and current distributions for a single feature."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    percentile_bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    percentile_bins = np.unique(percentile_bins)
    if len(percentile_bins) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=percentile_bins)
    cur_counts, _ = np.histogram(current, bins=percentile_bins)
    ref_pct = ref_counts / (len(reference) + epsilon)
    cur_pct = cur_counts / (len(current) + epsilon)
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(max(0.0, psi))


def compute_per_feature_psi(
    training_ref_path: str,
    db_path: str | None = None,
    window_days: int = 30,
) -> dict[str, float]:
    """Compute per-feature PSI against the training reference distribution.

    Returns a dict mapping feature name to PSI value.
    """
    from config.settings import settings
    from detection.feature_engineering import FEATURE_NAMES

    db_path = db_path or settings.db_path
    _init_db(db_path)

    try:
        training_df = pd.read_csv(training_ref_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Training reference not found at {training_ref_path}. "
            "Run the training pipeline first to generate training_reference.csv."
        )

    cutoff_time = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    conn = sqlite3.connect(db_path)
    scored_df = pd.read_sql_query(
        "SELECT feature_name, feature_value FROM feature_distribution_snapshots WHERE recorded_at >= ?",
        conn,
        params=(cutoff_time,),
    )
    conn.close()

    if scored_df.empty:
        logger.warning("No scored features found in the last %d days; returning all-zero PSI", window_days)
        return {f: 0.0 for f in FEATURE_NAMES if f in training_df.columns}

    scored_names = set(scored_df["feature_name"].unique())
    result: dict[str, float] = {}

    for feature_name in FEATURE_NAMES:
        if feature_name not in training_df.columns:
            continue
        ref_vals = training_df[feature_name].dropna().values
        if feature_name not in scored_names or len(ref_vals) == 0:
            result[feature_name] = 0.0
            continue
        cur_vals = scored_df[scored_df["feature_name"] == feature_name]["feature_value"].values
        result[feature_name] = compute_psi_for_feature(ref_vals, cur_vals)

    return result


def _init_psi_history_table(db_path: str) -> None:
    """Create feature_psi_history table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_psi_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature_name TEXT NOT NULL,
            psi_value REAL NOT NULL,
            window_days INTEGER NOT NULL DEFAULT 30,
            n_reference_samples INTEGER,
            n_current_samples INTEGER,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_psi_history_feature ON feature_psi_history(feature_name, computed_at)"
    )
    conn.commit()
    conn.close()


def record_psi_snapshot(
    psi_dict: dict[str, float],
    window_days: int = 30,
    db_path: str | None = None,
) -> None:
    """Persist per-feature PSI values to the feature_psi_history table."""
    from config.settings import settings

    db_path = db_path or settings.db_path
    _init_psi_history_table(db_path)

    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    rows = [
        (feature_name, psi_value, window_days, now)
        for feature_name, psi_value in psi_dict.items()
    ]
    conn.executemany(
        "INSERT INTO feature_psi_history (feature_name, psi_value, window_days, computed_at) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def load_psi_history(
    days_back: int = 90,
    feature_name: str | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    """Load PSI history from the database."""
    from config.settings import settings

    db_path = db_path or settings.db_path
    _init_psi_history_table(db_path)

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
    conn = sqlite3.connect(db_path)

    query = "SELECT feature_name, psi_value, window_days, computed_at FROM feature_psi_history WHERE computed_at >= ?"
    params: list = [cutoff]
    if feature_name:
        query += " AND feature_name = ?"
        params.append(feature_name)
    query += " ORDER BY computed_at"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if not df.empty:
        df["computed_at_date"] = pd.to_datetime(df["computed_at"]).dt.date.astype(str)
    return df


def check_psi_and_alert(
    psi_dict: dict[str, float],
    psi_threshold: float = 0.20,
    min_drifted_features: int = 3,
    db_path: str | None = None,
) -> bool:
    """Check if PSI alert should fire and write to degradation_alerts if so."""
    from config.settings import settings

    db_path = db_path or settings.db_path

    drifted = [f for f, v in psi_dict.items() if v > psi_threshold]
    if len(drifted) < min_drifted_features:
        return False

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS degradation_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT NOT NULL,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cooldown_cutoff = (datetime.now(timezone.utc) - timedelta(hours=PSI_ALERT_COOLDOWN_HOURS)).isoformat()
    recent = conn.execute(
        "SELECT COUNT(*) FROM degradation_alerts WHERE alert_type = 'feature_drift' AND created_at >= ?",
        (cooldown_cutoff,),
    ).fetchone()[0]

    if recent > 0:
        conn.close()
        logger.info("PSI alert suppressed (cooldown: %d hours)", PSI_ALERT_COOLDOWN_HOURS)
        return False

    details = json.dumps({
        "n_drifted": len(drifted),
        "affected_features": drifted,
        "psi_values": {f: psi_dict[f] for f in drifted},
    })
    conn.execute(
        "INSERT INTO degradation_alerts (alert_type, details) VALUES (?, ?)",
        ("feature_drift", details),
    )
    conn.commit()
    conn.close()

    logger.warning("Feature drift alert: %d features exceed PSI threshold", len(drifted))
    return True


def export_psi_heatmap(
    output_path: Path,
    days_back: int = 90,
    db_path: str | None = None,
) -> Path:
    """Generate a (n_features x n_dates) heatmap of PSI values as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    df = load_psi_history(days_back=days_back, db_path=db_path)
    if df.empty:
        logger.warning("No PSI history data for heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No PSI data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Feature PSI Heatmap (last {days_back} days)")
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    pivot = df.pivot_table(
        index="feature_name", columns="computed_at_date", values="psi_value", aggfunc="mean"
    )
    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 0.5), max(6, len(pivot.index) * 0.3))
    )
    cmap = LinearSegmentedColormap.from_list("psi", ["white", "yellow", "orange", "red"], N=256)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0.0, vmax=0.30)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title(f"Feature PSI Heatmap (last {days_back} days)")
    fig.colorbar(im, ax=ax, label="PSI")
    plt.tight_layout()
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def is_drift_detected(
    report: dict[str, float],
    psi_threshold: float = 0.20,
    min_drifted_features: int = 3,
) -> bool:
    """Determine if drift is detected based on the drift report.

    Drift is detected if at least `min_drifted_features` features have
    PSI > `psi_threshold`.

    Args:
        report: Dict mapping feature names to PSI values.
        psi_threshold: PSI threshold above which a feature is considered drifted.
        min_drifted_features: Minimum number of drifted features to trigger retraining.

    Returns:
        True if drift is detected, False otherwise.
    """
    drifted_count = sum(1 for psi in report.values() if psi > psi_threshold)
    is_drifted = drifted_count >= min_drifted_features

    if is_drifted:
        logger.info(
            "Drift detected: %d features exceed PSI threshold (%.3f)",
            drifted_count,
            psi_threshold,
        )
    else:
        logger.info(
            "No drift detected: %d features exceed PSI threshold (%.3f)",
            drifted_count,
            psi_threshold,
        )

    return is_drifted
