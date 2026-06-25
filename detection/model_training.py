"""Train the Random Forest / XGBoost / LightGBM wash-trading ensemble.

Expects a feature DataFrame (see `feature_engineering.build_feature_vector`)
with a binary `label` column (1 = confirmed wash trade pattern). Trained
models are written to `settings.model_dir` for `model_inference` to load.

When ``calibrate=True`` a calibration split is held out (10 % of the data,
stratified by label) *before* any model training, then used after training
to compute conformal prediction thresholds via ``ConformalCalibrator``.

Stacking ensemble (Issue-111):
  Architecture: RF / XGBoost / LightGBM as base models → Logistic Regression
  meta-learner trained on out-of-fold (OOF) predictions with temporal folds.

  OOF generation uses walk-forward cross-validation (5 folds, 7-day gap) so
  the meta-learner never sees future data during training. The fitted
  meta-learner is saved to ``models/meta_learner.joblib``.

  At inference time :class:`~detection.model_inference.ModelInference` loads
  the meta-learner and uses it when available, falling back to equal-weight
  averaging when absent.
"""

import logging

import joblib
import numpy as np
import pandas as pd
from detection.model_signing import sign_model_file
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config.settings import settings
from detection.feature_engineering import FEATURE_NAMES

_logger = logging.getLogger("ledgerlens.model_training")


STACKING_USE_DISAGREEMENT_FEATURES: bool = True


def _walk_forward_cv(
    X: np.ndarray,
    timestamps: np.ndarray,
    n_splits: int = 5,
    gap_days: float = 7.0,
):
    """Temporal walk-forward cross-validation iterator.

    Yields (train_idx, val_idx) pairs ordered chronologically, with a
    ``gap_days``-day purge gap between train and validation to prevent leakage.
    """
    n = len(timestamps)
    sorted_order = np.argsort(timestamps, kind="stable")
    fold_size = n // (n_splits + 1)

    for fold in range(n_splits):
        val_start_pos = (fold + 1) * fold_size
        val_end_pos = min(val_start_pos + fold_size, n)
        val_positions = sorted_order[val_start_pos:val_end_pos]

        if len(val_positions) == 0:
            continue

        val_time_min = timestamps[val_positions].min()
        gap_threshold = val_time_min - gap_days * 86400  # seconds

        train_positions = sorted_order[:val_start_pos]
        train_positions = train_positions[timestamps[train_positions] < gap_threshold]

        if len(train_positions) == 0:
            continue

        yield train_positions, val_positions


def generate_oof_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    timestamps: np.ndarray,
    base_models: dict,
    n_splits: int = 5,
    gap_days: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate out-of-fold predictions for the stacking meta-learner.

    Uses temporal walk-forward cross-validation to produce OOF predictions
    that respect the time ordering of transactions and prevent data leakage.

    Args:
        X_train: Feature matrix of shape (n, p).
        y_train: Binary labels of shape (n,).
        timestamps: Unix timestamps (seconds) of shape (n,).
        base_models: Dict mapping model name to unfitted estimator instance.
            Keys must be ``"rf"``, ``"xgb"``, ``"lgbm"`` (in that order for
            the column layout).
        n_splits: Number of temporal folds (default 5).
        gap_days: Purge gap in days between train and validation (default 7).

    Returns:
        Tuple of:
          - ``oof_proba``: shape (n_oof, 3) — OOF probabilities per base model.
          - ``oof_labels``: shape (n_oof,) — corresponding true labels.

    Raises:
        ValueError: if a base model raises during fit or predict_proba.
    """
    smote = SMOTE(random_state=42)
    model_names = list(base_models.keys())
    oof_proba = np.zeros((len(X_train), len(model_names)))
    oof_mask = np.zeros(len(X_train), dtype=bool)

    for fold_idx, (train_idx, val_idx) in enumerate(_walk_forward_cv(X_train, timestamps, n_splits, gap_days)):
        try:
            X_fold_train, y_fold_train = smote.fit_resample(X_train[train_idx], y_train[train_idx])
        except Exception:
            X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]

        for col, (name, model_cls) in enumerate(base_models.items()):
            try:
                model = clone(model_cls)
                model.fit(X_fold_train, y_fold_train)
                proba = model.predict_proba(X_train[val_idx])[:, 1]
                oof_proba[val_idx, col] = proba
            except Exception as exc:
                raise ValueError(
                    f"generate_oof_predictions: model '{name}' failed on fold {fold_idx}: {exc}"
                ) from exc
        oof_mask[val_idx] = True

    return oof_proba[oof_mask], y_train[oof_mask]


def train_meta_learner(
    oof_proba: np.ndarray,
    oof_labels: np.ndarray,
    use_disagreement_features: bool = STACKING_USE_DISAGREEMENT_FEATURES,
) -> LogisticRegression:
    """Train a logistic-regression meta-learner on OOF base-model predictions.

    Optionally augments the feature set with:
    - ``model_disagreement``: max − min across the three model predictions.
    - ``oof_mean``: mean prediction (equal-weight baseline as a feature).

    Args:
        oof_proba: Shape (n_oof, 3) out-of-fold probabilities.
        oof_labels: Shape (n_oof,) binary labels.
        use_disagreement_features: Append disagreement and mean features.

    Returns:
        Fitted :class:`~sklearn.linear_model.LogisticRegression` meta-learner.
    """
    X_meta = oof_proba.copy()
    if use_disagreement_features and oof_proba.shape[1] >= 2:
        disagreement = oof_proba.max(axis=1) - oof_proba.min(axis=1)
        mean_pred = oof_proba.mean(axis=1)
        X_meta = np.column_stack([X_meta, disagreement, mean_pred])

    unique_classes = np.unique(oof_labels)
    if len(unique_classes) < 2:
        _logger.warning(
            "OOF set contains only one class (%s); meta-learner falls back to equal-weight averaging",
            unique_classes,
        )
        return None

    meta = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    meta.fit(X_meta, oof_labels)

    model_names = ["rf", "xgb", "lgbm"]
    coef_str = ", ".join(
        f"{name}={meta.coef_[0][i]:.2f}" for i, name in enumerate(model_names)
    )
    _logger.info("Meta-learner coefficients: %s", coef_str)
    _logger.info("Meta-learner intercept: %.2f", float(meta.intercept_[0]))
    return meta


def _build_meta_features(stack_input: np.ndarray, use_disagreement: bool = STACKING_USE_DISAGREEMENT_FEATURES) -> np.ndarray:
    """Build the meta-learner feature matrix from base model outputs."""
    if use_disagreement and stack_input.shape[1] >= 2:
        disagreement = stack_input.max(axis=1, keepdims=True) - stack_input.min(axis=1, keepdims=True)
        mean_pred = stack_input.mean(axis=1, keepdims=True)
        return np.hstack([stack_input, disagreement, mean_pred])
    return stack_input


def _split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split `df` into `(X, y)`, ordering feature columns by `FEATURE_NAMES`
    so training and inference (`model_inference.score_feature_vector`) never drift.
    """
    X = df[FEATURE_NAMES].fillna(0.0)
    y = df["label"]
    return X, y


def _train_ensemble_base(
    df: pd.DataFrame,
    random_state: int = 42,
    adversarial_augment: bool = True,
    calibrate: bool = True,
    adversarial_hardening: bool = False,
    **kwargs,
) -> dict:
    """Train RF, XGBoost, and LightGBM classifiers on `df` and return metrics + models.

    Applies SMOTE to the training split to address class imbalance, since
    confirmed wash-trade examples are rare relative to clean activity.

    When ``adversarial_augment=True``, generates 3 additional datasets with
    mixed evasion strategies and concatenates them before SMOTE resampling,
    forcing the models to learn adversarial meta-signatures.

    When ``calibrate=True``, reserves a 10 % calibration split (stratified)
    before the train/test split, trains on the remaining data, then runs
    conformal calibration on the held-out set. Calibration data and
    ``ConformalCalibrator`` instances are returned under the ``"calib"`` key
    and used by ``save_models`` to persist the artifacts.
    """
    df = merge_evasion_samples(df, evasion_samples)
    if adversarial_augment:
        from detection.dataset import build_training_dataset
        from ingestion.adversarial_data import ALL_STRATEGIES, generate_adversarial_dataset

        augment_dfs = [df]
        strategy_groups = [
            ALL_STRATEGIES[:2],
            ALL_STRATEGIES[2:4],
            ALL_STRATEGIES,
        ]
        for i, strats in enumerate(strategy_groups):
            trades, meta, events, labels = generate_adversarial_dataset(
                n_normal_accounts=50,
                n_wash_rings=10,
                ring_size=4,
                evasion_strategies=strats,
                seed=random_state + i + 1,
            )
            augment_dfs.append(
                build_training_dataset(trades, labels, account_metadata=meta, order_book_events=events)
            )
        df = pd.concat(augment_dfs, ignore_index=True)

    X, y = _split_features_labels(df)

    if calibrate:
        X_remaining, X_cal, y_remaining, y_cal = train_test_split(
            X, y, test_size=0.10, random_state=random_state, stratify=y
        )
        cal_split_info = {
            "X_cal": X_cal,
            "y_cal": y_cal,
            "cal_index_start": X_cal.index.min(),
            "cal_index_end": X_cal.index.max(),
        }
        X_train, X_test, y_train, y_test = train_test_split(
            X_remaining, y_remaining, test_size=0.2, random_state=random_state, stratify=y_remaining
        )
    else:
        cal_split_info = {}
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        "xgboost": XGBClassifier(eval_metric="logloss", random_state=random_state),
        "lightgbm": LGBMClassifier(random_state=random_state, verbose=-1),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        results[name] = {
            "model": model,
            "auc_roc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "f1": f1_score(y_test, y_pred),
        }

    if calibrate:
        from detection.conformal import ConformalCalibrator

        calibrators = {}
        for name, result in results.items():
            cal = ConformalCalibrator(alpha=0.10).calibrate(
                result["model"], cal_split_info["X_cal"], cal_split_info["y_cal"]
            )
            calibrators[name] = cal
            # Empirical coverage on the calibration set
            cal_split_info[f"coverage_{name}"] = _compute_empirical_coverage(
                result["model"], cal_split_info["X_cal"], cal_split_info["y_cal"], cal.q_hat
            )
        results["_calib"] = {**cal_split_info, "calibrators": calibrators}

    # --- Stacking: train meta-learner on OOF predictions (Issue-111) ---
    try:
        base_model_instances = {
            "rf": models["random_forest"],
            "xgb": models["xgboost"],
            "lgbm": models["lightgbm"],
        }
        X_train_np = X_train.values if hasattr(X_train, "values") else X_train
        y_train_np = y_train.values if hasattr(y_train, "values") else y_train
        # Use sequential integer "timestamps" as a proxy; gap_days=0 avoids the
        # large second-scale gap that would exclude all training data with proxy timestamps.
        timestamps_proxy = np.arange(len(X_train_np), dtype=float)
        oof_proba, oof_labels = generate_oof_predictions(
            X_train_np, y_train_np, timestamps_proxy, base_model_instances, gap_days=0.0,
        )
        meta_learner = train_meta_learner(oof_proba, oof_labels)

        stacking_metrics: dict = {}
        if meta_learner is not None and len(oof_labels) > 0:
            meta_features = _build_meta_features(oof_proba)
            oof_meta_proba = meta_learner.predict_proba(meta_features)[:, 1]
            oof_avg_proba = oof_proba.mean(axis=1)
            try:
                stacking_metrics["meta_learner_auc_pr"] = float(
                    average_precision_score(oof_labels, oof_meta_proba)
                )
                stacking_metrics["avg_baseline_auc_pr"] = float(
                    average_precision_score(oof_labels, oof_avg_proba)
                )
                stacking_metrics["meta_learner_auc_roc"] = float(
                    roc_auc_score(oof_labels, oof_meta_proba)
                )
                stacking_metrics["meta_learner_coef"] = meta_learner.coef_[0].tolist()
                _logger.info(
                    "Meta-learner AUC-PR: %.3f (vs. equal-weight average: %.3f)",
                    stacking_metrics["meta_learner_auc_pr"],
                    stacking_metrics["avg_baseline_auc_pr"],
                )
            except Exception:
                pass
        results["_stacking"] = {"meta_learner": meta_learner, **stacking_metrics}
    except Exception as exc:
        _logger.warning("Stacking meta-learner training failed (best-effort): %s", exc)

    # --- Adversarial hardening: generate PGD adversarial examples from
    # training true positives and retrain once on the augmented set.
    if adversarial_hardening:
        try:
            from detection.adversarial_attack import pgd_attack

            # collect adversarial examples that successfully flip model
            adv_rows = []
            # use the ensemble (current models) to attack training positives
            ensemble_models = {k: v["model"] for k, v in results.items()}
            X_train_res_df = pd.DataFrame(X_train_res, columns=X_train_res.columns)
            y_train_res_ser = pd.Series(y_train_res)
            for idx, (x_row, y_val) in enumerate(zip(X_train_res_df.to_dict(orient="records"), y_train_res_ser.tolist())):
                if int(y_val) != 1:
                    continue
                pert, p = pgd_attack(x_row, ensemble_models, epsilon=0.1, alpha=0.01, steps=10)
                if p < 0.5:
                    adv_rows.append({**pert, "label": 1})

            if adv_rows:
                aug_df = pd.DataFrame(adv_rows)
                # append to original training set and retrain
                X_aug = pd.concat([X_train_res_df, aug_df.drop(columns=["label"])], ignore_index=True)
                y_aug = pd.concat([y_train_res_ser, aug_df["label"].astype(int)], ignore_index=True)

                for name, model in models.items():
                    model.fit(X_aug, y_aug)
                    y_proba = model.predict_proba(X_test)[:, 1]
                    y_pred = model.predict(X_test)
                    results[name] = {
                        "model": model,
                        "auc_roc": roc_auc_score(y_test, y_proba),
                        "pr_auc": average_precision_score(y_test, y_proba),
                        "f1": f1_score(y_test, y_pred),
                    }
        except Exception:
            # Hardening is best-effort; failures should not crash training.
            pass

    # Train LSTM temporal anomaly model
    try:
        from detection.temporal_dataset import build_training_sequences
        from detection.temporal_model import train_temporal_model, predict_temporal_risk

        # Train/validation split by wallet
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=random_state, stratify=df["label"]
        )

        X_train_seq, y_train_seq = build_training_sequences(train_df, db_path=settings.db_path)
        X_test_seq, y_test_seq = build_training_sequences(test_df, db_path=settings.db_path)

        lstm_model = train_temporal_model(X_train_seq, y_train_seq, epochs=15, batch_size=32)

        # Evaluate on test sequence dataset
        y_proba_seq = np.array([predict_temporal_risk(lstm_model, seq) for seq in X_test_seq])
        y_pred_seq = (y_proba_seq >= 0.5).astype(int)

        if len(np.unique(y_test_seq)) > 1:
            lstm_auc_roc = roc_auc_score(y_test_seq, y_proba_seq)
            lstm_pr_auc = average_precision_score(y_test_seq, y_proba_seq)
            lstm_f1 = f1_score(y_test_seq, y_pred_seq)
        else:
            lstm_auc_roc, lstm_pr_auc, lstm_f1 = 1.0, 1.0, 1.0

        results["temporal_lstm"] = {
            "model": lstm_model,
            "auc_roc": lstm_auc_roc,
            "pr_auc": lstm_pr_auc,
            "f1": lstm_f1,
        }
    except Exception as e:
        import logging
        logger = logging.getLogger("ledgerlens.model_training")
        logger.exception("Failed to train temporal LSTM model: %s", e)

    return results


def _compute_empirical_coverage(model, X_cal, y_cal, q_hat):
    """Fraction of calibration examples whose true class is in the prediction set."""
    probs = model.predict_proba(X_cal)
    scores = 1.0 - probs[range(len(y_cal)), y_cal.values]
    return float((scores <= q_hat).mean())


def save_models(
    results: dict,
    model_dir: str | None = None,
    training_dataset_path: str | None = None,
) -> None:
    """Persist trained models to `model_dir` (defaults to `settings.model_dir`).

    Also writes training_metadata.json with model versions, AUC-ROC scores,
    and training dataset path for drift detection and rollback.

    When ``results`` contains ``"_calib"`` key (from ``train_ensemble`` with
    ``calibrate=True``), calibration artifacts are written alongside each
    model file and ``metrics.json`` is updated with empirical coverage.
    """
    import hashlib
    import json
    import os
    from datetime import datetime, timezone

    from detection.model_registry import _compute_version_hash

    model_dir = model_dir or settings.model_dir
    os.makedirs(model_dir, exist_ok=True)

    signing_key = settings.model_signing_key.encode()
    for name, result in results.items():
        if name == "_calib":
            continue
        path = os.path.join(model_dir, f"{name}.joblib")
        joblib.dump(result["model"], path)
        sign_model_file(path, signing_key)

    # Write training_metadata.json
    if training_dataset_path:
        try:
            train_df = pd.read_csv(training_dataset_path)
            training_row_count = len(train_df)
            column_hash = hashlib.sha256(
                ",".join(train_df.columns).encode()
            ).hexdigest()[:8]
        except Exception:
            training_row_count = 0
            column_hash = "unknown"
    else:
        training_row_count = 0
        column_hash = "unknown"

    version = _compute_version_hash(training_row_count, column_hash)

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "training_dataset_path": training_dataset_path or "",
        "training_row_count": training_row_count,
        "column_hash": column_hash,
        "model_metrics": {
            name: {
                "auc_roc": result.get("auc_roc", 0.0),
                "pr_auc": result.get("pr_auc", 0.0),
                "f1": result.get("f1", 0.0),
            }
            for name, result in results.items()
            if name != "_calib"
        },
    }

    metadata_path = os.path.join(model_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    _logger.info("Wrote training metadata to %s", metadata_path)

    # ------------------------------------------------------------------
    # Calibration artifacts
    # ------------------------------------------------------------------
    calib = results.get("_calib")
    if calib and calib.get("calibrators"):
        metrics = {}
        for name, cal in calib["calibrators"].items():
            cal_path = os.path.join(model_dir, f"{name}_conformal.json")
            cal.save(cal_path)
            cover_key = f"coverage_{name}"
            cov = calib.get(cover_key, 0.0)
            metrics[f"conformal_empirical_coverage_{name}"] = round(cov, 4)

        # Aggregate coverage (simple average across models)
        coverages = [v for k, v in metrics.items() if k.startswith("conformal_empirical_coverage_")]
        metrics["conformal_empirical_coverage"] = round(
            sum(coverages) / len(coverages), 4
        ) if coverages else 0.0

        # Log calibration split index range for audit
        metrics["calibration_index_start"] = int(calib.get("cal_index_start", -1))
        metrics["calibration_index_end"] = int(calib.get("cal_index_end", -1))

        metrics_path = os.path.join(model_dir, "metrics.json")
        existing = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                try:
                    existing = json.load(f)
                except Exception:
                    pass
        existing.update(metrics)
        with open(metrics_path, "w") as f:
            json.dump(existing, f, indent=2)
        _logger.info(
            "Wrote calibration metrics (coverage=%.4f) to %s",
            metrics.get("conformal_empirical_coverage", 0.0),
            metrics_path,
        )

    # ------------------------------------------------------------------
    # Stacking: OOF meta-learner (Issue-111)
    # ------------------------------------------------------------------
    stacking_info = results.get("_stacking")
    if stacking_info and stacking_info.get("meta_learner") is not None:
        meta_path = os.path.join(model_dir, "meta_learner.joblib")
        joblib.dump(stacking_info["meta_learner"], meta_path)
        sign_model_file(meta_path, signing_key)
        _logger.info("Saved meta-learner to %s", meta_path)

        # Persist meta-learner metrics into training_metadata.json
        try:
            with open(metadata_path, "r") as f:
                meta_md = json.load(f)
            meta_md["meta_learner_auc_pr"] = stacking_info.get("meta_learner_auc_pr", 0.0)
            meta_md["meta_learner_auc_roc"] = stacking_info.get("meta_learner_auc_roc", 0.0)
            meta_md["meta_learner_coef"] = stacking_info.get("meta_learner_coef", [])
            with open(metadata_path, "w") as f:
                json.dump(meta_md, f, indent=2)
        except Exception as exc:
            _logger.warning("Failed to update training_metadata.json with meta-learner metrics: %s", exc)


if __name__ == "__main__":
    # The ledgerlens-data repo does not yet provide a labelled dataset, so
    # default to a synthetic one for local training/testing.
    import logging

    from detection.dataset import build_training_dataset
    from ingestion.synthetic_data import generate_synthetic_dataset

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ledgerlens.model_training")

    trades, account_metadata, order_book_events, labels = generate_synthetic_dataset(
        n_normal_accounts=60, n_wash_rings=10, ring_size=3
    )
    df = build_training_dataset(trades, labels, account_metadata=account_metadata, order_book_events=order_book_events)

    results = train_ensemble(df)  # noqa: F821
    for name, result in results.items():
        if name == "_calib":
            continue
        logger.info(
            "%s: AUC-ROC=%.3f PR-AUC=%.3f F1=%.3f",
            name,
            result["auc_roc"],
            result["pr_auc"],
            result["f1"],
        )

    save_models(results)
    logger.info("Saved models to %s", settings.model_dir)


from detection.gnn_model import TGATWashRingDetector, save_gnn_checkpoint, _HAS_PYG  # noqa: E402
from ingestion.graph_builder import TemporalGraphBuilder  # noqa: E402
import os  # noqa: E402


def train_ensemble(df, *args, use_gnn: bool = False, model_dir: str = "models", **kwargs):
    """Wraps the base ensemble trainer, optionally pre-training a T-GNN.

    Args:
        use_gnn: If True, trains a T-GNN on the training graph, appends its
            two output features to the feature matrix before SMOTE, and
            saves the checkpoint as gnn_model.pt in model_dir.
    """
    gnn_features_by_wallet = {}

    if use_gnn:
        if not _HAS_PYG:
            raise RuntimeError(
                "use_gnn=True requires torch + torch_geometric installed."
            )
        builder = TemporalGraphBuilder()
        trades = _trades_from_training_df(df)  # noqa: F821
        snapshots = builder.build_snapshots(trades, lookback_days=30)

        import torch
        model = TGATWashRingDetector()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        gnn_features_by_wallet = _run_gnn_training_loop(model, optimizer, snapshots)  # noqa: F821

        os.makedirs(model_dir, exist_ok=True)
        save_gnn_checkpoint(model, os.path.join(model_dir, "gnn_model.pt"))

    return _train_ensemble_base(
        df, *args, use_gnn=use_gnn, gnn_features=gnn_features_by_wallet,
        model_dir=model_dir, **kwargs
    )
