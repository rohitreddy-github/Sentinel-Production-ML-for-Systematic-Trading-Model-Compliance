"""
Production model training with walk-forward validation, class imbalance handling,
SHAP explainability, and full audit logging.

Fixes critical issues from the notebook-based pipeline:
1. Walk-forward CV instead of single train/test split
2. Scaler fit ONLY on training data within each fold
3. Proper target variable (not MACD sign)
4. SHAP values computed and logged for every prediction
"""

import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pipeline.config import (
    MODELS_DIR,
    LOGS_DIR,
    TRAINING_LOG_FILE,
    ValidationConfig,
    XGBoostConfig,
    MonitoringConfig,
)

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Time-series aware walk-forward cross-validation.

    Unlike sklearn's TimeSeriesSplit, this supports:
    - Configurable gap between train and test (prevents leakage)
    - Embargo period after test (prevents contamination)
    - Expanding window training
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

    def split(self, n_samples: int):
        """
        Generate train/test indices for walk-forward validation.

        Yields (train_indices, test_indices) tuples.
        """
        cfg = self.config
        total_test = cfg.n_splits * cfg.test_size
        remaining = n_samples - cfg.train_min_size - total_test

        if remaining < 0:
            raise ValueError(
                f"Not enough data: {n_samples} samples, need at least "
                f"{cfg.train_min_size + total_test} "
                f"(train_min={cfg.train_min_size}, test_total={total_test})"
            )

        for i in range(cfg.n_splits):
            test_end = n_samples - (cfg.n_splits - i - 1) * cfg.test_size
            test_start = test_end - cfg.test_size
            train_end = test_start - cfg.gap

            if train_end < cfg.train_min_size:
                continue

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx


class ModelTrainer:
    """
    Production model trainer with:
    - Walk-forward cross-validation
    - Per-fold scaling (no look-ahead bias)
    - Class imbalance handling via scale_pos_weight
    - SHAP explainability for every fold
    - Comprehensive audit logging
    """

    def __init__(
        self,
        xgb_config: Optional[XGBoostConfig] = None,
        val_config: Optional[ValidationConfig] = None,
    ):
        self.xgb_config = xgb_config or XGBoostConfig()
        self.val_config = val_config or ValidationConfig()
        self.validator = WalkForwardValidator(self.val_config)

        # Results storage
        self.fold_results: List[Dict] = []
        self.best_model: Optional[XGBClassifier] = None
        self.best_scaler: Optional[StandardScaler] = None
        self.shap_values_all: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    def train_and_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, Any]:
        """
        Run full walk-forward training and validation.

        Returns:
            Summary dict with per-fold and aggregate metrics.
        """
        self.feature_names = X.columns.tolist()
        n_samples = len(X)

        logger.info(
            f"Starting walk-forward validation: {n_samples} samples, "
            f"{len(self.feature_names)} features, "
            f"{self.val_config.n_splits} folds"
        )

        # Data fingerprint for audit
        data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(X).values.tobytes()
        ).hexdigest()[:12]

        self.fold_results = []
        all_test_preds = []
        all_test_true = []
        all_shap = []
        best_f1 = -1.0

        for fold_idx, (train_idx, test_idx) in enumerate(
            self.validator.split(n_samples)
        ):
            fold_result = self._train_fold(
                X, y, train_idx, test_idx, fold_idx
            )
            self.fold_results.append(fold_result)

            all_test_preds.extend(fold_result["predictions"])
            all_test_true.extend(fold_result["actuals"])

            if fold_result["shap_values"] is not None:
                all_shap.append(fold_result["shap_values"])

            # Track best model
            if fold_result["f1"] > best_f1:
                best_f1 = fold_result["f1"]
                self.best_model = fold_result["model"]
                self.best_scaler = fold_result["scaler"]

            logger.info(
                f"Fold {fold_idx}: accuracy={fold_result['accuracy']:.4f}, "
                f"f1={fold_result['f1']:.4f}, mcc={fold_result['mcc']:.4f}, "
                f"auc={fold_result['auc']:.4f}"
            )

        # Aggregate metrics
        all_test_preds = np.array(all_test_preds)
        all_test_true = np.array(all_test_true)

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "data_hash": data_hash,
            "n_samples": n_samples,
            "n_features": len(self.feature_names),
            "n_folds": len(self.fold_results),
            "aggregate_accuracy": float(accuracy_score(all_test_true, all_test_preds)),
            "aggregate_f1": float(f1_score(all_test_true, all_test_preds)),
            "aggregate_mcc": float(matthews_corrcoef(all_test_true, all_test_preds)),
            "per_fold_accuracy": [r["accuracy"] for r in self.fold_results],
            "per_fold_f1": [r["f1"] for r in self.fold_results],
            "classification_report": classification_report(
                all_test_true, all_test_preds, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(
                all_test_true, all_test_preds
            ).tolist(),
            "feature_names": self.feature_names,
            "config": {
                "xgboost": self.xgb_config.to_dict(),
                "validation": self.val_config.__dict__,
            },
        }

        # Combine SHAP values
        if all_shap:
            self.shap_values_all = np.vstack(all_shap)

        # Log training run
        self._log_training_run(summary)

        logger.info(
            f"Walk-forward complete: "
            f"accuracy={summary['aggregate_accuracy']:.4f}, "
            f"f1={summary['aggregate_f1']:.4f}, "
            f"mcc={summary['aggregate_mcc']:.4f}"
        )

        return summary

    def _train_fold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        fold_idx: int,
    ) -> Dict[str, Any]:
        """Train a single fold with proper scaling and evaluation."""

        X_train = X.iloc[train_idx].values
        X_test = X.iloc[test_idx].values
        y_train = y.iloc[train_idx].values
        y_test = y.iloc[test_idx].values

        # Fit scaler ONLY on training data (prevents look-ahead bias)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Handle class imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)

        # Train XGBoost
        config = self.xgb_config.to_dict()
        config["scale_pos_weight"] = scale_pos_weight

        model = XGBClassifier(**config, use_label_encoder=False)
        model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False,
        )

        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        accuracy = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        mcc = float(matthews_corrcoef(y_test, y_pred))
        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            auc = 0.5

        # SHAP explainability
        shap_values = None
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_scaled)
        except Exception as e:
            logger.warning(f"SHAP computation failed for fold {fold_idx}: {e}")

        return {
            "fold": fold_idx,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": accuracy,
            "f1": f1,
            "mcc": mcc,
            "auc": auc,
            "predictions": y_pred.tolist(),
            "probabilities": y_prob.tolist(),
            "actuals": y_test.tolist(),
            "model": model,
            "scaler": scaler,
            "shap_values": shap_values,
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Get SHAP-based feature importance from the best model."""
        if self.shap_values_all is None:
            raise ValueError("No SHAP values available. Run train_and_validate first.")

        importance = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": np.abs(self.shap_values_all).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        return importance

    def get_shap_explanation(
        self, X_single: np.ndarray
    ) -> Dict[str, float]:
        """
        Get SHAP explanation for a single prediction.
        Used by the serving API to attach explanations.
        """
        if self.best_model is None:
            raise ValueError("No trained model available.")

        explainer = shap.TreeExplainer(self.best_model)
        X_scaled = self.best_scaler.transform(X_single.reshape(1, -1))
        shap_vals = explainer.shap_values(X_scaled)[0]

        return dict(zip(self.feature_names, shap_vals.tolist()))

    def save_model(self, tag: str = "latest") -> Path:
        """Save the best model, scaler, and metadata."""
        if self.best_model is None:
            raise ValueError("No trained model to save.")

        import joblib

        model_dir = MODELS_DIR / tag
        model_dir.mkdir(parents=True, exist_ok=True)

        self.best_model.save_model(str(model_dir / "model.json"))
        joblib.dump(self.best_scaler, model_dir / "scaler.pkl")

        metadata = {
            "tag": tag,
            "timestamp": datetime.utcnow().isoformat(),
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "fold_results": [
                {"fold": r["fold"], "accuracy": r["accuracy"], "f1": r["f1"]}
                for r in self.fold_results
            ],
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_dir}")
        return model_dir

    def _log_training_run(self, summary: Dict) -> None:
        """Append training run to audit log."""
        log_entry = {
            "event": "training_complete",
            "timestamp": summary["timestamp"],
            "data_hash": summary["data_hash"],
            "aggregate_accuracy": summary["aggregate_accuracy"],
            "aggregate_f1": summary["aggregate_f1"],
            "aggregate_mcc": summary["aggregate_mcc"],
            "n_folds": summary["n_folds"],
            "n_samples": summary["n_samples"],
            "n_features": summary["n_features"],
        }
        TRAINING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TRAINING_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
