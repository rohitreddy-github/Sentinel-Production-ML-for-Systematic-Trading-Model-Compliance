"""
Model drift detection using Population Stability Index (PSI)
and rolling performance monitoring.

Detects:
1. Data drift — feature distributions shifting from training baseline
2. Concept drift — model accuracy degrading over time
3. Prediction drift — output distribution changing

Designed for SR 11-7 / MiFID II compliance: all drift events are
logged with timestamps, severity, and recommended actions.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.config import LOGS_DIR, MonitoringConfig

logger = logging.getLogger(__name__)

DRIFT_LOG_FILE = LOGS_DIR / "drift_events.jsonl"


class DriftDetector:
    """
    Production drift detection for financial ML models.

    Methods:
    - compute_psi: Population Stability Index per feature
    - monitor_accuracy: Rolling accuracy tracking
    - monitor_prediction_distribution: Output distribution shifts
    - generate_drift_report: Comprehensive drift assessment
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self._baseline_distributions: Dict[str, np.ndarray] = {}

    # ── PSI (Population Stability Index) ──────────────────────────────────

    def set_baseline(self, X_train: pd.DataFrame) -> None:
        """
        Store training data distributions as the baseline for PSI.

        Call this once after training. PSI will compare new data
        against these distributions.
        """
        for col in X_train.columns:
            values = X_train[col].dropna().values
            self._baseline_distributions[col] = values
        logger.info(
            f"Baseline set with {len(self._baseline_distributions)} features"
        )

    def compute_psi(
        self,
        X_new: pd.DataFrame,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Compute PSI for each feature comparing new data to baseline.

        PSI interpretation:
        - < 0.1: No significant shift
        - 0.1 - 0.2: Moderate shift (warning)
        - > 0.2: Significant shift (action required)

        Returns:
            Dict of feature_name → PSI value
        """
        if not self._baseline_distributions:
            raise ValueError("Baseline not set. Call set_baseline() first.")

        psi_scores = {}

        for col in X_new.columns:
            if col not in self._baseline_distributions:
                continue

            baseline = self._baseline_distributions[col]
            current = X_new[col].dropna().values

            if len(current) < 10:
                logger.warning(f"Too few samples for PSI on {col}")
                continue

            psi = self._calculate_psi(baseline, current, n_bins)
            psi_scores[col] = psi

        return psi_scores

    def _calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Calculate PSI between two distributions."""
        # Create bins from baseline
        eps = 1e-4
        breakpoints = np.quantile(
            baseline, np.linspace(0, 1, n_bins + 1)
        )
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        # Compute bin proportions
        baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
        current_counts = np.histogram(current, bins=breakpoints)[0]

        baseline_pct = (baseline_counts / len(baseline)) + eps
        current_pct = (current_counts / len(current)) + eps

        # PSI formula
        psi = np.sum(
            (current_pct - baseline_pct)
            * np.log(current_pct / baseline_pct)
        )

        return float(psi)

    # ── Rolling Accuracy Monitor ──────────────────────────────────────────

    def monitor_accuracy(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: pd.DatetimeIndex,
        window: int = 63,
    ) -> pd.DataFrame:
        """
        Compute rolling accuracy, precision, and recall.

        Returns:
            DataFrame with rolling metrics indexed by date.
        """
        results = pd.DataFrame(index=dates)
        results["correct"] = (predictions == actuals).astype(float)
        results["prediction"] = predictions
        results["actual"] = actuals

        results["rolling_accuracy"] = (
            results["correct"].rolling(window, min_periods=20).mean()
        )

        # Rolling precision (of positive predictions)
        tp = ((results["prediction"] == 1) & (results["actual"] == 1)).astype(float)
        fp = ((results["prediction"] == 1) & (results["actual"] == 0)).astype(float)
        results["rolling_precision"] = (
            tp.rolling(window, min_periods=20).sum()
            / (
                tp.rolling(window, min_periods=20).sum()
                + fp.rolling(window, min_periods=20).sum()
            ).replace(0, np.nan)
        )

        # Rolling recall
        fn = ((results["prediction"] == 0) & (results["actual"] == 1)).astype(float)
        results["rolling_recall"] = (
            tp.rolling(window, min_periods=20).sum()
            / (
                tp.rolling(window, min_periods=20).sum()
                + fn.rolling(window, min_periods=20).sum()
            ).replace(0, np.nan)
        )

        return results

    # ── Prediction Distribution Monitor ───────────────────────────────────

    def monitor_prediction_distribution(
        self,
        probabilities: np.ndarray,
        baseline_probabilities: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Check if the model's prediction confidence distribution has shifted.
        """
        psi = self._calculate_psi(baseline_probabilities, probabilities)

        return {
            "prediction_psi": psi,
            "baseline_mean": float(np.mean(baseline_probabilities)),
            "current_mean": float(np.mean(probabilities)),
            "baseline_std": float(np.std(baseline_probabilities)),
            "current_std": float(np.std(probabilities)),
            "drift_detected": psi > self.config.psi_warning,
        }

    # ── Comprehensive Drift Report ────────────────────────────────────────

    def generate_drift_report(
        self,
        X_new: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Generate a full drift assessment report.

        Returns a structured report suitable for compliance review.
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_samples_evaluated": len(X_new),
            "data_drift": {},
            "concept_drift": {},
            "severity": "LOW",
            "recommended_action": "NONE",
        }

        # Data drift (PSI)
        psi_scores = self.compute_psi(X_new)
        report["data_drift"]["psi_scores"] = psi_scores

        critical_features = {
            k: v for k, v in psi_scores.items()
            if v > self.config.psi_critical
        }
        warning_features = {
            k: v for k, v in psi_scores.items()
            if self.config.psi_warning <= v <= self.config.psi_critical
        }

        report["data_drift"]["critical_count"] = len(critical_features)
        report["data_drift"]["warning_count"] = len(warning_features)
        report["data_drift"]["critical_features"] = critical_features
        report["data_drift"]["warning_features"] = warning_features

        # Concept drift (if actuals available)
        if predictions is not None and actuals is not None:
            from sklearn.metrics import accuracy_score, f1_score

            accuracy = accuracy_score(actuals, predictions)
            f1 = f1_score(actuals, predictions, zero_division=0)

            report["concept_drift"] = {
                "accuracy": float(accuracy),
                "f1": float(f1),
                "accuracy_below_threshold": accuracy < self.config.accuracy_min,
            }

        # Determine severity
        if len(critical_features) > 0:
            report["severity"] = "CRITICAL"
            report["recommended_action"] = "RETRAIN_IMMEDIATELY"
        elif len(warning_features) > 3:
            report["severity"] = "HIGH"
            report["recommended_action"] = "SCHEDULE_RETRAIN"
        elif len(warning_features) > 0:
            report["severity"] = "MEDIUM"
            report["recommended_action"] = "MONITOR_CLOSELY"

        # Log the drift event
        self._log_drift_event(report)

        return report

    def _log_drift_event(self, report: Dict) -> None:
        """Append drift event to audit log."""
        log_entry = {
            "event": "drift_assessment",
            "timestamp": report["timestamp"],
            "severity": report["severity"],
            "action": report["recommended_action"],
            "critical_features": report["data_drift"].get("critical_count", 0),
            "warning_features": report["data_drift"].get("warning_count", 0),
        }
        DRIFT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DRIFT_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
