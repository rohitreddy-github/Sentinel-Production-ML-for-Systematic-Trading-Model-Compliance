"""
Compliance-grade audit logging for ML predictions and model lifecycle events.

Every prediction, training run, model promotion, and drift event is logged
as a structured JSON line for regulatory traceability.

Meets requirements of:
- SR 11-7 (Model Risk Management)
- MiFID II (Algorithmic Trading Decision Audit Trail)
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.config import LOGS_DIR, AUDIT_LOG_FILE, PREDICTION_LOG_FILE

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Structured audit logging for financial ML compliance.

    Every log entry contains:
    - Unique event ID (UUID4)
    - ISO-8601 timestamp (UTC)
    - Event type classification
    - Full payload with inputs, outputs, and explanations
    - Model version used
    """

    def __init__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def log_prediction(
        self,
        model_version: str,
        features: Dict[str, float],
        prediction: int,
        probability: float,
        shap_explanation: Dict[str, float],
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Log a single prediction with full explainability context.

        This creates the audit trail required by MiFID II for
        algorithmic trading decisions.

        Returns:
            Event ID (UUID4 string)
        """
        event_id = str(uuid.uuid4())

        entry = {
            "event_id": event_id,
            "event_type": "prediction",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_version": model_version,
            "input_features": features,
            "output": {
                "prediction": prediction,
                "probability": probability,
                "signal": "BUY" if prediction == 1 else "SELL",
            },
            "explanation": {
                "shap_values": shap_explanation,
                "top_3_drivers": self._top_drivers(shap_explanation, 3),
            },
            "metadata": metadata or {},
        }

        self._append_log(PREDICTION_LOG_FILE, entry)

        logger.debug(
            f"Prediction logged: {event_id} | "
            f"signal={'BUY' if prediction == 1 else 'SELL'} | "
            f"prob={probability:.4f}"
        )

        return event_id

    def log_training_event(
        self,
        event_subtype: str,
        model_version: str,
        details: Dict[str, Any],
    ) -> str:
        """Log a model lifecycle event (training, validation, promotion)."""
        event_id = str(uuid.uuid4())

        entry = {
            "event_id": event_id,
            "event_type": "model_lifecycle",
            "event_subtype": event_subtype,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_version": model_version,
            "details": details,
        }

        self._append_log(AUDIT_LOG_FILE, entry)
        logger.info(f"Lifecycle event: {event_subtype} for {model_version}")

        return event_id

    def log_drift_event(
        self,
        severity: str,
        drift_report: Dict[str, Any],
        action_taken: str,
    ) -> str:
        """Log a drift detection event."""
        event_id = str(uuid.uuid4())

        entry = {
            "event_id": event_id,
            "event_type": "drift_detection",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "severity": severity,
            "drift_summary": {
                "critical_features": drift_report.get("data_drift", {}).get(
                    "critical_count", 0
                ),
                "warning_features": drift_report.get("data_drift", {}).get(
                    "warning_count", 0
                ),
            },
            "action_taken": action_taken,
        }

        self._append_log(AUDIT_LOG_FILE, entry)
        return event_id

    def log_risk_breach(
        self,
        metric_name: str,
        value: float,
        threshold: float,
        portfolio_state: Dict[str, Any],
    ) -> str:
        """Log a risk limit breach (VaR exceeded, drawdown limit hit)."""
        event_id = str(uuid.uuid4())

        entry = {
            "event_id": event_id,
            "event_type": "risk_breach",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metric": metric_name,
            "value": value,
            "threshold": threshold,
            "breach_severity": abs(value - threshold) / threshold,
            "portfolio_state": portfolio_state,
        }

        self._append_log(AUDIT_LOG_FILE, entry)
        logger.warning(
            f"RISK BREACH: {metric_name}={value:.4f} "
            f"(threshold={threshold:.4f})"
        )

        return event_id

    def get_prediction_log(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict]:
        """Read prediction log entries, optionally filtered by date range."""
        return self._read_log(PREDICTION_LOG_FILE, start_date, end_date, limit)

    def get_audit_log(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict]:
        """Read audit log entries."""
        return self._read_log(AUDIT_LOG_FILE, start_date, end_date, limit)

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _top_drivers(
        shap_values: Dict[str, float], n: int
    ) -> List[Dict[str, Any]]:
        """Extract top N SHAP feature drivers."""
        sorted_features = sorted(
            shap_values.items(), key=lambda x: abs(x[1]), reverse=True
        )
        return [
            {
                "feature": name,
                "shap_value": value,
                "direction": "positive" if value > 0 else "negative",
            }
            for name, value in sorted_features[:n]
        ]

    @staticmethod
    def _append_log(filepath: Path, entry: Dict) -> None:
        """Append a JSON line to a log file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    @staticmethod
    def _read_log(
        filepath: Path,
        start_date: Optional[str],
        end_date: Optional[str],
        limit: int,
    ) -> List[Dict]:
        """Read and filter log entries."""
        if not filepath.exists():
            return []

        entries = []
        with open(filepath) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)

                if start_date and entry.get("timestamp", "") < start_date:
                    continue
                if end_date and entry.get("timestamp", "") > end_date:
                    continue

                entries.append(entry)
                if len(entries) >= limit:
                    break

        return entries
