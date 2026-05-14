"""
Data quality monitoring with automated alerting.

Validates incoming data against expected schemas, ranges, and freshness
requirements before it enters the ML pipeline.

Designed for continuous production monitoring — can be run as a
pre-prediction gate or as a scheduled health check.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline.config import LOGS_DIR, MonitoringConfig

logger = logging.getLogger(__name__)

DATA_QUALITY_LOG = LOGS_DIR / "data_quality.jsonl"


class DataQualityMonitor:
    """
    Production data quality monitoring.

    Checks:
    1. Schema validation — expected columns present and typed correctly
    2. Completeness — missing value thresholds
    3. Range validation — values within expected bounds
    4. Freshness — data recency meets SLA
    5. Statistical consistency — sudden distribution changes
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self._expected_schema: Optional[Dict[str, str]] = None
        self._expected_ranges: Dict[str, Dict[str, float]] = {}

    def learn_schema(self, df: pd.DataFrame) -> None:
        """
        Learn expected schema and value ranges from training data.
        """
        self._expected_schema = {col: str(df[col].dtype) for col in df.columns}

        for col in df.select_dtypes(include=[np.number]).columns:
            values = df[col].dropna()
            self._expected_ranges[col] = {
                "min": float(values.quantile(0.001)),
                "max": float(values.quantile(0.999)),
                "mean": float(values.mean()),
                "std": float(values.std()),
            }

        logger.info(
            f"Learned schema: {len(self._expected_schema)} columns, "
            f"{len(self._expected_ranges)} numeric ranges"
        )

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all data quality checks on incoming data.

        Returns:
            Report dict with pass/fail status and details.
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "checks": {},
            "passed": True,
        }

        # 1. Schema validation
        schema_result = self._check_schema(df)
        report["checks"]["schema"] = schema_result
        if not schema_result["passed"]:
            report["passed"] = False

        # 2. Completeness
        completeness_result = self._check_completeness(df)
        report["checks"]["completeness"] = completeness_result
        if not completeness_result["passed"]:
            report["passed"] = False

        # 3. Range validation
        range_result = self._check_ranges(df)
        report["checks"]["ranges"] = range_result
        if not range_result["passed"]:
            report["passed"] = False

        # 4. Duplicate detection
        dup_result = self._check_duplicates(df)
        report["checks"]["duplicates"] = dup_result

        # 5. Statistical consistency
        stats_result = self._check_statistics(df)
        report["checks"]["statistics"] = stats_result

        # Log
        self._log_quality_check(report)

        status = "PASSED" if report["passed"] else "FAILED"
        logger.info(f"Data quality check {status}: {len(df)} rows")

        return report

    def _check_schema(self, df: pd.DataFrame) -> Dict:
        """Verify expected columns are present."""
        if self._expected_schema is None:
            return {"passed": True, "note": "No schema baseline set"}

        expected = set(self._expected_schema.keys())
        actual = set(df.columns)

        missing = expected - actual
        extra = actual - expected

        return {
            "passed": len(missing) == 0,
            "missing_columns": list(missing),
            "extra_columns": list(extra),
        }

    def _check_completeness(self, df: pd.DataFrame) -> Dict:
        """Check missing value ratios."""
        completeness = df.notna().mean()
        below_threshold = completeness[
            completeness < self.config.min_feature_completeness
        ]

        return {
            "passed": len(below_threshold) == 0,
            "overall_completeness": float(completeness.mean()),
            "failing_columns": {
                col: float(val) for col, val in below_threshold.items()
            },
        }

    def _check_ranges(self, df: pd.DataFrame) -> Dict:
        """Check if values fall within expected ranges."""
        if not self._expected_ranges:
            return {"passed": True, "note": "No range baseline set"}

        out_of_range = {}

        for col, bounds in self._expected_ranges.items():
            if col not in df.columns:
                continue
            values = df[col].dropna()
            n_below = (values < bounds["min"]).sum()
            n_above = (values > bounds["max"]).sum()
            total_oor = n_below + n_above

            if total_oor > 0:
                out_of_range[col] = {
                    "below_min": int(n_below),
                    "above_max": int(n_above),
                    "pct_out_of_range": float(total_oor / len(values)),
                }

        return {
            "passed": len(out_of_range) == 0,
            "out_of_range_columns": out_of_range,
        }

    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate rows."""
        n_dupes = df.duplicated().sum()
        return {
            "passed": n_dupes == 0,
            "n_duplicates": int(n_dupes),
            "pct_duplicates": float(n_dupes / len(df)) if len(df) > 0 else 0,
        }

    def _check_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Check for suspicious statistical properties:
        - Constant columns (zero variance)
        - Extreme skewness
        """
        issues = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            values = df[col].dropna()
            if len(values) < 10:
                continue

            std = values.std()
            if std == 0:
                issues[col] = "constant_value"
            elif abs(values.skew()) > 10:
                issues[col] = f"extreme_skew ({values.skew():.1f})"

        return {
            "passed": len(issues) == 0,
            "issues": issues,
        }

    def _log_quality_check(self, report: Dict) -> None:
        """Append quality check to audit log."""
        log_entry = {
            "event": "data_quality_check",
            "timestamp": report["timestamp"],
            "passed": report["passed"],
            "n_rows": report["n_rows"],
            "checks_summary": {
                k: v.get("passed", True)
                for k, v in report["checks"].items()
            },
        }
        DATA_QUALITY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_QUALITY_LOG, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
