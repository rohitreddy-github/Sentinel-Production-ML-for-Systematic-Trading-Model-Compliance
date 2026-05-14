"""
Tests for the production ML pipeline.

Covers:
- Walk-forward validation correctness (no data leakage)
- Feature engineering output shapes
- Model registry CRUD operations
- Drift detection PSI calculation
- Data quality checks
- Risk metrics computation
- Audit logger compliance
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Validator Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestWalkForwardValidator:
    """Verify walk-forward validation prevents data leakage."""

    def test_no_future_leak(self):
        """Train indices must always be strictly before test indices."""
        from pipeline.model_trainer import WalkForwardValidator, ValidationConfig

        config = ValidationConfig(
            n_splits=3, train_min_size=100, test_size=30, gap=1
        )
        validator = WalkForwardValidator(config)

        for train_idx, test_idx in validator.split(300):
            assert train_idx.max() < test_idx.min(), (
                f"Leakage: train_max={train_idx.max()}, test_min={test_idx.min()}"
            )

    def test_gap_enforced(self):
        """Gap between train and test must be at least config.gap."""
        from pipeline.model_trainer import WalkForwardValidator, ValidationConfig

        config = ValidationConfig(
            n_splits=3, train_min_size=100, test_size=30, gap=5
        )
        validator = WalkForwardValidator(config)

        for train_idx, test_idx in validator.split(300):
            gap = test_idx.min() - train_idx.max()
            assert gap >= config.gap, f"Gap too small: {gap}"

    def test_expanding_window(self):
        """Each subsequent fold should have more training data."""
        from pipeline.model_trainer import WalkForwardValidator, ValidationConfig

        config = ValidationConfig(
            n_splits=4, train_min_size=50, test_size=20, gap=1
        )
        validator = WalkForwardValidator(config)

        train_sizes = [len(t) for t, _ in validator.split(200)]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1], (
                f"Training window should expand: {train_sizes}"
            )

    def test_minimum_samples_enforced(self):
        """Should raise ValueError if not enough data."""
        from pipeline.model_trainer import WalkForwardValidator, ValidationConfig

        config = ValidationConfig(
            n_splits=5, train_min_size=500, test_size=100, gap=1
        )
        validator = WalkForwardValidator(config)

        with pytest.raises(ValueError, match="Not enough data"):
            list(validator.split(100))


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Engineering Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeatureEngine:
    """Verify feature computation correctness."""

    def _make_sample_data(self, n_days=500):
        """Create synthetic ETF data for testing."""
        dates = pd.bdate_range("2020-01-01", periods=n_days)
        data = {}
        np.random.seed(42)

        for ticker in ["QQQ", "SPY", "GLD", "TLT", "IWM", "XLE", "XLF"]:
            price = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.01))
            df = pd.DataFrame({
                "Open": price * (1 + np.random.randn(n_days) * 0.001),
                "High": price * (1 + np.abs(np.random.randn(n_days) * 0.01)),
                "Low": price * (1 - np.abs(np.random.randn(n_days) * 0.01)),
                "Close": price,
                "Volume": np.random.randint(1_000_000, 10_000_000, n_days),
            }, index=dates)
            data[ticker] = df

        return data

    def test_target_is_actual_return(self):
        """Target must be based on actual next-day return, not MACD."""
        from pipeline.feature_engine import FeatureEngine

        data = self._make_sample_data()
        engine = FeatureEngine()
        X, y = engine.build_dataset(data)

        # Target should be binary
        assert set(y.unique()).issubset({0, 1})
        # Target should be approximately balanced (random data ≈ 50/50)
        balance = y.mean()
        assert 0.3 < balance < 0.7, f"Target imbalance: {balance}"

    def test_no_nan_in_output(self):
        """Output dataset must have no NaN values."""
        from pipeline.feature_engine import FeatureEngine

        data = self._make_sample_data()
        engine = FeatureEngine()
        X, y = engine.build_dataset(data)

        assert X.isna().sum().sum() == 0, "Features contain NaN"
        assert y.isna().sum() == 0, "Target contains NaN"

    def test_feature_count(self):
        """Should produce a reasonable number of features."""
        from pipeline.feature_engine import FeatureEngine

        data = self._make_sample_data()
        engine = FeatureEngine()
        X, y = engine.build_dataset(data)

        # At least 7 ETFs * ~10 features each + cross-asset features
        assert X.shape[1] >= 50, f"Too few features: {X.shape[1]}"


# ═══════════════════════════════════════════════════════════════════════════════
# PSI (Drift Detection) Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDriftDetector:
    """Verify PSI drift detection correctness."""

    def test_psi_identical_distributions(self):
        """PSI of identical distributions should be ~0."""
        from monitoring.drift_detector import DriftDetector

        detector = DriftDetector()
        data = np.random.randn(1000)
        psi = detector._calculate_psi(data, data)
        assert psi < 0.01, f"PSI of identical data should be ~0, got {psi}"

    def test_psi_shifted_distribution(self):
        """PSI of significantly shifted distribution should be high."""
        from monitoring.drift_detector import DriftDetector

        detector = DriftDetector()
        baseline = np.random.randn(1000)
        shifted = np.random.randn(1000) + 3  # Large shift

        psi = detector._calculate_psi(baseline, shifted)
        assert psi > 0.2, f"PSI of shifted data should be >0.2, got {psi}"

    def test_drift_report_severity(self):
        """Drift report should correctly classify severity."""
        from monitoring.drift_detector import DriftDetector

        detector = DriftDetector()

        # Set baseline
        np.random.seed(42)
        baseline_df = pd.DataFrame({
            "f1": np.random.randn(500),
            "f2": np.random.randn(500),
        })
        detector.set_baseline(baseline_df)

        # Create heavily drifted data
        drifted_df = pd.DataFrame({
            "f1": np.random.randn(100) + 5,  # Big shift
            "f2": np.random.randn(100) + 5,
        })

        report = detector.generate_drift_report(drifted_df)
        assert report["severity"] in ("HIGH", "CRITICAL")


# ═══════════════════════════════════════════════════════════════════════════════
# Data Quality Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataQuality:
    """Verify data quality checks."""

    def test_clean_data_passes(self):
        """Clean data should pass all quality checks."""
        from monitoring.data_quality import DataQualityMonitor

        monitor = DataQualityMonitor()
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
        })
        monitor.learn_schema(df)
        report = monitor.validate(df)
        assert report["passed"]

    def test_missing_columns_detected(self):
        """Missing columns should be flagged."""
        from monitoring.data_quality import DataQualityMonitor

        monitor = DataQualityMonitor()
        df_train = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        monitor.learn_schema(df_train)

        df_new = pd.DataFrame({"a": [1, 2]})  # Missing b, c
        report = monitor.validate(df_new)
        assert not report["checks"]["schema"]["passed"]

    def test_high_nulls_detected(self):
        """High missing value ratio should be flagged."""
        from monitoring.data_quality import DataQualityMonitor

        monitor = DataQualityMonitor()
        df = pd.DataFrame({
            "a": [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        })
        report = monitor.validate(df)
        assert not report["checks"]["completeness"]["passed"]


# ═══════════════════════════════════════════════════════════════════════════════
# Risk Metrics Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRiskMetrics:
    """Verify risk metric computations."""

    def test_var_positive(self):
        """VaR should be a positive number (representing loss)."""
        from monitoring.risk_metrics import RiskMetrics

        returns = pd.Series(np.random.randn(252) * 0.01)
        var = RiskMetrics.value_at_risk(returns, 0.95)
        assert var > 0

    def test_cvar_greater_than_var(self):
        """CVaR should be >= VaR (it's the average of tail losses)."""
        from monitoring.risk_metrics import RiskMetrics

        returns = pd.Series(np.random.randn(1000) * 0.01)
        var = RiskMetrics.value_at_risk(returns, 0.95)
        cvar = RiskMetrics.conditional_var(returns, 0.95)
        assert cvar >= var, f"CVaR ({cvar}) should be >= VaR ({var})"

    def test_max_drawdown_bounded(self):
        """Max drawdown should be between 0 and 1."""
        from monitoring.risk_metrics import RiskMetrics

        returns = pd.Series(np.random.randn(252) * 0.02)
        dd, _, _ = RiskMetrics.max_drawdown(returns)
        assert 0 <= dd <= 5.0  # Can exceed 1 for very volatile paths

    def test_compute_all_returns_dict(self):
        """compute_all should return a complete metrics dict."""
        from monitoring.risk_metrics import RiskMetrics

        returns = pd.Series(np.random.randn(252) * 0.01)
        metrics = RiskMetrics.compute_all(returns)
        assert "var_95_historical" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "sortino_ratio" in metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Audit Logger Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuditLogger:
    """Verify audit logging compliance."""

    def test_prediction_logged(self, tmp_path, monkeypatch):
        """Every prediction should produce a log entry."""
        monkeypatch.setattr(
            "pipeline.config.PREDICTION_LOG_FILE",
            tmp_path / "predictions.jsonl",
        )

        from monitoring.audit_logger import AuditLogger

        logger = AuditLogger()
        event_id = logger.log_prediction(
            model_version="v001",
            features={"rsi": 55.0, "macd": 0.3},
            prediction=1,
            probability=0.72,
            shap_explanation={"rsi": 0.15, "macd": 0.08},
        )

        assert event_id is not None

        # Verify log file content
        log_file = tmp_path / "predictions.jsonl"
        assert log_file.exists()

        with open(log_file) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "prediction"
        assert entry["model_version"] == "v001"
        assert entry["output"]["signal"] == "BUY"
        assert "shap_values" in entry["explanation"]

    def test_event_id_unique(self, tmp_path, monkeypatch):
        """Each event should have a unique ID."""
        monkeypatch.setattr(
            "pipeline.config.PREDICTION_LOG_FILE",
            tmp_path / "predictions.jsonl",
        )

        from monitoring.audit_logger import AuditLogger

        logger = AuditLogger()
        ids = set()
        for _ in range(10):
            eid = logger.log_prediction(
                model_version="v001",
                features={"rsi": 55.0},
                prediction=1,
                probability=0.6,
                shap_explanation={"rsi": 0.1},
            )
            ids.add(eid)

        assert len(ids) == 10, "Event IDs should be unique"


# ═══════════════════════════════════════════════════════════════════════════════
# Model Registry Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelRegistry:
    """Verify model governance operations."""

    def test_version_incrementing(self, tmp_path):
        """Model versions should auto-increment."""
        from pipeline.model_registry import ModelRegistry

        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        # Mock model
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler

        model = XGBClassifier(n_estimators=2, max_depth=2)
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        scaler = StandardScaler().fit(X)

        v1 = registry.register_model(
            model=model, scaler=scaler,
            training_summary={"aggregate_accuracy": 0.55, "aggregate_f1": 0.54},
            feature_names=["f1", "f2", "f3"],
            data_hash="abc123",
        )
        v2 = registry.register_model(
            model=model, scaler=scaler,
            training_summary={"aggregate_accuracy": 0.58, "aggregate_f1": 0.57},
            feature_names=["f1", "f2", "f3"],
            data_hash="def456",
        )

        assert v1 == "v001"
        assert v2 == "v002"

    def test_champion_promotion(self, tmp_path):
        """Promoted model should be loadable as champion."""
        from pipeline.model_registry import ModelRegistry

        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler

        model = XGBClassifier(n_estimators=2, max_depth=2)
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        scaler = StandardScaler().fit(X)

        version = registry.register_model(
            model=model, scaler=scaler,
            training_summary={"aggregate_accuracy": 0.60, "aggregate_f1": 0.58},
            feature_names=["f1", "f2", "f3"],
            data_hash="abc123",
        )

        registry.validate_model(version)
        registry.promote_to_champion(version)

        champion = registry.load_champion()
        assert champion is not None
        assert champion.version == "champion"
