"""
Orchestrator: End-to-end training pipeline.

Usage:
    python -m pipeline.train_pipeline

This script runs the full production pipeline:
1. Load and validate data
2. Engineer features (with leakage-free target)
3. Walk-forward train + validate
4. Register model in governance registry
5. Validate against minimum thresholds
6. Optionally promote to champion
7. Set monitoring baselines
"""

import argparse
import logging
import sys
from datetime import datetime

from pipeline.config import (
    DATA_START_DATE,
    DATA_END_DATE,
    XGBoostConfig,
    ValidationConfig,
    MonitoringConfig,
)
from pipeline.data_loader import DataLoader
from pipeline.feature_engine import FeatureEngine
from pipeline.model_trainer import ModelTrainer
from pipeline.model_registry import ModelRegistry
from monitoring.drift_detector import DriftDetector
from monitoring.data_quality import DataQualityMonitor
from monitoring.audit_logger import AuditLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    auto_promote: bool = False,
    refresh_data: bool = False,
):
    """
    Execute the full training pipeline.

    Args:
        auto_promote: Automatically promote model if validation passes
        refresh_data: Re-download data from Yahoo Finance
    """
    audit = AuditLogger()
    started_at = datetime.utcnow().isoformat()
    logger.info("=" * 70)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 70)

    # ── Step 1: Load Data ─────────────────────────────────────────────────
    logger.info("Step 1/7: Loading market data...")
    loader = DataLoader()
    etf_data = loader.load_etf_universe(
        start=DATA_START_DATE,
        end=DATA_END_DATE,
        refresh=refresh_data,
    )
    vix = loader.load_vix(start=DATA_START_DATE, end=DATA_END_DATE)

    # ── Step 2: Data Quality ──────────────────────────────────────────────
    logger.info("Step 2/7: Running data quality checks...")
    dq_monitor = DataQualityMonitor()

    import pandas as pd
    combined_close = pd.DataFrame({
        t: df["Close"] for t, df in etf_data.items()
    })
    dq_report = dq_monitor.validate(combined_close)
    if not dq_report["passed"]:
        logger.warning("Data quality issues detected — proceeding with caution")

    # ── Step 3: Feature Engineering ───────────────────────────────────────
    logger.info("Step 3/7: Engineering features (leakage-free)...")
    engine = FeatureEngine()
    X, y = engine.build_dataset(etf_data, vix=vix)

    logger.info(f"  Dataset shape: {X.shape}")
    logger.info(f"  Target balance: {y.value_counts().to_dict()}")

    # ── Step 4: Train & Validate ──────────────────────────────────────────
    logger.info("Step 4/7: Walk-forward training and validation...")
    trainer = ModelTrainer(
        xgb_config=XGBoostConfig(),
        val_config=ValidationConfig(),
    )
    summary = trainer.train_and_validate(X, y)

    logger.info(f"  Aggregate accuracy: {summary['aggregate_accuracy']:.4f}")
    logger.info(f"  Aggregate F1:       {summary['aggregate_f1']:.4f}")
    logger.info(f"  Aggregate MCC:      {summary['aggregate_mcc']:.4f}")

    # Feature importance
    importance = trainer.get_feature_importance()
    logger.info("  Top 10 features by SHAP importance:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"    {row['feature']:40s} {row['mean_abs_shap']:.6f}")

    # ── Step 5: Register Model ────────────────────────────────────────────
    logger.info("Step 5/7: Registering model in governance registry...")
    registry = ModelRegistry()
    version = registry.register_model(
        model=trainer.best_model,
        scaler=trainer.best_scaler,
        training_summary=summary,
        feature_names=trainer.feature_names,
        data_hash=summary["data_hash"],
    )

    audit.log_training_event(
        event_subtype="model_registered",
        model_version=version,
        details={
            "accuracy": summary["aggregate_accuracy"],
            "f1": summary["aggregate_f1"],
            "mcc": summary["aggregate_mcc"],
            "n_samples": summary["n_samples"],
            "n_features": summary["n_features"],
        },
    )

    # ── Step 6: Validate Against Thresholds ───────────────────────────────
    logger.info("Step 6/7: Validating model against governance thresholds...")
    monitoring_config = MonitoringConfig()
    validation_report = registry.validate_model(version, monitoring_config)

    if validation_report["passed"]:
        logger.info(f"  Model {version} PASSED validation ✓")

        if auto_promote:
            registry.promote_to_champion(version)
            audit.log_training_event(
                event_subtype="model_promoted",
                model_version=version,
                details={"auto_promoted": True},
            )
            logger.info(f"  Model {version} promoted to CHAMPION")
        else:
            logger.info(
                f"  Run with --auto-promote to promote, or manually call "
                f"registry.promote_to_champion('{version}')"
            )
    else:
        logger.warning(f"  Model {version} FAILED validation ✗")
        for check, result in validation_report["checks"].items():
            if not result["passed"]:
                logger.warning(
                    f"    {check}: value={result['value']:.4f}, "
                    f"threshold={result['threshold']:.4f}"
                )

    # ── Step 7: Set Monitoring Baselines ──────────────────────────────────
    logger.info("Step 7/7: Setting monitoring baselines...")
    drift_detector = DriftDetector(monitoring_config)
    drift_detector.set_baseline(X)

    dq_monitor.learn_schema(X)

    logger.info("=" * 70)
    logger.info(f"PIPELINE COMPLETE — Model {version}")
    logger.info(f"  Accuracy: {summary['aggregate_accuracy']:.4f}")
    logger.info(f"  F1 Score: {summary['aggregate_f1']:.4f}")
    logger.info(f"  MCC:      {summary['aggregate_mcc']:.4f}")
    logger.info("=" * 70)

    return {
        "version": version,
        "summary": summary,
        "validation": validation_report,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the production ML training pipeline"
    )
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Automatically promote model if validation passes",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Re-download data from Yahoo Finance",
    )
    args = parser.parse_args()

    result = run_pipeline(
        auto_promote=args.auto_promote,
        refresh_data=args.refresh_data,
    )

    sys.exit(0 if result["validation"]["passed"] else 1)


if __name__ == "__main__":
    main()
