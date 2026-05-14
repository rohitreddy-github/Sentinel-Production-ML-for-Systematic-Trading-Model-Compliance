"""
Data loading with validation, quality checks, and staleness detection.

Replaces hardcoded paths and adds production-grade data quality gates.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from pipeline.config import (
    DATA_DIR,
    DATA_START_DATE,
    DATA_END_DATE,
    ETF_UNIVERSE,
    MIN_HISTORY_DAYS,
    MonitoringConfig,
)

logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Raised when data fails quality gates."""
    pass


class DataLoader:
    """
    Production-grade financial data loader with quality validation.

    Features:
    - Schema validation on every load
    - Staleness detection
    - Missing value reporting
    - Outlier flagging (returns > 5 sigma)
    - Idempotent caching to disk
    """

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self._cache: Dict[str, pd.DataFrame] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def load_etf_universe(
        self,
        start: str = DATA_START_DATE,
        end: str = DATA_END_DATE,
        refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for all ETFs in the universe.

        Returns:
            Dict mapping ticker → DataFrame with validated OHLCV data.
        """
        data = {}
        quality_report = []

        for ticker in ETF_UNIVERSE:
            df = self._load_single(ticker, start, end, refresh)
            report = self._validate(df, ticker)
            quality_report.append(report)
            data[ticker] = df

        self._log_quality_report(quality_report)
        return data

    def load_returns(
        self,
        start: str = DATA_START_DATE,
        end: str = DATA_END_DATE,
    ) -> pd.DataFrame:
        """
        Load daily log returns for all ETFs. Returns aligned DataFrame.
        """
        data = self.load_etf_universe(start, end)
        closes = pd.DataFrame({t: df["Close"] for t, df in data.items()})
        returns = np.log(closes / closes.shift(1)).dropna()
        return returns

    def load_vix(
        self,
        start: str = DATA_START_DATE,
        end: str = DATA_END_DATE,
    ) -> pd.Series:
        """Load VIX index data."""
        df = self._load_single("^VIX", start, end)
        return df["Close"].rename("VIX")

    # ── Internal ──────────────────────────────────────────────────────────

    def _load_single(
        self,
        ticker: str,
        start: str,
        end: str,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Load a single ticker, using disk cache when available."""
        safe_name = ticker.replace("^", "").upper()
        cache_path = DATA_DIR / f"{safe_name}_ohlcv.parquet"

        if not refresh and cache_path.exists():
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded {ticker} from cache ({len(df)} rows)")
            return df

        logger.info(f"Downloading {ticker} from Yahoo Finance...")
        df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            raise DataQualityError(f"No data returned for {ticker}")

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.to_parquet(cache_path)
        logger.info(f"Saved {ticker} to {cache_path} ({len(df)} rows)")
        return df

    def _validate(self, df: pd.DataFrame, ticker: str) -> Dict:
        """
        Run data quality checks. Returns a quality report dict.

        Checks:
        1. Schema: required columns present
        2. Completeness: % of non-null values
        3. Staleness: last data point recency
        4. Sufficiency: minimum history length
        5. Outliers: returns beyond 5 sigma
        """
        report = {"ticker": ticker, "rows": len(df), "issues": []}

        # 1. Schema
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            report["issues"].append(f"Missing columns: {missing_cols}")

        # 2. Completeness
        completeness = df[self.REQUIRED_COLUMNS].notna().mean().min()
        report["completeness"] = float(completeness)
        if completeness < self.config.min_feature_completeness:
            report["issues"].append(
                f"Completeness {completeness:.2%} < {self.config.min_feature_completeness:.2%}"
            )

        # 3. Staleness
        if hasattr(df.index, 'max'):
            last_date = pd.Timestamp(df.index.max())
            staleness = (pd.Timestamp.now() - last_date).total_seconds() / 3600
            report["staleness_hours"] = staleness
            # Only warn on business days
            if staleness > self.config.data_staleness_hours * 3:  # ~3 business days
                report["issues"].append(f"Data is {staleness:.0f}h old")

        # 4. Sufficiency
        if len(df) < MIN_HISTORY_DAYS:
            report["issues"].append(
                f"Only {len(df)} rows, need {MIN_HISTORY_DAYS}"
            )

        # 5. Outlier detection (returns > 5 sigma)
        if "Close" in df.columns:
            returns = df["Close"].pct_change().dropna()
            sigma = returns.std()
            outliers = (returns.abs() > 5 * sigma).sum()
            report["outlier_count"] = int(outliers)
            if outliers > 0:
                report["issues"].append(f"{outliers} return outliers (>5σ)")

        if report["issues"]:
            logger.warning(f"Data quality issues for {ticker}: {report['issues']}")

        return report

    def _log_quality_report(self, reports: list) -> None:
        """Log a summary of all data quality checks."""
        total_issues = sum(len(r["issues"]) for r in reports)
        if total_issues == 0:
            logger.info("✓ All data quality checks passed")
        else:
            logger.warning(f"⚠ {total_issues} data quality issues detected")
            for r in reports:
                if r["issues"]:
                    for issue in r["issues"]:
                        logger.warning(f"  {r['ticker']}: {issue}")
