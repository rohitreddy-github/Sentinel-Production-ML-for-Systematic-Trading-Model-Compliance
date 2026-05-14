"""
Production feature engineering with stationarity handling and leakage prevention.

Fixes the critical data leakage in the original pipeline where:
1. Target was next-day MACD sign (while MACD was also a feature)
2. Now uses actual next-day return direction as target
3. All features use only past data (no look-ahead)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange

from pipeline.config import (
    ETF_UNIVERSE,
    PRIMARY_TARGET,
    TECHNICAL_INDICATORS as TI,
    ROLLING_ZSCORE_WINDOW,
)

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Builds a supervised learning dataset from raw OHLCV data.

    Design principles:
    - NO future information in features (strict look-back only)
    - Target is actual next-day return direction (not MACD sign)
    - Features are made stationary via rolling z-scores
    - Cross-asset features capture inter-market dynamics
    - Feature metadata is tracked for audit/explainability
    """

    def __init__(self):
        self.feature_names: List[str] = []
        self.feature_metadata: Dict[str, Dict] = {}

    def build_dataset(
        self,
        etf_data: Dict[str, pd.DataFrame],
        regimes: Optional[pd.Series] = None,
        vix: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build the complete supervised dataset.

        Args:
            etf_data: Dict of ticker → OHLCV DataFrame
            regimes: Optional HMM regime labels (0=Bull, 1=Bear)
            vix: Optional VIX close series

        Returns:
            (features_df, target_series) — aligned, clean, no NaNs
        """
        logger.info("Building supervised dataset...")

        all_features = []

        # Per-ETF technical features
        for ticker in ETF_UNIVERSE:
            if ticker not in etf_data:
                logger.warning(f"Missing data for {ticker}, skipping")
                continue
            df = etf_data[ticker].copy()
            features = self._compute_technical_features(df, ticker)
            all_features.append(features)

        # Merge all feature DataFrames
        merged = pd.concat(all_features, axis=1)

        # Cross-asset features
        cross_features = self._compute_cross_asset_features(etf_data)
        merged = merged.join(cross_features, how="left")

        # VIX features
        if vix is not None:
            vix_features = self._compute_vix_features(vix)
            merged = merged.join(vix_features, how="left")

        # Regime features
        if regimes is not None:
            merged["regime"] = regimes
            merged["regime"] = merged["regime"].ffill()

        # Target: actual next-day return direction for PRIMARY_TARGET
        target_ticker = PRIMARY_TARGET
        if target_ticker in etf_data:
            close = etf_data[target_ticker]["Close"]
            # Next-day return = close[t+1] / close[t] - 1
            next_day_return = close.pct_change().shift(-1)
            target = (next_day_return > 0).astype(int)
            target.name = "target"
        else:
            raise ValueError(f"Primary target {target_ticker} not in data")

        # Align features and target
        merged, target = merged.align(target, join="inner", axis=0)

        # Drop the LAST row (target is NaN because we shifted -1)
        valid_mask = target.notna()
        merged = merged[valid_mask]
        target = target[valid_mask].astype(int)

        # Drop rows with NaN features (from rolling indicators warmup)
        valid_mask = merged.notna().all(axis=1)
        merged = merged[valid_mask]
        target = target[valid_mask]

        self.feature_names = merged.columns.tolist()
        logger.info(
            f"Dataset built: {len(merged)} samples, {len(self.feature_names)} features"
        )
        logger.info(
            f"Target distribution: {target.value_counts().to_dict()}"
        )

        return merged, target

    # ── Technical Features ────────────────────────────────────────────────

    def _compute_technical_features(
        self, df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """Compute technical indicators for a single ETF."""
        features = pd.DataFrame(index=df.index)
        prefix = ticker

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # Daily returns (log)
        features[f"{prefix}_log_return"] = np.log(close / close.shift(1))

        # RSI
        rsi = RSIIndicator(close=close, window=TI["rsi_window"])
        features[f"{prefix}_rsi"] = rsi.rsi()

        # MACD (use histogram, not raw MACD to reduce autocorrelation)
        macd = MACD(
            close=close,
            window_fast=TI["macd_fast"],
            window_slow=TI["macd_slow"],
            window_sign=TI["macd_signal"],
        )
        features[f"{prefix}_macd_hist"] = macd.macd_diff()

        # Bollinger Band Width (normalized)
        bb = BollingerBands(
            close=close,
            window=TI["bollinger_window"],
            window_dev=TI["bollinger_std"],
        )
        features[f"{prefix}_bb_width"] = bb.bollinger_wband()
        features[f"{prefix}_bb_pct"] = bb.bollinger_pband()

        # ATR (normalized by close price)
        atr = AverageTrueRange(
            high=high, low=low, close=close, window=TI["atr_window"]
        )
        features[f"{prefix}_atr_norm"] = atr.average_true_range() / close

        # Rolling volatility (annualized)
        log_ret = features[f"{prefix}_log_return"]
        features[f"{prefix}_volatility"] = (
            log_ret.rolling(TI["volatility_window"]).std() * np.sqrt(252)
        )

        # Rolling Sharpe ratio
        features[f"{prefix}_sharpe"] = (
            log_ret.rolling(TI["sharpe_window"]).mean()
            / log_ret.rolling(TI["sharpe_window"]).std()
        ) * np.sqrt(252)

        # Momentum (5-day, 21-day returns)
        features[f"{prefix}_mom_5d"] = close.pct_change(5)
        features[f"{prefix}_mom_21d"] = close.pct_change(21)

        # Apply rolling z-score for stationarity
        zscore_cols = [
            f"{prefix}_macd_hist",
            f"{prefix}_bb_width",
            f"{prefix}_atr_norm",
            f"{prefix}_volatility",
        ]
        for col in zscore_cols:
            if col in features.columns:
                rolling_mean = features[col].rolling(ROLLING_ZSCORE_WINDOW).mean()
                rolling_std = features[col].rolling(ROLLING_ZSCORE_WINDOW).std()
                features[f"{col}_zscore"] = (
                    (features[col] - rolling_mean) / rolling_std.replace(0, np.nan)
                )

        return features

    # ── Cross-Asset Features ──────────────────────────────────────────────

    def _compute_cross_asset_features(
        self, etf_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Compute inter-ETF features: relative strength, correlations."""
        features = pd.DataFrame()

        # Build close price matrix
        closes = pd.DataFrame({
            t: df["Close"] for t, df in etf_data.items() if t in ETF_UNIVERSE
        })
        returns = np.log(closes / closes.shift(1))

        if PRIMARY_TARGET in returns.columns:
            target_ret = returns[PRIMARY_TARGET]

            for ticker in ETF_UNIVERSE:
                if ticker == PRIMARY_TARGET or ticker not in returns.columns:
                    continue

                # Relative strength (rolling ratio of cumulative returns)
                features[f"rs_{PRIMARY_TARGET}_vs_{ticker}"] = (
                    target_ret.rolling(21).sum()
                    - returns[ticker].rolling(21).sum()
                )

                # Rolling correlation
                features[f"corr_{PRIMARY_TARGET}_{ticker}"] = (
                    target_ret.rolling(TI["correlation_window"]).corr(
                        returns[ticker]
                    )
                )

        return features

    # ── VIX Features ──────────────────────────────────────────────────────

    def _compute_vix_features(self, vix: pd.Series) -> pd.DataFrame:
        """Compute VIX-derived features."""
        features = pd.DataFrame(index=vix.index)

        features["vix_level"] = vix
        features["vix_change"] = vix.pct_change()
        features["vix_ma_ratio"] = vix / vix.rolling(21).mean()
        features["vix_zscore"] = (
            (vix - vix.rolling(ROLLING_ZSCORE_WINDOW).mean())
            / vix.rolling(ROLLING_ZSCORE_WINDOW).std()
        )

        return features
