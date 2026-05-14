"""
Risk metrics computation for portfolio monitoring.

Implements institutional-grade risk measures:
- Value at Risk (VaR) — Historical and Parametric
- Conditional VaR (CVaR / Expected Shortfall)
- Maximum Drawdown
- Portfolio Sharpe and Sortino ratios
- Regime-conditional risk assessment
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskMetrics:
    """
    Compute portfolio risk metrics for monitoring and compliance.

    All methods accept portfolio returns as input and return
    named metrics suitable for dashboarding and alerting.
    """

    @staticmethod
    def value_at_risk(
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Compute Value at Risk.

        Args:
            returns: Daily portfolio returns
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            method: "historical" or "parametric"

        Returns:
            VaR as a positive number (represents potential loss)
        """
        if method == "historical":
            var = -np.percentile(returns.dropna(), (1 - confidence) * 100)
        elif method == "parametric":
            from scipy.stats import norm
            mu = returns.mean()
            sigma = returns.std()
            var = -(mu + sigma * norm.ppf(1 - confidence))
        else:
            raise ValueError(f"Unknown VaR method: {method}")

        return float(var)

    @staticmethod
    def conditional_var(
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Compute Conditional VaR (Expected Shortfall).

        CVaR is the expected loss given that the loss exceeds VaR.
        More conservative than VaR and better captures tail risk.
        """
        threshold = np.percentile(returns.dropna(), (1 - confidence) * 100)
        tail_returns = returns[returns <= threshold]
        cvar = -tail_returns.mean() if len(tail_returns) > 0 else 0.0
        return float(cvar)

    @staticmethod
    def max_drawdown(returns: pd.Series) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        Compute maximum drawdown.

        Returns:
            (max_drawdown, peak_date, trough_date)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        max_dd = float(drawdown.min())
        trough_idx = drawdown.idxmin()
        peak_idx = cumulative[:trough_idx].idxmax() if trough_idx is not None else None

        return abs(max_dd), peak_idx, trough_idx

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.04,
        annualize: bool = True,
    ) -> float:
        """
        Compute Sharpe ratio.

        Args:
            risk_free_rate: Annual risk-free rate (default 4%)
            annualize: Whether to annualize the ratio
        """
        daily_rf = risk_free_rate / 252
        excess = returns - daily_rf
        ratio = excess.mean() / excess.std() if excess.std() > 0 else 0.0

        if annualize:
            ratio *= np.sqrt(252)

        return float(ratio)

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.04,
        annualize: bool = True,
    ) -> float:
        """
        Compute Sortino ratio (uses downside deviation only).
        """
        daily_rf = risk_free_rate / 252
        excess = returns - daily_rf
        downside = excess[excess < 0]
        downside_std = downside.std() if len(downside) > 0 else 1e-8
        ratio = excess.mean() / downside_std

        if annualize:
            ratio *= np.sqrt(252)

        return float(ratio)

    @classmethod
    def compute_all(
        cls,
        returns: pd.Series,
        risk_free_rate: float = 0.04,
    ) -> Dict[str, float]:
        """
        Compute all risk metrics in one call.

        Returns:
            Dict with all named risk metrics.
        """
        max_dd, peak, trough = cls.max_drawdown(returns)

        metrics = {
            "var_95_historical": cls.value_at_risk(returns, 0.95, "historical"),
            "var_99_historical": cls.value_at_risk(returns, 0.99, "historical"),
            "var_95_parametric": cls.value_at_risk(returns, 0.95, "parametric"),
            "cvar_95": cls.conditional_var(returns, 0.95),
            "cvar_99": cls.conditional_var(returns, 0.99),
            "max_drawdown": max_dd,
            "max_drawdown_peak": str(peak) if peak else None,
            "max_drawdown_trough": str(trough) if trough else None,
            "sharpe_ratio": cls.sharpe_ratio(returns, risk_free_rate),
            "sortino_ratio": cls.sortino_ratio(returns, risk_free_rate),
            "annualized_return": float((1 + returns.mean()) ** 252 - 1),
            "annualized_volatility": float(returns.std() * np.sqrt(252)),
            "skewness": float(returns.skew()),
            "kurtosis": float(returns.kurtosis()),
            "positive_days_pct": float((returns > 0).mean()),
            "n_observations": len(returns),
        }

        return metrics
