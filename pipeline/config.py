"""
Configuration module for the Risk-Aware Portfolio Optimization pipeline.
Centralizes all constants, paths, and hyperparameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import os


# ─── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
FIGURES_DIR = PROJECT_ROOT / "figures"
REGISTRY_DIR = PROJECT_ROOT / "model_registry"

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR, REGISTRY_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Universe ────────────────────────────────────────────────────────────────

ETF_UNIVERSE: List[str] = ["QQQ", "SPY", "GLD", "TLT", "IWM", "XLE", "XLF"]
PRIMARY_TARGET: str = "QQQ"
BENCHMARK: str = "SPY"

# ─── Data ─────────────────────────────────────────────────────────────────────

DATA_START_DATE: str = "2015-01-01"
DATA_END_DATE: str = "2025-12-31"
MIN_HISTORY_DAYS: int = 252  # At least 1 year of trading days

# ─── Feature Engineering ─────────────────────────────────────────────────────

TECHNICAL_INDICATORS = {
    "rsi_window": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_window": 20,
    "bollinger_std": 2,
    "atr_window": 14,
    "volatility_window": 21,
    "sharpe_window": 63,  # ~3 months
    "correlation_window": 63,
}

ROLLING_ZSCORE_WINDOW: int = 63  # Stationarity transform window


# ─── Model Hyperparameters ───────────────────────────────────────────────────

@dataclass
class XGBoostConfig:
    """XGBoost classifier hyperparameters."""
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0  # Adjusted dynamically for class imbalance
    random_state: int = 42
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 20

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class LSTMConfig:
    """LSTM model hyperparameters."""
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    sequence_length: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 50
    patience: int = 10  # Early stopping patience
    bidirectional: bool = False


# ─── Walk-Forward Validation ──────────────────────────────────────────────────

@dataclass
class ValidationConfig:
    """Walk-forward cross-validation settings."""
    n_splits: int = 5
    train_min_size: int = 504   # ~2 years minimum training data
    test_size: int = 63          # ~3 months test window
    gap: int = 1                 # 1-day gap to prevent leakage
    embargo_days: int = 5        # Extra buffer after test period


# ─── Monitoring Thresholds ────────────────────────────────────────────────────

@dataclass
class MonitoringConfig:
    """Thresholds for drift detection and alerting."""
    psi_warning: float = 0.1
    psi_critical: float = 0.2
    accuracy_min: float = 0.52    # Below this = model is no better than random
    sharpe_min: float = 0.3
    max_drawdown_limit: float = 0.20
    var_95_limit: float = 0.03    # 3% daily VaR limit
    data_staleness_hours: int = 26  # Alert if data older than this
    min_feature_completeness: float = 0.95  # 95% non-null required


# ─── Risk Management ─────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    """Portfolio risk constraints."""
    max_position_size: float = 0.30      # Max 30% in any single asset
    min_position_size: float = 0.02      # Min 2% if allocated
    max_sector_concentration: float = 0.50
    max_leverage: float = 1.0            # No leverage by default
    rebalance_frequency_days: int = 5    # Weekly rebalancing
    transaction_cost_bps: float = 5.0    # 5 basis points per trade
    slippage_bps: float = 2.0            # 2 basis points slippage


# ─── Audit ────────────────────────────────────────────────────────────────────

AUDIT_LOG_FILE = LOGS_DIR / "audit_log.jsonl"
TRAINING_LOG_FILE = LOGS_DIR / "training_log.jsonl"
PREDICTION_LOG_FILE = LOGS_DIR / "prediction_log.jsonl"

# ─── API ──────────────────────────────────────────────────────────────────────

API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_WORKERS: int = int(os.getenv("API_WORKERS", "2"))
