import numpy as np
import pandas as pd

# === Technical Indicators ===

# RSI (Relative Strength Index) calculation
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# MACD (Moving Average Convergence Divergence) calculation
def compute_macd(series, short=12, long=26, signal=9):
    exp1 = series.ewm(span=short, adjust=False).mean()
    exp2 = series.ewm(span=long, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Bollinger Bands calculation
def compute_bollinger(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    return sma, upper_band, lower_band

# === Risk Metrics ===

def sharpe_ratio(returns, risk_free_rate=0.01):
    excess = returns - (risk_free_rate / 252)
    return (excess.mean() / returns.std()) * np.sqrt(252)

def sortino_ratio(returns, risk_free_rate=0.01):
    downside = returns[returns < 0]
    return (returns.mean() - risk_free_rate / 252) / downside.std() * np.sqrt(252)

def max_drawdown(returns):
    cum_return = (1 + returns).cumprod()
    peak = cum_return.cummax()
    drawdown = (cum_return - peak) / peak
    return drawdown.min()

