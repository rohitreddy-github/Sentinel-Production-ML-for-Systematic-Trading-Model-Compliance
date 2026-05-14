# System Architecture: Risk-Aware Portfolio Optimization

## Production-Grade ML System for Financial Institutions

---

## 1. Business Problem Definition

### Use Case: Multi-Asset Directional Signal Generation

**Problem Statement:** Generate daily buy/hold/sell signals for a universe of 7 ETFs
(QQQ, SPY, GLD, TLT, IWM, XLE, XLF) using ML-driven return forecasting, market
regime awareness, and risk management — suitable for integration into a systematic
trading desk or robo-advisory platform.

### Business KPIs

| KPI | Target | Measurement |
|-----|--------|-------------|
| **Signal Accuracy (Directional)** | >55% out-of-sample | Walk-forward cross-validation |
| **Information Ratio** | >0.5 annualized | Risk-adjusted excess return vs. benchmark |
| **Maximum Drawdown** | <15% | Worst peak-to-trough decline |
| **Sharpe Ratio** | >1.0 annualized | Excess return / portfolio volatility |
| **Value at Risk (95%)** | <2% daily | Historical + parametric VaR |
| **Model Staleness** | Retrain within 5 business days of drift alert | PSI > 0.2 triggers retraining |
| **Prediction Latency** | <200ms p99 | API response time |
| **Audit Completeness** | 100% | Every prediction logged with features + SHAP values |

### Regulatory Context

- **MiFID II (EU):** Requires explainability for algorithmic trading decisions
- **SR 11-7 (US Fed):** Model risk management — validation, monitoring, governance
- **SEC Rule 15c3-5:** Risk controls for market access
- This system provides the **audit trail, explainability, and monitoring** required

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RISK-AWARE PORTFOLIO SYSTEM                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐ │
│  │  Data Layer   │──▶│  ML Pipeline  │──▶│  Serving API  │──▶│  Dashboard  │ │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └─────────────┘ │
│         │                  │                   │                            │
│  ┌──────▼───────┐   ┌──────▼───────┐   ┌──────▼───────┐                   │
│  │ Data Quality  │   │   Model      │   │   Audit      │                   │
│  │ Monitoring    │   │   Registry   │   │   Logger     │                   │
│  └──────────────┘   └──────────────┘   └──────────────┘                   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Monitoring & Alerting Stack                        │   │
│  │    Model Drift  │  Data Quality  │  Performance  │  Risk Metrics     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 Data Layer (`pipeline/data_loader.py`)
- **Market data:** Yahoo Finance (daily OHLCV) with validation
- **Macro data:** FRED API (GDP, CPI, Fed Funds, Unemployment)
- **Regime data:** HMM-based market regime classification
- **Data quality gates:** Schema validation, staleness checks, outlier detection

### 3.2 Feature Engineering (`pipeline/feature_engine.py`)
- **Technical indicators:** RSI, MACD, Bollinger Bands, ATR
- **Cross-asset features:** Inter-ETF correlations, relative strength
- **Risk features:** Rolling volatility, VIX ratio, drawdown
- **Regime features:** HMM state probabilities
- **Stationarity handling:** Returns-based features, differencing, rolling z-scores

### 3.3 ML Pipeline (`pipeline/model_trainer.py`)
- **Target variable:** Next-day return direction (fixed from MACD leakage)
- **Class imbalance:** SMOTE, class weights, threshold optimization
- **Walk-forward validation:** No future data leakage
- **Models:** XGBoost (primary), LSTM (secondary), ensemble
- **Adversarial robustness:** Feature perturbation testing, regime-conditional evaluation

### 3.4 Model Governance (`pipeline/model_registry.py`)
- Version-controlled model artifacts with metadata
- Champion/challenger model comparison
- Automated validation against minimum performance thresholds
- Full lineage tracking: data version → features → model → predictions

### 3.5 Serving Layer (`api/serve.py`)
- FastAPI-based prediction endpoint
- SHAP explanations attached to every prediction
- Request/response audit logging
- Health checks and readiness probes

### 3.6 Monitoring (`monitoring/drift_detector.py`, `monitoring/data_quality.py`)
- **Data drift:** Population Stability Index (PSI) per feature
- **Concept drift:** Rolling accuracy, precision/recall tracking
- **Data quality:** Missing values, schema violations, range checks
- **Risk metrics:** Real-time VaR, CVaR, portfolio drawdown

---

## 4. Handling Financial ML Challenges

### 4.1 Non-Stationarity
- All features computed as returns or rolling z-scores (not raw prices)
- Regime-conditional model evaluation (separate accuracy for bull/bear)
- Periodic retraining with expanding window
- PSI-based drift detection triggers automatic retraining

### 4.2 Class Imbalance
- Financial returns are approximately balanced (up/down ~50/50)
- For tail-event prediction: SMOTE oversampling + class_weight adjustment
- Threshold optimization via precision-recall curve
- Evaluation with F1, MCC, and profit-weighted metrics (not just accuracy)

### 4.3 Adversarial Behavior / Regime Shifts
- HMM regime detection conditions model behavior
- Stress testing on historical crisis periods (2020 COVID, 2022 rate hikes)
- Feature importance stability checks across time windows
- Ensemble of models trained on different market regimes

---

## 5. Docker Architecture

```
docker-compose.yml
├── trainer        # ML training pipeline (batch job)
├── api            # FastAPI prediction server
├── dashboard      # Streamlit monitoring dashboard
└── prometheus     # Metrics collection (optional)
```

---

## 6. Portfolio Presentation Strategy

### What Makes This Impressive to Fintech/Trading Firms

1. **Demonstrates awareness of real failure modes** (leakage, non-stationarity, look-ahead bias)
2. **Production mindset** — not just a notebook, but a deployable system
3. **Regulatory awareness** — audit logging, explainability, model governance
4. **Risk management** — VaR, CVaR, drawdown monitoring built-in
5. **Clean software engineering** — typed Python, Docker, API design, tests
6. **Honest metrics** — walk-forward validation, not inflated accuracy

### Talking Points for Interviews
- "I discovered and fixed data leakage that inflated accuracy from 98% to realistic 55-60%"
- "The system includes SHAP-based explainability for every prediction, meeting MiFID II requirements"
- "Model drift detection uses PSI with configurable thresholds and automated retraining triggers"
- "Walk-forward cross-validation ensures no future information leaks into training"
