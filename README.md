# Risk-Aware Portfolio Optimization

### Production-Grade ML System for Financial Signal Generation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/deploy-Docker-2496ED.svg)](https://docker.com)

---

## Problem Statement

Generate **daily directional trading signals** for a 7-ETF universe (QQQ, SPY, GLD, TLT, IWM, XLE, XLF) using machine learning, with full **explainability**, **audit logging**, and **model governance** — suitable for integration into a systematic trading desk or robo-advisory platform.

### Business KPIs

| Metric | Target | How Measured |
|--------|--------|-------------|
| Directional Accuracy | >55% OOS | Walk-forward cross-validation |
| Information Ratio | >0.5 | Risk-adjusted excess return vs. SPY |
| Max Drawdown | <15% | Peak-to-trough decline |
| Prediction Latency | <200ms p99 | API response time |
| Audit Completeness | 100% | Every prediction logged with SHAP |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                  RISK-AWARE PORTFOLIO SYSTEM                         │
│                                                                      │
│  Data Layer ──▶ Feature Engine ──▶ ML Pipeline ──▶ Serving API       │
│      │               │                │               │              │
│  Validation      Stationarity    Walk-Forward      SHAP-based        │
│  & Quality       Transforms      Validation       Explanations       │
│                                       │                              │
│                               Model Registry                        │
│                            (Champion/Challenger)                     │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │              Monitoring & Compliance Layer                     │  │
│  │  PSI Drift Detection │ Data Quality │ Risk Metrics │ Audit    │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### ML Engineering (Not Just Notebooks)
- **Fixed data leakage**: Original pipeline used next-day MACD sign as target while MACD was a feature (98% accuracy → inflated). Now uses actual next-day return direction (~55-60% realistic accuracy)
- **Walk-forward cross-validation**: 5-fold expanding window with configurable gap to prevent information leakage
- **Per-fold scaling**: StandardScaler fit only on training data within each fold (no look-ahead bias)
- **Class imbalance handling**: Dynamic `scale_pos_weight` adjustment per fold
- **Stationarity**: Rolling z-score transforms on non-stationary features

### Explainability & Compliance
- **SHAP explanations** attached to every prediction (MiFID II ready)
- **Structured audit logging**: Every prediction, training run, and model promotion logged as JSON lines
- **Model governance registry**: Versioned models with champion/challenger promotion workflow
- **Risk breach logging**: Automatic alerts when VaR/drawdown limits are exceeded

### Monitoring
- **PSI-based drift detection**: Per-feature Population Stability Index with severity classification
- **Data quality gates**: Schema validation, completeness checks, range validation, staleness detection
- **Rolling performance tracking**: Accuracy, precision, recall monitored over time
- **Risk metrics**: VaR (historical + parametric), CVaR, Maximum Drawdown, Sharpe, Sortino

### Production Infrastructure
- **FastAPI serving**: `/predict` endpoint with SHAP explanations and <200ms latency
- **Docker Compose**: Training pipeline, API server, and dashboard as separate services
- **Streamlit dashboard**: 5-tab monitoring UI (signals, governance, risk, drift, audit)
- **Health checks**: Readiness probes for container orchestrators

---

## Project Structure

```
Risk-Aware-Portfolio-Optimization/
│
├── pipeline/                        # Production ML pipeline
│   ├── config.py                    # Centralized configuration
│   ├── data_loader.py               # Data loading with quality validation
│   ├── feature_engine.py            # Leakage-free feature engineering
│   ├── model_trainer.py             # Walk-forward training + SHAP
│   ├── model_registry.py            # Model governance & versioning
│   └── train_pipeline.py            # End-to-end orchestrator
│
├── monitoring/                      # Monitoring & compliance
│   ├── drift_detector.py            # PSI-based drift detection
│   ├── data_quality.py              # Data quality gates
│   ├── risk_metrics.py              # VaR, CVaR, drawdown computation
│   └── audit_logger.py              # Compliance-grade audit logging
│
├── api/                             # Serving layer
│   └── serve.py                     # FastAPI prediction server
│
├── streamlit_app/                   # Dashboard
│   ├── app.py                       # Original dashboard
│   └── app_production.py            # Production monitoring dashboard
│
├── notebooks/                       # Research & exploration
│   ├── data_collection.ipynb
│   ├── feature_engineering.ipynb
│   ├── regime_classification_hmm.ipynb
│   ├── return_prediction_xgboost_classifier.ipynb
│   ├── lstm_predictor.ipynb
│   ├── LSTM_vs_XGBOOST.ipynb
│   ├── shap_explainability_for_xgboost_&_lstm.ipynb
│   ├── backtesting.ipynb
│   └── sentiment_analysis_finbert.ipynb
│
├── tests/                           # Test suite
│   └── test_pipeline.py             # Pipeline, drift, registry tests
│
├── docs/                            # Documentation
│   └── SYSTEM_ARCHITECTURE.md       # Full system design document
│
├── Dockerfile                       # API server container
├── Dockerfile.trainer               # Training pipeline container
├── docker-compose.yml               # Multi-service orchestration
├── requirements-prod.txt            # Pinned production dependencies
└── requirements.txt                 # Original dependencies
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-prod.txt
```

### 2. Run Training Pipeline

```bash
python -m pipeline.train_pipeline --auto-promote
```

This will:
- Download ETF data from Yahoo Finance
- Run data quality validation
- Engineer features (leakage-free)
- Walk-forward train XGBoost (5 folds)
- Compute SHAP feature importance
- Register and validate the model
- Promote to champion if validation passes

### 3. Start API Server

```bash
uvicorn api.serve:app --host 0.0.0.0 --port 8000
```

### 4. Start Dashboard

```bash
streamlit run streamlit_app/app_production.py
```

### 5. Docker Deployment

```bash
# Train model
docker compose --profile train up trainer

# Start API + Dashboard
docker compose up api dashboard
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Get trading signal with SHAP explanation |
| GET | `/health` | Health check + model status |
| GET | `/model/info` | Current model version and performance |
| GET | `/model/versions` | All registered model versions |
| GET | `/audit/recent` | Recent prediction audit trail |

### Example: Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"QQQ_rsi": 55.2, "QQQ_macd_hist": 0.3, ...}}'
```

### Example: Response

```json
{
  "event_id": "a1b2c3d4-...",
  "signal": "BUY",
  "probability": 0.623,
  "confidence": "MEDIUM",
  "explanation": {
    "top_drivers": [
      {"feature": "QQQ_rsi", "shap_value": 0.15, "direction": "positive"},
      {"feature": "SPY_volatility", "shap_value": -0.08, "direction": "negative"}
    ]
  },
  "latency_ms": 12.5
}
```

---

## Issues Fixed from Original Implementation

| Issue | Severity | Fix |
|-------|----------|-----|
| **Data leakage** — target was next-day MACD sign while MACD was a feature | Critical | Target is now actual next-day return direction |
| **Look-ahead bias** — scaler fit on all data before split | High | Scaler fit only on training fold data |
| **Single split** — no proper time-series CV | High | Walk-forward validation with gap/embargo |
| **Inflated accuracy (98%)** — due to leakage | High | Realistic ~55-60% directional accuracy |
| **Hardcoded paths** — absolute Windows paths everywhere | Medium | Relative paths from `config.py` `PROJECT_ROOT` |
| **API keys in code** — Alpha Vantage + NewsAPI keys exposed | Medium | Moved to environment variables |
| **Missing requirements** — hmmlearn, scikit-learn, xgboost, etc. | Medium | Complete `requirements-prod.txt` |
| **Disconnected components** — macro/sentiment data never used | Medium | Documented as optional integrations |

---

## Regulatory Alignment

| Regulation | Requirement | Implementation |
|-----------|-------------|----------------|
| **SR 11-7** | Model risk management | Model registry, validation gates, champion/challenger |
| **MiFID II** | Algo trading decision audit | SHAP explanation on every prediction, full audit trail |
| **SEC 15c3-5** | Risk controls | VaR/CVaR limits, drawdown monitoring, position limits |

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| ML/Modeling | XGBoost, LSTM (PyTorch), SHAP, scikit-learn |
| Data | yfinance, pandas, NumPy, ta (technical analysis) |
| Regime Detection | Hidden Markov Models (hmmlearn) |
| API Serving | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit |
| Infrastructure | Docker, Docker Compose |
| Testing | pytest |
| Monitoring | Custom PSI drift detector, data quality gates |

---

## Running Tests

```bash
pytest tests/ -v
```

---


