"""
FastAPI prediction server with SHAP explanations and audit logging.

Endpoints:
- POST /predict        → Get trading signal with SHAP explanation
- GET  /health         → Health check + model status
- GET  /model/info     → Current model version and performance
- GET  /risk/metrics   → Current portfolio risk metrics
- GET  /audit/recent   → Recent prediction audit trail

Every prediction is:
1. Validated (input data quality checks)
2. Explained (SHAP values for top feature drivers)
3. Logged (full audit trail for compliance)
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline.config import API_HOST, API_PORT
from pipeline.model_registry import ModelRegistry
from monitoring.audit_logger import AuditLogger
from monitoring.data_quality import DataQualityMonitor
from monitoring.risk_metrics import RiskMetrics

logger = logging.getLogger(__name__)

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Risk-Aware Portfolio Signal API",
    description=(
        "Production ML serving for financial signal generation "
        "with SHAP explainability and compliance-grade audit logging."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────────────────

registry = ModelRegistry()
audit_logger = AuditLogger()
data_quality_monitor = DataQualityMonitor()

# Load champion model at startup
_champion_model = None
_model_version = None
_feature_names = []
_shap_explainer = None


@app.on_event("startup")
async def load_model():
    """Load the champion model from the registry at startup."""
    global _champion_model, _model_version, _feature_names, _shap_explainer

    try:
        champion = registry.load_champion()
        if champion:
            _champion_model = champion
            _model_version = champion.version
            _feature_names = champion.metadata.get("feature_names", [])
            logger.info(f"Loaded champion model: {_model_version}")

            # Pre-warm SHAP explainer
            import shap
            _shap_explainer = shap.TreeExplainer(champion.model)
            logger.info("SHAP explainer initialized")
        else:
            logger.warning("No champion model found in registry")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


# ─── Request/Response Models ─────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    """Input features for prediction."""
    features: Dict[str, float] = Field(
        ...,
        description="Feature name → value mapping",
        example={"QQQ_rsi": 55.2, "QQQ_macd_hist": 0.3, "QQQ_bb_width": 0.02},
    )
    include_explanation: bool = Field(
        default=True,
        description="Whether to include SHAP explanations",
    )


class PredictionResponse(BaseModel):
    """Prediction output with explanation."""
    event_id: str
    timestamp: str
    model_version: str
    signal: str  # "BUY" or "SELL"
    probability: float
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    explanation: Optional[Dict[str, Any]] = None
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: float
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    version: str
    feature_count: int
    feature_names: List[str]
    training_accuracy: Optional[float]
    training_f1: Optional[float]
    registered_at: Optional[str]


class RiskResponse(BaseModel):
    """Portfolio risk metrics."""
    metrics: Dict[str, Any]
    timestamp: str


# ─── Startup time ─────────────────────────────────────────────────────────────

_startup_time = time.time()


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate a trading signal with SHAP-based explanation.

    The prediction, its inputs, and SHAP values are logged to the
    audit trail for regulatory compliance.
    """
    start_time = time.time()

    if _champion_model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Validate input features
    missing = set(_feature_names) - set(request.features.keys())
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {list(missing)[:5]}... ({len(missing)} total)",
        )

    # Build feature vector in correct order
    feature_vector = np.array(
        [request.features[f] for f in _feature_names]
    ).reshape(1, -1)

    # Scale
    X_scaled = _champion_model.scaler.transform(feature_vector)

    # Predict
    prediction = int(_champion_model.model.predict(X_scaled)[0])
    probability = float(_champion_model.model.predict_proba(X_scaled)[0, 1])

    # Determine confidence
    if probability > 0.65 or probability < 0.35:
        confidence = "HIGH"
    elif probability > 0.55 or probability < 0.45:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # SHAP explanation
    explanation = None
    shap_dict = {}
    if request.include_explanation and _shap_explainer is not None:
        try:
            shap_values = _shap_explainer.shap_values(X_scaled)[0]
            shap_dict = dict(zip(_feature_names, shap_values.tolist()))
            sorted_features = sorted(
                shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
            )
            explanation = {
                "top_drivers": [
                    {"feature": k, "shap_value": round(v, 6), "direction": "positive" if v > 0 else "negative"}
                    for k, v in sorted_features[:5]
                ],
                "base_value": float(_shap_explainer.expected_value)
                if isinstance(_shap_explainer.expected_value, (int, float))
                else float(_shap_explainer.expected_value[0]),
            }
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")

    latency_ms = (time.time() - start_time) * 1000

    # Audit log
    event_id = audit_logger.log_prediction(
        model_version=_model_version,
        features=request.features,
        prediction=prediction,
        probability=probability,
        shap_explanation=shap_dict,
        metadata={"latency_ms": latency_ms, "confidence": confidence},
    )

    return PredictionResponse(
        event_id=event_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        model_version=_model_version,
        signal="BUY" if prediction == 1 else "SELL",
        probability=round(probability, 6),
        confidence=confidence,
        explanation=explanation,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for load balancers and orchestrators."""
    return HealthResponse(
        status="healthy" if _champion_model is not None else "degraded",
        model_loaded=_champion_model is not None,
        model_version=_model_version,
        uptime_seconds=round(time.time() - _startup_time, 1),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return current model metadata."""
    if _champion_model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    return ModelInfoResponse(
        version=_model_version,
        feature_count=len(_feature_names),
        feature_names=_feature_names,
        training_accuracy=_champion_model.metadata.get("aggregate_accuracy"),
        training_f1=_champion_model.metadata.get("aggregate_f1"),
        registered_at=_champion_model.metadata.get("timestamp"),
    )


@app.get("/audit/recent")
async def recent_audit(limit: int = 20):
    """Retrieve recent prediction audit entries."""
    entries = audit_logger.get_prediction_log(limit=limit)
    return {"count": len(entries), "entries": entries}


@app.get("/model/versions")
async def model_versions():
    """List all registered model versions."""
    versions = registry.list_versions()
    champion = registry.get_champion_version()
    return {
        "champion": champion,
        "versions": versions,
    }


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "api.serve:app",
        host=API_HOST,
        port=API_PORT,
        workers=1,
        reload=False,
    )
