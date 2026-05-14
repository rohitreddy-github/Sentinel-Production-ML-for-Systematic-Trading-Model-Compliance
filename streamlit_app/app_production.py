"""
Production Streamlit Dashboard for Risk-Aware Portfolio Optimization.

Tabs:
1. Signal Generator — Live predictions with SHAP explanations
2. Model Governance — Registry, versions, champion status
3. Risk Monitor — VaR, CVaR, drawdown, Sharpe tracking
4. Drift Detection — PSI monitoring, data quality
5. Audit Trail — Prediction and lifecycle event logs
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import (
    LOGS_DIR,
    REGISTRY_DIR,
    PREDICTION_LOG_FILE,
    AUDIT_LOG_FILE,
)

st.set_page_config(
    page_title="Risk-Aware Portfolio System",
    page_icon="📊",
    layout="wide",
)


def main():
    st.title("📊 Risk-Aware Portfolio Optimization")
    st.caption("Production ML System with Explainability & Compliance")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Signal Generator",
        "📦 Model Governance",
        "⚠️ Risk Monitor",
        "📉 Drift Detection",
        "📋 Audit Trail",
    ])

    with tab1:
        render_signal_tab()

    with tab2:
        render_governance_tab()

    with tab3:
        render_risk_tab()

    with tab4:
        render_drift_tab()

    with tab5:
        render_audit_tab()


# ─── Tab 1: Signal Generator ─────────────────────────────────────────────────


def render_signal_tab():
    st.header("Trading Signal Generator")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Model Status")

        # Check if champion model exists
        champion_dir = REGISTRY_DIR / "champion"
        if champion_dir.exists():
            try:
                with open(champion_dir / "metadata.json") as f:
                    meta = json.load(f)
                st.success(f"Champion: {meta.get('version', 'unknown')}")
                st.metric("Accuracy", f"{meta.get('aggregate_accuracy', 0):.2%}")
                st.metric("F1 Score", f"{meta.get('aggregate_f1', 0):.2%}")
                st.metric("MCC", f"{meta.get('aggregate_mcc', 0):.4f}")
                st.metric("Features", meta.get("n_features", 0))
            except Exception as e:
                st.error(f"Error loading model metadata: {e}")
        else:
            st.warning("No champion model found. Run the training pipeline first.")
            st.code("python -m pipeline.train_pipeline --auto-promote", language="bash")

    with col2:
        st.subheader("SHAP Feature Importance")

        # Try to load SHAP figures
        figures_dir = PROJECT_ROOT / "figures"
        shap_images = list(figures_dir.glob("*shap*")) if figures_dir.exists() else []

        if shap_images:
            for img in shap_images[:2]:
                st.image(str(img), use_container_width=True)
        else:
            st.info("Run the training pipeline to generate SHAP visualizations.")


# ─── Tab 2: Model Governance ─────────────────────────────────────────────────


def render_governance_tab():
    st.header("Model Governance & Registry")

    registry_file = REGISTRY_DIR / "registry.json"
    if not registry_file.exists():
        st.warning("No models registered yet.")
        return

    with open(registry_file) as f:
        registry = json.load(f)

    # Champion status
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Champion")
        champion = registry.get("champion")
        if champion:
            st.success(f"Version: {champion}")
        else:
            st.warning("No champion designated")

    with col2:
        st.subheader("Promotion History")
        history = registry.get("promotion_history", [])
        if history:
            df = pd.DataFrame(history)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No promotions yet")

    # All versions
    st.subheader("Registered Versions")
    versions = registry.get("versions", {})
    if versions:
        rows = []
        for v, info in sorted(versions.items()):
            rows.append({
                "Version": v,
                "Status": info.get("status", "unknown"),
                "Accuracy": f"{info.get('accuracy', 0):.4f}",
                "F1": f"{info.get('f1', 0):.4f}",
                "Registered": info.get("timestamp", "unknown"),
                "Data Hash": info.get("data_hash", "unknown")[:8],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ─── Tab 3: Risk Monitor ─────────────────────────────────────────────────────


def render_risk_tab():
    st.header("Risk Metrics Monitor")

    st.info(
        "Risk metrics are computed from portfolio returns. "
        "Run the backtesting pipeline to populate these metrics."
    )

    # Show example risk metrics layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("VaR (95%)", "—", help="Value at Risk at 95% confidence")
    with col2:
        st.metric("CVaR (95%)", "—", help="Conditional VaR (Expected Shortfall)")
    with col3:
        st.metric("Max Drawdown", "—", help="Worst peak-to-trough decline")
    with col4:
        st.metric("Sharpe Ratio", "—", help="Risk-adjusted return")

    st.subheader("Risk Limits Configuration")
    st.json({
        "max_position_size": "30%",
        "max_drawdown_limit": "20%",
        "var_95_limit": "3%",
        "max_leverage": "1.0x",
        "transaction_cost_bps": "5 bps",
    })


# ─── Tab 4: Drift Detection ──────────────────────────────────────────────────


def render_drift_tab():
    st.header("Model & Data Drift Detection")

    drift_log = LOGS_DIR / "drift_events.jsonl"
    if not drift_log.exists():
        st.info("No drift events logged yet. Drift detection runs during prediction.")
        return

    events = []
    with open(drift_log) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    if events:
        df = pd.DataFrame(events)
        st.dataframe(df, use_container_width=True)

        # Severity distribution
        if "severity" in df.columns:
            st.subheader("Drift Severity Distribution")
            severity_counts = df["severity"].value_counts()
            st.bar_chart(severity_counts)
    else:
        st.info("No drift events recorded")


# ─── Tab 5: Audit Trail ──────────────────────────────────────────────────────


def render_audit_tab():
    st.header("Compliance Audit Trail")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction Log")
        if PREDICTION_LOG_FILE.exists():
            entries = []
            with open(PREDICTION_LOG_FILE) as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))

            if entries:
                # Show recent entries
                recent = entries[-20:]
                display_rows = []
                for e in recent:
                    display_rows.append({
                        "Time": e.get("timestamp", "")[:19],
                        "Signal": e.get("output", {}).get("signal", ""),
                        "Probability": f"{e.get('output', {}).get('probability', 0):.4f}",
                        "Model": e.get("model_version", ""),
                        "Event ID": e.get("event_id", "")[:8],
                    })
                st.dataframe(pd.DataFrame(display_rows), use_container_width=True)
                st.caption(f"Showing {len(recent)} of {len(entries)} entries")
            else:
                st.info("No predictions logged yet")
        else:
            st.info("Prediction log not found")

    with col2:
        st.subheader("Lifecycle Events")
        if AUDIT_LOG_FILE.exists():
            entries = []
            with open(AUDIT_LOG_FILE) as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))

            if entries:
                recent = entries[-20:]
                display_rows = []
                for e in recent:
                    display_rows.append({
                        "Time": e.get("timestamp", "")[:19],
                        "Type": e.get("event_type", ""),
                        "Subtype": e.get("event_subtype", ""),
                        "Model": e.get("model_version", ""),
                    })
                st.dataframe(pd.DataFrame(display_rows), use_container_width=True)
            else:
                st.info("No lifecycle events logged")
        else:
            st.info("Audit log not found")


if __name__ == "__main__":
    main()
