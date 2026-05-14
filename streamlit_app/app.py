import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import torch
import numpy as np
import xgboost as xgb

st.set_page_config(page_title="Risk-Aware Portfolio Dashboard", layout="wide")
st.title("📈 Risk-Aware Portfolio Optimization Dashboard")
st.markdown("""
Welcome to the interactive dashboard showcasing model outputs from:
- **XGBoost Classifier** with SHAP Explainability
- **Attention-based LSTM** for sequence modeling
""")

# SHAP Feature Importance
st.subheader("🔍 SHAP Feature Importance (XGBoost)")
bar_path = r"C:\Users\saira\Desktop\Risk-Aware Optimization\Risk-Aware-Portfolio-Optimization\figures\day10_xgb_shap_summary_bar.png"
beeswarm_path = r"C:\Users\saira\Desktop\Risk-Aware Optimization\Risk-Aware-Portfolio-Optimization\figures\day10_xgb_shap_beeswarm.png"

col1, col2 = st.columns(2)
with col1:
    st.image(bar_path, caption="Top Features - SHAP Bar Plot", use_column_width=True)
with col2:
    st.image(beeswarm_path, caption="SHAP Beeswarm Plot", use_column_width=True)

# Attention Heatmap 
st.subheader("🧠 Attention Visualization (LSTM)")
attn_path = r"C:\Users\saira\Desktop\Risk-Aware Optimization\Risk-Aware-Portfolio-Optimization\figures\day11_lstm_attention_weights.png"
st.image(attn_path, caption="Attention Weights over Past 20 Days", use_column_width=True)

# Prediction Demo (XGBoost) 
st.subheader("📊 Demo: XGBoost Prediction for Latest QQQ Snapshot")

try:
    # Load model and scaler SEPARATELY
    model = xgb.XGBClassifier()
    model.load_model(r"C:\Users\saira\Desktop\Risk-Aware Optimization\Risk-Aware-Portfolio-Optimization\models\day7_xgb_multi_etf_classifier.json")  # Make sure this is your actual model file
    scaler = joblib.load(r"C:\Users\saira\Desktop\Risk-Aware Optimization\Risk-Aware-Portfolio-Optimization\models\day7_xgb_scaler.pkl")
    
    # Load data
    df = pd.read_csv(r"C:\Users\saira\Desktop\Risk-Aware Optimization\Risk-Aware-Portfolio-Optimization\data\qqq_supervised.csv")
    
    # Prepare input data - use only features the scaler was trained on
    required_features = scaler.feature_names_in_
    latest_input = df[required_features].iloc[-1:]
    
    # Scale the input
    latest_scaled = scaler.transform(latest_input)
    
    # Make prediction using the MODEL (not the scaler)
    prediction = model.predict(latest_scaled)[0]
    proba = model.predict_proba(latest_scaled)[0][prediction]
    
    label = "⬆️ Buy Signal" if prediction == 1 else "⬇️ Sell Signal"
    st.metric(label="Prediction", value=label, delta=f"Confidence: {proba:.2%}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Debug info:")
    if 'df' in locals():
        st.write(f"DataFrame columns: {df.columns.tolist()}")
    if 'scaler' in locals():
        st.write(f"Scaler features: {scaler.feature_names_in_}")
    if 'latest_input' in locals():
        st.write(f"Input features: {latest_input.columns.tolist()}")

# Footer 
st.markdown("---")
st.markdown("Built using Streamlit. Data source: Yahoo Finance & FRED.")