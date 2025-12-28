# app.py
import streamlit as st
import numpy as np
import joblib
import os

# 🔹 Load model & scaler safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH  = os.path.join(MODEL_DIR, "chlorophyll_model.pkl")

scaler = joblib.load(SCALER_PATH)
model  = joblib.load(MODEL_PATH)

# 🔹 App title
st.set_page_config(page_title="Water Purity Prediction", layout="centered")
st.title("🌊 Water Purity Prediction due to Algae Growth")
st.write("Predict **Chlorophyll-a concentration** using water quality parameters.")

st.divider()

# 🔹 Input form
st.subheader("Enter Water Quality Parameters")

temp = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
do   = st.number_input("💧 Dissolved Oxygen (mg/L)", min_value=0.0, max_value=15.0, value=6.5)
ph   = st.number_input("⚗️ pH", min_value=0.0, max_value=14.0, value=8.0)
sal  = st.number_input("🧂 Salinity (ppt)", min_value=0.0, max_value=50.0, value=35.0)
turb = st.number_input("🌫️ Turbidity (NTU)", min_value=0.0, max_value=100.0, value=5.0)
cond = st.number_input("🔌 Conductivity (µS/cm)", min_value=0.0, max_value=2000.0, value=55.0)

# 🔹 Predict button
if st.button("🔍 Predict Chlorophyll-a"):
    x = np.array([[temp, do, ph, sal, turb, cond]], dtype=float)
    x_scaled = scaler.transform(x)
    prediction = model.predict(x_scaled)[0]

    st.success(f"Predicted Chlorophyll-a: **{prediction:.2f} µg/L**")

    # 🔹 Water quality interpretation
    if prediction < 10:
        st.info("✅ Water Status: SAFE (Low algae presence)")
    elif 10 <= prediction < 30:
        st.warning("⚠️ Water Status: MODERATE (Algae growth starting)")
    else:
        st.error("🚨 Water Status: DANGEROUS (High algae bloom risk)")

st.divider()
st.caption("📊 AI-based Water Quality Monitoring System")
