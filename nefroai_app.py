import streamlit as st
import joblib
import numpy as np
import time, psutil, platform, os, csv
from datetime import datetime

# =========================
# Utility Functions
# =========================
def now_ms():
    return time.perf_counter() * 1000

def system_snapshot():
    v = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_mb": int((v.total - v.available) / (1024*1024)),
        "ram_total_mb": int(v.total / (1024*1024)),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }

def log_latency(times, extra):
    os.makedirs("logs", exist_ok=True)
    path = "logs/latency_log.csv"
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list({**times, **extra}.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow({**times, **extra})

# =========================
# Load Model
# =========================
model = joblib.load("best_rf_model.pkl")

# Prediction function with timing
def predict(input_data):
    t0 = now_ms()
    prediction = model.predict([input_data])
    t1 = now_ms()
    probability = model.predict_proba([input_data])
    t2 = now_ms()

    times = {
        "inference_ms": round(t1 - t0, 2),
        "probability_ms": round(t2 - t1, 2),
        "total_ms": round(t2 - t0, 2)
    }
    return prediction[0], probability[0][1], times

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="NefroAi: CKD Predictor", layout="centered")
st.title("NefroAi: A Real-Time Framework for Predicting Chronic Kidney Disease")

st.markdown("""
Welcome to **NefroAi**, an intelligent system to help predict the risk of **Chronic Kidney Disease (CKD)** using clinical values.  
Please fill in your health parameters and click **Predict** to see the results.
""")

col1, col2 = st.columns(2)

with col1:
    hemo = st.number_input("Hemoglobin (g/dL)", min_value=3.1, max_value=17.8, step=0.1, value=10.1)
    pcv = st.number_input("Packed Cell Volume (%)", min_value=9, max_value=54, step=1, value=30)
    rc = st.number_input("Red Blood Cell Count (million cells/uL)", min_value=2.0, max_value=8.0, step=0.1, value=4.5)
    sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.4, max_value=76.1, step=0.1, value=1.2)
    sg = st.number_input("Urine Specific Gravity", min_value=1.005, max_value=1.025, step=0.005, value=1.015)

with col2:
    bgr = st.number_input("Blood Glucose Random (mg/dL)", min_value=22, max_value=490, step=1, value=110)
    al = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=5.0, step=0.1, value=2.5)
    sod = st.number_input("Sodium - Na⁺ (mEq/L)", min_value=104, max_value=163, step=1, value=135)
    pot = st.number_input("Potassium - K⁺ (mEq/L)", min_value=2.5, max_value=7.0, step=0.1, value=4.5)
    bu = st.number_input("Blood Urea (mg/dL)", min_value=1.5, max_value=391.1, step=1.0, value=45.0)

input_data = [hemo, pcv, rc, sc, sg, bgr, al, sod, pot, bu]

# =========================
# Prediction Button
# =========================
if st.button("🔍 Predict"):
    prediction, probability, times = predict(input_data)
    snap = system_snapshot()

    # Log result
    log_latency(times, snap)

    # Show Prediction
    if prediction == 1:
        st.error("⚠️ You may have Chronic Kidney Disease. Please consult a healthcare provider.")
        st.write(f"Model Confidence: **{probability * 100:.2f}%** chance of CKD.")
    else:
        st.success("✅ You are safe from CKD based on the current data.")
        st.write(f"Model Confidence: **{(1 - probability) * 100:.2f}%** chance of being healthy.")

    # Show Latency & Resource Metrics in Sidebar
    st.sidebar.subheader("⏱ Performance Metrics")
    st.sidebar.metric("Inference (ms)", times["inference_ms"])
    st.sidebar.metric("Probability Calc (ms)", times["probability_ms"])
    st.sidebar.metric("Total Latency (ms)", times["total_ms"])

    st.sidebar.subheader("💻 System Resources")
    st.sidebar.metric("CPU (%)", snap["cpu_percent"])
    st.sidebar.metric("RAM Used (MB)", snap["ram_used_mb"])
    st.sidebar.metric("RAM Total (MB)", snap["ram_total_mb"])
    st.sidebar.caption(f"OS: {snap['os']}, CPU: {snap['processor']}")