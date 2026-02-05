import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD MODEL & FILE
# =========================
model = joblib.load("model_rf_reduced.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    layout="centered"
)

st.title("🏨 Hotel Booking Cancellation Prediction")
st.write("Masukkan data reservasi untuk memprediksi kemungkinan pembatalan booking.")

# =========================
# USER INPUT
# =========================
lead_time = st.number_input("Lead Time (hari)", 0, 500, 50)
adr = st.number_input("ADR (harga per malam)", 0.0, 500.0, 100.0)
total_nights = st.number_input("Total Nights", 1, 30, 3)
adults = st.number_input("Adults", 1, 5, 2)

hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
meal = st.selectbox("Meal", ["BB", "HB", "FB", "SC"])
market_segment = st.selectbox(
    "Market Segment",
    ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"]
)
customer_type = st.selectbox(
    "Customer Type",
    ["Transient", "Transient-Party", "Contract", "Group"]
)

# =========================
# BUILD FEATURE MATRIX
# =========================
# Ground truth: model_features.pkl
X = pd.DataFrame(
    np.zeros((1, len(model_features))),
    columns=model_features
)

# ---- numeric
numeric_inputs = {
    "lead_time": lead_time,
    "adr": adr,
    "total_nights": total_nights,
    "adults": adults
}

for col, val in numeric_inputs.items():
    if col in X.columns:
        X[col] = val

# ---- manual one-hot categorical
categorical_inputs = {
    "hotel": hotel,
    "meal": meal,
    "market_segment": market_segment,
    "customer_type": customer_type
}

for feature, value in categorical_inputs.items():
    onehot_col = f"{feature}_{value}"
    if onehot_col in X.columns:
        X[onehot_col] = 1

# =========================
# SCALING (CRITICAL FIX)
# =========================
# scaler dilatih dengan NumPy → inference juga NumPy
X_scaled = scaler.transform(X.values)

# =========================
# PREDICTION
# =========================
if st.button("🔮 Predict"):
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    if prediction == 1:
        st.error(f"❌ Booking berpotensi **DIBATALKAN** ({probability:.2%})")
    else:
        st.success(f"✅ Booking **AMAN** ({1 - probability:.2%})")
