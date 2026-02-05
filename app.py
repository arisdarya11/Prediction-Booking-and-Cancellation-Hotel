import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD FILE
# =========================
model = joblib.load("model_rf_reduced.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Hotel Booking Cancellation", layout="centered")
st.title("🏨 Hotel Booking Cancellation Prediction")

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
# BUILD FEATURE MATRIX (FOLLOW model_features)
# =========================
X = pd.DataFrame(
    np.zeros((1, len(model_features))),
    columns=model_features
)

# ---- numeric
numeric_values = {
    "lead_time": lead_time,
    "adr": adr,
    "total_nights": total_nights,
    "adults": adults
}

for col, val in numeric_values.items():
    if col in X.columns:
        X[col] = val

# ---- manual one-hot categorical
categorical_map = {
    "hotel": hotel,
    "meal": meal,
    "market_segment": market_segment,
    "customer_type": customer_type
}

for feature, value in categorical_map.items():
    col_name = f"{feature}_{value}"
    if col_name in X.columns:
        X[col_name] = 1

# =========================
# SCALING
# =========================
X_scaled = scaler.transform(X)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    if prediction == 1:
        st.error(f"❌ Booking berpotensi **DIBATALKAN** ({prob:.2%})")
    else:
        st.success(f"✅ Booking **AMAN** ({1 - prob:.2%})")
