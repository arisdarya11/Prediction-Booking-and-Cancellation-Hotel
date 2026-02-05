import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# LOAD MODEL & TOOLS
# ======================
model = joblib.load("model_rf_reduced.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    layout="centered"
)

st.title("🏨 Hotel Booking Cancellation Prediction")

# ======================
# USER INPUT
# ======================
lead_time = st.number_input("Lead Time (hari)", 0, 500, 50)
adr = st.number_input("ADR", 0.0, 500.0, 100.0)
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

# ======================
# RAW INPUT (WAJIB SESUAI TRAINING)
# ======================
raw_df = pd.DataFrame({
    "lead_time": [lead_time],
    "adr": [adr],
    "total_nights": [total_nights],
    "adults": [adults],
    "hotel": [hotel],
    "meal": [meal],
    "market_segment": [market_segment],
    "customer_type": [customer_type],
})

# ======================
# PREPROCESSING
# ======================
num_cols = scaler.feature_names_in_
cat_cols = encoder.feature_names_in_

X_num = scaler.transform(raw_df[num_cols])
X_cat = encoder.transform(raw_df[cat_cols])

X_final = np.hstack([X_num, X_cat])

# ======================
# PREDICTION
# ======================
if st.button("🔮 Predict"):
    pred = model.predict(X_final)[0]
    prob = model.predict_proba(X_final)[0][1]

    if pred == 1:
        st.error(f"❌ Booking kemungkinan **DIBATALKAN** ({prob:.2%})")
    else:
        st.success(f"✅ Booking **TIDAK DIBATALKAN** ({1 - prob:.2%})")
