import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================
# Load artifacts
# =========================
model = joblib.load("model_rf_reduced.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("model_features.pkl")  # <- PALING PENTING

# =========================
# UI
# =========================
st.title("Prediksi Pembatalan Booking Hotel")
st.write("Masukkan data reservasi untuk memprediksi kemungkinan cancel")

lead_time = st.number_input("Lead Time (hari)", 0, 500, 50)
adr = st.number_input("ADR (Harga per malam)", 0.0, 500.0, 100.0)
adults = st.number_input("Jumlah Dewasa", 1, 5, 2)
children = st.number_input("Jumlah Anak", 0, 5, 0)
babies = st.number_input("Jumlah Bayi", 0, 3, 0)

deposit_type = st.selectbox(
    "Tipe Deposit",
    ["No Deposit", "Non Refund", "Refundable"]
)

market_segment = st.selectbox(
    "Market Segment",
    ["Online TA", "Offline TA/TO", "Direct", "Corporate"]
)

# =========================
# Prediction
# =========================
if st.button("Prediksi"):

    # -------------------------
    # Raw input
    # -------------------------
    raw_input = pd.DataFrame([{
        "lead_time": lead_time,
        "adr": adr,
        "adults": adults,
        "children": children,
        "babies": babies,
        "deposit_type": deposit_type,
        "market_segment": market_segment
    }])

    # -------------------------
    # Split numeric & categorical
    # -------------------------
    numeric_cols = ["lead_time", "adr", "adults", "children", "babies"]
    categorical_cols = ["deposit_type", "market_segment"]

    X_num = raw_input[numeric_cols]
    X_cat = raw_input[categorical_cols]

    # -------------------------
    # Encoding & scaling
    # -------------------------
    X_cat_enc = encoder.transform(X_cat)
    X_num_scaled = scaler.transform(X_num)

    # -------------------------
    # Gabungkan fitur
    # -------------------------
    X_final = np.hstack([X_num_scaled, X_cat_enc])

    # -------------------------
    # SAMAKAN dengan fitur model
    # -------------------------
    X_final_df = pd.DataFrame(X_final, columns=model_features)

    # -------------------------
    # Predict
    # -------------------------
    prediction = model.predict(X_final_df)
    prob = model.predict_proba(X_final_df)[0][1]

    # -------------------------
    # Output
    # -------------------------
    if prediction[0] == 1:
        st.error(f"⚠️ Booking Berpotensi Cancel ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Booking Aman ({(1-prob)*100:.2f}%)")
