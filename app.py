import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# LOAD MODEL & OBJECT
# ======================
model = joblib.load("model_rf_reduced.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

num_cols = list(scaler.feature_names_in_)
cat_cols = list(encoder.feature_names_in_)

# ======================
# STREAMLIT UI
# ======================
st.title("Prediksi Pembatalan Booking Hotel")
st.write("Masukkan data reservasi (input terbatas, model tetap lengkap)")

lead_time = st.number_input("Lead Time (hari)", 0, 500, 50)
adr = st.number_input("ADR (harga per malam)", 0.0, 1000.0, 100.0)
adults = st.number_input("Jumlah Dewasa", 1, 5, 2)
children = st.number_input("Jumlah Anak", 0, 5, 0)
babies = st.number_input("Jumlah Bayi", 0, 3, 0)

hotel = st.selectbox("Tipe Hotel", ["City Hotel", "Resort Hotel"])
market_segment = st.selectbox(
    "Market Segment",
    ["Online TA", "Offline TA/TO", "Direct", "Corporate"]
)
deposit_type = st.selectbox(
    "Tipe Deposit",
    ["No Deposit", "Non Refund", "Refundable"]
)

# ======================
# PREDICT
# ======================
if st.button("Prediksi"):

    # ------------------
    # RAW INPUT USER
    # ------------------
    user_input = {
        "lead_time": lead_time,
        "adr": adr,
        "adults": adults,
        "children": children,
        "babies": babies,
        "hotel": hotel,
        "market_segment": market_segment,
        "deposit_type": deposit_type,
    }

    raw_df = pd.DataFrame([user_input])

    # ------------------
    # NUMERIC FEATURES
    # ------------------
    X_num = pd.DataFrame(0, index=[0], columns=num_cols)

    for col in raw_df.columns:
        if col in num_cols:
            X_num[col] = raw_df[col]

    X_num_scaled = scaler.transform(X_num)

    # ------------------
    # CATEGORICAL FEATURES
    # ------------------
    X_cat = pd.DataFrame({
        col: raw_df[col] if col in raw_df.columns else encoder.categories_[i][0]
        for i, col in enumerate(cat_cols)
    })

    X_cat_encoded = encoder.transform(X_cat)

    # ------------------
    # FINAL INPUT
    # ------------------
    X_final = np.hstack([X_num_scaled, X_cat_encoded])

    # ------------------
    # PREDICTION
    # ------------------
    prediction = model.predict(X_final)[0]
    probability = model.predict_proba(X_final)[0][1]

    # ------------------
    # OUTPUT
    # ------------------
    if prediction == 1:
        st.error(f"⚠️ Booking Berpotensi Cancel ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Booking Aman ({(1-probability)*100:.2f}%)")
