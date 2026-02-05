import streamlit as st
import pandas as pd
import joblib

# Load model & preprocessing
model = joblib.load("model_rf.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")


# Judul App
st.title("Prediksi Pembatalan Booking Hotel")
st.write("Masukkan data reservasi untuk memprediksi kemungkinan cancel")

# Input User
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


# Button Predict
if st.button("Prediksi"):

    # Buat DataFrame
    input_data = pd.DataFrame([{
        "lead_time": lead_time,
        "adr": adr,
        "adults": adults,
        "children": children,
        "babies": babies,
        "deposit_type": deposit_type,
        "market_segment": market_segment
    }])

    # Encoding
    encoded = encoder.transform(input_data)

    # Scaling
    scaled = scaler.transform(encoded)

    # Predict
    prediction = model.predict(scaled)
    prob = model.predict_proba(scaled)[0][1]

    # Output
    if prediction[0] == 1:
        st.error(f"⚠️ Booking Berpotensi Cancel ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Booking Aman ({(1-prob)*100:.2f}%)")
