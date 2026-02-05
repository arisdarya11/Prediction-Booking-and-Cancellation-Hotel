import streamlit as st
import pandas as pd
import joblib

pipeline = joblib.load("pipeline.pkl")

st.title("Prediksi Pembatalan Booking Hotel")

lead_time = st.number_input("Lead Time", 0, 500, 50)
adr = st.number_input("ADR", 0.0, 500.0, 100.0)
adults = st.number_input("Dewasa", 1, 5, 2)
children = st.number_input("Anak", 0, 5, 0)
babies = st.number_input("Bayi", 0, 3, 0)

deposit_type = st.selectbox(
    "Tipe Deposit",
    ["No Deposit", "Non Refund", "Refundable"]
)

market_segment = st.selectbox(
    "Market Segment",
    ["Online TA", "Offline TA/TO", "Direct", "Corporate"]
)

if st.button("Prediksi"):
    input_data = pd.DataFrame([{
        "lead_time": lead_time,
        "adr": adr,
        "adults": adults,
        "children": children,
        "babies": babies,
        "deposit_type": deposit_type,
        "market_segment": market_segment
    }])

    prediction = pipeline.predict(input_data)
    prob = pipeline.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Booking Berpotensi Cancel ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Booking Aman ({(1-prob)*100:.2f}%)")
