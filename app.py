import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL ONLY
# =========================
model = joblib.load("model_rf_reduced.pkl")

st.set_page_config(page_title="Hotel Booking Cancellation", layout="centered")
st.title("🏨 Hotel Booking Cancellation Prediction")

st.write("⚠️ Versi sederhana tanpa preprocessing eksternal")

# =========================
# USER INPUT
# =========================
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

# =========================
# RAW DATAFRAME (NO ENCODING, NO SCALING)
# =========================
input_df = pd.DataFrame([{
    "lead_time": lead_time,
    "adr": adr,
    "total_nights": total_nights,
    "adults": adults,
    "hotel": hotel,
    "meal": meal,
    "market_segment": market_segment,
    "customer_type": customer_type
}])

st.write("📥 Input ke model:")
st.dataframe(input_df)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"❌ Booking berpotensi DIBATALKAN ({prob:.2%})")
        else:
            st.success(f"✅ Booking AMAN ({1 - prob:.2%})")

    except Exception as e:
        st.error("❌ Model tidak kompatibel dengan input mentah.")
        st.code(str(e))
