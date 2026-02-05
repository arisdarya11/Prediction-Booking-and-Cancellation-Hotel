import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================
# LOAD MODEL ARTIFACTS
# =========================
model = joblib.load("model_rf_reduced.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Hotel Booking Cancellation Prediction")

st.title("🏨 Hotel Booking Cancellation Prediction")

# =========================
# INPUT USER
# =========================
st.subheader("Booking Information")

lead_time = st.number_input("Lead Time (days)", min_value=0, value=50)
adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0, value=100.0)
total_nights = st.number_input("Total Nights", min_value=1, value=3)
adults = st.number_input("Adults", min_value=1, value=2)

hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
meal = st.selectbox("Meal Type", ["BB", "HB", "FB", "SC"])
market_segment = st.selectbox(
    "Market Segment",
    ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"]
)
customer_type = st.selectbox(
    "Customer Type",
    ["Transient", "Transient-Party", "Contract", "Group"]
)

# =========================
# BUILD INPUT DATAFRAME
# =========================
input_dict = {
    "lead_time": lead_time,
    "adr": adr,
    "total_nights": total_nights,
    "adults": adults,
    "hotel": hotel,
    "meal": meal,
    "market_segment": market_segment,
    "customer_type": customer_type,
}

input_df = pd.DataFrame([input_dict])

# =========================
# FEATURE ENGINEERING
# =========================
num_cols = ["lead_time", "adr", "total_nights", "adults"]
cat_cols = ["hotel", "meal", "market_segment", "customer_type"]

X_num = input_df[num_cols]
X_cat = input_df[cat_cols]

# ⚠️ PENTING: pastikan kolom sama persis
X_cat = X_cat[encoder.feature_names_in_]

# Encode & Scale
X_cat_enc = encoder.transform(X_cat)
X_num_scaled = scaler.transform(X_num)

# Gabungkan
X_final = np.hstack([X_num_scaled, X_cat_enc])

# =========================
# SAMAKAN URUTAN FEATURE MODEL
# =========================
X_final_df = pd.DataFrame(X_final, columns=model_features)
X_final_df = X_final_df[model_features]

# =========================
# PREDICTION
# =========================
if st.button("Predict Cancellation"):
    prediction = model.predict(X_final_df)[0]
    probability = model.predict_proba(X_final_df)[0][1]

    if prediction == 1:
        st.error(f"❌ Booking kemungkinan DIBATALKAN ({probability:.2%})")
    else:
        st.success(f"✅ Booking kemungkinan TIDAK dibatalkan ({1 - probability:.2%})")
