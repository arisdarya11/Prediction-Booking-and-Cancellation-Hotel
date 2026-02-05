import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD ARTIFACTS
# =========================
model = joblib.load("model_rf_reduced.pkl")
encoder = joblib.load("encoder.pkl")
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
lead_time = st.number_input("Lead Time (hari)", min_value=0, value=50)
adr = st.number_input("ADR (harga per malam)", min_value=0.0, value=100.0)
total_nights = st.number_input("Total Nights", min_value=1, value=3)
adults = st.number_input("Jumlah Dewasa", min_value=1, value=2)

hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
meal = st.selectbox("Meal Plan", ["BB", "HB", "FB", "SC"])
market_segment = st.selectbox(
    "Market Segment",
    ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"]
)
customer_type = st.selectbox(
    "Customer Type",
    ["Transient", "Transient-Party", "Contract", "Group"]
)

# =========================
# RAW INPUT
# =========================
raw_df = pd.DataFrame([{
    "lead_time": lead_time,
    "adr": adr,
    "total_nights": total_nights,
    "adults": adults,
    "hotel": hotel,
    "meal": meal,
    "market_segment": market_segment,
    "customer_type": customer_type
}])

# =========================
# PREPROCESSING
# =========================
# ---- NUMERIC (SAFE MODE)
num_safe = pd.DataFrame(
    np.zeros((1, len(scaler.feature_names_in_))),
    columns=scaler.feature_names_in_
)

for col in raw_df.columns:
    if col in num_safe.columns:
        num_safe[col] = raw_df[col].values

X_num = scaler.transform(num_safe)

# ---- CATEGORICAL (SAFE MODE)
cat_safe = pd.DataFrame(
    np.zeros((1, len(encoder.feature_names_in_))),
    columns=encoder.feature_names_in_
)

for col in raw_df.columns:
    if col in cat_safe.columns:
        cat_safe[col] = raw_df[col].values

X_cat = encoder.transform(cat_safe)

# =========================
# FINAL FEATURE MATRIX
# =========================
X_final = np.hstack([X_num, X_cat])
X_final = pd.DataFrame(X_final, columns=model_features)

# =========================
# PREDICTION
# =========================
if st.button("🔮 Predict"):
    prediction = model.predict(X_final)[0]
    probability = model.predict_proba(X_final)[0][1]

    if prediction == 1:
        st.error(f"❌ Booking berpotensi **DIBATALKAN** ({probability:.2%})")
    else:
        st.success(f"✅ Booking **AMAN** ({1 - probability:.2%})")
