import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD FILE
# =========================
model = joblib.load("model_rf_reduced.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Hotel Booking Cancellation", layout="centered")
st.title("🏨 Hotel Booking Cancellation Prediction")

# =========================
# INPUT
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
# BUILD FEATURE MATRIX (FOLLOW MODEL_FEATURES)
# =========================
X = pd.DataFrame(
    np.zeros((1, len(model_features))),
    columns=model_features
)

# numeric
for col in ["lead_time", "adr", "total_nights", "adults"]:
    if col in X.columns:
        X[col] = raw_df[col].values

# categorical
cat_cols = raw_df.select_dtypes(include="object")
cat_encoded = encoder.transform(cat_cols)
cat_feature_names = encoder.get_feature_names_out(cat_cols.columns)
cat_df = pd.DataFrame(cat_encoded, columns=cat_feature_names)

for col in cat_df.columns:
    if col in X.columns:
        X[col] = cat_df[col].values

# scaling (NO feature_names_in_)
X_scaled = scaler.transform(X)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    if pred == 1:
        st.error(f"❌ Booking berpotensi DIBATALKAN ({prob:.2%})")
    else:
        st.success(f"✅ Booking AMAN ({1 - prob:.2%})")
