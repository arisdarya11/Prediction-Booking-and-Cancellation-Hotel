import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# LOAD OBJECT
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

lead_time = st.number_input("Lead Time", 0, 500, 50)
adr = st.number_input("ADR", 0.0, 1000.0, 100.0)
adults = st.number_input("Adults", 1, 5, 2)
children = st.number_input("Children", 0, 5, 0)
babies = st.number_input("Babies", 0, 3, 0)

hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
market_segment = st.selectbox(
    "Market Segment",
    ["Online TA", "Offline TA/TO", "Direct", "Corporate"]
)
deposit_type = st.selectbox(
    "Deposit Type",
    ["No Deposit", "Non Refund", "Refundable"]
)

# ======================
# PREDICT
# ======================
if st.button("Prediksi"):

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

    # ------------------
    # NUMERIC
    # ------------------
    X_num = pd.DataFrame(0, index=[0], columns=num_cols)
    for k, v in user_input.items():
        if k in num_cols:
            X_num.at[0, k] = v

    X_num_scaled = scaler.transform(X_num)
    X_num_scaled = np.asarray(X_num_scaled)

    # ------------------
    # CATEGORICAL
    # ------------------
    X_cat = pd.DataFrame(index=[0], columns=cat_cols)
    for i, col in enumerate(cat_cols):
        if col in user_input:
            X_cat.at[0, col] = user_input[col]
        else:
            X_cat.at[0, col] = encoder.categories_[i][0]

    X_cat_encoded = encoder.transform(X_cat)
    if hasattr(X_cat_encoded, "toarray"):
        X_cat_encoded = X_cat_encoded.toarray()

    X_cat_encoded = np.asarray(X_cat_encoded)

    # ------------------
    # FINAL FEATURE VECTOR
    # ------------------
    X_final = np.concatenate([X_num_scaled, X_cat_encoded], axis=1)

    # 🔐 PAKSA SESUAI MODEL
    expected = model.n_features_in_
    current = X_final.shape[1]

    if current > expected:
        X_final = X_final[:, :expected]
    elif current < expected:
        pad = expected - current
        X_final = np.hstack([X_final, np.zeros((1, pad))])

    # ------------------
    # PREDICTION
    # ------------------
    prediction = model.predict(X_final)[0]
    prob = model.predict_proba(X_final)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Booking Berpotensi Cancel ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Booking Aman ({(1-prob)*100:.2f}%)")
