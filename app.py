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

lead_time = st.number_input("Lead Time (hari)", 0, 500, 50)
adr = st.number_input("ADR", 0.0, 1000.0, 100.0)

country = st.selectbox(
    "Country",
    ["PRT", "GBR", "ESP", "FRA", "DEU", "OTHERS"]
)

deposit_type = st.selectbox(
    "Deposit Type",
    ["No Deposit", "Non Refund", "Refundable"]
)

total_of_special_requests = st.slider(
    "Total Special Requests",
    0, 5, 1
)

# ======================
# PREDICT
# ======================
if st.button("Prediksi"):

    user_input = {
        "lead_time": lead_time,
        "adr": adr,
        "country": country,
        "deposit_type": deposit_type,
        "total_of_special_requests": total_of_special_requests,
    }

    # ------------------
    # NUMERIC FEATURES
    # ------------------
    X_num = pd.DataFrame(0, index=[0], columns=num_cols)
    for k, v in user_input.items():
        if k in num_cols:
            X_num.at[0, k] = v

    X_num_scaled = scaler.transform(X_num)
    X_num_scaled = np.asarray(X_num_scaled)

    # ------------------
    # CATEGORICAL FEATURES
    # ------------------
    X_cat = pd.DataFrame(index=[0], columns=cat_cols)
    for i, col in enumerate(cat_cols):
        if col in user_input:
            X_cat.at[0, col] = user_input[col]
        else:
            # default ke kategori pertama saat training
            X_cat.at[0, col] = encoder.categories_[i][0]

    X_cat_encoded = encoder.transform(X_cat)
    if hasattr(X_cat_encoded, "toarray"):
        X_cat_encoded = X_cat_encoded.toarray()

    X_cat_encoded = np.asarray(X_cat_encoded)

    # ------------------
    # FINAL FEATURE VECTOR
    # ------------------
    X_final = np.concatenate([X_num_scaled, X_cat_encoded], axis=1)

    # 🔐 PAKSA SESUAI JUMLAH FITUR MODEL
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
