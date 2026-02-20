import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ─────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Pembatalan Booking Hotel",
    page_icon="🏨",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD MODEL & PREPROCESSING
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Gagal memuat model: {e}")

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🏨 Prediksi Pembatalan Booking Hotel")
st.markdown("""
Aplikasi ini menggunakan model **XGBoost** untuk memprediksi apakah sebuah 
pemesanan hotel berisiko dibatalkan. Masukkan informasi booking di bawah ini.
""")
st.divider()

# ─────────────────────────────────────────
# FORM INPUT
# ─────────────────────────────────────────
st.subheader("📋 Informasi Booking")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Informasi Tamu**")
    adults = st.number_input("Jumlah Tamu Dewasa", min_value=0, max_value=10, value=2)
    children = st.number_input("Jumlah Anak-anak", min_value=0, max_value=10, value=0)
    babies = st.number_input("Jumlah Bayi", min_value=0, max_value=10, value=0)
    customer_type = st.selectbox("Tipe Customer", ["Transient", "Contract", "Group", "Transient-Party"])

with col2:
    st.markdown("**Informasi Menginap**")
    lead_time = st.number_input("Lead Time (hari sebelum check-in)", min_value=0, max_value=700, value=30)
    stays_in_weekend_nights = st.number_input("Malam Weekend", min_value=0, max_value=20, value=1)
    stays_in_week_nights = st.number_input("Malam Weekday", min_value=0, max_value=50, value=2)
    adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0, max_value=5000.0, value=100.0)

with col3:
    st.markdown("**Informasi Pemesanan**")
    deposit_type = st.selectbox("Tipe Deposit", ["No Deposit", "Non Refund", "Refundable"])
    market_segment = st.selectbox("Segmen Market", ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Aviation"])
    distribution_channel = st.selectbox("Channel Distribusi", ["Direct", "Corporate", "TA/TO", "GDS", "Undefined"])
    total_of_special_requests = st.number_input("Jumlah Permintaan Khusus", min_value=0, max_value=5, value=0)

col4, col5 = st.columns(2)

with col4:
    st.markdown("**Riwayat Tamu**")
    previous_cancellations = st.number_input("Jumlah Pembatalan Sebelumnya", min_value=0, max_value=50, value=0)
    previous_bookings_not_canceled = st.number_input("Booking Sebelumnya (Tidak Dibatalkan)", min_value=0, max_value=50, value=0)
    is_repeated_guest = st.selectbox("Tamu Berulang?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")

with col5:
    st.markdown("**Informasi Kamar**")
    reserved_room_type = st.selectbox("Tipe Kamar Dipesan", ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"])
    booking_changes = st.number_input("Jumlah Perubahan Booking", min_value=0, max_value=20, value=0)
    days_in_waiting_list = st.number_input("Hari di Waiting List", min_value=0, max_value=400, value=0)
    required_car_parking_spaces = st.number_input("Tempat Parkir Dibutuhkan", min_value=0, max_value=8, value=0)

st.divider()

# ─────────────────────────────────────────
# PREDIKSI
# ─────────────────────────────────────────
if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):
    if not model_loaded:
        st.error("Model tidak berhasil dimuat. Pastikan file model tersedia.")
    else:
        # Mapping kategorikal ke angka (sesuaikan dengan encoding saat training)
        deposit_map = {"No Deposit": 0, "Non Refund": 1, "Refundable": 2}
        customer_map = {"Transient": 0, "Contract": 1, "Group": 2, "Transient-Party": 3}
        market_map = {"Direct": 0, "Corporate": 1, "Online TA": 2, "Offline TA/TO": 3,
                      "Complementary": 4, "Groups": 5, "Aviation": 6}
        channel_map = {"Direct": 0, "Corporate": 1, "TA/TO": 2, "GDS": 3, "Undefined": 4}
        room_map = {r: i for i, r in enumerate(["A","B","C","D","E","F","G","H","L","P"])}

        input_data = {
            "lead_time": lead_time,
            "stays_in_weekend_nights": stays_in_weekend_nights,
            "stays_in_week_nights": stays_in_week_nights,
            "adults": adults,
            "children": children,
            "babies": babies,
            "is_repeated_guest": is_repeated_guest,
            "previous_cancellations": previous_cancellations,
            "previous_bookings_not_canceled": previous_bookings_not_canceled,
            "booking_changes": booking_changes,
            "days_in_waiting_list": days_in_waiting_list,
            "adr": adr,
            "required_car_parking_spaces": required_car_parking_spaces,
            "total_of_special_requests": total_of_special_requests,
            "deposit_type": deposit_map.get(deposit_type, 0),
            "customer_type": customer_map.get(customer_type, 0),
            "market_segment": market_map.get(market_segment, 0),
            "distribution_channel": channel_map.get(distribution_channel, 0),
            "reserved_room_type": room_map.get(reserved_room_type, 0),
        }

        # Buat DataFrame dan sesuaikan dengan feature_columns
        input_df = pd.DataFrame([input_data])

        # Tambah kolom yang hilang dengan nilai 0
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Urutkan kolom sesuai training
        input_df = input_df[feature_columns]

        # Scaling
        try:
            input_scaled = scaler.transform(input_df)
        except Exception:
            input_scaled = input_df.values

        # Prediksi
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        cancel_prob = proba[1] * 100
        not_cancel_prob = proba[0] * 100

        # Tampilkan hasil
        st.subheader("📊 Hasil Prediksi")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediction == 1:
                st.error(f"⚠️ **BOOKING BERISIKO DIBATALKAN**")
                st.metric("Probabilitas Pembatalan", f"{cancel_prob:.1f}%")
            else:
                st.success(f"✅ **BOOKING TIDAK AKAN DIBATALKAN**")
                st.metric("Probabilitas Tidak Dibatalkan", f"{not_cancel_prob:.1f}%")

        with res_col2:
            st.markdown("**Distribusi Probabilitas:**")
            st.progress(int(cancel_prob), text=f"Dibatalkan: {cancel_prob:.1f}%")
            st.progress(int(not_cancel_prob), text=f"Tidak Dibatalkan: {not_cancel_prob:.1f}%")

        # Interpretasi risiko
        st.divider()
        st.subheader("💡 Interpretasi Risiko")
        if cancel_prob >= 70:
            st.warning("🔴 **Risiko Tinggi** – Pertimbangkan kebijakan deposit ketat atau konfirmasi ulang tamu.")
        elif cancel_prob >= 40:
            st.info("🟡 **Risiko Sedang** – Pantau booking ini dan berikan penawaran retensi jika perlu.")
        else:
            st.success("🟢 **Risiko Rendah** – Booking ini kemungkinan besar akan berjalan lancar.")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.caption("🤖 Model: XGBoost | Dataset: Hotel Booking | Dibuat untuk deployment prediksi pembatalan hotel")
