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
    st.stop()

# ─────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────
NUMERIC_COLS = [
    "adr", "adults", "agent", "arrival_date_day_of_month",
    "arrival_date_week_number", "arrival_date_year", "babies",
    "booking_changes", "children", "days_in_waiting_list",
    "lead_time", "previous_bookings_not_canceled",
    "previous_cancellations", "required_car_parking_spaces",
    "stays_in_week_nights", "stays_in_weekend_nights",
    "total_guests", "total_of_special_requests", "total_stay"
]

HOTEL_LIST = [
    "Albuquerque Airport Courtyard Albuquerque, NM",
    "Anaheim Marriott Anaheim, CA",
    "Baltimore BWI Airport Courtyard Linthicum, MD",
    "Baton Rouge Acadian Centre/LSU Area Courtyard Baton Rouge, LA",
    "Berlin Marriott Hotel Berlin, Germany",
    "Cape Town Marriott Hotel Crystal Towers Cape Town, South Africa",
    "Chicago O'Hare Courtyard Des Plaines, IL",
    "Colony Club, Barbados Barbados",
    "Courtyard Las Vegas Convention Center Las Vegas, NV ",
    "Courtyard by Marriott Aberdeen Airport Aberdeen, United Kingdom",
    "Courtyard by Marriott Paris Gare de Lyon Paris, France",
    "Courtyard by Marriott Rio de Janeiro Barra da Tijuca Barra da Tijuca, Brazil",
    "Courtyard by Marriott Toulouse Airport Toulouse, France",
    "Crystal Cove, Barbados Barbados",
    "Des Moines West/Clive Courtyard Clive, IA",
    "Fort Worth University Drive Courtyard Fort Worth, TX",
    "Frankfurt Marriott Hotel Frankfurt, Germany",
    "Greensboro Courtyard Greensboro, NC",
    "Grosvenor House, A JW Marriott Hotel London, United Kingdom",
    "Heidelberg Marriott Hotel Heidelberg, Germany",
    "Hotel Alfonso XIII, a Luxury Collection Hotel, Seville Seville, Spain",
    "Hotel Maria Cristina, San Sebastian San Sebastian, Spain",
    "Indianapolis Airport Courtyard Indianapolis, IN",
    "Irvine John Wayne Airport/Orange County Courtyard Irvine, CA",
    "Las Vegas Marriott Las Vegas, NV ",
    "Leipzig Marriott Hotel Leipzig, Germany",
    "Louisville East Courtyard Louisville, KY",
    "Marriott Puerto Vallarta Resort & Spa Puerto Vallarta, Mexico",
    "Mt. Laurel Courtyard Mt. Laurel, NJ",
    "Newark Liberty International Airport Courtyard Newark, NJ",
    "Orlando Airport Courtyard Orlando, FL",
    "Orlando International Drive/Convention Center Courtyard Orlando, FL",
    "Protea Hotel Fire & Ice! by Marriott Cape Town Cape Town, South Africa",
    "Protea Hotel Fire & Ice! by Marriott Johannesburg Melrose Arch Johannesburg, South Africa",
    "Protea Hotel by Marriott Cape Town Sea Point Cape Town, South Africa",
    "Protea Hotel by Marriott Midrand Midrand, South Africa",
    "Protea Hotel by Marriott O.R. Tambo Airport Johannesburg, South Africa",
    "Renaissance Hamburg Hotel Hamburg, Germany",
    "Renaissance New York Times Square Hotel New York, NY",
    "Renaissance Santo Domingo Jaragua Hotel & Casino Santo Domingo, Dominican Republic",
    "Residence Inn Las Vegas Convention Center Las Vegas, NV ",
    "Residence Inn Rio de Janeiro Barra da Tijuca Barra da Tijuca, Brazil",
    "Sacramento Airport Natomas Courtyard Sacramento, CA",
    "San Diego Sorrento Valley Courtyard San Diego, CA",
    "Sheraton Diana Majestic, Milan Milan, Italy",
    "Sheraton Grand Rio Hotel & Resort Rio de Janeiro, Brazil",
    "Sheraton Lima Hotel & Convention Center Lima, Peru",
    "Sheraton Mexico City Maria Isabel Hotel Mexico City, Mexico",
    "Spokane Downtown at the Convention Center Courtyard Spokane, WA",
    "St. Louis Downtown West Courtyard St. Louis, MO",
    "Tamarind, Barbados Barbados",
    "The House, Barbados Barbados",
    "The Ritz-Carlton, Berlin Berlin, Germany",
    "The Ritz-Carlton, Tokyo Tokyo, Japan",
    "The St. Regis Osaka Osaka, Japan",
    "The Westin Peachtree Plaza, Atlanta Atlanta, GA ",
    "Treasure Beach, Barbados Barbados",
    "Turtle Beach, Barbados Barbados",
    "W Barcelona Barcelona, Spain",
    "W London \u2013 Leicester Square London, United Kingdom",
    "W New York \u2013 Times Square New York, NY",
    "W New York \u2013 Union Square New York, NY",
    "Waves, Barbados Barbados",
]

COUNTRY_LIST = sorted([
    "PRT","GBR","ESP","IRL","FRA","DEU","USA","ITA","BEL","NLD","BRA","CHE",
    "AUT","POL","RUS","SWE","NOR","DNK","FIN","AUS","CHN","IND","JPN","KOR",
    "ARG","COL","MEX","ZAF","MAR","EGY","TUR","GRC","HRV","CZE","HUN","ROU",
    "BGR","SRB","UKR","ISR","SAU","ARE","QAT","KWT","SGP","MYS","THA","VNM",
    "IDN","PHL","NZL","CAN","AGO","AIA","ALB","AND","ARM","ASM","AZE","BEN",
    "BFA","BGD","BHR","BIH","BLR","BOL","BRB","CAF","CHL","CIV","CMR","CN",
    "COM","CPV","CRI","CUB","CYM","CYP","DJI","DMA","DOM","DZA","ECU","EST",
    "ETH","FJI","FRO","GAB","GEO","GHA","GIB","GLP","GNB","GTM","HKG","HND",
    "IRN","IRQ","ISL","JAM","JEY","JOR","KAZ","KEN","KHM","KIR","KNA","LAO",
    "LBN","LBY","LIE","LKA","LTU","LUX","LVA","MAC","MCO","MDG","MDV","MKD",
    "MLI","MLT","MMR","MNE","MOZ","MRT","MUS","MWI","MYT","NAM","NGA","NIC",
    "NPL","OMN","PAK","PAN","PER","PLW","PRI","PRY","RWA","SDN","SEN","SLE",
    "SLV","SMR","STP","SUR","SVK","SVN","SYC","SYR","TGO","TJK","TMP","TUN",
    "TWN","TZA","UGA","UMI","URY","UZB","VEN","ZMB","ZWE"
])

MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🏨 Prediksi Pembatalan Booking Hotel")
st.markdown("Aplikasi ini menggunakan model **XGBoost** untuk memprediksi apakah sebuah pemesanan hotel berisiko dibatalkan.")
st.divider()

# ─────────────────────────────────────────
# FORM INPUT
# ─────────────────────────────────────────
st.subheader("📋 Informasi Booking")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🏠 Hotel & Kamar**")
    hotel = st.selectbox("Nama Hotel", HOTEL_LIST, index=13)
    reserved_room_type = st.selectbox("Tipe Kamar Dipesan", ["A","B","C","D","E","F","G","H","L","P"])
    assigned_room_type = st.selectbox("Tipe Kamar Diberikan", ["A","B","C","D","E","F","G","H","I","K","P"])
    meal = st.selectbox("Tipe Meal", ["BB","FB","HB","SC","Undefined"])

with col2:
    st.markdown("**👤 Informasi Tamu**")
    adults = st.number_input("Jumlah Dewasa", min_value=0, max_value=10, value=2)
    children = st.number_input("Jumlah Anak-anak", min_value=0, max_value=10, value=0)
    babies = st.number_input("Jumlah Bayi", min_value=0, max_value=10, value=0)
    country = st.selectbox("Negara Asal", COUNTRY_LIST, index=COUNTRY_LIST.index("IDN") if "IDN" in COUNTRY_LIST else 0)
    is_repeated_guest = st.selectbox("Tamu Berulang?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    customer_type = st.selectbox("Tipe Customer", ["Transient","Contract","Group","Transient-Party"])

with col3:
    st.markdown("**📅 Tanggal & Durasi**")
    arrival_date_year = st.selectbox("Tahun Kedatangan", [2017,2018,2019,2020,2021,2022,2023,2024,2025], index=6)
    arrival_date_month = st.selectbox("Bulan Kedatangan", MONTHS)
    arrival_date_day_of_month = st.number_input("Tanggal Kedatangan", min_value=1, max_value=31, value=15)
    arrival_date_week_number = st.number_input("Nomor Minggu", min_value=1, max_value=53, value=20)
    stays_in_weekend_nights = st.number_input("Malam Weekend", min_value=0, max_value=20, value=1)
    stays_in_week_nights = st.number_input("Malam Weekday", min_value=0, max_value=50, value=2)

st.divider()
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("**💰 Informasi Pemesanan**")
    lead_time = st.number_input("Lead Time (hari)", min_value=0, max_value=700, value=30)
    adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0, max_value=5000.0, value=100.0)
    agent = st.number_input("ID Agent (0 jika tidak ada)", min_value=0, max_value=600, value=0)
    deposit_type = st.selectbox("Tipe Deposit", ["No Deposit","Non Refund","Refundable"])
    market_segment = st.selectbox("Segmen Market", ["Direct","Corporate","Online TA","Offline TA/TO","Complementary","Groups","Undefined"])
    distribution_channel = st.selectbox("Channel Distribusi", ["Direct","Corporate","TA/TO","GDS","Undefined"])

with col5:
    st.markdown("**📊 Riwayat & Perubahan**")
    previous_cancellations = st.number_input("Pembatalan Sebelumnya", min_value=0, max_value=50, value=0)
    previous_bookings_not_canceled = st.number_input("Booking Sebelumnya (Tidak Batal)", min_value=0, max_value=50, value=0)
    booking_changes = st.number_input("Jumlah Perubahan Booking", min_value=0, max_value=20, value=0)
    days_in_waiting_list = st.number_input("Hari di Waiting List", min_value=0, max_value=400, value=0)

with col6:
    st.markdown("**🚗 Fasilitas**")
    required_car_parking_spaces = st.number_input("Tempat Parkir", min_value=0, max_value=8, value=0)
    total_of_special_requests = st.number_input("Permintaan Khusus", min_value=0, max_value=5, value=0)

st.divider()

# ─────────────────────────────────────────
# PREDIKSI
# ─────────────────────────────────────────
if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):

    # Hitung fitur turunan
    total_guests = adults + children + babies
    total_stay = stays_in_weekend_nights + stays_in_week_nights
    has_deposit = 1 if deposit_type != "No Deposit" else 0
    high_cancel_history = 1 if previous_cancellations > 0 else 0
    is_family = 1 if (children > 0 or babies > 0) else 0
    is_loyal = 1 if previous_bookings_not_canceled > 3 else 0

    # Semua kolom default 0
    input_dict = {col: 0 for col in feature_columns}

    # Isi numerik
    input_dict.update({
        "adr": adr,
        "adults": adults,
        "agent": agent,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_year": arrival_date_year,
        "babies": babies,
        "booking_changes": booking_changes,
        "children": children,
        "days_in_waiting_list": days_in_waiting_list,
        "lead_time": lead_time,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "previous_cancellations": previous_cancellations,
        "required_car_parking_spaces": required_car_parking_spaces,
        "stays_in_week_nights": stays_in_week_nights,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "total_guests": total_guests,
        "total_of_special_requests": total_of_special_requests,
        "total_stay": total_stay,
        "has_deposit": has_deposit,
        "high_cancel_history": high_cancel_history,
        "is_family": is_family,
        "is_loyal": is_loyal,
        "is_repeated_guest": is_repeated_guest,
    })

    # One-hot encoding
    def set_onehot(key):
        if key in input_dict:
            input_dict[key] = 1

    set_onehot(f"arrival_date_month_{arrival_date_month}")

    if assigned_room_type != "A":
        set_onehot(f"assigned_room_type_{assigned_room_type}")

    set_onehot(f"country_{country}")

    if customer_type != "Contract":
        set_onehot(f"customer_type_{customer_type}")

    if deposit_type != "No Deposit":
        set_onehot(f"deposit_type_{deposit_type}")

    if distribution_channel != "Corporate":
        set_onehot(f"distribution_channel_{distribution_channel}")

    set_onehot(f"hotel_{hotel}")

    ms_map = {
        "Complementary": "market_segment_Complementary",
        "Corporate": "market_segment_Corporate",
        "Direct": "market_segment_Direct",
        "Groups": "market_segment_Groups",
        "Offline TA/TO": "market_segment_Offline TA/TO",
        "Online TA": "market_segment_Online TA",
        "Undefined": "market_segment_Undefined",
    }
    if market_segment in ms_map:
        set_onehot(ms_map[market_segment])

    meal_map = {"FB":"meal_FB","HB":"meal_HB","SC":"meal_SC","Undefined":"meal_Undefined"}
    if meal in meal_map:
        set_onehot(meal_map[meal])

    if reserved_room_type != "A":
        set_onehot(f"reserved_room_type_{reserved_room_type}")

    # Buat DataFrame sesuai urutan feature_columns
    input_df = pd.DataFrame([input_dict])[feature_columns]

    # Scale kolom numerik
    input_df[NUMERIC_COLS] = scaler.transform(input_df[NUMERIC_COLS])

    # Prediksi
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    cancel_prob = proba[1] * 100
    not_cancel_prob = proba[0] * 100

    # Tampilkan hasil
    st.subheader("📊 Hasil Prediksi")
    res1, res2 = st.columns(2)

    with res1:
        if prediction == 1:
            st.error("⚠️ **BOOKING BERISIKO DIBATALKAN**")
        else:
            st.success("✅ **BOOKING TIDAK AKAN DIBATALKAN**")
        st.metric("Probabilitas Dibatalkan", f"{cancel_prob:.1f}%")
        st.metric("Probabilitas Tidak Dibatalkan", f"{not_cancel_prob:.1f}%")

    with res2:
        st.markdown("**Distribusi Probabilitas:**")
        st.progress(int(cancel_prob), text=f"🔴 Dibatalkan: {cancel_prob:.1f}%")
        st.progress(int(not_cancel_prob), text=f"🟢 Tidak Dibatalkan: {not_cancel_prob:.1f}%")

    st.divider()
    st.subheader("💡 Rekomendasi")
    if cancel_prob >= 70:
        st.warning("🔴 **Risiko Tinggi** – Terapkan kebijakan deposit ketat atau konfirmasi ulang kepada tamu.")
    elif cancel_prob >= 40:
        st.info("🟡 **Risiko Sedang** – Pantau booking ini dan berikan penawaran retensi jika diperlukan.")
    else:
        st.success("🟢 **Risiko Rendah** – Booking ini kemungkinan besar akan berjalan lancar.")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.caption("🤖 Model: XGBoost | Dataset: Hotel Booking | Prediksi Pembatalan Hotel")
