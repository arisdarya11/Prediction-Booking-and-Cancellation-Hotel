import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ─────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #FFF9F0 0%, #FFF3E4 30%, #F0F7FF 70%, #EEF4FF 100%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }

/* HERO */
.hero-banner {
    background: linear-gradient(135deg, #1A56DB 0%, #0EA5E9 50%, #06B6D4 100%);
    border-radius: 24px; padding: 44px 52px; margin-bottom: 28px;
    position: relative; overflow: hidden;
    box-shadow: 0 20px 60px rgba(26,86,219,0.25);
}
.hero-banner::before {
    content:''; position:absolute; top:-50px; right:-50px;
    width:300px; height:300px; background:rgba(255,255,255,0.08); border-radius:50%;
}
.hero-title {
    font-family:'Playfair Display',serif; font-size:2.6rem; font-weight:700;
    color:#FFF; margin:0 0 10px; line-height:1.2; position:relative; z-index:1;
}
.hero-subtitle {
    font-size:1rem; color:rgba(255,255,255,0.85); margin:0;
    position:relative; z-index:1; max-width:580px; line-height:1.6;
}
.hero-badge {
    display:inline-flex; align-items:center; gap:8px;
    background:rgba(255,255,255,0.2); border:1px solid rgba(255,255,255,0.3);
    border-radius:100px; padding:5px 14px; font-size:0.8rem; color:white;
    font-weight:500; margin-bottom:16px; position:relative; z-index:1;
}

/* TOP FEATURES BANNER */
.top-feat-banner {
    background: #FFFFFF;
    border: 1.5px solid #E2E8F0;
    border-radius: 16px;
    padding: 16px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.top-feat-label {
    font-size: 0.75rem; font-weight: 700; color: #64748B;
    text-transform: uppercase; letter-spacing: 0.08em;
    white-space: nowrap;
}
.feat-rank {
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border: 1.5px solid #93C5FD; border-radius: 100px;
    padding: 4px 12px; font-size: 0.78rem; font-weight: 600; color: #1D4ED8;
}
.feat-rank-num {
    background: #1D4ED8; color: white; border-radius: 50%;
    width: 18px; height: 18px; display: inline-flex;
    align-items: center; justify-content: center;
    font-size: 0.65rem; font-weight: 800; flex-shrink: 0;
}

/* FORM CARD */
.form-card {
    background: #FFFFFF; border-radius: 20px; padding: 24px 24px 16px;
    margin-bottom: 18px; border: 1px solid rgba(226,232,240,0.8);
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
}
.card-header {
    display:flex; align-items:center; gap:12px;
    margin-bottom:16px; padding-bottom:14px; border-bottom:2px solid #F1F5F9;
}
.card-icon {
    width:40px; height:40px; border-radius:12px;
    display:flex; align-items:center; justify-content:center; font-size:1.1rem; flex-shrink:0;
}
.icon-blue   { background:linear-gradient(135deg,#DBEAFE,#BFDBFE); }
.icon-orange { background:linear-gradient(135deg,#FFEDD5,#FED7AA); }
.icon-purple { background:linear-gradient(135deg,#EDE9FE,#DDD6FE); }
.icon-green  { background:linear-gradient(135deg,#DCFCE7,#BBF7D0); }
.card-title  { font-size:1rem; font-weight:700; color:#1E293B; margin:0; }
.card-desc   { font-size:0.75rem; color:#94A3B8; margin:2px 0 0; }

/* HIGH IMPACT BADGE */
.impact-badge {
    display:inline-flex; align-items:center; gap:5px;
    background:linear-gradient(135deg,#FEF3C7,#FDE68A);
    border:1.5px solid #F59E0B; border-radius:100px;
    padding:3px 10px; font-size:0.7rem; font-weight:700; color:#92400E;
    margin-left:8px; vertical-align:middle;
}

/* FEATURE ENGINEERING BOX */
.fe-box {
    background: linear-gradient(135deg,#EFF6FF,#DBEAFE);
    border: 1.5px solid #BFDBFE; border-radius: 12px;
    padding: 12px 16px; margin-top: 12px;
}
.fe-title {
    font-size:0.72rem; font-weight:700; color:#1D4ED8;
    text-transform:uppercase; letter-spacing:0.06em; margin:0 0 8px;
}
.fe-chips { display:flex; flex-wrap:wrap; gap:6px; }
.fe-chip {
    background:white; border:1.5px solid #93C5FD; border-radius:100px;
    padding:3px 10px; font-size:0.75rem; font-weight:600; color:#1D4ED8;
    display:inline-flex; align-items:center; gap:5px;
}
.fe-chip-val {
    background:#1D4ED8; color:white; border-radius:100px;
    padding:1px 7px; font-size:0.68rem; font-weight:700;
}

/* INFO BOX */
.info-box {
    background:#FFFBEB; border:1.5px solid #FCD34D;
    border-radius:12px; padding:12px 16px; margin-top:10px;
}
.info-box-title {
    font-size:0.72rem; font-weight:700; color:#92400E;
    text-transform:uppercase; letter-spacing:0.05em; margin:0 0 8px;
}
.info-row { display:flex; gap:6px; align-items:flex-start; margin-bottom:5px; }
.info-badge       { background:#F59E0B; color:white; border-radius:6px; padding:1px 8px; font-size:0.7rem; font-weight:700; flex-shrink:0; margin-top:1px; }
.info-badge-green { background:#10B981; color:white; border-radius:6px; padding:1px 8px; font-size:0.7rem; font-weight:700; flex-shrink:0; margin-top:1px; }
.info-text { font-size:0.76rem; color:#78350F; line-height:1.5; }

/* WIDGET */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    font-size:0.78rem !important; font-weight:600 !important;
    color:#374151 !important; text-transform:uppercase; letter-spacing:0.03em;
}
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] > div > div > input {
    border-radius:10px !important; border:1.5px solid #E2E8F0 !important;
    background:#F8FAFC !important; color:#1E293B !important;
    font-size:0.9rem !important; font-family:'Plus Jakarta Sans',sans-serif !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stNumberInput"] > div > div > input:focus {
    border-color:#1A56DB !important; background:#FFF !important;
    box-shadow:0 0 0 3px rgba(26,86,219,0.1) !important;
}

/* BUTTON */
div[data-testid="stButton"] > button {
    background:linear-gradient(135deg,#1A56DB,#0EA5E9) !important;
    color:white !important; border:none !important; border-radius:14px !important;
    padding:16px 32px !important; font-size:1.05rem !important;
    font-weight:700 !important; font-family:'Plus Jakarta Sans',sans-serif !important;
    box-shadow:0 8px 24px rgba(26,86,219,0.35) !important; width:100%;
}
div[data-testid="stButton"] > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 12px 32px rgba(26,86,219,0.45) !important;
}

/* RESULT */
.result-danger  { background:linear-gradient(135deg,#FEF2F2,#FFE4E6); border:2px solid #FECACA; border-radius:20px; padding:28px; text-align:center; }
.result-success { background:linear-gradient(135deg,#F0FDF4,#DCFCE7); border:2px solid #BBF7D0; border-radius:20px; padding:28px; text-align:center; }
.result-title { font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:700; margin:12px 0 6px; }
.result-title-danger  { color:#DC2626; }
.result-title-success { color:#16A34A; }
.result-emoji { font-size:3rem; }
.result-desc  { font-size:0.88rem; color:#64748B; margin:0; }

/* PROB BAR */
.prob-container { background:#FFF; border-radius:20px; padding:24px; border:1px solid #E2E8F0; box-shadow:0 4px 16px rgba(0,0,0,0.05); }
.prob-label { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
.prob-label-text { font-size:0.83rem; font-weight:600; color:#374151; }
.prob-value { font-size:1.1rem; font-weight:800; }
.bar-track { height:12px; background:#F1F5F9; border-radius:100px; overflow:hidden; margin-bottom:18px; }
.bar-fill-red   { height:100%; background:linear-gradient(90deg,#EF4444,#F97316); border-radius:100px; }
.bar-fill-green { height:100%; background:linear-gradient(90deg,#10B981,#06B6D4); border-radius:100px; }

/* REC */
.rec-high   { background:linear-gradient(135deg,#FFF7ED,#FFEDD5); border-left:5px solid #F97316; border-radius:0 16px 16px 0; padding:18px 22px; margin-top:14px; }
.rec-medium { background:linear-gradient(135deg,#FFFBEB,#FEF3C7); border-left:5px solid #F59E0B; border-radius:0 16px 16px 0; padding:18px 22px; margin-top:14px; }
.rec-low    { background:linear-gradient(135deg,#F0FDF4,#DCFCE7); border-left:5px solid #10B981; border-radius:0 16px 16px 0; padding:18px 22px; margin-top:14px; }
.rec-title  { font-size:0.95rem; font-weight:700; color:#1E293B; margin:0 0 4px; }
.rec-desc   { font-size:0.83rem; color:#475569; margin:0; line-height:1.6; }

/* METRIC */
[data-testid="metric-container"] {
    background:#F8FAFC; border:1.5px solid #E2E8F0; border-radius:14px; padding:14px 18px;
}
[data-testid="metric-container"] label {
    color:#64748B !important; font-size:0.75rem !important;
    font-weight:600 !important; text-transform:uppercase; letter-spacing:0.05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#1E293B !important; font-size:1.8rem !important; font-weight:800 !important;
}

/* FOOTER */
.footer-bar {
    background:linear-gradient(135deg,#1E293B,#334155); border-radius:16px;
    padding:18px 28px; margin-top:28px;
    display:flex; justify-content:space-between; align-items:center;
}
.footer-text  { color:#94A3B8; font-size:0.8rem; margin:0; }
.footer-badge { background:rgba(255,255,255,0.1); border-radius:100px; padding:4px 12px; color:#CBD5E1; font-size:0.75rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODEL
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
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

NUMERIC_COLS = [
    "adr","adults","agent","arrival_date_day_of_month","arrival_date_week_number",
    "arrival_date_year","babies","booking_changes","children","days_in_waiting_list",
    "lead_time","previous_bookings_not_canceled","previous_cancellations",
    "required_car_parking_spaces","stays_in_week_nights","stays_in_weekend_nights",
    "total_guests","total_of_special_requests","total_stay"
]

MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]

HOTEL_LIST = [
    "Albuquerque Airport Courtyard Albuquerque, NM","Anaheim Marriott Anaheim, CA",
    "Baltimore BWI Airport Courtyard Linthicum, MD",
    "Baton Rouge Acadian Centre/LSU Area Courtyard Baton Rouge, LA",
    "Berlin Marriott Hotel Berlin, Germany",
    "Cape Town Marriott Hotel Crystal Towers Cape Town, South Africa",
    "Chicago O'Hare Courtyard Des Plaines, IL","Colony Club, Barbados Barbados",
    "Courtyard Las Vegas Convention Center Las Vegas, NV ",
    "Courtyard by Marriott Aberdeen Airport Aberdeen, United Kingdom",
    "Courtyard by Marriott Paris Gare de Lyon Paris, France",
    "Courtyard by Marriott Rio de Janeiro Barra da Tijuca Barra da Tijuca, Brazil",
    "Courtyard by Marriott Toulouse Airport Toulouse, France",
    "Crystal Cove, Barbados Barbados","Des Moines West/Clive Courtyard Clive, IA",
    "Fort Worth University Drive Courtyard Fort Worth, TX",
    "Frankfurt Marriott Hotel Frankfurt, Germany","Greensboro Courtyard Greensboro, NC",
    "Grosvenor House, A JW Marriott Hotel London, United Kingdom",
    "Heidelberg Marriott Hotel Heidelberg, Germany",
    "Hotel Alfonso XIII, a Luxury Collection Hotel, Seville Seville, Spain",
    "Hotel Maria Cristina, San Sebastian San Sebastian, Spain",
    "Indianapolis Airport Courtyard Indianapolis, IN",
    "Irvine John Wayne Airport/Orange County Courtyard Irvine, CA",
    "Las Vegas Marriott Las Vegas, NV ","Leipzig Marriott Hotel Leipzig, Germany",
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
    "Tamarind, Barbados Barbados","The House, Barbados Barbados",
    "The Ritz-Carlton, Berlin Berlin, Germany","The Ritz-Carlton, Tokyo Tokyo, Japan",
    "The St. Regis Osaka Osaka, Japan",
    "The Westin Peachtree Plaza, Atlanta Atlanta, GA ",
    "Treasure Beach, Barbados Barbados","Turtle Beach, Barbados Barbados",
    "W Barcelona Barcelona, Spain",
    "W London \u2013 Leicester Square London, United Kingdom",
    "W New York \u2013 Times Square New York, NY",
    "W New York \u2013 Union Square New York, NY","Waves, Barbados Barbados",
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

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🤖 Powered by XGBoost · Top Features Only</div>
    <h1 class="hero-title">🏨 Hotel Cancellation<br>Risk Predictor</h1>
    <p class="hero-subtitle">
        Isi hanya <strong>7 fitur paling berpengaruh</strong> — sistem akan menghitung
        sisanya secara otomatis. Prediksi instan, akurat, dan efisien.
    </p>
</div>
""", unsafe_allow_html=True)

# ── TOP FEATURES INFO BAR ──
st.markdown("""
<div class="top-feat-banner">
    <span class="top-feat-label">🏆 Fitur Paling Berpengaruh:</span>
    <span class="feat-rank"><span class="feat-rank-num">1</span> Lead Time</span>
    <span class="feat-rank"><span class="feat-rank-num">2</span> Tipe Deposit</span>
    <span class="feat-rank"><span class="feat-rank-num">3</span> Harga Kamar (ADR)</span>
    <span class="feat-rank"><span class="feat-rank-num">4</span> Riwayat Pembatalan</span>
    <span class="feat-rank"><span class="feat-rank-num">5</span> Channel Pemesanan</span>
    <span class="feat-rank"><span class="feat-rank-num">6</span> Permintaan Khusus</span>
    <span class="feat-rank"><span class="feat-rank-num">7</span> Tipe Customer</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FORM — 2 KOLOM RINGKAS
# ─────────────────────────────────────────
col1, col2 = st.columns(2, gap="medium")

# ── KOLOM KIRI ──
with col1:

    # Card 1: Waktu & Harga
    st.markdown("""
    <div class="form-card">
        <div class="card-header">
            <div class="card-icon icon-orange">⏱️</div>
            <div>
                <p class="card-title">Waktu & Harga <span class="impact-badge">🔥 Pengaruh Tertinggi</span></p>
                <p class="card-desc">Lead time & ADR adalah prediktor #1 dan #3</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    lead_time = st.number_input(
        "Lead Time — Berapa hari sebelum check-in booking dibuat?",
        min_value=0, max_value=700, value=30,
        help="Semakin lama lead time, semakin tinggi risiko batal"
    )
    adr = st.number_input(
        "ADR — Harga rata-rata per malam (Rp/USD)",
        min_value=0.0, max_value=5000.0, value=100.0,
        help="Average Daily Rate — harga kamar per malam"
    )

    # Card 2: Deposit & Channel
    st.markdown("""
    <div class="form-card" style="margin-top:4px;">
        <div class="card-header">
            <div class="card-icon icon-purple">💳</div>
            <div>
                <p class="card-title">Deposit & Channel Pemesanan <span class="impact-badge">🔥 Pengaruh Tinggi</span></p>
                <p class="card-desc">Kebijakan deposit & asal booking sangat menentukan risiko</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    deposit_type = st.selectbox(
        "Tipe Deposit",
        ["No Deposit", "Non Refund", "Refundable"],
        help="Non Refund → risiko batal lebih rendah. No Deposit → risiko paling tinggi"
    )
    distribution_channel = st.selectbox(
        "Channel Distribusi",
        ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
        help="Booking via TA/TO cenderung lebih mudah dibatalkan"
    )
    market_segment = st.selectbox(
        "Segmen Market",
        ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups", "Complementary", "Undefined"],
        help="Online TA memiliki tingkat pembatalan lebih tinggi"
    )

    has_deposit = 1 if deposit_type != "No Deposit" else 0
    st.markdown(f"""
    <div class="fe-box">
        <p class="fe-title">⚙️ Dihitung Otomatis</p>
        <div class="fe-chips">
            <span class="fe-chip">Has Deposit <span class="fe-chip-val">{'Ya ✓' if has_deposit else 'Tidak ✗'}</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── KOLOM KANAN ──
with col2:

    # Card 3: Riwayat Tamu
    st.markdown("""
    <div class="form-card">
        <div class="card-header">
            <div class="card-icon icon-blue">📋</div>
            <div>
                <p class="card-title">Riwayat Tamu <span class="impact-badge">🔥 Pengaruh Tinggi</span></p>
                <p class="card-desc">Histori pembatalan adalah sinyal risiko terkuat</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    previous_cancellations = st.number_input(
        "Jumlah Pembatalan Sebelumnya",
        min_value=0, max_value=50, value=0
    )
    previous_bookings_not_canceled = st.number_input(
        "Jumlah Booking Berhasil Sebelumnya",
        min_value=0, max_value=50, value=0
    )
    is_repeated_guest = st.selectbox(
        "Apakah Tamu Pernah Menginap di Sini Sebelumnya?",
        [0, 1], format_func=lambda x: "✅ Ya — Returning Guest" if x == 1 else "❌ Tidak — New Guest"
    )

    high_cancel_history = 1 if previous_cancellations > 0 else 0
    is_loyal            = 1 if previous_bookings_not_canceled > 3 else 0

    st.markdown(f"""
    <div class="fe-box">
        <p class="fe-title">⚙️ Dihitung Otomatis</p>
        <div class="fe-chips">
            <span class="fe-chip">High Cancel History <span class="fe-chip-val">{'Ya ⚠️' if high_cancel_history else 'Tidak ✓'}</span></span>
            <span class="fe-chip">Loyal Guest <span class="fe-chip-val">{'Ya ✓' if is_loyal else 'Tidak'}</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <p class="info-box-title">ℹ️ Panduan Pengisian</p>
        <div class="info-row">
            <span class="info-badge">0</span>
            <span class="info-text"><strong>Pembatalan = 0</strong> → Belum pernah batal (risiko rendah)</span>
        </div>
        <div class="info-row">
            <span class="info-badge">≥1</span>
            <span class="info-text"><strong>Pembatalan ≥ 1</strong> → Pernah batal → ditandai <em>High Cancel History</em></span>
        </div>
        <div class="info-row">
            <span class="info-badge-green">≥4</span>
            <span class="info-text"><strong>Booking Berhasil ≥ 4</strong> → Dikategorikan sebagai <em>Loyal Guest</em></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Card 4: Tamu & Permintaan
    st.markdown("""
    <div class="form-card" style="margin-top:4px;">
        <div class="card-header">
            <div class="card-icon icon-green">🎯</div>
            <div>
                <p class="card-title">Tamu & Permintaan Khusus</p>
                <p class="card-desc">Engagement tamu & tipe customer</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    customer_type = st.selectbox(
        "Tipe Customer",
        ["Transient", "Contract", "Group", "Transient-Party"],
        help="Transient memiliki kebebasan batal lebih tinggi dibanding Contract"
    )
    total_of_special_requests = st.number_input(
        "Jumlah Permintaan Khusus (0–5)",
        min_value=0, max_value=5, value=0,
        help="Misal: kamar non-smoking, extra bed, lantai tinggi, dll."
    )

    st.markdown("""
    <div class="info-box">
        <p class="info-box-title">ℹ️ Panduan Pengisian</p>
        <div class="info-row">
            <span class="info-badge">0</span>
            <span class="info-text"><strong>Permintaan = 0</strong> → Tidak ada kebutuhan khusus</span>
        </div>
        <div class="info-row">
            <span class="info-badge-green">1–5</span>
            <span class="info-text"><strong>Permintaan ≥ 1</strong> → Tamu lebih engaged → risiko batal lebih rendah</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# NILAI DEFAULT UNTUK FITUR LAIN
# (tidak ditampilkan ke user, pakai median/default)
# ─────────────────────────────────────────
# Fitur tamu (default)
adults   = 2; children = 0; babies = 0
# Tanggal (default nilai tengah)
arrival_date_year = 2019; arrival_date_month = "July"
arrival_date_day_of_month = 15; arrival_date_week_number = 27
# Durasi (default 2 malam weekday, 1 weekend)
stays_in_weekend_nights = 1; stays_in_week_nights = 2
# Hotel & kamar (default paling umum)
hotel = "Crystal Cove, Barbados Barbados"
reserved_room_type = "A"; assigned_room_type = "A"; meal = "BB"
country = "PRT"; agent = 0
# Fitur minor
booking_changes = 0; days_in_waiting_list = 0
required_car_parking_spaces = 0

# Feature engineering
total_guests = adults + children + babies
total_stay   = stays_in_weekend_nights + stays_in_week_nights
is_family    = 0

# ─────────────────────────────────────────
# TOMBOL PREDIKSI
# ─────────────────────────────────────────
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
predict_btn = st.button("🔍  Prediksi Risiko Pembatalan Sekarang", use_container_width=True)

# ─────────────────────────────────────────
# LOGIKA PREDIKSI
# ─────────────────────────────────────────
if predict_btn:
    input_dict = {col: 0 for col in feature_columns}
    input_dict.update({
        "adr": adr, "adults": adults, "agent": agent,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_year": arrival_date_year,
        "babies": babies, "booking_changes": booking_changes,
        "children": children, "days_in_waiting_list": days_in_waiting_list,
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

    def set_onehot(key):
        if key in input_dict: input_dict[key] = 1

    set_onehot(f"arrival_date_month_{arrival_date_month}")
    if assigned_room_type != "A": set_onehot(f"assigned_room_type_{assigned_room_type}")
    set_onehot(f"country_{country}")
    if customer_type != "Contract": set_onehot(f"customer_type_{customer_type}")
    if deposit_type != "No Deposit": set_onehot(f"deposit_type_{deposit_type}")
    if distribution_channel != "Corporate": set_onehot(f"distribution_channel_{distribution_channel}")
    set_onehot(f"hotel_{hotel}")

    ms_map = {
        "Complementary":"market_segment_Complementary","Corporate":"market_segment_Corporate",
        "Direct":"market_segment_Direct","Groups":"market_segment_Groups",
        "Offline TA/TO":"market_segment_Offline TA/TO","Online TA":"market_segment_Online TA",
        "Undefined":"market_segment_Undefined",
    }
    if market_segment in ms_map: set_onehot(ms_map[market_segment])

    meal_map = {"FB":"meal_FB","HB":"meal_HB","SC":"meal_SC","Undefined":"meal_Undefined"}
    if meal in meal_map: set_onehot(meal_map[meal])
    if reserved_room_type != "A": set_onehot(f"reserved_room_type_{reserved_room_type}")

    input_df = pd.DataFrame([input_dict])[feature_columns]
    input_df[NUMERIC_COLS] = scaler.transform(input_df[NUMERIC_COLS])

    prediction      = model.predict(input_df)[0]
    proba           = model.predict_proba(input_df)[0]
    cancel_prob     = proba[1] * 100
    not_cancel_prob = proba[0] * 100

    # ── HASIL ──
    st.markdown("<hr style='border:none;border-top:2px solid #E2E8F0;margin:28px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;margin-bottom:20px;'>
        <h2 style='font-family:Playfair Display,serif;font-size:1.8rem;color:#1E293B;margin:0 0 4px;'>📊 Hasil Prediksi</h2>
        <p style='font-size:0.85rem;color:#64748B;margin:0;'>Berdasarkan 7 fitur utama + 6 fitur engineered otomatis</p>
    </div>
    """, unsafe_allow_html=True)

    r1, r2 = st.columns([1,1], gap="medium")

    with r1:
        if prediction == 1:
            st.markdown("""
            <div class="result-danger">
                <div class="result-emoji">⚠️</div>
                <p class="result-title result-title-danger">Berisiko Dibatalkan</p>
                <p class="result-desc">Model memprediksi booking ini memiliki probabilitas tinggi untuk dibatalkan.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-success">
                <div class="result-emoji">✅</div>
                <p class="result-title result-title-success">Tidak Akan Dibatalkan</p>
                <p class="result-desc">Model memprediksi tamu kemungkinan besar akan check-in sesuai jadwal.</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1: st.metric("🔴 Prob. Dibatalkan",      f"{cancel_prob:.1f}%")
        with m2: st.metric("🟢 Prob. Tidak Dibatalkan", f"{not_cancel_prob:.1f}%")

    with r2:
        st.markdown(f"""
        <div class="prob-container">
            <p style="font-weight:700;color:#1E293B;margin:0 0 18px;font-size:0.92rem;">📈 Distribusi Probabilitas</p>
            <div class="prob-label">
                <span class="prob-label-text">🔴 Risiko Dibatalkan</span>
                <span class="prob-value" style="color:#EF4444;">{cancel_prob:.1f}%</span>
            </div>
            <div class="bar-track"><div class="bar-fill-red" style="width:{cancel_prob}%"></div></div>
            <div class="prob-label">
                <span class="prob-label-text">🟢 Kemungkinan Aman</span>
                <span class="prob-value" style="color:#10B981;">{not_cancel_prob:.1f}%</span>
            </div>
            <div class="bar-track"><div class="bar-fill-green" style="width:{not_cancel_prob}%"></div></div>
            <p style="font-size:0.72rem;color:#94A3B8;margin:14px 0 0;text-align:center;">
                Total = 100% &nbsp;|&nbsp; XGBoost Classifier &nbsp;|&nbsp; Accuracy ~88.5%
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if cancel_prob >= 70:
        st.markdown(f"""
        <div class="rec-high">
            <p class="rec-title">🔴 Risiko Tinggi — Tindakan Segera Diperlukan</p>
            <p class="rec-desc">Probabilitas pembatalan <strong>{cancel_prob:.1f}%</strong>. Terapkan <strong>Non-Refundable Deposit</strong>, konfirmasi ulang via telepon, dan pertimbangkan upgrade kamar sebagai insentif agar tamu tetap check-in.</p>
        </div>""", unsafe_allow_html=True)
    elif cancel_prob >= 40:
        st.markdown(f"""
        <div class="rec-medium">
            <p class="rec-title">🟡 Risiko Sedang — Perlu Dipantau</p>
            <p class="rec-desc">Probabilitas pembatalan <strong>{cancel_prob:.1f}%</strong>. Kirimkan reminder booking dan tawarkan layanan tambahan atau diskon early check-in untuk meningkatkan komitmen tamu.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="rec-low">
            <p class="rec-title">🟢 Risiko Rendah — Booking Aman</p>
            <p class="rec-desc">Probabilitas pembatalan hanya <strong>{cancel_prob:.1f}%</strong>. Booking ini kemungkinan besar berjalan lancar. Fokus pada pengalaman tamu terbaik saat check-in.</p>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
    <p class="footer-text">🏨 Hotel Cancellation Risk Predictor &nbsp;·&nbsp; XGBoost · Streamlit</p>
    <span class="footer-badge">Accuracy ~88.5% · 7 Input Utama</span>
</div>
""", unsafe_allow_html=True)
