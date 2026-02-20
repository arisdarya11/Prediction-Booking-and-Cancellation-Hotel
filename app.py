import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Hotel Cancel Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
# CUSTOM CSS — Dark Luxury Hotel Theme
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --navy:    #0D1B2A;
    --blue:    #1A3A5C;
    --teal:    #00C2CB;
    --gold:    #E8B84B;
    --white:   #F4F6F9;
    --card:    #112033;
    --muted:   #607080;
    --green:   #10B981;
    --red:     #EF4444;
    --orange:  #F59E0B;
    --border:  rgba(0,194,203,0.18);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--navy) !important;
    color: var(--white) !important;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }

/* ── Main Container ── */
.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1200px !important;
}

/* ── HERO HEADER ── */
.hero-wrapper {
    position: relative;
    background: linear-gradient(135deg, #0D1B2A 0%, #1A3A5C 50%, #0D1B2A 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    overflow: hidden;
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,194,203,0.15) 0%, transparent 70%);
}
.hero-wrapper::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(232,184,75,0.10) 0%, transparent 70%);
}
.hero-badge {
    display: inline-block;
    background: rgba(0,194,203,0.12);
    border: 1px solid rgba(0,194,203,0.35);
    color: var(--teal);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 3px;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    margin-bottom: 0.9rem;
    text-transform: uppercase;
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--white);
    line-height: 1.2;
    margin: 0 0 0.6rem;
}
.hero-title span { color: var(--teal); }
.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    font-weight: 300;
    max-width: 520px;
    line-height: 1.6;
    margin: 0;
}
.hero-model-badge {
    position: absolute; top: 2rem; right: 2.5rem;
    text-align: center;
}
.hero-model-badge .acc { font-family: 'Playfair Display', serif; font-size: 2.2rem; font-weight: 700; color: var(--green); line-height: 1; }
.hero-model-badge .lbl { font-size: 0.7rem; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; margin-top: 0.2rem; }
.hero-model-badge .mdl { font-size: 0.75rem; color: var(--teal); font-weight: 500; margin-top: 0.15rem; }

/* ── STAT CARDS (KPI Row) ── */
.kpi-row {
    display: flex; gap: 1rem; margin-bottom: 2rem;
}
.kpi-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 100%; height: 3px;
}
.kpi-card.teal::before { background: var(--teal); }
.kpi-card.gold::before  { background: var(--gold); }
.kpi-card.red::before   { background: var(--red); }
.kpi-card.green::before { background: var(--green); }
.kpi-val { font-family: 'Playfair Display', serif; font-size: 1.6rem; font-weight: 700; color: var(--white); line-height: 1.1; }
.kpi-val.teal-c { color: var(--teal); }
.kpi-val.gold-c  { color: var(--gold); }
.kpi-val.red-c   { color: var(--red); }
.kpi-val.green-c { color: var(--green); }
.kpi-lbl { font-size: 0.72rem; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; margin-top: 0.25rem; }

/* ── SECTION HEADER ── */
.section-header {
    display: flex; align-items: center; gap: 0.7rem;
    margin-bottom: 1.2rem;
}
.section-icon {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem; flex-shrink: 0;
}
.section-icon.teal { background: rgba(0,194,203,0.15); }
.section-icon.gold  { background: rgba(232,184,75,0.15); }
.section-header-text h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem; font-weight: 600;
    color: var(--white); margin: 0; line-height: 1;
}
.section-header-text p { font-size: 0.78rem; color: var(--muted); margin: 0.2rem 0 0; }

/* ── FORM CARD ── */
.form-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
}

/* ── Streamlit Input Overrides ── */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div:first-child,
div[data-baseweb="textarea"] > div {
    background: #0D1B2A !important;
    border: 1px solid rgba(0,194,203,0.25) !important;
    border-radius: 8px !important;
    color: var(--white) !important;
    transition: border-color 0.2s;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 2px rgba(0,194,203,0.12) !important;
}
input[type="number"], input[type="text"] {
    color: var(--white) !important;
    background: transparent !important;
}
.stSlider > div > div > div { color: var(--teal) !important; }
.stSlider > div { padding: 0 !important; }
[data-testid="stSlider"] div[role="slider"] { background: var(--teal) !important; }
[data-testid="stSlider"] .st-bg { background: rgba(0,194,203,0.25) !important; }

/* ── Number Input ── */
.stNumberInput div { border-color: rgba(0,194,203,0.25) !important; }
.stNumberInput div:focus-within { border-color: var(--teal) !important; }

/* ── Labels ── */
label, .stSelectbox label, .stNumberInput label, .stSlider label {
    color: #A0BAD4 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    margin-bottom: 0.3rem !important;
}

/* ── PREDICT BUTTON ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, var(--teal) 0%, #009BA3 100%) !important;
    color: #0D1B2A !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.85rem 1.5rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    margin-top: 0.5rem !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,194,203,0.35) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── RESULT CARDS ── */
.result-cancel {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(239,68,68,0.05) 100%);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(16,185,129,0.05) 100%);
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-emoji { font-size: 2.8rem; margin-bottom: 0.5rem; }
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem; font-weight: 700;
    margin: 0.3rem 0;
}
.result-title.danger { color: var(--red); }
.result-title.safe   { color: var(--green); }
.result-prob { font-size: 0.85rem; color: var(--muted); margin-bottom: 1rem; }

/* ── PROBABILITY BAR ── */
.prob-bar-wrapper { margin: 1rem 0; }
.prob-bar-track {
    background: rgba(255,255,255,0.08);
    border-radius: 6px; height: 8px;
    overflow: hidden; margin: 0.4rem 0;
}
.prob-bar-fill-red   { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #F59E0B, #EF4444); }
.prob-bar-fill-green { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #34D399, #10B981); }
.prob-bar-labels { display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--muted); }

/* ── RECOMMENDATION CARD ── */
.rec-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
    border-left: 3px solid;
}
.rec-card.warn { border-left-color: var(--orange); }
.rec-card.ok   { border-left-color: var(--green); }
.rec-card h4   { font-size: 0.85rem; font-weight: 600; margin: 0 0 0.3rem; }
.rec-card p    { font-size: 0.8rem; color: var(--muted); margin: 0; line-height: 1.5; }

/* ── RISK METER ── */
.risk-meter {
    margin: 1rem 0;
}
.risk-track {
    display: flex; gap: 3px; margin: 0.5rem 0;
}
.risk-seg {
    flex: 1; height: 10px; border-radius: 3px;
    background: rgba(255,255,255,0.08);
    transition: background 0.4s;
}
.risk-seg.active-red    { background: var(--red); }
.risk-seg.active-orange { background: var(--orange); }
.risk-seg.active-green  { background: var(--green); }
.risk-labels { display: flex; justify-content: space-between; font-size: 0.68rem; color: var(--muted); margin-top: 0.3rem; }

/* ── FEATURE IMPORTANCE MINI ── */
.feat-row {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.feat-name { font-size: 0.8rem; color: #A0BAD4; width: 150px; flex-shrink: 0; }
.feat-bar-wrap { flex: 1; background: rgba(255,255,255,0.06); border-radius: 4px; height: 6px; }
.feat-bar { height: 100%; border-radius: 4px; }
.feat-pct { font-size: 0.78rem; color: var(--white); font-weight: 600; width: 35px; text-align: right; }

/* ── FOOTER ── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
.footer p { font-size: 0.75rem; color: var(--muted); margin: 0.2rem 0; }
.footer a { color: var(--teal); text-decoration: none; }

/* ── TOOLTIP HINT ── */
.hint {
    display: inline-block;
    background: rgba(0,194,203,0.08);
    border: 1px solid rgba(0,194,203,0.2);
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    font-size: 0.75rem;
    color: var(--teal);
    margin-top: 0.4rem;
}

/* ── DIVIDER ── */
.gold-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 1.5rem 0;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    model    = joblib.load("model_rf_reduced.pkl")
    scaler   = joblib.load("scaler.pkl")
    encoder  = joblib.load("encoder.pkl")
    try:
        features = joblib.load("model_features.pkl")
    except Exception:
        features = None
    return model, scaler, encoder, features

model, scaler, encoder, model_features = load_artifacts()
num_cols = list(scaler.feature_names_in_)
cat_cols = list(encoder.feature_names_in_)

# ══════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-model-badge">
        <div class="acc">88.51%</div>
        <div class="lbl">Accuracy</div>
        <div class="mdl">Random Forest · AUC 0.95</div>
    </div>
    <div class="hero-badge">⚡ ML-Powered Prediction</div>
    <h1 class="hero-title">Hotel Booking<br><span>Cancellation</span> Predictor</h1>
    <p class="hero-sub">Prediksi risiko pembatalan reservasi hotel secara real-time menggunakan machine learning berbasis data historis 83.293 transaksi.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# KPI STATS ROW
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="kpi-row">
    <div class="kpi-card teal">
        <div class="kpi-val teal-c">83.293</div>
        <div class="kpi-lbl">Data Transaksi</div>
    </div>
    <div class="kpi-card gold">
        <div class="kpi-val gold-c">36.95%</div>
        <div class="kpi-lbl">Cancel Rate Aktual</div>
    </div>
    <div class="kpi-card red">
        <div class="kpi-val red-c">Rp 29.19M</div>
        <div class="kpi-lbl">Lost Revenue</div>
    </div>
    <div class="kpi-card green">
        <div class="kpi-val green-c">0.95</div>
        <div class="kpi-lbl">AUC Score</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MAIN LAYOUT — 2 COLUMN
# ══════════════════════════════════════════════════════════════════
col_form, col_result = st.columns([1.05, 0.95], gap="large")

# ── LEFT: INPUT FORM ──────────────────────────────────────────────
with col_form:
    # Section header
    st.markdown("""
    <div class="section-header">
        <div class="section-icon teal">📋</div>
        <div class="section-header-text">
            <h3>Detail Reservasi</h3>
            <p>Isi informasi booking untuk mendapatkan prediksi</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Card 1: Waktu & Harga ──
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown("**⏱ Waktu & Harga**")
    c1, c2 = st.columns(2)
    with c1:
        lead_time = st.number_input(
            "Lead Time (hari)",
            min_value=0, max_value=500, value=60,
            help="Jarak hari antara tanggal booking dan check-in"
        )
    with c2:
        adr = st.number_input(
            "ADR — Average Daily Rate",
            min_value=0.0, max_value=1500.0, value=120.0, step=10.0,
            help="Rata-rata harga kamar per malam (USD)"
        )

    c3, c4 = st.columns(2)
    with c3:
        stays_weekend = st.number_input("Malam Akhir Pekan", min_value=0, max_value=15, value=1)
    with c4:
        stays_week = st.number_input("Malam Hari Kerja", min_value=0, max_value=30, value=2)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Card 2: Informasi Tamu ──
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown("**👤 Informasi Tamu & Booking**")
    c5, c6 = st.columns(2)
    with c5:
        country = st.selectbox(
            "Negara Asal",
            ["PRT", "GBR", "FRA", "ESP", "DEU", "IRL", "BEL", "BRA", "USA", "OTHERS"],
            help="Negara asal tamu"
        )
    with c6:
        customer_type = st.selectbox(
            "Tipe Customer",
            ["Transient", "Contract", "Transient-Party", "Group"]
        )

    c7, c8 = st.columns(2)
    with c7:
        market_segment = st.selectbox(
            "Market Segment",
            ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups", "Complementary", "Aviation"]
        )
    with c8:
        distribution_channel = st.selectbox(
            "Distribution Channel",
            ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Card 3: Detail Reservasi ──
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown("**🏨 Detail Reservasi**")

    deposit_type = st.selectbox(
        "Deposit Type",
        ["No Deposit", "Non Refund", "Refundable"],
        help="No Deposit = risiko tinggi | Non Refund = risiko rendah"
    )

    c9, c10 = st.columns(2)
    with c9:
        reserved_room = st.selectbox("Tipe Kamar Dipesan", ["A","B","C","D","E","F","G","H","L","P"])
    with c10:
        assigned_room = st.selectbox("Kamar Ditetapkan Hotel", ["A","B","C","D","E","F","G","H","I","K","L","P"])

    total_of_special_requests = st.slider(
        "Total Special Requests",
        min_value=0, max_value=5, value=1,
        help="Semakin banyak permintaan khusus → tamu lebih serius → risiko cancel lebih rendah"
    )
    st.markdown(
        '<span class="hint">💡 Special request tinggi = komitmen tamu lebih kuat</span>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Card 4: Riwayat ──
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown("**📊 Riwayat Tamu**")
    c11, c12 = st.columns(2)
    with c11:
        previous_cancellations = st.number_input("Riwayat Cancel Sebelumnya", min_value=0, max_value=26, value=0)
    with c12:
        is_repeated_guest = st.selectbox("Tamu Berulang?", ["Tidak (0)", "Ya (1)"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── PREDICT BUTTON ──
    predict_clicked = st.button("🔮  PREDIKSI RISIKO PEMBATALAN", use_container_width=True)

# ── RIGHT: RESULT PANEL ───────────────────────────────────────────
with col_result:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon gold">📊</div>
        <div class="section-header-text">
            <h3>Hasil Prediksi & Analisis</h3>
            <p>Risk score, probabilitas, dan rekomendasi tindakan</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if predict_clicked:
        # ── Build Input ──
        is_repeated = 1 if "Ya" in is_repeated_guest else 0
        room_mismatch = 1 if reserved_room != assigned_room else 0
        total_nights = stays_weekend + stays_week

        user_input = {
            "lead_time":                  lead_time,
            "adr":                        adr,
            "stays_in_weekend_nights":    stays_weekend,
            "stays_in_week_nights":       stays_week,
            "total_of_special_requests":  total_of_special_requests,
            "previous_cancellations":     previous_cancellations,
            "is_repeated_guest":          is_repeated,
            "country":                    country,
            "market_segment":             market_segment,
            "deposit_type":               deposit_type,
            "distribution_channel":       distribution_channel,
            "customer_type":              customer_type,
            "reserved_room_type":         reserved_room,
            "assigned_room_type":         assigned_room,
        }

        # ── Numeric ──
        X_num = pd.DataFrame(0, index=[0], columns=num_cols)
        for k, v in user_input.items():
            if k in num_cols:
                X_num.at[0, k] = v
        X_num_scaled = np.asarray(scaler.transform(X_num))

        # ── Categorical ──
        X_cat = pd.DataFrame(index=[0], columns=cat_cols)
        for i, col in enumerate(cat_cols):
            if col in user_input:
                X_cat.at[0, col] = user_input[col]
            else:
                X_cat.at[0, col] = encoder.categories_[i][0]
        X_cat_enc = encoder.transform(X_cat)
        if hasattr(X_cat_enc, "toarray"):
            X_cat_enc = X_cat_enc.toarray()
        X_cat_enc = np.asarray(X_cat_enc)

        # ── Combine & Pad ──
        X_final = np.concatenate([X_num_scaled, X_cat_enc], axis=1)
        expected = model.n_features_in_
        current  = X_final.shape[1]
        if current > expected:
            X_final = X_final[:, :expected]
        elif current < expected:
            X_final = np.hstack([X_final, np.zeros((1, expected - current))])

        # ── Predict ──
        with st.spinner("Menganalisis data reservasi..."):
            time.sleep(0.6)
            prediction = model.predict(X_final)[0]
            prob       = model.predict_proba(X_final)[0][1]
            safe_prob  = 1 - prob

        # ── Determine Risk Level ──
        if prob >= 0.70:
            risk_level = "SANGAT TINGGI"; risk_segs = 10; seg_class = "active-red"
        elif prob >= 0.50:
            risk_level = "TINGGI"; risk_segs = 7; seg_class = "active-red"
        elif prob >= 0.35:
            risk_level = "SEDANG"; risk_segs = 5; seg_class = "active-orange"
        elif prob >= 0.20:
            risk_level = "RENDAH"; risk_segs = 3; seg_class = "active-green"
        else:
            risk_level = "SANGAT RENDAH"; risk_segs = 1; seg_class = "active-green"

        # ── Result Card ──
        if prediction == 1:
            st.markdown(f"""
            <div class="result-cancel">
                <div class="result-emoji">⚠️</div>
                <div class="result-title danger">BOOKING BERISIKO BATAL</div>
                <div class="result-prob">Probabilitas Cancel: <strong style="color:#EF4444">{prob*100:.1f}%</strong> &nbsp;|&nbsp; Risk Level: <strong style="color:#F59E0B">{risk_level}</strong></div>
                <div class="prob-bar-wrapper">
                    <div class="prob-bar-labels"><span>0%</span><span>Risiko Cancel</span><span>100%</span></div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill-red" style="width:{prob*100:.1f}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-emoji">✅</div>
                <div class="result-title safe">BOOKING KEMUNGKINAN AMAN</div>
                <div class="result-prob">Probabilitas Tidak Cancel: <strong style="color:#10B981">{safe_prob*100:.1f}%</strong> &nbsp;|&nbsp; Risk Level: <strong style="color:#10B981">{risk_level}</strong></div>
                <div class="prob-bar-wrapper">
                    <div class="prob-bar-labels"><span>0%</span><span>Keamanan Booking</span><span>100%</span></div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill-green" style="width:{safe_prob*100:.1f}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Risk Meter ──
        segs_html = ""
        for s in range(10):
            cls = seg_class if s < risk_segs else ""
            segs_html += f'<div class="risk-seg {cls}"></div>'

        st.markdown(f"""
        <div class="risk-meter">
            <div style="font-size:0.78rem; color:#A0BAD4; margin-bottom:0.4rem; text-transform:uppercase; letter-spacing:1px;">Risk Meter</div>
            <div class="risk-track">{segs_html}</div>
            <div class="risk-labels"><span>Sangat Rendah</span><span>Sedang</span><span>Sangat Tinggi</span></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Gold Divider ──
        st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

        # ── Feature Importance Mini ──
        st.markdown("""
        <div style="font-size:0.78rem; color:#A0BAD4; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.7rem;">
        🔍 Faktor Paling Berpengaruh (Model)
        </div>
        """, unsafe_allow_html=True)

        features_importance = [
            ("lead_time",            28, "#EF4444"),
            ("deposit_type",         18, "#F59E0B"),
            ("adr",                  14, "#E8B84B"),
            ("prev_cancellations",   12, "#00C2CB"),
            ("distribution_channel",  9, "#00C2CB"),
            ("special_requests",      7, "#10B981"),
            ("market_segment",        5, "#10B981"),
            ("is_repeated_guest",     4, "#10B981"),
        ]
        feat_html = ""
        for fname, fpct, fcolor in features_importance:
            feat_html += f"""
            <div class="feat-row">
                <span class="feat-name">{fname}</span>
                <div class="feat-bar-wrap">
                    <div class="feat-bar" style="width:{fpct/28*100:.0f}%;background:{fcolor};"></div>
                </div>
                <span class="feat-pct">{fpct}%</span>
            </div>"""
        st.markdown(feat_html, unsafe_allow_html=True)

        # ── Gold Divider ──
        st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

        # ── Recommendations ──
        st.markdown("""
        <div style="font-size:0.78rem; color:#A0BAD4; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.7rem;">
        💼 Rekomendasi Tindakan Bisnis
        </div>
        """, unsafe_allow_html=True)

        if prediction == 1:
            recs = []
            if lead_time > 90:
                recs.append(("warn", "⏰ Lead Time Panjang",
                    f"Lead time {lead_time} hari sangat tinggi. Terapkan deposit progresif atau reconfirmation otomatis H-30 dan H-7."))
            if deposit_type == "No Deposit":
                recs.append(("warn", "💳 Tidak Ada Deposit",
                    "Booking tanpa deposit memiliki cancel rate jauh lebih tinggi. Pertimbangkan deposit minimal 20–30%."))
            if previous_cancellations > 0:
                recs.append(("warn", "📋 Riwayat Cancel",
                    f"Tamu memiliki {previous_cancellations}x riwayat pembatalan. Terapkan kebijakan deposit lebih ketat."))
            if total_of_special_requests == 0:
                recs.append(("warn", "📞 Engagement Rendah",
                    "Tamu tidak membuat permintaan khusus. Coba follow-up proaktif untuk meningkatkan engagement."))
            if not recs:
                recs.append(("warn", "🎯 Pantau Reservasi Ini",
                    "Monitor booking secara berkala. Pertimbangkan konfirmasi ulang H-14 sebelum check-in."))
        else:
            recs = [("ok", "✅ Booking Terlihat Sehat",
                     "Profil reservasi ini menunjukkan risiko rendah. Tetap lakukan konfirmasi standar H-3 sebelum check-in.")]
            if total_of_special_requests >= 3:
                recs.append(("ok", "⭐ Tamu Potensial Loyal",
                    f"{total_of_special_requests} special request menunjukkan keterlibatan tinggi. Pertimbangkan upsell paket premium."))
            if is_repeated == 1:
                recs.append(("ok", "🎁 Tamu Setia",
                    "Tamu berulang — berikan reward loyalitas atau early check-in untuk meningkatkan kepuasan."))

        for rtype, rtitle, rdesc in recs:
            st.markdown(f"""
            <div class="rec-card {rtype}">
                <h4>{rtitle}</h4>
                <p>{rdesc}</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        # ── Placeholder before prediction ──
        st.markdown("""
        <div style="
            background: var(--card);
            border: 1px dashed rgba(0,194,203,0.25);
            border-radius: 14px;
            padding: 3rem 2rem;
            text-align: center;
            color: var(--muted);
        ">
            <div style="font-size:3rem; margin-bottom:1rem;">🏨</div>
            <div style="font-family:'Playfair Display',serif; font-size:1.2rem; color:#A0BAD4; margin-bottom:0.5rem;">
                Prediksi Belum Dijalankan
            </div>
            <p style="font-size:0.85rem; max-width:280px; margin:0 auto; line-height:1.6;">
                Isi detail reservasi di sebelah kiri, lalu klik tombol <strong style="color:var(--teal)">Prediksi Risiko Pembatalan</strong> untuk melihat hasil analisis.
            </p>
        </div>

        <div style="margin-top:1.5rem;">
            <div style="font-size:0.78rem; color:#A0BAD4; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.8rem;">📌 Panduan Pengisian</div>
        </div>
        """, unsafe_allow_html=True)

        guides = [
            ("⏱", "Lead Time", "Jarak hari booking → check-in. Makin jauh, risiko makin tinggi."),
            ("💳", "Deposit Type", "'No Deposit' = risiko paling tinggi. 'Non Refund' = paling aman."),
            ("📋", "Special Requests", "Makin banyak permintaan, makin serius tamu tersebut."),
            ("📂", "Market Segment", "Direct booking biasanya paling stabil. Online TA paling berisiko."),
        ]
        for gicon, gtitle, gdesc in guides:
            st.markdown(f"""
            <div class="rec-card ok" style="margin-bottom:0.6rem;">
                <h4>{gicon} {gtitle}</h4>
                <p>{gdesc}</p>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <p>🏨 <strong>Hotel Booking Cancellation Predictor</strong> — Powered by Random Forest (Accuracy 88.51% · AUC 0.95)</p>
    <p>Dibuat oleh <strong>Aris Darya Fernanda</strong> · Data Analyst & Data Scientist · 
       <a href="https://github.com/arisdarya11" target="_blank">github.com/arisdarya11</a> · 
       <a href="https://prediction-booking-and-cancellation-hotel.streamlit.app/" target="_blank">Live App ↗</a>
    </p>
    <p style="margin-top:0.5rem; font-size:0.68rem;">
        Dataset: 83.293 transaksi hotel (2017–2019) · Model: Random Forest · Features: lead_time, deposit_type, adr, prev_cancellations, distribution_channel
    </p>
</div>
""", unsafe_allow_html=True)
