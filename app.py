"""
Fraud Early Warning System - Streamlit app
Model: LightGBM (pipeline preprocessing + model) dari cc_fraud_project_perbaikan.ipynb

Jalankan lokal:  streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Early Warning", page_icon="🛡️", layout="wide")

MODEL_PATH = Path(__file__).parent / "fraud_model_final.joblib"
THRESHOLD_DEFAULT = 0.15          # pengaturan "Seimbang" (recall ~63%), sesuai presentasi BOD
BIAYA_FN, BIAYA_FP = 136.18, 7.50


# ---------------------------------------------------------------- model
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["model"]
    prep = pipe.named_steps["prep"]
    # Ambil daftar kategori langsung dari OneHotEncoder di dalam pipeline
    kategori = {}
    for name, trans, cols in prep.transformers_:
        if name == "browser":
            kategori["browser"] = list(trans.named_steps["onehot"].categories_[0])
        elif name == "onehot":
            for c, cats in zip(cols, trans.categories_):
                kategori[c] = list(cats)
    nama_out = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    return bundle, pipe, prep, kategori, nama_out


bundle, pipe, prep, KATEGORI, NAMA_OUT = load_model()
FITUR = bundle["fitur"]


def kolom_asli(n):
    """Petakan nama hasil one-hot (mis. 'merchant_country_USA') ke kolom aslinya."""
    return max([c for c in FITUR if n == c or n.startswith(c + "_")], key=len)


PETA = np.array([kolom_asli(n) for n in NAMA_OUT])


def siapkan(df):
    df = df.copy()
    if "browser" in df:
        df["browser"] = df["browser"].fillna("offline")
    hilang = [c for c in FITUR if c not in df.columns]
    if hilang:
        raise ValueError(f"Kolom berikut tidak ada: {', '.join(hilang)}")
    return df[FITUR]


def skor(df):
    return pipe.predict_proba(siapkan(df))[:, 1]


def alasan(df_satu, top=6):
    """Kontribusi SHAP (TreeSHAP bawaan LightGBM, tanpa library shap) per kolom asli."""
    X = prep.transform(siapkan(df_satu))
    contrib = pipe.named_steps["model"].predict(X, raw_score=True, pred_contrib=True)[0, :-1]
    s = pd.Series(contrib, index=PETA).groupby(level=0).sum()
    return s.reindex(s.abs().sort_values(ascending=False).index).head(top)


def label_risiko(p, t):
    if p >= 0.50:
        return "🔴 Risiko sangat tinggi", "Tahan transaksi dan telepon nasabah segera"
    if p >= t:
        return "🟠 Risiko tinggi", "Tahan transaksi (maks. 24 jam) dan konfirmasi via SMS/aplikasi"
    return "🟢 Risiko rendah", "Setujui transaksi"


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🛡️ Fraud Early Warning")
st.sidebar.caption(f"Model: {bundle['nama_model']}")
threshold = st.sidebar.slider(
    "Threshold (skor minimal untuk ditahan)", 0.05, 0.50, THRESHOLD_DEFAULT, 0.01,
    help="0,15 = Seimbang (recall ~63%, ~40 alert/hari). 0,21 = biaya terendah. 0,10 = agresif.",
)
st.sidebar.markdown(
    """
**Pengaturan referensi (data uji)**
| Threshold | Recall | Precision |
|---|---|---|
| 0,10 | 71% | 6,9% |
| **0,15** | **63%** | **8,8%** |
| 0,21 | 55% | 11,5% |
| 0,30 | 40% | 14,8% |
"""
)

st.title("Deteksi Dini Transaksi Fraud Kartu Kredit")
tab1, tab2 = st.tabs(["Cek satu transaksi", "Cek banyak transaksi (CSV)"])

# ---------------------------------------------------------------- tab 1
with tab1:
    st.write("Isi data transaksi, lalu klik **Hitung skor risiko**. Nilai awal adalah nilai tipikal (median) dari data.")
    with st.form("form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Transaksi")
            transaction_amount = st.number_input("Nominal transaksi ($)", 0.5, 15000.0, 52.26)
            transaction_hour = st.slider("Jam transaksi", 0, 23, 14)
            transaction_day_of_week = st.selectbox("Hari", KATEGORI["transaction_day_of_week"],
                                                   index=KATEGORI["transaction_day_of_week"].index("Tuesday"))
            merchant_category = st.selectbox("Kategori merchant", KATEGORI["merchant_category"],
                                             index=KATEGORI["merchant_category"].index("Online Retail"))
            merchant_country = st.selectbox("Negara merchant", KATEGORI["merchant_country"],
                                            index=KATEGORI["merchant_country"].index("USA"))
            transaction_channel = st.selectbox("Channel", KATEGORI["transaction_channel"],
                                               index=KATEGORI["transaction_channel"].index("Online"))
            device_type = st.selectbox("Perangkat", KATEGORI["device_type"],
                                       index=KATEGORI["device_type"].index("Mobile"))
            browser = st.selectbox("Browser", KATEGORI["browser"],
                                   index=KATEGORI["browser"].index("Chrome") if "Chrome" in KATEGORI["browser"] else 0)
            card_present = st.checkbox("Kartu fisik hadir")
            used_chip = st.checkbox("Memakai chip EMV")
            used_pin = st.checkbox("Memakai PIN")
        with c2:
            st.subheader("Lokasi dan risiko")
            distance_from_home_km = st.number_input("Jarak dari rumah (km)", 0.0, 15000.0, 1949.95)
            distance_from_last_tx_km = st.number_input("Jarak dari transaksi terakhir (km)", 0.0, 12000.0, 13.3)
            international_transaction = st.checkbox("Transaksi lintas negara")
            high_risk_country = st.checkbox("Negara berisiko tinggi")
            num_declined_last_30d = st.number_input("Ditolak 30 hari terakhir", 0, 5, 0)
            num_chargebacks_ever = st.number_input("Jumlah chargeback", 0, 4, 0)
            st.subheader("Kecepatan transaksi")
            transactions_last_1h = st.number_input("Transaksi 1 jam terakhir", 0, 8, 0)
            transactions_last_24h = st.number_input("Transaksi 24 jam terakhir", 0, 18, 2)
            transactions_last_7d = st.number_input("Transaksi 7 hari terakhir", 0, 42, 7)
            amount_last_1h = st.number_input("Nominal 1 jam terakhir ($)", 0.0, 11500.0, 0.0)
            amount_last_24h = st.number_input("Nominal 24 jam terakhir ($)", 0.0, 15200.0, 197.7)
        with c3:
            st.subheader("Profil nasabah")
            customer_age = st.number_input("Umur", 18, 85, 42)
            customer_gender = st.selectbox("Gender", KATEGORI["customer_gender"])
            customer_state = st.selectbox("Negara bagian", KATEGORI["customer_state"])
            email_domain = st.selectbox("Domain email", KATEGORI["email_domain"])
            account_age_days = st.number_input("Umur akun (hari)", 1, 3650, 554)
            credit_limit = st.number_input("Limit kredit ($)", 500.0, 36670.0, 8500.0)
            credit_score = st.number_input("Credit score", 300, 850, 683)
            num_cards_held = st.number_input("Jumlah kartu", 1, 5, 2)
            avg_transaction_amount_30d = st.number_input("Rata-rata transaksi 30 hari ($)", 5.0, 2000.0, 89.37)
            std_transaction_amount_30d = st.number_input("Std transaksi 30 hari ($)", 2.0, 1340.0, 48.17)
            days_since_last_transaction = st.number_input("Hari sejak transaksi terakhir", 0.0, 50.0, 2.4)
        submitted = st.form_submit_button("Hitung skor risiko", type="primary")

    if submitted:
        is_weekend = int(transaction_day_of_week in ("Saturday", "Sunday"))
        amount_vs_avg_ratio = transaction_amount / max(avg_transaction_amount_30d, 0.01)
        data = {k: v for k, v in locals().items() if k in FITUR}
        data.update(is_weekend=is_weekend, amount_vs_avg_ratio=amount_vs_avg_ratio)
        for b in ["card_present", "used_chip", "used_pin", "international_transaction", "high_risk_country"]:
            data[b] = int(data[b])
        df1 = pd.DataFrame([data])[FITUR]

        p = float(skor(df1)[0])
        status, aksi = label_risiko(p, threshold)
        m1, m2, m3 = st.columns(3)
        m1.metric("Skor risiko fraud", f"{p:.3f}")
        m2.metric("Status", status)
        m3.metric("Threshold", f"{threshold:.2f}")
        (st.error if p >= threshold else st.success)(f"**Rekomendasi:** {aksi}")

        st.subheader("Alasan utama (SHAP)")
        r = alasan(df1)
        tabel = pd.DataFrame({
            "Faktor": r.index,
            "Nilai": [str(round(v, 2)) if isinstance(v, float) else str(v) for v in (df1.iloc[0][c] for c in r.index)],
            "Pengaruh": ["↑ menaikkan risiko" if v > 0 else "↓ menurunkan risiko" for v in r.values],
            "Kontribusi (log-odds)": r.values.round(3),
        })
        st.dataframe(tabel, hide_index=True, width="stretch")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.write(
        "Upload CSV dengan kolom yang sama seperti dataset (kolom tambahan seperti `transaction_id` "
        "boleh ada dan ikut ditampilkan). Kolom `browser` yang kosong otomatis diisi `offline`."
    )
    up = st.file_uploader("File CSV", type="csv")
    if up is not None:
        try:
            raw = pd.read_csv(up)
            raw["skor_fraud"] = skor(raw).round(4)
            raw["ditahan"] = raw["skor_fraud"] >= threshold
            n, n_tahan = len(raw), int(raw["ditahan"].sum())
            k1, k2, k3 = st.columns(3)
            k1.metric("Jumlah transaksi", f"{n:,}")
            k2.metric("Ditahan", f"{n_tahan:,} ({n_tahan / n:.1%})")
            k3.metric("Skor rata-rata", f"{raw['skor_fraud'].mean():.3f}")

            if "is_fraud" in raw:   # jika label tersedia, tampilkan evaluasi
                y = raw["is_fraud"].astype(bool)
                tp = int((raw["ditahan"] & y).sum()); fp = int((raw["ditahan"] & ~y).sum())
                fn = int((~raw["ditahan"] & y).sum())
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Recall", f"{tp / max(tp + fn, 1):.1%}")
                e2.metric("Precision", f"{tp / max(tp + fp, 1):.1%}")
                e3.metric("TP / FP / FN", f"{tp} / {fp} / {fn}")
                e4.metric("Biaya (FN×136,18 + FP×7,50)", f"${fn * BIAYA_FN + fp * BIAYA_FP:,.0f}")

            depan = [c for c in ["transaction_id", "skor_fraud", "ditahan"] if c in raw]
            hasil = raw[depan + [c for c in raw.columns if c not in depan]]
            st.dataframe(hasil.sort_values("skor_fraud", ascending=False), width="stretch", height=420)
            st.download_button("Download hasil (CSV)", hasil.to_csv(index=False).encode(),
                               "hasil_skor_fraud.csv", "text/csv")
        except ValueError as e:
            st.error(str(e))

st.caption("Skor adalah probabilitas fraud dari model LightGBM. Keputusan akhir tetap melalui konfirmasi tim Fraud Authorization.")
