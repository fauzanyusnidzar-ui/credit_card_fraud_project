# Fraud Early Warning System

Aplikasi Streamlit untuk memberi skor risiko fraud pada transaksi kartu kredit, memakai model **LightGBM** dari notebook `cc_fraud_project_perbaikan.ipynb`.

- **Threshold default 0,15** (pengaturan Seimbang): recall ±63%, ±40 alert per hari per 100.000 transaksi/tahun
- Skor ≥ 0,50: tahan dan telepon segera · 0,15–0,50: tahan dan konfirmasi SMS/aplikasi · < 0,15: setujui
- Setiap prediksi disertai alasan (kontribusi SHAP bawaan LightGBM, tanpa library `shap`)

## Isi repo

| File | Fungsi |
|---|---|
| `app.py` | Aplikasi Streamlit (cek satu transaksi atau upload CSV) |
| `fraud_model_final.joblib` | Pipeline preprocessing + model LightGBM |
| `requirements.txt` | Library (versi scikit-learn dan LightGBM dikunci agar model bisa dibuka) |
| `contoh_transaksi.csv` | 200 transaksi contoh dari data uji untuk mencoba fitur upload |

## Menjalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Push repo ini ke GitHub.
2. Buka https://share.streamlit.io, login dengan GitHub, lalu klik **New app**.
3. Pilih repo, branch `main`, dan file `app.py`, lalu klik **Deploy**.

## Format CSV

CSV harus berisi 35 kolom fitur dataset asli (misalnya `transaction_amount`, `distance_from_home_km`, `merchant_country`, dan seterusnya). Kolom lain seperti `transaction_id` boleh ada. Jika ada kolom `is_fraud`, aplikasi juga menampilkan recall, precision, dan biaya.
