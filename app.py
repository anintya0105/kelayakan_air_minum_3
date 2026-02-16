import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Uji Kelayakan Air Minum",
    page_icon="💧",
    layout="wide"
)

# ==========================================
# 2. LOAD MODEL (RF) & SCALER
# ==========================================
@st.cache_resource
def load_resources():
    try:
        # Load Random Forest & Scaler
        model = joblib.load('model_random_forest.pkl')
        scaler = joblib.load('minmax_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_resources()

if model is None:
    st.error("❌ Error: File 'rf_model.pkl' atau 'scaler.pkl' tidak ditemukan.")
    st.stop()

# ==========================================
# 3. SIDEBAR (INPUT 9 PARAMETER)
# ==========================================
st.sidebar.header("📝 Panel Input Data")
st.sidebar.markdown("Masukkan data laboratorium air:")

# --- DEFAULT VALUE (AMAN / POTABLE) ---
def_ph = 7.08
def_hard = 196.0
def_solid = 22000.0
def_chloro = 7.10
def_sulfate = 333.0
def_conduct = 426.0
def_carbon = 14.0
def_trihalo = 66.0
def_turbid = 3.96

# Input 1-3 (Parameter Utama)
ph = st.sidebar.number_input("1. pH Level", 0.0, 14.0, def_ph, help="Netral: 7.0. Aman: 6.5-8.5")
hardness = st.sidebar.number_input("2. Hardness (mg/L)", 0.0, 400.0, def_hard, help="Kekerasan air")
solids = st.sidebar.number_input("3. Solids (ppm)", 0.0, 60000.0, def_solid, help="Total padatan terlarut")

st.sidebar.markdown("---")
st.sidebar.markdown("**Parameter Tambahan (Opsional):**")

# Input 4-9 (Parameter Pendukung)
chloramines = st.sidebar.number_input("4. Chloramines (ppm)", 0.0, 14.0, def_chloro)
sulfate = st.sidebar.number_input("5. Sulfate (mg/L)", 0.0, 500.0, def_sulfate)
conductivity = st.sidebar.number_input("6. Conductivity (μS/cm)", 0.0, 800.0, def_conduct)
organic_carbon = st.sidebar.number_input("7. Organic Carbon (ppm)", 0.0, 30.0, def_carbon)
trihalomethanes = st.sidebar.number_input("8. Trihalomethanes (μg/L)", 0.0, 125.0, def_trihalo)
turbidity = st.sidebar.number_input("9. Turbidity (NTU)", 0.0, 7.0, def_turbid)

# ==========================================
# 4. HALAMAN UTAMA (TENGAH)
# ==========================================
st.title("💧 Aplikasi Uji Kelayakan Air Minum")
st.markdown("""
Selamat datang! Aplikasi ini menggunakan **Algoritma (Random Forest)** untuk memprediksi apakah air layak untuk diminum atau tidak.
Silakan masukkan data hasil uji lab di panel sebelah kiri.
""")

# --- FITUR EDUKASI ---
st.info("ℹ️ **Panduan Standar Kualitas Air (WHO & Kemenkes)**")
with st.expander("Klik di sini untuk melihat Tabel Kriteria Aman", expanded=True):
    st.markdown("""
    | Parameter | Satuan | Rentang Aman (Ideal) | Bahaya Jika |
    | :--- | :--- | :--- | :--- |
    | **pH** | - | **6.5 - 8.5** | < 6 (Asam) atau > 8.5 (Basa) |
    | **Hardness** | mg/L | **< 300** | Terlalu tinggi (Kerak) |
    | **Solids (TDS)** | ppm | **< 500 - 1000** | Air keruh / berasa |
    | **Chloramines** | ppm | **< 4.0** | Bau kaporit menyengat |
    | **Sulfate** | mg/L | **< 250** | Rasa pahit / Diare |
    | **Turbidity** | NTU | **< 5.0** | Air tampak keruh/kotor |
    """)

# ==========================================
# 5. TOMBOL & LOGIKA PREDIKSI
# ==========================================
st.markdown("---")

if st.button("🔍 CEK KELAYAKAN AIR", type="primary", use_container_width=True):
    
    # 1. Kumpulkan Data
    input_data = np.array([[
        ph, hardness, solids, chloramines, sulfate, 
        conductivity, organic_carbon, trihalomethanes, turbidity
    ]])
    
    # 2. Scaling
    input_scaled = scaler.transform(input_data)
    
    # 3. Prediksi
    prediksi = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    confidence = proba[prediksi] * 100
    
    # 4. TAMPILAN HASIL (LAYOUT 2 KOLOM YANG ANDA SUKAI)
    st.markdown("---")
    
    # Membagi layar jadi 2 kolom: Kiri (Hasil), Kanan (Meteran Keyakinan)
    col_hasil, col_meter = st.columns([2, 1])

    with col_hasil:
        st.subheader("Hasil Analisis:")
        if prediksi == 1:
            st.success("✅ **STATUS: AIR LAYAK MINUM (POTABLE)**")
            st.write("**Kesimpulan:** Parameter air berada dalam batas wajar dan **AMAN** dikonsumsi.")
        else:
            st.error("⛔ **STATUS: AIR TIDAK LAYAK MINUM (NOT POTABLE)**")
            st.write("**Kesimpulan:** Terdapat indikator kimia yang **BERBAHAYA** bagi kesehatan.")

    with col_meter:
        st.subheader("Tingkat Keyakinan:")
        # Menampilkan angka besar dan progress bar
        st.metric(label="Probabilitas Model", value=f"{confidence:.1f}%")
        st.progress(int(confidence))
        st.caption("Semakin tinggi %, semakin yakin model dengan prediksinya.")

    # Tampilkan Data Input (Review)
    with st.expander("Lihat Rincian Data Input Anda"):
        st.dataframe(pd.DataFrame(input_data, columns=[
            'pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
            'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity'
        ]))

else:
    st.write("👈 *Ubah nilai di Sidebar kiri, lalu tekan tombol di atas.*")