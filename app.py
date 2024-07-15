import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Memuat model dan scaler
with open('model_random_forest.sav', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.sav', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Fungsi untuk melakukan prediksi
def predict_outcome(data):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    outcome_map = {0: 'DISCHARGE', 1: 'EXPIRY', 2: 'DAMA'}
    return outcome_map[prediction[0]]

# Judul aplikasi
st.title('Prediksi Outcome Pasien Rumah Sakit')

# Input dari pengguna
age = st.number_input('Usia', min_value=0, max_value=120, value=65)
gender = st.selectbox('Jenis Kelamin', [0, 1], format_func=lambda x: 'Laki-laki' if x == 0 else 'Perempuan')
rural = st.selectbox('Lokasi Tempat Tinggal', [0, 1], format_func=lambda x: 'Rural' if x == 0 else 'Urban')
admission_type = st.selectbox('Tipe Admission', [0, 1], format_func=lambda x: 'Emergency' if x == 0 else 'OPD')
duration_of_stay = st.number_input('Durasi Tinggal', min_value=1, max_value=365, value=10)
dm = st.selectbox('Diabetes Mellitus', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
htn = st.selectbox('Hipertensi', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
cad = st.selectbox('Penyakit Arteri Koroner', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
prior_cmp = st.selectbox('Riwayat Penyakit Jantung', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
ckd = st.selectbox('Penyakit Ginjal Kronis', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
smoking = st.selectbox('Kebiasaan Merokok', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
alcohol = st.selectbox('Kebiasaan Minum Alkohol', [0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')

# Membuat DataFrame untuk prediksi
input_data = pd.DataFrame([[age, gender, rural, admission_type, duration_of_stay, dm, htn, cad, prior_cmp, ckd, smoking, alcohol]],
                          columns=['AGE', 'GENDER', 'RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD', 'DURATION OF STAY', 'DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD', 'SMOKING', 'ALCOHOL'])

# Tombol prediksi
if st.button('Prediksi'):
    result = predict_outcome(input_data)
    st.write(f'Hasil prediksi: {result}')
