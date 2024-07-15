import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Fungsi untuk memuat dan mempersiapkan data
@st.cache
def load_and_prepare_data():
    data = pd.read_csv('data_rumah_sakit.csv')
    
    data['D.O.A'] = pd.to_datetime(data['D.O.A'])
    data['D.O.D'] = pd.to_datetime(data['D.O.D'])
    data['year'] = data['D.O.A'].dt.year

    data.fillna(data.median(numeric_only=True), inplace=True)

    data['GENDER'] = data['GENDER'].map({'M': 0, 'F': 1})
    data['RURAL'] = data['RURAL'].map({'R': 0, 'U': 1})
    data['TYPE OF ADMISSION-EMERGENCY/OPD'] = data['TYPE OF ADMISSION-EMERGENCY/OPD'].map({'E': 0, 'O': 1})
    data['OUTCOME'] = data['OUTCOME'].map({'DISCHARGE': 0, 'EXPIRY': 1, 'DAMA': 2})
    data['SMOKING'] = data['SMOKING'].map({'Tidak': 0, 'Ya': 1})
    data['ALCOHOL'] = data['ALCOHOL'].map({'Tidak': 0, 'Ya': 1})
    data['DM'] = data['DM'].map({'Tidak': 0, 'Ya': 1})
    data['HTN'] = data['HTN'].map({'Tidak': 0, 'Ya': 1})
    data['CAD'] = data['CAD'].map({'Tidak': 0, 'Ya': 1})
    data['PRIOR CMP'] = data['PRIOR CMP'].map({'Tidak': 0, 'Ya': 1})
    data['CKD'] = data['CKD'].map({'Tidak': 0, 'Ya': 1})

    selected_columns = ['AGE', 'GENDER', 'RURAL', 'TYPE OF ADMISSION-EMERGENCY/OPD', 'DURATION OF STAY', 'DM', 'HTN', 'CAD', 'PRIOR CMP', 'CKD', 'SMOKING', 'ALCOHOL']
    X = data[selected_columns]
    y = data['OUTCOME']

    return X, y

# Fungsi untuk melatih model
@st.cache(allow_output_mutation=True)
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, scaler, accuracy, report

# Fungsi untuk melakukan prediksi
def predict_outcome(model, scaler, data):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    outcome_map = {0: 'DISCHARGE', 1: 'EXPIRY', 2: 'DAMA'}
    return outcome_map[prediction[0]]

# Judul aplikasi
st.title('Prediksi Outcome Pasien Rumah Sakit')

# Load dan persiapkan data
X, y = load_and_prepare_data()

# Latih model
model, scaler, accuracy, report = train_model(X, y)

# Tampilkan metrik evaluasi
st.write(f'Akurasi Model: {accuracy:.2f}')
st.write('Laporan Klasifikasi:')
st.text(report)

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
    result = predict_outcome(model, scaler, input_data)
    st.write(f'Hasil prediksi: {result}')
