import pickle
import streamlit as st
import numpy as np

# Membaca Dataset
diabetes_model = pickle.load(open('modelsvmsmote.sav', 'rb'))

# Judul Web
st.title('Sistem Klasifikasi Diabetes')

# Membagi Kolom
col1, col2 = st.columns(2)
with col1:
    Pregnancies = st.text_input('Nilai Pregnancies')
with col1:
    Glucose = st.text_input('Nilai Glucose (mg/dL)')
with col1:
    BloodPressure = st.text_input('Nilai BloodPressure (mmHg)')
with col1:
    SkinThickness = st.text_input('Nilai SkinThickness')
with col2:
    Insulin = st.text_input('Nilai Insulin')
with col2:
    BMI = st.text_input('Nilai BMI')
with col2:
    DiabetesPedigreeFunction = st.text_input('Nilai DiabetesPedigreeFunction')
with col2:
    Age = st.text_input('Nilai Age (tahun)')

# Pemeriksaan kesalahan dan Normalisasi input menggunakan Min-Max scaler
input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
# Code untuk Klasifikasi
diab_diagnosis = ''

# Membuat Tombol untuk Klasifikasi
if st.button('Cek Hasil'):
    try:
        class_diabetes = diabetes_model.predict(input_data)

        # Memastikan hasil prediksi merupakan 0 atau 1
        if(class_diabetes[0] == 0):
            diab_diagnosis = 'Pasien Tidak Terkena Diabetes'
        else:
            diab_diagnosis = 'Pasien Terkena Diabetes'
    
    except ValueError as e:
        st.error(f"Error: {e}. Pastikan Anda Mengisi Semua Nilai Dengan Benar.")

st.success(diab_diagnosis)
