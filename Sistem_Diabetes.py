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
    Pregnancies = st.text_input('Jumlah Berapa Kali Hamil')
with col1:
    Glucose = st.text_input('Konsentrasi Kandungan Glukosa (mg/dL)')
with col1:
    BloodPressure = st.text_input('Tekanan Darah Diastolik (mmHg)')
with col1:
    SkinThickness = st.text_input('Ketebalan Lipatan Kulit (mm)')
with col2:
    Insulin = st.text_input('Insulin (muU/mL)')
with col2:
    BMI = st.text_input('Massa Tubuh (kg/m2)')
with col2:
    DiabetesPedigreeFunction = st.text_input('Silsilah Keturunan Diabetes')
with col2:
    Age = st.text_input('Umur (tahun)')

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
