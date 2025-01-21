import streamlit as st
import pickle
import numpy as np
from sklearn.exceptions import NotFittedError

# Streamlit app
st.title("Prediksi Biaya Asuransi")
st.write("Nama: Aulia Adiba")  # Replace with your name
st.write("NIM: 2021230023")    # Replace with your NIM

st.write("Masukkan data Anda untuk memprediksi biaya asuransi.")

# Meminta input dari pengguna
age = st.number_input("Usia (age)", min_value=0, max_value=100, step=1, value=25)
sex = st.selectbox("Jenis Kelamin (sex)", ["Laki-laki", "Perempuan"])
bmi = st.number_input("BMI (bmi)", min_value=0.0, max_value=100.0, step=0.1, value=25.0)
children = st.number_input("Jumlah Anak (children)", min_value=0, max_value=10, step=1, value=0)
smoker = st.selectbox("Apakah Anda Perokok? (smoker)", ["Tidak", "Ya"])

# Preprocess input
sex_encoded = 0 if sex == "Laki-laki" else 1
smoker_encoded = 0 if smoker == "Tidak" else 1

# Membentuk array untuk input
X = np.array([age, sex_encoded, bmi, children, smoker_encoded]).reshape(1, -1)

# Load model yang telah disimpan
model_path = 'model_uas.pkl'  # Pastikan model Anda berada di path yang benar
try:
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file tidak ditemukan. Pastikan file `model_uas.pkl` ada di lokasi yang benar.")
    loaded_model = None
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    loaded_model = None

# Menampilkan prediksi biaya asuransi
if st.button("Prediksi Biaya Asuransi"):
    if loaded_model is not None:
        try:
            charges_pred = loaded_model.predict(X)
            st.success(f"Prediksi Biaya Asuransi: ${charges_pred[0]:,.2f}")
        except NotFittedError:
            st.error("Model belum di-train dengan data. Pastikan model sudah di-train sebelum digunakan.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
    else:
        st.error("Model tidak tersedia. Tidak dapat melakukan prediksi.")
