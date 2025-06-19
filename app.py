import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("🎓 Prediksi Nilai Akhir Mahasiswa")

uts = st.number_input("Nilai UTS", 0, 100, 75)
tugas = st.number_input("Nilai Tugas", 0, 100, 80)
absensi = st.number_input("Kehadiran (%)", 0, 100, 90)

np.random.seed(42)
n = 100
x_uts = np.random.randint(50, 100, n)
x_tugas = np.random.randint(50, 100, n)
x_absen = np.random.randint(70, 100, n)
y_nilai = 0.4*x_uts + 0.4*x_tugas + 0.2*x_absen + np.random.normal(0, 2, n)

X = np.column_stack((x_uts, x_tugas, x_absen))
model = LinearRegression()
model.fit(X, y_nilai)

pred = model.predict([[uts, tugas, absensi]])[0]

st.subheader("📊 Hasil Prediksi:")
st.success(f"Nilai akhir diprediksi: {pred:.2f}")
