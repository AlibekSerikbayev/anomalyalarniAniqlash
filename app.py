import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# .pkl fayllarni yuklash
@st.cache_data
def load_models():
    with open('naive_bayes_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('pca_model.pkl', 'rb') as file:
        pca = pickle.load(file)
    return model, scaler, pca

# Model, Scaler va PCA yuklash
model, scaler, pca = load_models()

# Foydalanuvchi interfeysi
st.title("Kiberxavfsizlik: Fraud Detection Ilovasi")
st.write("Kredit kartadagi firibgarlikni aniqlash uchun ma'lumotlaringizni kiriting.")

# Xususiyatlarni kiritish
st.sidebar.header("Ma'lumotlarni kiriting:")
feature_1 = st.sidebar.number_input("Xususiyat 1", value=0.0)
feature_2 = st.sidebar.number_input("Xususiyat 2", value=0.0)
feature_3 = st.sidebar.number_input("Xususiyat 3", value=0.0)
feature_4 = st.sidebar.number_input("Xususiyat 4", value=0.0)
feature_5 = st.sidebar.number_input("Xususiyat 5", value=0.0)

# Kiritilgan ma'lumotlarni array shakliga keltirish
input_data = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5]])

# Bashorat funksiyasi
def predict_fraud(data):
    # Standartlashtirish
    data_scaled = scaler.transform(data)
    # PCA yordamida dimensiyani qisqartirish
    data_pca = pca.transform(data_scaled)
    # Bashorat qilish
    prediction = model.predict(data_pca)
    probability = model.predict_proba(data_pca)[:, 1]
    return prediction[0], probability[0]

# Bashorat qilish tugmasi
if st.button("Bashorat qiling"):
    prediction, probability = predict_fraud(input_data)
    if prediction == 1:
        st.error(f"Natija: Firibgarlik ehtimoli. Ishonch darajasi: {probability:.2%}")
    else:
        st.success(f"Natija: Firibgarlik aniqlanmadi. Ishonch darajasi: {probability:.2%}")

# Ma'lumotlar haqida qo'shimcha
st.write("---")
st.subheader("Qo'shimcha ma'lumot")
st.write("Ushbu ilova **Streamlit** yordamida ishlab chiqilgan va ma'lumotlar bo'yicha tahlil qilish imkonini beradi.")
