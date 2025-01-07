import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# .pkl fayllarni yuklash
@st.cache_data
def load_models():
    with open('naive_bayes_model1.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler1.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('pca_model1.pkl', 'rb') as file:
        pca = pickle.load(file)
    return model, scaler, pca

# Model, Scaler va PCA yuklash
model, scaler, pca = load_models()

# Foydalanuvchi interfeysi
st.title("Kiberxavfsizlik: Fraud Detection Ilovasi")
st.write("Kredit kartadagi firibgarlikni aniqlash uchun ma'lumotlaringizni kiriting.")

# Xususiyatlarni kiritish (tasodifiy qiymat)
np.random.seed()  # Har safar turli qiymatlar bo'lishi uchun
feature_1 = np.random.uniform(-3.49, 8.08)
feature_2 = np.random.uniform(-3.49, 8.08)
feature_3 = np.random.uniform(-3.49, 8.08)
feature_4 = np.random.uniform(-3.49, 8.08)
feature_5 = np.random.uniform(-3.49, 8.08)

# Streamlit sidebarda tasodifiy qiymatlarni ko'rsatish
st.sidebar.header("Ma'lumotlar (tasodifiy qiymatlar bilan):")
st.sidebar.write(f"Xususiyat 1: {feature_1:.2f}")
st.sidebar.write(f"Xususiyat 2: {feature_2:.2f}")
st.sidebar.write(f"Xususiyat 3: {feature_3:.2f}")
st.sidebar.write(f"Xususiyat 4: {feature_4:.2f}")
st.sidebar.write(f"Xususiyat 5: {feature_5:.2f}")

# Kiritilgan ma'lumotlarni array shakliga keltirish
# 22 xususiyatni tasodifiy qiymatlar bilan to'ldirish
# Kiritilgan ma'lumotlarni array shakliga keltirish (22 xususiyatga to'ldirish)
# 22 ta tasodifiy qiymatni -3.49 dan 8.08 gacha generatsiya qilish
input_data = np.random.uniform(-3.49, 8.08, size=(1, 22))

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

# Foydalanuvchi interfeysida tasodifiy qiymatlarni ko'rsatish
st.sidebar.header("Ma'lumotlar (tasodifiy qiymatlar bilan):")
for i, value in enumerate(input_data[0]):
    st.sidebar.write(f"Xususiyat {i+1}: {value:.2f}")

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

