import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Modeli yükle
model = joblib.load("model.pkl")

st.set_page_config(page_title="Titanic Tahmin Uygulaması")
st.title("🚢 Titanic Hayatta Kalma Tahmini")
st.markdown("Lütfen aşağıdaki bilgileri girin:")

# Girdiler
sex = st.radio("Cinsiyetiniz", ("Erkek", "Kadın"))
age = st.slider("Yaşınız", 0, 80, 30)
sibsp = st.number_input("Kardeş/Eş Sayısı (SibSp)", 0, 10, 0)
parch = st.number_input("Ebeveyn/Çocuk Sayısı (Parch)", 0, 10, 0)
fare = st.number_input("Bilet Ücreti (USD)", 0.0, 600.0, 30.0)
embarked = st.selectbox("Biniş Limanı", ("Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"))
pclass = st.selectbox("Yolcu Sınıfı", ("1", "2", "3"))
title = st.selectbox("Unvan (Title)", ["Mr", "Mrs", "Miss", "Rare"])  # Basitleştirilmiş set

# Özellik mühendisliği
fare_log = np.log1p(fare)
sex_male = 1 if sex == "Erkek" else 0
embarked_q = 1 if embarked == "Queenstown (Q)" else 0
embarked_s = 1 if embarked == "Southampton (S)" else 0
pclass_2 = 1 if pclass == "2" else 0
pclass_3 = 1 if pclass == "3" else 0

title_miss = 1 if title == "Miss" else 0
title_mr = 1 if title == "Mr" else 0
title_mrs = 1 if title == "Mrs" else 0
title_rare = 1 if title == "Rare" else 0

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# Girdileri sıraya göre oluştur
input_data = pd.DataFrame([[
    age, sibsp, parch, fare_log,
    sex_male, embarked_q, embarked_s,
    pclass_2, pclass_3,
    title_miss, title_mr, title_mrs, title_rare,
    family_size, is_alone
]], columns=[
    'Age', 'SibSp', 'Parch', 'Fare',
    'Sex_male', 'Embarked_Q', 'Embarked_S',
    'Pclass_2', 'Pclass_3',
    'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare',
    'FamilySize', 'IsAlone'
])

# Tahmin
if st.button("Tahmin Yap"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Sonuç")
    if prediction == 1:
        st.success("✅ Hayatta kalma ihtimali yüksek.")
        st.balloons()
    else:
        st.error("❌ Maalesef hayatta kalma ihtimali düşük.")
    
    st.write(f"**Hayatta Kalma Olasılığı:** %{proba * 100:.2f}")
