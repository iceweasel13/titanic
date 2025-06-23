import streamlit as st
import pandas as pd
import numpy as np
import joblib # Kaydedilen modeli yüklemek için

# 1. Model Yükleme
# train.csv ve model.pkl dosyalarının app.py ile aynı dizinde olduğundan emin olun.
try:
    model = joblib.load('model.pkl')
    st.success("Model başarıyla yüklendi!")
except FileNotFoundError:
    st.error("HATA: 'model.pkl' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    st.stop() # Dosya yoksa uygulamayı durdur

# 2. Streamlit Uygulamasının Başlığı ve Açıklaması
st.set_page_config(page_title="Titanic Hayatta Kalma Tahmini", layout="centered")

st.title("🚢 Titanic Hayatta Kalma Tahmini Uygulaması")
st.markdown("Bu uygulama, Titanic veri setini kullanarak bir yolcunun hayatta kalıp kalmayacağını tahmin eder.")
st.markdown("**Lütfen aşağıdaki bilgileri girin:**")

# 3. Kullanıcıdan Girdi Alma (Input Widget'ları)

# Cinsiyet
sex = st.radio("Cinsiyetiniz:", ("Erkek", "Kadın"))

# Yaş
age = st.slider("Yaşınız:", min_value=0, max_value=80, value=30)

# Yolcu Sınıfı (Pclass)
pclass_map = {"1. Sınıf (Üst Sınıf)": 1, "2. Sınıf (Orta Sınıf)": 2, "3. Sınıf (Alt Sınıf)": 3}
pclass_selection = st.selectbox("Yolcu Sınıfınız:", list(pclass_map.keys()))
pclass = pclass_map[pclass_selection]

# Kardeş/Eş Sayısı (SibSp)
sibsp = st.number_input("Gemideki kardeş/eş sayınız:", min_value=0, max_value=8, value=0)

# Ebeveyn/Çocuk Sayısı (Parch)
parch = st.number_input("Gemideki ebeveyn/çocuk sayınız:", min_value=0, max_value=6, value=0)

# Bilet Fiyatı (Fare) - Kullanıcıdan ham fiyatı alacağız, dönüşümü biz yapacağız
fare = st.number_input("Bilet Fiyatınız (USD):", min_value=0.0, max_value=600.0, value=30.0, format="%.2f")

# Biniş Limanı (Embarked)
embarked = st.selectbox("Biniş Limanınız:", ("Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"))
# Streamlit'te gösterim için daha okunaklı isimler, kodu işlerken kısa kodları kullanacağız.

# 4. Tahmin Yapma Butonu
if st.button("Hayatta Kalma Tahmini Yap"):
    # 5. Kullanıcı Girişlerini Modele Hazır Hale Getirme (Ön İşleme ve Özellik Mühendisliği)

    # DataFrame oluştur
    # Modelimizin beklediği tüm sütunları ve sıralamayı dikkate almalıyız!
    # Jupyter Notebook'taki X_train.columns çıktısını kullanabiliriz.
    # Örnek: ['Age', 'Fare', 'FamilySize', 'IsAlone', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Pclass_2', 'Pclass_3', 'Title_Mr', 'Title_Mrs', 'Title_Miss', 'Title_Rare']
    
    # Not: Buradaki sütun isimleri, Jupyter Notebook'taki son X_train.columns çıktınızla birebir aynı olmalı.
    # Eğer farklılıklar varsa bu kısımda hata alırsınız.
    # Varsayımsal olarak, aşağıdaki sütunları bekliyoruz.
    
    input_data = pd.DataFrame([[
        age,
        np.log1p(fare), # Fare'yi log dönüşümüne uğrat
        sibsp + parch + 1, # FamilySize
        1 if (sibsp + parch + 1) == 1 else 0, # IsAlone
        1 if sex == "Erkek" else 0, # Sex_male
        1 if embarked == "Queenstown (Q)" else 0, # Embarked_Q
        1 if embarked == "Southampton (S)" else 0, # Embarked_S
        1 if pclass == 2 else 0, # Pclass_2
        1 if pclass == 3 else 0, # Pclass_3
        # Title için varsayımlar (kullanıcıdan unvan almadığımız için cinsiyet ve yaşa göre varsayabiliriz)
        # Daha iyi bir yaklaşım: Kullanıcıdan unvan al, veya yaş/cinsiyete göre en yaygın unvanı atayarak basitçe handle et
        # Basitlik adına, burada Sex ve Age'e göre kaba bir Title varsayımı yapabiliriz.
        # Örneğin: Erkek ve yaş > 15 ise Mr, Kadın ve evli ise Mrs, Kadın ve bekar/çocuk ise Miss, Erkek çocuk ise Master.
        # Bu kısım modelin Title özelliğini nasıl öğrendiğine bağlı olarak hassastır.
        # Şu an için basitçe varsayımsal değerler atayalım:
        # Eğer kullanıcının ismini alıp unvanı oradan çekeceksek burası değişir.
        # Ancak web arayüzünde isim almadığımız için, cinsiyet ve yaşa dayalı Title tahmini yapalım.
        # Ya da daha basit: doğrudan Title'ı kullanıcıdan input olarak alabiliriz veya bu özelliği şimdilik atlayabiliriz.
        # En doğrusu, model eğitiminde kullandığımız Title çıkarma mantığını burada da uygulamak.
        # Modelin beklediği Title One-Hot Encoded kolonları: 'Title_Mr', 'Title_Mrs', 'Title_Miss', 'Title_Rare'
        # Basit bir yaklaşımla, en yaygın unvanlara göre 1/0 atayalım:

        # Title_Mr: Erkek ve 15 yaşından büyükse
        1 if (sex == "Erkek" and age > 15) else 0,
        # Title_Mrs: Kadın ve 15 yaşından büyükse (evli varsayımı)
        1 if (sex == "Kadın" and age > 15) else 0,
        # Title_Miss: Kadın ve 15 yaşından küçükse (veya bekar varsayımı, daha karmaşık olabilir)
        1 if (sex == "Kadın" and age <= 15) else 0,
        # Title_Rare: (Bu senaryoda kullanıcı inputundan çıkarmak zor, varsayalım ki 0)
        0 # Nadir unvanları burada tahmin etmek zor olduğundan varsayımsal olarak 0.
          # Gerçek uygulamada daha sofistike bir yaklaşım gerekir (ör: kullanıcıdan unvan al)
    ]], columns=['Age', 'Fare', 'FamilySize', 'IsAlone', 'Sex_male', 'Embarked_Q', 'Embarked_S',
                 'Pclass_2', 'Pclass_3', 'Title_Mr', 'Title_Mrs', 'Title_Miss', 'Title_Rare'])
    
    # NOT: Yukarıdaki input_data DataFrame'inin sütunları,
    # modeli eğittiğiniz X_train'deki sütunlarla aynı SIRADA ve İSİMDE olmalıdır!
    # Eğer eksik veya fazla sütun varsa hata alırsınız.
    # Örnek Jupyter çıktınızdan X_train.columns'ı alıp buraya yapıştırmak en güvenlisidir.


    # Tahmin yap
    prediction = model.predict(input_data)[0] # predict tek bir değer döner
    prediction_proba = model.predict_proba(input_data)[0] # olasılıkları al (hayatta kalma olasılığı için)

    st.write("---")
    st.subheader("Tahmin Sonucu:")

    if prediction == 1:
        st.success(f"🎉 Tebrikler! Modelimiz bu yolcunun hayatta kalma olasılığının yüksek olduğunu düşünüyor.")
        st.balloons() # Balonlar fırlat :)
    else:
        st.error(f"😔 Maalesef, modelimiz bu yolcunun hayatta kalma olasılığının düşük olduğunu tahmin ediyor.")

    st.write(f"**Hayatta Kalma Olasılığı:** %{(prediction_proba[1] * 100):.2f}")
    st.write(f"**Hayatta Kalmama Olasılığı:** %{(prediction_proba[0] * 100):.2f}")

    st.markdown("---")
    st.info("Bu tahminler eğitimli bir makine öğrenimi modeline dayanmaktadır ve kesinlik garantisi vermez.")