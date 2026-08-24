"""
Müşteri Ayrılma Tahmini (Churn Prediction) - Makine Öğrenmesi Ara Ödevi
Amaç: Temel makine öğrenmesi akışını (veri okuma, ön işleme, model eğitimi, değerlendirme) tek bir dosyada pratik etmek.
Kullanılan Kütüphaneler: pandas, numpy, scikit-learn
Çalıştırma Adımları: 'python odev.py' komutu ile terminal üzerinden çalıştırılabilir.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Soru 2: Veri setini pandas DataFrame olarak hazırlayın
# ---------------------------------------------------------
# Gerçek veri seti olmadığı için istenen kurallara uygun (en az 100 satır) sentetik veri üretiyoruz.
np.random.seed(42)
n_samples = 300

data = {
    'yas': np.random.randint(18, 70, n_samples),
    'gelir': np.random.randint(3000, 20000, n_samples),
    'abonelik_suresi': np.random.randint(1, 60, n_samples),
    'destek_talebi_sayisi': np.random.randint(0, 10, n_samples),
    'sehir': np.random.choice(['Istanbul', 'Ankara', 'Izmir'], n_samples),
    'uyelik_tipi': np.random.choice(['Premium', 'Standart'], n_samples),
    'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]) # Hedef değişken: 0=Kalıcı, 1=Ayrıldı
}
df = pd.DataFrame(data)

# Soru 4'teki eksik veri doldurma işlemi için kasıtlı olarak veriye boşluklar (NaN) ekliyoruz
df.loc[10:20, 'gelir'] = np.nan
df.loc[50:60, 'abonelik_suresi'] = np.nan

# ---------------------------------------------------------
# Soru 3: Veri setinin ilk satırlarını, boyutunu ve hedef değişkeni inceleyin
# ---------------------------------------------------------
print("\n--- 3. VERİ İNCELEME ---")
print("Veri Setinin İlk 5 Satırı:\n", df.head())
print(f"\nSatır sayısı: {df.shape[0]}, Sütun sayısı: {df.shape[1]}")
print("\nHedef Değişken (Churn) Dağılımı:\n", df['churn'].value_counts())

# ---------------------------------------------------------
# Soru 4: Eksik değer kontrolü yapın ve uygun şekilde doldurun
# ---------------------------------------------------------
print("\n--- 4. EKSİK DEĞER KONTROLÜ VE TEMİZLİK ---")
print("Doldurmadan Önce Eksik Değer Sayıları:\n", df.isnull().sum())

# Sayısal değerlerdeki eksiklikleri medyan (ortanca) ile dolduruyoruz
df['gelir'] = df['gelir'].fillna(df['gelir'].median())
df['abonelik_suresi'] = df['abonelik_suresi'].fillna(df['abonelik_suresi'].median())

print("\nDoldurduktan Sonra Eksik Değer Sayıları:\n", df.isnull().sum())

# ---------------------------------------------------------
# Soru 7: En az 1 tane basit öznitelik üretin
# ---------------------------------------------------------
print("\n--- 7. YENİ ÖZNİTELİK (FEATURE) ÜRETME ---")
# Gelir değişkenini medyan üzerinden 'Yüksek' ve 'Düşük' olarak gruplayalım
df['gelir_grubu'] = np.where(df['gelir'] > df['gelir'].median(), 'Yuksek_Gelir', 'Dusuk_Gelir')
print("Oluşturulan 'gelir_grubu' değişkeninden örnekler:\n", df[['gelir', 'gelir_grubu']].head())

# ---------------------------------------------------------
# Soru 5: Kategorik değişkenleri One-Hot Encoding ile sayısal forma dönüştürün
# ---------------------------------------------------------
print("\n--- 5. ONE-HOT ENCODING ---")
# 'sehir', 'uyelik_tipi' ve 'gelir_grubu' sütunları kategorik, onları dönüşürüyoruz
df = pd.get_dummies(df, columns=['sehir', 'uyelik_tipi', 'gelir_grubu'], drop_first=True, dtype=int)
print("Dönüşüm sonrası yeni sütun isimleri:\n", df.columns.tolist())

# ---------------------------------------------------------
# Soru 8: Veriyi train, validation ve test kümelerine ayırın (Stratify ile)
# ---------------------------------------------------------
print("\n--- 8. VERİYİ BÖLME (TRAIN, VALIDATION, TEST) ---")
X = df.drop('churn', axis=1)
y = df['churn']

# Veriyi önce %70 Train, %30 Geçici Set olarak stratify (sınıf dengesini koruyarak) bölüyoruz
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
# Kalan %30'luk seti ikiye bölüyoruz: %15 Validation, %15 Test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Train verisi boyutu: {X_train.shape}")
print(f"Validation verisi boyutu: {X_val.shape}")
print(f"Test verisi boyutu: {X_test.shape}")

# ---------------------------------------------------------
# Soru 6: Sayısal değişkenlerde ölçekleme (Scaling) uygulayın
# ---------------------------------------------------------
print("\n--- 6. SAYISAL DEĞİŞKENLERDE ÖLÇEKLEME (SCALING) ---")
# Sadece kategorik olmayan sayısal sütunlara scaling yapıyoruz
num_cols = ['yas', 'gelir', 'abonelik_suresi', 'destek_talebi_sayisi']
scaler = StandardScaler()

# Scaler'ı SADECE Train setinden öğreniyoruz (fit), sonra tüm setlere uyguluyoruz (transform)
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
print("Ölçekleme başarıyla tamamlandı.")

# ---------------------------------------------------------
# Soru 9: En az 2 model eğitin (Logistic Regression ve KNN)
# Soru 10: Validation sonuçlarına göre modelleri karşılaştırın
# ---------------------------------------------------------
print("\n--- 9 & 10. MODEL EĞİTİMİ VE VALIDATION KARŞILAŞTIRMASI ---")

# 1. Lojistik Regresyon
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_val_pred_log = log_reg.predict(X_val)
log_val_acc = accuracy_score(y_val, y_val_pred_log)

# 2. KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_val_pred_knn = knn.predict(X_val)
knn_val_acc = accuracy_score(y_val, y_val_pred_knn)

print(f"-> Lojistik Regresyon Validation Accuracy: {log_val_acc:.4f}")
print(f"-> KNN Validation Accuracy: {knn_val_acc:.4f}")

# En iyi modeli if-else ile programatik olarak seçme
if log_val_acc >= knn_val_acc:
    best_model = log_reg
    best_model_name = "Lojistik Regresyon"
else:
    best_model = knn
    best_model_name = "KNN"

print(f"\nSeçilen En İyi Model: ** {best_model_name} **")

# ---------------------------------------------------------
# Soru 11: Seçtiğiniz modeli test verisi üzerinde değerlendirin ve metrikleri yazdırın
# ---------------------------------------------------------
print("\n--- 11. TEST SETİ ÜZERİNDE DEĞERLENDİRME ---")
y_test_pred = best_model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print(f"Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"F1-Score : {f1_score(y_test, y_test_pred, zero_division=0):.4f}")

# ---------------------------------------------------------
# Soru 12: Kodun sonunda kısa bir yorum çıktısı üretin
# ---------------------------------------------------------
print("\n--- 12. SONUÇ YORUMU ---")
print(f"Hangi model daha iyi?: Validation (Doğrulama) setindeki doğruluk oranlarına bakıldığında {best_model_name} modeli daha iyi performans göstermiştir.")
print("Neden?: Kullandığımız veri seti (Yaş, Gelir, Abonelik Süresi vb.) numpy ile tamamen rastgele sayılarla (random) oluşturulduğu için özellikler ile hedef değişken (Churn) arasında belirgin, gerçekçi bir örüntü (pattern) yoktur. Bu nedenle modellerin tahmin başarısı (F1, Recall vb.) düşük kalmıştır. Ancak baştan sona veri işleme ve makine öğrenmesi akışı teknik olarak hatasız bir şekilde tamamlanmıştır.")
print("========================================================\n")
