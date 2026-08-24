# 📞 Müşteri Ayrılma Tahmini (Churn Prediction) & Makine Öğrenmesi Akışı

<br>

## 📌 Proje Özeti
Bu proje, makine öğrenmesinde **Uçtan Uca (End-to-End)** modelleme akışını pratik etmek amacıyla geliştirilmiştir. Bir müşterinin şirketi terk etme (churn) ihtimalini tahminlemek üzerine kurulan bu çalışmada; veri üretimi, eksik değer yönetimi, özellik mühendisliği (feature engineering) ve model seçimi adımları tek bir akış (pipeline) mantığıyla entegre edilmiştir.

<br>

## 🎯 İş Problemi ve Yaklaşım
Gerçek dünyada veriler hiçbir zaman temiz gelmez. Bu projede de veri bilimi pipeline'ının en kritik adımlarını simüle etmek hedeflenmiştir:

* Gerçeğe uygun (Yaş, Gelir, Abonelik Süresi, Destek Talebi) müşteri profilleri oluşturuldu.
* Eksik veriler analiz edilerek **Medyan (Ortanca)** yöntemiyle uygun şekilde dolduruldu (Imputation).
* Çıplak veriyi zenginleştirmek için iş kurallarına dayalı **Yeni Öznitelik (Feature Engineering)** üretildi (Örn: `gelir_grubu`).
* Sınıflar arası dengesizliği korumak adına veri seti **Stratified Split** ile Train (%70), Validation (%15) ve Test (%15) olarak profesyonelce üçe bölündü.
* Mesafe tabanlı ve doğrusal algoritmaların sapma (bias) yapmaması için **StandardScaler** ile sadece Train setinden öğrenilen ölçekler diğer setlere uygulandı (Data Leakage önlendi).

<br>

## 📈 Elde Edilen Temel Sonuçlar
* **Karşılaştırmalı Modeller:** Logistic Regression ve K-Nearest Neighbors (KNN)
* **Model Seçimi:** Her iki modelin **Validation (Doğrulama)** setindeki Accuracy (Doğruluk) oranları karşılaştırıldı ve kod tarafından otomatik olarak en yüksek performanslı model (Logistic Regression) nihai model seçildi.
* **Değerlendirme:** En iyi model hiç görmediği **Test Seti** üzerinde çalıştırılarak; `Confusion Matrix`, `Accuracy`, `Precision`, `Recall` ve `F1-Score` metrikleri raporlandı.

<br>

## 🛠️ Kullanılan Teknolojiler
* **Programlama Dili:** Python
* **Veri Manipülasyonu & Üretimi:** Pandas, NumPy
* **Makine Öğrenmesi:** Scikit-Learn (LogisticRegression, KNeighborsClassifier, StandardScaler)
* **Model Değerlendirme:** `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`

<br>

## 📂 Klasör Yapısı
* **`odev.py`** : Veri üretiminden başlayıp makine öğrenmesi metriklerinin raporlanmasına kadar giden ana pipeline (çalışma) dosyasıdır. Terminalden `python odev.py` yazılarak tek tuşla çalıştırılabilir.
* **`requirements.txt`** : Projenin bağımlılıklarını barındırır.
* **`README.md`** : Proje dokümantasyonudur.

<br>

## 👨‍💻 Geliştirici
**Bereket İŞ**  
Veri Bilimi ve Yapay Zeka alanında çalışmalar yapıyorum. Kodun mantığı ve geliştirilmesiyle ilgili tüm iletişim kanallarından bana ulaşabilirsiniz.
