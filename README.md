# Makine Öğrenmesi Ara Ödevi - Müşteri Ayrılma (Churn) Tahmini

Bu proje, Türkiye Yapay Zeka Akademisi "Makine Öğrenmesi Ara Ödevi" kapsamında hazırlanmıştır.

## Projenin Amacı
Projenin temel amacı, baştan sona (uçtan uca) bir makine öğrenmesi akışını tek bir Python dosyası içinde pratik etmektir. Proje kapsamında sentetik bir müşteri veri seti üretilmiş, bu veri seti üzerinde eksik veri doldurma, kategorik veri kodlama (One-Hot Encoding) ve ölçekleme (Scaling) gibi veri ön işleme adımları uygulanmıştır. Daha sonra veriler Eğitim (Train), Doğrulama (Validation) ve Test kümelerine ayrılarak Lojistik Regresyon ve K-En Yakın Komşu (KNN) modelleri eğitilmiştir.

## İçerik
- `odev.py`: Tüm veri üretim, ön işleme, model eğitimi ve değerlendirme aşamalarını içeren ana Python dosyasıdır.
- `requirements.txt`: Projenin çalışması için gereken kütüphaneleri barındırır.
- `README.md`: Proje açıklamasını içeren belge.

## Nasıl Çalıştırılır?
1. Repoyu bilgisayarınıza indirin veya klonlayın.
2. Proje dizininde bir terminal açın.
3. Gerekli kütüphaneleri yüklemek için aşağıdaki komutu çalıştırın:
   `pip install -r requirements.txt`
4. Ödevi ve sonuçlarını görmek için Python dosyasını çalıştırın:
   `python odev.py`

## Sonuç Yorumu
Oluşturulan sentetik veri seti üzerinden Lojistik Regresyon ve KNN modelleri eğitilmiş ve Validation setindeki doğruluk (accuracy) skorlarına göre en iyi model seçilmiştir. Seçilen nihai model, daha önce hiç görmediği Test setine sokularak başarı metrikleri (Confusion Matrix, Accuracy, Precision, Recall, F1-Score) hesaplanmıştır. Veriler tamamen rastgele üretildiği için model metrikleri temel seviyede çıkmıştır, ancak makine öğrenmesi akışı (pipeline) tamamen sorunsuz çalışmaktadır.
