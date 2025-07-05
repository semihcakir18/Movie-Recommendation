![image](https://github.com/user-attachments/assets/bfbd9cde-62d2-4de2-8c5a-351a5c8f9e27)

# Akıllı Hibrit Film Tavsiye Sistemi

## Açıklama
Bu proje, kullanıcılara hem izledikleri filmlere içerik olarak benzeyen (Content-Based Filtering) hem de kendilerine benzer zevklere sahip diğer kullanıcıların sevdikleri (Collaborative Filtering) filmleri öneren hibrit bir tavsiye sistemi sunar. Proje, kullanıcıların kolayca etkileşimde bulunabilmesi için Streamlit ile geliştirilmiş interaktif bir web arayüzüne sahiptir.

## Özellikler
- **İçerik Tabanlı Tavsiye:** Bir filmin türüne (genre) göre ona en çok benzeyen diğer filmleri önerir.
- **İşbirlikçi Filtreleme:** Kullanıcıların geçmişteki puanlama davranışlarına dayanarak kişiselleştirilmiş tavsiyeler sunar.
- **Hibrit Yaklaşım:** İçerik tabanlı modelin ürettiği aday filmleri, işbirlikçi filtreleme ile kullanıcının zevklerine göre yeniden sıralayarak daha isabetli ve çeşitli öneriler sunar.
- **İnteraktif Arayüz:** Streamlit kullanılarak geliştirilmiş, kullanıcı dostu bir web arayüzü.

## Kullanılan Teknolojiler
- **Python:** Ana programlama dili.
- **Pandas & Numpy:** Veri manipülasyonu ve analizi.
- **Scikit-learn:** İçerik tabanlı modelleme için (TF-IDF, Cosine Similarity).
- **Surprise:** İşbirlikçi filtreleme için (SVD algoritması).
- **Streamlit:** İnteraktif web uygulamasını oluşturmak için.

## Kurulum
Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1. **Proje Deposunu Klonlayın:**
   ```bash
   git clone <proje_reposu_adresi>
   cd Akıllı-Hibrit-Film-Tavsiye-Sistemi
   ```

2. **Sanal Ortam Oluşturun ve Aktive Edin:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Gerekli Kütüphaneleri Yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

## Kullanım
Uygulamayı başlatmak için projenin ana dizininde aşağıdaki komutu çalıştırın:

```bash
streamlit run app.py
```

Bu komutu çalıştırdıktan sonra web tarayıcınızda uygulamanın arayüzü açılacaktır. Arayüz üzerinden beğendiğiniz bir filmi ve kendi kullanıcı ID'nizi seçerek hibrit tavsiyeler alabilirsiniz.


