META TALİMAT: Gemini için Kurallar
Başlamadan Önce Oku: Sen, deneyimli bir yazılım mühendisisin. Görevin, aşağıdaki plana harfiyen uyarak modüler, temiz ve iyi belgelenmiş bir Film Tavsiye Sistemi projesi oluşturmak. Her adımı sırayla takip et. Kodları Python dilinde ve belirtilen kütüphaneleri kullanarak yaz. Her kod bloğunu ayrı ve çalıştırılabilir şekilde sun. Fonksiyonların ve sınıfların içine ne işe yaradıklarını açıklayan docstring'ler ekle.
PROJE PLANI: Akıllı Hibrit Film Tavsiye Sistemi
Projenin Amacı: Kullanıcılara hem izledikleri filmlere içerik olarak benzeyen (Content-Based) hem de kendilerine benzer zevklere sahip diğer kullanıcıların sevdikleri (Collaborative Filtering) filmleri öneren hibrit bir sistem oluşturmak. Proje, Streamlit kullanılarak interaktif bir web arayüzü ile sunulacaktır.
Aşama 0: Proje Kurulumu ve Ortam Hazırlığı
Görev: Projenin temel iskeletini, gerekli kütüphaneleri ve veri setini hazırlamak.
1.Dizin Yapısı Oluştur:
Terminalde aşağıdaki komutları çalıştırarak proje klasör yapısını oluştur:
    mkdir film_tavsiye_projesi
    cd film_tavsiye_projesi
    mkdir src data notebooks
    touch src/__init__.py
2.Sanal Ortam (Virtual Environment) Oluştur:
Proje bağımlılıklarını izole etmek için bir sanal ortam oluştur ve aktive et.
    python -m venv venv
    venv\Scripts\activate
Gerekli Kütüphaneleri Belirle ve Kur:
requirements.txt adında bir dosya oluştur ve içine aşağıdaki kütüphaneleri yaz:
    pandas
    numpy
    scikit-learn
    streamlit
    surprise
Ardından bu kütüphaneleri kur:
    pip install -r requirements.txt
Aşama 1: Veri Keşfi ve Ön İşleme (Jupyter Notebook)
Görev: Veriyi anlamak, temizlemek ve modelleme için hazır hale getirmek. Bu adımlar notebooks/01_Veri_Analizi.ipynb dosyasında yapılacak.
Veriyi Yükle: pandas kullanarak data/ml-latest-small/ratings.csv ve data/ml-latest-small/movies.csv dosyalarını yükle.
Veriyi Birleştir: ratings ve movies veri çerçevelerini movieId sütunu üzerinden birleştirerek hangi kullanıcının hangi filme ne puan verdiğini tek bir yerde topla.
Keşifsel Veri Analizi (EDA):
Boş (NaN) değer olup olmadığını kontrol et.
Toplam benzersiz kullanıcı ve film sayısını bul.
Puanların (ratings) dağılımını görselleştir.
En çok puan alan ilk 20 filmi listele.
En çok puanlama yapan ilk 20 kullanıcıyı listele.
Sonuç: Bu analizlerin çıktılarını ve gözlemlerini notebook'a markdown hücreleri olarak ekle.
Aşama 2: İçerik Tabanlı Tavsiye Modeli (Content-Based Filtering)
Görev: Bir filmin türüne (genre) göre ona en çok benzeyen diğer filmleri bulan bir sistem kurmak.
Modül Oluştur: src/content_based_recommender.py adında bir dosya oluştur.
Fonksiyonu Tasarla: get_content_based_recommendations(movie_title, movies_df, top_n=10) adında bir fonksiyon yaz.
Fonksiyonun Adımları:
Özellik Çıkarımı: scikit-learn'ün TfidfVectorizer sınıfını kullanarak genres sütununu vektörlere dönüştür. (Genre'lardaki | karakterini boşlukla değiştirerek kelimeleri ayır).
Benzerlik Hesabı: cosine_similarity fonksiyonunu kullanarak tüm filmler arasındaki benzerlik skorlarını içeren bir matris oluştur.
Tavsiye Üretme:
a. Fonksiyona girdi olarak verilen movie_title'ın index'ini bul.
b. Bu filmin diğer tüm filmlerle olan benzerlik skorlarını al.
c. Skorları büyükten küçüğe sırala.
d. En yüksek skora sahip ilk top_n filmin başlığını (kendisi hariç) döndür.
Aşama 3: İşbirlikçi Filtreleme Modeli (Collaborative Filtering)
Görev: Kullanıcıların puanlama davranışlarına göre, bir kullanıcıya sevebileceği yeni filmler önermek. Bunun için surprise kütüphanesini kullanacağız.
Modül Oluştur: src/collaborative_recommender.py adında bir dosya oluştur.
Model Eğitme Fonksiyonu Tasarla: train_svd_model(ratings_df) adında bir fonksiyon yaz.
surprise.Reader ile puanlama ölçeğini tanımla.
surprise.Dataset.load_from_df ile pandas veri çerçevesini surprise formatına çevir.
Veri setini eğitim setine dönüştür (build_full_trainset).
surprise.SVD algoritmasını başlat ve eğitim seti üzerinde eğit (fit).
Eğitilmiş modeli (algo) ve eğitim setini (trainset) döndür.
Tavsiye Üretme Fonksiyonu Tasarla: get_collaborative_recommendations(user_id, model, trainset, movies_df, top_n=10) adında bir fonksiyon yaz.
Verilen user_id'nin puanlamadığı filmlerin bir listesini oluştur.
Bu listedeki her film için model.predict(user_id, movie_id) kullanarak kullanıcının o filme vereceği tahmini puanı hesapla.
Tahminleri büyükten küçüğe sırala.
En yüksek tahmini puana sahip ilk top_n filmin başlığını döndür.
Aşama 4: Hibrit Tavsiye Sistemi
Görev: İçerik tabanlı ve işbirlikçi filtreleme modellerini birleştirerek daha isabetli ve çeşitli öneriler sunmak.
Modül Oluştur: src/hybrid_recommender.py adında bir dosya oluştur.
Hibrit Fonksiyonu Tasarla: get_hybrid_recommendations(user_id, movie_title, top_n=10) adında bir fonksiyon yaz. Bu fonksiyon, diğer modüllerdeki fonksiyonları çağıracak.
Hibrit Mantığı (Yeniden Sıralama - Re-ranking):
Adım 1 (Aday Üretme): İçerik tabanlı model (get_content_based_recommendations) ile movie_title'a benzeyen 50 filmden oluşan bir aday listesi al.
Adım 2 (Kişiselleştirme): İşbirlikçi filtreleme modelini (model.predict) kullanarak, bu 50 aday filmin her biri için user_id'nin vereceği tahmini puanı hesapla.
Adım 3 (Yeniden Sıralama): 50 filmlik aday listesini, kullanıcının tahmini puanlarına göre büyükten küçüğe sırala.
Adım 4 (Sonuç): Yeniden sıralanmış listenin ilk top_n filmini nihai tavsiye olarak döndür. Bu yöntem, "Bu filme benzeyenler arasından senin en çok seveceklerin şunlar" demiş olur.
Aşama 5: İnteraktif Web Arayüzü (Streamlit)
Görev: Kullanıcıların kolayca etkileşime girebileceği bir arayüz oluşturmak.
Ana Dosyayı Oluştur: Projenin ana dizininde app.py adında bir dosya oluştur.
Arayüz Elemanları:
st.title() ile bir başlık ekle: "Akıllı Hibrit Film Tavsiye Sistemi".
@st.cache_data decorator'ünü kullanarak veriyi ve eğitilmiş modeli sadece bir kere yükleyip bellekte tutan fonksiyonlar yaz. Bu, uygulamanın hızını artırır.
st.selectbox ile kullanıcıların film listesinden bir film seçmesini sağla.
st.selectbox ile kullanıcıların ID listesinden bir kullanıcı seçmesini sağla.
st.button("Tavsiye Getir") ile tavsiye sürecini tetikleyecek bir buton ekle.
İş Akışı:
Butona tıklandığında, seçilen film ve kullanıcı ID'si ile get_hybrid_recommendations fonksiyonunu çağır.
Sonuçları üç sütun halinde göster: st.subheader("İçerik Tabanlı Öneriler"), st.subheader("Size Özel Öneriler (İşbirlikçi)") ve st.subheader("Hibrit Öneriler").
Her bir modelin sonuçlarını ilgili başlığın altında st.write() ile listele.
Aşama 6: Proje Belgeleri (README)
Görev: Projenin ne olduğunu, nasıl kurulduğunu ve nasıl kullanıldığını açıklayan profesyonel bir README.md dosyası oluşturmak.
İçerik:
Proje Başlığı
Açıklama: Projenin amacı ve hibrit yaklaşımın ne olduğu.
Özellikler: İçerik tabanlı, işbirlikçi ve hibrit tavsiye yetenekleri.
Kullanılan Teknolojiler: Python, Pandas, Scikit-learn, Surprise, Streamlit.
Kurulum: git clone, sanal ortam oluşturma, pip install -r requirements.txt adımları.
Kullanım: streamlit run app.py komutu ile uygulamanın nasıl çalıştırılacağı.
Ekran Görüntüsü: Uygulamanın çalışan halinden bir GIF veya resim.