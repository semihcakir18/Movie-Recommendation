import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_content_based_recommendations(movie_title: str, movies_df: pd.DataFrame, top_n: int = 10):
    """
    Bir film başlığına göre içerik tabanlı film tavsiyeleri üretir.

    Args:
        movie_title (str): Tavsiye alınmak istenen filmin başlığı.
        movies_df (pd.DataFrame): Film bilgilerini içeren DataFrame (movieId, title, genres).
        top_n (int, optional): Döndürülecek tavsiye sayısı. Defaults to 10.

    Returns:
        list: Tavsiye edilen film başlıklarının bir listesi.
    """
    # 1. Özellik Çıkarımı (TF-IDF)
    # Genre'lardaki '|' karakterini boşlukla değiştirerek kelimeleri ayır
    movies_df['genres'] = movies_df['genres'].str.replace('|', ' ', regex=False)

    # TF-IDF Vectorizer'ı oluştur
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

    # 2. Benzerlik Hesabı (Kosinüs Benzerliği)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 3. Tavsiye Üretme
    try:
        # Film başlığının index'ini bul
        movie_idx = movies_df[movies_df['title'] == movie_title].index[0]

        # O filmin benzerlik skorlarını al
        sim_scores = list(enumerate(cosine_sim[movie_idx]))

        # Benzerlik skorlarına göre sırala
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # En benzer ilk N filmi al (kendisi hariç)
        sim_scores = sim_scores[1:top_n+1]

        # Film index'lerini al
        movie_indices = [i[0] for i in sim_scores]

        # Film başlıklarını döndür
        return movies_df['title'].iloc[movie_indices].tolist()

    except IndexError:
        return [f"'{movie_title}' filmi veri setinde bulunamadı."]
