import pandas as pd
from surprise.trainset import Trainset
from surprise import SVD

from src.content_based_recommender import get_content_based_recommendations

def get_hybrid_recommendations(user_id: int, movie_title: str, movies_df: pd.DataFrame, collab_model: SVD, top_n: int = 10) -> list:
    """
    İçerik tabanlı ve işbirlikçi filtrelemeyi birleştiren hibrit bir tavsiye sunar.

    Args:
        user_id (int): Tavsiye yapılacak kullanıcı ID'si.
        movie_title (str): Benzerleri bulunacak referans film başlığı.
        movies_df (pd.DataFrame): Film bilgilerini içeren DataFrame.
        collab_model (SVD): Eğitilmiş işbirlikçi filtreleme modeli.
        top_n (int, optional): Döndürülecek tavsiye sayısı. Defaults to 10.

    Returns:
        list: Hibrit olarak tavsiye edilen film başlıklarının listesi.
    """
    # Adım 1: Aday Üretme (İçerik Tabanlı)
    # Referans filme benzeyen 50 aday film al
    content_recs = get_content_based_recommendations(movie_title, movies_df.copy(), top_n=50)
    if not content_recs or isinstance(content_recs[0], str) and 'bulunamadı' in content_recs[0]:
        return content_recs # Hata mesajı veya boş liste döndür

    # Aday filmlerin ID'lerini bul
    candidate_movies = movies_df[movies_df['title'].isin(content_recs)]

    # Adım 2: Kişiselleştirme ve Yeniden Sıralama (İşbirlikçi)
    # Aday filmler için kullanıcının tahmini puanlarını hesapla
    candidate_movies['est_rating'] = candidate_movies['movieId'].apply(
        lambda x: collab_model.predict(user_id, x).est
    )

    # Adım 3: Yeniden Sıralama
    # Tahmini puana göre büyükten küçüğe sırala
    reranked_movies = candidate_movies.sort_values('est_rating', ascending=False)

    # Adım 4: Sonuç
    # En iyi N filmin başlığını döndür
    return reranked_movies.head(top_n)['title'].tolist()
