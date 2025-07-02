import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.trainset import Trainset

def train_svd_model(ratings_df: pd.DataFrame) -> tuple[SVD, Trainset]:
    """
    Verilen puanlama verisiyle bir SVD modelini eğitir.

    Args:
        ratings_df (pd.DataFrame): Kullanıcı, film ve puan bilgilerini içeren DataFrame.

    Returns:
        tuple[SVD, Trainset]: Eğitilmiş SVD modeli ve eğitim seti.
    """
    # Surprise Reader ile puanlama ölçeğini tanımla (1-5 arası)
    reader = Reader(rating_scale=(1, 5))

    # Pandas DataFrame'i Surprise formatına çevir
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

    # Veri setini tam eğitim setine dönüştür
    trainset = data.build_full_trainset()

    # SVD algoritmasını başlat ve eğit
    algo = SVD(n_factors=50, n_epochs=20, random_state=42)
    algo.fit(trainset)

    return algo, trainset

def get_collaborative_recommendations(user_id: int, model: SVD, trainset: Trainset, movies_df: pd.DataFrame, top_n: int = 10) -> list:
    """
    Bir kullanıcı için işbirlikçi filtreleme tabanlı film tavsiyeleri üretir.

    Args:
        user_id (int): Tavsiye yapılacak kullanıcı ID'si.
        model (SVD): Eğitilmiş Surprise SVD modeli.
        trainset (Trainset): Modelin eğitildiği eğitim seti.
        movies_df (pd.DataFrame): Film bilgilerini içeren DataFrame.
        top_n (int, optional): Döndürülecek tavsiye sayısı. Defaults to 10.

    Returns:
        list: Tavsiye edilen film başlıklarının bir listesi.
    """
    # Kullanıcının puanlamadığı filmleri bul
    user_movies = trainset.ur[trainset.to_inner_uid(user_id)]
    rated_movie_iids = [iid for (iid, _) in user_movies]
    all_movie_iids = trainset.all_items()
    unrated_movie_iids = [iid for iid in all_movie_iids if iid not in rated_movie_iids]

    # Puanlanmamış filmler için tahmin yap
    predictions = [model.predict(user_id, trainset.to_raw_iid(iid)) for iid in unrated_movie_iids]

    # Tahminleri puana göre büyükten küçüğe sırala
    predictions.sort(key=lambda x: x.est, reverse=True)

    # En iyi N tavsiyeyi al
    top_predictions = predictions[:top_n]

    # Tahminlerden film ID'lerini çıkar
    top_movie_ids = [pred.iid for pred in top_predictions]

    # Film ID'lerinden film başlıklarını bul
    recommended_movies = movies_df[movies_df['movieId'].isin(top_movie_ids)]['title'].tolist()

    return recommended_movies
