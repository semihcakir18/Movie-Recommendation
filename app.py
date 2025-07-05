import streamlit as st
import pandas as pd

# Proje modüllerini import et
from src.content_based_recommender import get_content_based_recommendations
from src.collaborative_recommender import train_svd_model, get_collaborative_recommendations
from src.hybrid_recommender import get_hybrid_recommendations

# --- Veri Yükleme ve Model Eğitimi (Cache ile) ---
@st.cache_data
def load_data():
    """Veri setlerini yükler ve birleştirir."""
    ratings_df = pd.read_csv('data/ml-latest-small/ratings.csv')
    movies_df = pd.read_csv('data/ml-latest-small/movies.csv')
    df = pd.merge(ratings_df, movies_df, on='movieId')
    return df, movies_df, ratings_df

@st.cache_data
def train_models(ratings_df):
    """İşbirlikçi filtreleme modelini eğitir."""
    collab_model, trainset = train_svd_model(ratings_df)
    return collab_model, trainset

# Veriyi ve modelleri yükle
df, movies_df, ratings_df = load_data()
collab_model, trainset = train_models(ratings_df)

# --- Streamlit Arayüzü ---
st.title("🎬 Akıllı Hibrit Film Tavsiye Sistemi")
st.write("Bu uygulama, size hem seçtiğiniz filme benzeyen hem de zevklerinize uyan filmleri önerir.")

# --- Kullanıcı Girdileri ---
st.sidebar.header("Seçimlerinizi Yapın")

# Film seçimi
all_movies = sorted(movies_df['title'].unique())
selected_movie = st.sidebar.selectbox(
    "Beğendiğiniz bir filmi seçin:",
    all_movies
)

# Kullanıcı ID seçimi
all_users = sorted(df['userId'].unique())
selected_user_id = st.sidebar.selectbox(
    "Kullanıcı ID'nizi seçin:",
    all_users
)

# Tavsiye butonu
if st.sidebar.button("Tavsiye Getir", key="recommend_button"):
    st.header(f"'{selected_movie}' için Öneriler")

    # --- Tavsiyeleri Hesapla ve Göster ---
    col1, col2, col3 = st.columns(3)

    # 1. İçerik Tabanlı Öneriler
    with col1:
        st.subheader("Bu Filme Benzeyenler")
        with st.spinner('Benzer filmler bulunuyor...'):
            content_recs = get_content_based_recommendations(selected_movie, movies_df.copy(), top_n=10)
            if content_recs and 'bulunamadı' not in content_recs[0]:
                for i, movie in enumerate(content_recs):
                    st.write(f"{i+1}. {movie}")
            else:
                st.warning(content_recs[0])

    # 2. İşbirlikçi Filtreleme Önerileri
    with col2:
        st.subheader("Size Özel Öneriler")
        with st.spinner('Zevklerinize göre filmler aranıyor...'):
            collab_recs = get_collaborative_recommendations(selected_user_id, collab_model, trainset, movies_df, top_n=10)
            if collab_recs:
                for i, movie in enumerate(collab_recs):
                    st.write(f"{i+1}. {movie}")
            else:
                st.info("Bu kullanıcı için işbirlikçi öneri bulunamadı.")

    # 3. Hibrit Öneriler
    with col3:
        st.subheader("⭐ Hibrit Öneriler")
        with st.spinner('Hibrit öneriler oluşturuluyor...'):
            hybrid_recs = get_hybrid_recommendations(selected_user_id, selected_movie, movies_df.copy(), collab_model, top_n=10)
            if hybrid_recs and 'bulunamadı' not in hybrid_recs[0]:
                for i, movie in enumerate(hybrid_recs):
                    st.success(f"**{i+1}. {movie}**")
            else:
                st.error(hybrid_recs[0])

    # --- Kullanıcının Geçmiş Yorumları ---
    st.divider()

    st.subheader("Önceki Yorumladığınız Filmler")

    user_rated_movies = df[df['userId'] == selected_user_id].sort_values(by="rating", ascending=False)

    if not user_rated_movies.empty:
        with st.expander("Listeyi görmek için tıklayın..."):
            for _, row in user_rated_movies.iterrows():
                st.write(f"- **{row['title']}** (Verdiğiniz Puan: {row['rating']})")
    else:
        st.info("Bu kullanıcının geçmişte yorumladığı bir film bulunamadı.")
else:
    st.info("Lütfen sol menüden bir film ve kullanıcı seçip 'Tavsiye Getir' butonuna tıklayın.")
