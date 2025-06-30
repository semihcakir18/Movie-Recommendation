import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
# Veri dosyalarının yolları
ratings_path = 'data\ml-latest-small/ratings.csv'
movies_path = 'data/ml-latest-small/movies.csv'

# Veriyi yükleme
ratings_df = pd.read_csv(ratings_path)
movies_df = pd.read_csv(movies_path)

print("Ratings Veri Seti:")
print(ratings_df.head(),"\n\n\n")
print("Movies Veri Seti:")
print(movies_df.head(),"\n\n\n")

df = pd.merge(ratings_df, movies_df, on='movieId')
print("Birleştirilmiş Veri Seti:")
print(df.head(),"\n\n\n")

print(df.info())

print("Boş değerlerin toplamı: ", df.isnull().sum(),"\n\n\n")

n_users = df['userId'].nunique()
n_movies = df['movieId'].nunique()

print(f'Toplam Benzersiz Kullanıcı Sayısı: {n_users}',"\n\n\n")
print(f'Toplam Benzersiz Film Sayısı: {n_movies}',"\n\n\n")

plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=df)
plt.title('Puanların Dağılımı', fontsize=15)
plt.xlabel('Puan')
plt.ylabel('Puan Sayısı')
plt.show()

movie_rating_counts = df['title'].value_counts()
print("En çok puan alan ilk 20 film:")
print(movie_rating_counts.head(20),"\n\n\n")

user_rating_counts = df['userId'].value_counts()
print("En çok puanlama yapan ilk 20 kullanıcı:")
print(user_rating_counts.head(20))

