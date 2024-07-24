import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load datasets
ratings = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")
movies = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")

# Summary statistics
n_ratings = len(ratings)
n_movies = len(ratings['movieId'].unique())
n_users = len(ratings['userId'].unique())
print(f"Number of ratings: {n_ratings}")
print(f"Number of unique movies: {n_movies}")
print(f"Number of unique users: {n_users}")
print(f"Average ratings per user: {round(n_ratings/n_users, 2)}")
print(f"Average ratings per movie: {round(n_ratings/n_movies, 2)}")

# User rating frequency
user_freq = ratings[['userId', 'movieId']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']
print(user_freq.head())

# Highest and lowest rated movies
mean_rating = ratings.groupby('movieId')[['rating']].mean()
lowest_rated = mean_rating['rating'].idxmin()
highest_rated = mean_rating['rating'].idxmax()
print(movies.loc[movies['movieId'] == lowest_rated])
print(movies.loc[movies['movieId'] == highest_rated])
print(ratings[ratings['movieId'] == highest_rated].shape[0])
print(ratings[ratings['movieId'] == lowest_rated].shape[0])

# Movie statistics
movie_stats = ratings.groupby('movieId')[['rating']].agg(['count', 'mean'])
movie_stats.columns = movie_stats.columns.droplevel()

# Create user-item matrix
def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]
    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

# Find similar movies using KNN
def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    neighbour_ids = []
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(1, k):
        n = neighbour[1].item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    return neighbour_ids

movie_titles = dict(zip(movies['movieId'], movies['title']))

movie_id = 3
similar_ids = find_similar_movies(movie_id, X, k=10)
movie_title = movie_titles[movie_id]

print(f"Since you watched {movie_title}")
for i in similar_ids:
    print(movie_titles[i])

# Recommend movies for a user
def recommend_movies_for_user(user_id, X, user_mapper, movie_mapper, movie_inv_mapper, k=10):
    user_ratings = ratings[ratings['userId'] == user_id]
    if user_ratings.empty:
        print(f"User with ID {user_id} does not exist.")
        return
    highest_rated_movie = user_ratings[user_ratings['rating'] == user_ratings['rating'].max()]['movieId'].iloc[0]
    similar_ids = find_similar_movies(highest_rated_movie, X, k)
    movie_title = movie_titles.get(highest_rated_movie, "Movie not found")
    if movie_title == "Movie not found":
        print(f"Movie with ID {highest_rated_movie} not found.")
        return
    print(f"Since you watched {movie_title}, you might also like:")
    for i in similar_ids:
        print(movie_titles.get(i, "Movie not found"))

# Example usage
recommend_movies_for_user(150, X, user_mapper, movie_mapper, movie_inv_mapper, k=10)
recommend_movies_for_user(2300, X, user_mapper, movie_mapper, movie_inv_mapper, k=10)
