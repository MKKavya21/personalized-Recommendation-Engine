import pandas as pd

ratings = pd.read_csv(
    "data/ratings.csv"
)

movies = pd.read_csv(
    "data/movies.csv"
)

movie_stats = ratings.groupby(
    "movieId"
).agg(
    avg_rating=("rating", "mean"),
    rating_count=("rating", "count")
)

popular_movies = movie_stats[
    movie_stats["rating_count"] > 50
]

popular_movies = popular_movies.sort_values(
    by="avg_rating",
    ascending=False
)

recommendations = popular_movies.merge(
    movies,
    on="movieId"
)

print("Top Movies For New Users:\n")

print(
    recommendations[
        ["title", "avg_rating", "rating_count"]
    ].head(10)
)