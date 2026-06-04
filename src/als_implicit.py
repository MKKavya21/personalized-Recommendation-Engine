import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

print("Loading ratings...")

ratings = pd.read_csv("data/ratings.csv")

print("Creating user-item matrix...")

user_item = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating",
    fill_value=0
)

print("Converting to sparse matrix...")

sparse_matrix = csr_matrix(
    user_item.values
)

print("Training ALS model...")

model = AlternatingLeastSquares(
    factors=50,
    regularization=0.01,
    iterations=20
)

model.fit(sparse_matrix)

print("ALS model trained successfully!")