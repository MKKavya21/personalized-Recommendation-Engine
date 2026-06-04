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

sparse_matrix = csr_matrix(
    user_item.values
)

print("Training ALS...")

model = AlternatingLeastSquares(
    factors=50,
    iterations=20,
    regularization=0.1
)

model.fit(sparse_matrix)

print("ALS Training Completed!")

print("\nModel Summary:")
print(f"Users: {user_item.shape[0]}")
print(f"Movies: {user_item.shape[1]}")
print(f"Factors: 50")
user_id = 0

ids, scores = model.recommend(
    user_id,
    sparse_matrix[user_id],
    N=10
)

print("\nTop Recommendations:")

for movie_id, score in zip(ids, scores):
    print(movie_id, score)