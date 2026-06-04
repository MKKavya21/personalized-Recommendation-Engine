import pandas as pd
import torch
import torch.nn as nn

print("Loading ratings...")

ratings = pd.read_csv("data/ratings.csv")

user_mapping = {
    uid: idx
    for idx, uid in enumerate(
        ratings["userId"].unique()
    )
}

movie_mapping = {
    mid: idx
    for idx, mid in enumerate(
        ratings["movieId"].unique()
    )
}

num_users = len(user_mapping)
num_movies = len(movie_mapping)

print("Users:", num_users)
print("Movies:", num_movies)

class TwoTowerModel(nn.Module):

    def __init__(
        self,
        num_users,
        num_movies,
        embedding_dim=32
    ):

        super().__init__()

        self.user_tower = nn.Embedding(
            num_users,
            embedding_dim
        )

        self.movie_tower = nn.Embedding(
            num_movies,
            embedding_dim
        )

    def forward(
        self,
        user_ids,
        movie_ids
    ):

        user_vec = self.user_tower(
            user_ids
        )

        movie_vec = self.movie_tower(
            movie_ids
        )

        similarity = (
            user_vec * movie_vec
        ).sum(dim=1)

        return similarity

model = TwoTowerModel(
    num_users,
    num_movies
)

print("\nTwo-Tower Model Created Successfully!")

sample_user = torch.tensor([0])
sample_movie = torch.tensor([0])

score = model(
    sample_user,
    sample_movie
)

print(
    f"\nSimilarity Score: {score.item():.4f}"
)