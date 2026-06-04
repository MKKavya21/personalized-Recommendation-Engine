import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

print("Loading ratings...")

ratings = pd.read_csv(
    "data/ratings.csv"
)

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

ratings["user_idx"] = ratings[
    "userId"
].map(user_mapping)

ratings["movie_idx"] = ratings[
    "movieId"
].map(movie_mapping)

num_users = len(user_mapping)
num_movies = len(movie_mapping)

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

        score = (
            user_vec * movie_vec
        ).sum(dim=1)

        return score


model = TwoTowerModel(
    num_users,
    num_movies
)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_fn = nn.MSELoss()

users = torch.tensor(
    ratings["user_idx"].values,
    dtype=torch.long
)

movies = torch.tensor(
    ratings["movie_idx"].values,
    dtype=torch.long
)

targets = torch.tensor(
    ratings["rating"].values,
    dtype=torch.float32
)

print("\nTraining Two-Tower Model...")

for epoch in range(3):

    optimizer.zero_grad()

    predictions = model(
        users,
        movies
    )

    loss = loss_fn(
        predictions,
        targets
    )

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1} Loss: {loss.item():.4f}"
    )

print("\nTraining Complete!")