import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

print("Loading ratings...")

ratings = pd.read_csv(
    "data/ratings.csv"
)

# Create mappings
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

print("Users:", num_users)
print("Movies:", num_movies)


class DeepRecommender(nn.Module):

    def __init__(
        self,
        num_users,
        num_movies,
        embedding_dim=32
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_dim
        )

        self.fc1 = nn.Linear(
            embedding_dim * 2,
            64
        )

        self.fc2 = nn.Linear(
            64,
            32
        )

        self.output = nn.Linear(
            32,
            1
        )

        self.relu = nn.ReLU()

    def forward(
        self,
        user_ids,
        movie_ids
    ):

        user_vec = self.user_embedding(
            user_ids
        )

        movie_vec = self.movie_embedding(
            movie_ids
        )

        x = torch.cat(
            [user_vec, movie_vec],
            dim=1
        )

        x = self.relu(
            self.fc1(x)
        )

        x = self.relu(
            self.fc2(x)
        )

        # Prediction between 0 and 5
        rating = torch.sigmoid(
            self.output(x)
        )

        rating = rating * 5

        return rating.squeeze()


model = DeepRecommender(
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

print("\nTraining Deep Learning Model...")

for epoch in range(20):

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

sample_user = torch.tensor([0])
sample_movie = torch.tensor([0])

prediction = model(
    sample_user,
    sample_movie
)

print(
    f"\nPredicted Rating: {prediction.item():.2f}"
)