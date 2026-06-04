import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

print("Loading ratings...")

ratings = pd.read_csv("data/ratings.csv")

# =====================================================
# Create continuous user indices
# =====================================================

user_mapping = {
    user_id: idx
    for idx, user_id in enumerate(
        ratings["userId"].unique()
    )
}

ratings["user_idx"] = ratings["userId"].map(
    user_mapping
)

# =====================================================
# Create continuous movie indices
# =====================================================

movie_mapping = {
    movie_id: idx
    for idx, movie_id in enumerate(
        ratings["movieId"].unique()
    )
}

ratings["movie_idx"] = ratings["movieId"].map(
    movie_mapping
)

num_users = len(user_mapping)
num_movies = len(movie_mapping)

print(f"Users: {num_users}")
print(f"Movies: {num_movies}")

# =====================================================
# Dataset
# =====================================================

class MovieLensDataset(Dataset):

    def __init__(self, dataframe):

        self.users = torch.tensor(
            dataframe["user_idx"].values,
            dtype=torch.long
        )

        self.movies = torch.tensor(
            dataframe["movie_idx"].values,
            dtype=torch.long
        )

        self.ratings = torch.tensor(
            dataframe["rating"].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):

        return (
            self.users[idx],
            self.movies[idx],
            self.ratings[idx]
        )

dataset = MovieLensDataset(ratings)

loader = DataLoader(
    dataset,
    batch_size=1024,
    shuffle=True
)

# =====================================================
# Deep Learning Recommendation Model
# =====================================================

class RecommenderNet(nn.Module):

    def __init__(
        self,
        num_users,
        num_movies,
        embedding_size=50
    ):

        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_size
        )

        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_size
        )

        self.fc1 = nn.Linear(
            embedding_size * 2,
            128
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            128,
            64
        )

        self.fc3 = nn.Linear(
            64,
            1
        )

    def forward(
        self,
        user,
        movie
    ):

        user_vec = self.user_embedding(user)

        movie_vec = self.movie_embedding(movie)

        x = torch.cat(
            [user_vec, movie_vec],
            dim=1
        )

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.relu(x)

        x = self.fc3(x)

        return x.squeeze()

# =====================================================
# Model
# =====================================================

model = RecommenderNet(
    num_users,
    num_movies
)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# =====================================================
# Training
# =====================================================

epochs = 5

print("\nTraining model...\n")

for epoch in range(epochs):

    total_loss = 0

    model.train()

    for users, movies, ratings_batch in loader:

        predictions = model(
            users,
            movies
        )

        loss = loss_fn(
            predictions,
            ratings_batch
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss: {total_loss:.4f}"
    )

print("\nTraining completed!")

# =====================================================
# Sample Prediction
# =====================================================

sample_user = torch.tensor([0])
sample_movie = torch.tensor([0])

model.eval()

with torch.no_grad():

    predicted_rating = model(
        sample_user,
        sample_movie
    )

print(
    f"\nSample Predicted Rating: "
    f"{predicted_rating.item():.2f}"
)